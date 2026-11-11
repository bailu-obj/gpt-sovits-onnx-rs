use hound::{WavReader, WavSpec};
use log::{debug, info};
use ndarray::{
    Array1, Array2, ArrayBase, ArrayD, ArrayView2, Axis, IxDyn, OwnedRepr, concatenate, s,
};
use ort::{
    inputs,
    session::{Session, SessionInputValue, SessionOutputs},
    value::TensorRef,
};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::borrow::Cow;
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;
use std::{fs::File, path::Path};
use tokio::task::block_in_place;

mod cpu_info;
mod error;
mod infer_params;
mod logits_sampler;
mod onnx_builder;
pub use onnx_builder::{OnnxSessionOptions, configure_onnx_sessions};
#[cfg(target_os = "android")]
mod qnn_pipeline;
mod postprocess;
mod preprocessor;
mod sv;
mod t2s_cache;
mod token_stream;
pub use token_stream::TokenStreamOptions;
mod vocoder;

use onnx_builder::create_onnx_cpu_session;
pub use postprocess::{PostprocessParams, audio_postprocess, recovery_order};
pub use preprocessor::LangId;
pub use preprocessor::lang::Lang;
pub use preprocessor::{TextProcessor, bert, en, phoneme_finalize, text_normalize, zh};

use logits_sampler::Sampler;
use preprocessor::{bert::BertModel, en::g2p_en::G2pEn, zh::g2pw::G2PW};
use t2s_cache::{CACHE_RESERVE, CacheLayout, LayerCache, SharedKvCache};
use vocoder::Vocoder;

pub use error::GSVError;
pub use infer_params::InferParams;
pub use logits_sampler::{SamplingParams, SamplingParamsBuilder};

use crate::{onnx_builder::BIG_CORES, sv::SvModel};

const T2S_DECODER_EOS: i64 = 1024;
const VOCAB_SIZE: usize = 1025;
const DEFAULT_NUM_LAYERS: usize = 24;
const MAX_DECODER_STEPS: usize = 1501;
/// Product of VITS `upsample_rates` for v2 / v2Pro / v2ProPlus (`[10, 8, 2, 2, 2]`).
const VITS_UPSAMPLE_RATE: usize = 640;
const VITS_FRAMES_PER_TOKEN: usize = 2;

/// Expected VITS waveform length before postprocess, matching Python
/// `pred_semantic.shape[0] * 2 * upsample_rate`.
fn vits_output_samples(semantic_token_count: usize) -> usize {
    semantic_token_count * VITS_FRAMES_PER_TOKEN * VITS_UPSAMPLE_RATE
}

fn semantic_at_stop(sequence: &[i64], prefix_len: usize, step: usize) -> (&[i64], Vec<i64>) {
    let legacy = logits_sampler::extract_semantic_tokens(sequence, prefix_len, step);
    // Keep the legacy output length and preserve tokens already sent to VITS.
    let start = prefix_len + 1;
    (&sequence[start..start + legacy.len()], legacy)
}

static STANDALONE_TOKIO_RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

fn t2s_num_layers_from_session(session: &Session) -> usize {
    let n = session
        .inputs()
        .iter()
        .filter(|input| input.name().starts_with("ik_cache_"))
        .count();
    if n == 0 { DEFAULT_NUM_LAYERS } else { n }
}

fn t2s_kv_io_names(num_layers: usize) -> (Vec<String>, Vec<String>, Vec<String>, Vec<String>) {
    let ik = (0..num_layers).map(|i| format!("ik_cache_{}", i)).collect();
    let iv = (0..num_layers).map(|i| format!("iv_cache_{}", i)).collect();
    let k = (0..num_layers).map(|i| format!("k_cache_{}", i)).collect();
    let v = (0..num_layers).map(|i| format!("v_cache_{}", i)).collect();
    (ik, iv, k, v)
}

fn take_kv_pair(
    output: &mut SessionOutputs<'_>,
    key: &str,
    value: &str,
) -> Result<LayerCache, GSVError> {
    let key = output
        .remove(key)
        .ok_or_else(|| GSVError::from(format!("Missing T2S decoder output {key}")))?;
    let value = output
        .remove(value)
        .ok_or_else(|| GSVError::from(format!("Missing T2S decoder output {value}")))?;
    Ok((key, value))
}

#[derive(Clone)]
pub struct ReferenceData {
    ref_seq: Array2<i64>,
    ref_bert: Array2<f32>,
    vits: VitsReference,
    ssl_shape: Vec<usize>,
    prompts: ArrayD<i64>,
    sv_emb: Option<ArrayD<f32>>,
}

#[derive(Clone)]
enum VitsReference {
    Audio(Array2<f32>),
    Features(ArrayD<f32>),
}

pub struct TTSModel {
    text_processor: TextProcessor,
    sovits: Option<Vocoder>,
    sovits_reference: Option<Session>,
    ssl: Session,
    t2s_encoder: Session,
    t2s_fs_decoder: Session,
    t2s_s_decoder: Session,
    t2s_shared_layout: Option<CacheLayout>,
    sv: Option<SvModel>,
    ref_data: Option<Arc<ReferenceData>>,
    t2s_dec_ik: Vec<String>,
    t2s_dec_iv: Vec<String>,
    t2s_k_cache_out: Vec<String>,
    t2s_v_cache_out: Vec<String>,
    num_layers: usize,
    output_spec: WavSpec,
}

impl TTSModel {
    /// Capture real text and semantic inputs for calibrating a QNN VITS generator.
    pub fn qnn_calibration_inputs(
        &mut self,
        text: &str,
        sampling: SamplingParams,
        lang: LangId,
    ) -> Result<Vec<serde_json::Value>, GSVError> {
        let reference = self
            .ref_data
            .as_ref()
            .ok_or_else(|| GSVError::from("Reference data not initialized"))?
            .clone();
        let VitsReference::Features(ge) = &reference.vits else {
            return Err(GSVError::from("QNN calibration requires a split VITS reference graph"));
        };
        let mut inputs = Vec::new();
        for (_, phones, bert) in self.text_processor.get_phone_and_bert(text, lang)? {
            let tokens = self.synthesize_semantic(&bert, &phones, &reference, sampling)?;
            inputs.push(serde_json::json!({
                "text_seq": phones,
                "pred_semantic": tokens.iter().copied().collect::<Vec<_>>(),
                "ge": ge.iter().copied().collect::<Vec<_>>(),
                "ge_shape": ge.shape(),
            }));
        }
        if inputs.is_empty() {
            return Err(GSVError::from("No QNN calibration inputs generated"));
        }
        Ok(inputs)
    }

    /// Load a TTS model with the VITS graph for the intended inference mode.
    /// Use an ordinary graph for [`Self::synthesize_sync`] or a graph with
    /// `stream_start_frame` for [`Self::synthesize_token_chunks_sync`].
    /// A decoder with cached `ge` loads `vits_ref.onnx` from the same directory.
    /// The BERT, Chinese G2P, and English G2P paths are optional.
    pub fn new<P: AsRef<Path>>(
        sovits_path: P,
        ssl_path: P,
        t2s_encoder_path: P,
        t2s_fs_decoder_path: P,
        t2s_s_decoder_path: P,
        bert_path: Option<P>,
        g2pw_path: Option<P>,
        g2p_en_path: Option<P>,
        sv_path: Option<P>,
    ) -> Result<Self, GSVError> {
        info!("Initializing TTSModel with ONNX sessions");
        info!("use cpu cores: {:?}", BIG_CORES.as_slice());

        // let create_session_with_profiling = |path: P| {
        //     Session::builder()?
        //         .with_execution_providers([CPUExecutionProvider::default()
        //             .with_arena_allocator(true)
        //             .build()])?
        //         .with_optimization_level(GraphOptimizationLevel::Level3)?
        //         .with_intra_threads(8)?
        //         .with_memory_pattern(true)?
        //         .with_prepacking(true)?
        //         .with_config_entry("session.enable_mem_reuse", "1")?
        //         .with_independent_thread_pool()?
        //         .with_intra_op_spinning(true)?
        //         // .with_profiling("t2sd")?
        //         .commit_from_file(path)
        // };

        let output_spec = WavSpec {
            channels: 1,
            sample_rate: 32000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let t2s_s_decoder = create_onnx_cpu_session(t2s_s_decoder_path)?;
        let t2s_shared_layout = CacheLayout::from_decoder(&t2s_s_decoder)?;
        let num_layers = t2s_num_layers_from_session(&t2s_s_decoder);
        info!("T2S decoder num_layers: {}", num_layers);

        let (t2s_dec_ik, t2s_dec_iv, t2s_k_cache_out, t2s_v_cache_out) =
            t2s_kv_io_names(num_layers);

        let sovits_path = sovits_path.as_ref();
        let sovits = create_onnx_cpu_session(sovits_path)?;
        let sovits_reference = if sovits.inputs().iter().any(|input| input.name() == "ge") {
            if sv_path.is_none() {
                return Err(GSVError::from("Split SoVITS requires an SV model"));
            }
            Some(create_onnx_cpu_session(
                sovits_path.with_file_name("vits_ref.onnx"),
            )?)
        } else {
            None
        };

        Ok(TTSModel {
            text_processor: TextProcessor::new(
                G2PW::new(g2pw_path)?,
                G2pEn::new(g2p_en_path)?,
                BertModel::new(bert_path)?,
            )?,
            sovits: Some(Vocoder::new(sovits)),
            sovits_reference,
            ssl: create_onnx_cpu_session(ssl_path)?,
            t2s_encoder: create_onnx_cpu_session(t2s_encoder_path)?,
            t2s_fs_decoder: create_onnx_cpu_session(t2s_fs_decoder_path)?,
            t2s_s_decoder,
            t2s_shared_layout,
            sv: match sv_path {
                Some(p) => Some(SvModel::new(create_onnx_cpu_session(p)?)),
                None => None,
            },
            ref_data: None,
            t2s_dec_ik,
            t2s_dec_iv,
            t2s_k_cache_out,
            t2s_v_cache_out,
            num_layers,
            output_spec,
        })
    }

    /// Load the standard ONNX files from the supplied model directory.
    pub fn from_bundle(root: &Path) -> Result<Self, GSVError> {
        let sv = root.join("sv.onnx");
        Self::new(
            root.join("vits.onnx"),
            root.join("ssl.onnx"),
            root.join("t2s_encoder.onnx"),
            root.join("t2s_fs_decoder.onnx"),
            root.join("t2s_s_decoder.onnx"),
            Some(root.join("bert.onnx")),
            Some(root.join("g2pW.onnx")),
            Some(root.join("g2p_en")),
            sv.is_file().then_some(sv),
        )
    }

    fn run_async_in_context<F, T>(fut: F) -> Result<T, GSVError>
    where
        F: std::future::Future<Output = Result<T, GSVError>>,
    {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => block_in_place(|| handle.block_on(fut)),
            Err(_) => {
                let rt = STANDALONE_TOKIO_RT.get_or_init(|| {
                    tokio::runtime::Runtime::new().expect("failed to create tokio runtime")
                });
                rt.block_on(fut)
            }
        }
    }

    /// run reference with async fn
    ///
    /// `reference_audio_path` shall be input wav(16khz) path
    ///
    /// `ref_text` is input ref text
    ///
    /// `lang_id` can be LangId::Auto(Mandarin) or LangId::AutoYue（cantonese）
    ///
    pub async fn process_reference<P: AsRef<Path>>(
        &mut self,
        reference_audio_path: P,
        ref_text: &str,
        lang_id: LangId,
    ) -> Result<(), GSVError> {
        info!("Processing reference audio and text: {}", ref_text);
        let ref_text = ensure_punctuation(ref_text);
        let (ref_norm, ref_phone_ids, ref_bert) = self
            .text_processor
            .get_phone_and_bert_whole(&ref_text, lang_id)?;
        let ref_seq = Array2::from_shape_vec((1, ref_phone_ids.len()), ref_phone_ids)?;
        debug!("Reference norm text: {}", ref_norm);
        let (ref_audio_16k, ref_audio_16k_raw, ref_audio_32k) =
            read_and_resample_audio(&reference_audio_path)?;
        let ssl_content = self.process_ssl(&ref_audio_16k)?;
        let ssl_shape = ssl_content.shape().to_vec();
        let prompts = {
            let output = self.t2s_encoder.run(inputs![
                "ssl_content" => TensorRef::from_array_view(&ssl_content)?
            ])?;
            output["prompts"].try_extract_array::<i64>()?.into_owned()
        };

        let sv_emb = match &mut self.sv {
            Some(sv_model) => {
                let row = ref_audio_16k_raw.row(0);
                let audio_slice = row
                    .as_slice()
                    .ok_or_else(|| GSVError::from("reference audio row must be contiguous"))?;
                let sv_emb = sv_model.infer(audio_slice)?;
                debug!("SV embedding shape: {:?}", sv_emb.shape());
                Some(sv_emb)
            }
            None => {
                debug!("SV model not provided, skipping SV embedding extraction");
                None
            }
        };

        let vits = if let Some(reference) = &mut self.sovits_reference {
            let sv_emb = sv_emb
                .as_ref()
                .ok_or_else(|| GSVError::from("Missing SV embedding for SoVITS reference"))?;
            let outputs = reference.run(inputs![
                "ref_audio" => TensorRef::from_array_view(&ref_audio_32k)?,
                "sv_emb" => TensorRef::from_array_view(sv_emb)?,
            ])?;
            let ge = outputs
                .get("ge")
                .ok_or_else(|| GSVError::from("Missing SoVITS reference output ge"))?
                .try_extract_array::<f32>()?;
            if ge.is_empty() || !ge.iter().all(|value| value.is_finite()) {
                return Err(GSVError::from("Invalid SoVITS reference features"));
            }
            VitsReference::Features(ge.into_owned())
        } else {
            VitsReference::Audio(ref_audio_32k)
        };

        self.ref_data = Some(Arc::new(ReferenceData {
            ref_seq,
            ref_bert,
            sv_emb,
            vits,
            ssl_shape,
            prompts,
        }));

        Ok(())
    }

    fn process_ssl(
        &mut self,
        ref_audio_16k: &Array2<f32>,
    ) -> Result<ArrayBase<OwnedRepr<f32>, IxDyn>, GSVError> {
        let time = SystemTime::now();
        let ssl_output = self
            .ssl
            .run(inputs!["ref_audio_16k" => TensorRef::from_array_view(ref_audio_16k).unwrap()])?;
        debug!("SSL processing time: {:?}", time.elapsed()?);
        Ok(ssl_output["ssl_content"]
            .try_extract_array::<f32>()?
            .into_owned())
    }

    /// run reference
    ///
    /// `reference_audio_path` shall be input wav(16khz) path
    ///
    /// `ref_text` is input ref text
    ///
    /// `lang_id` can be LangId::Auto(Mandarin) or LangId::AutoYue（cantonese）
    ///
    pub fn process_reference_sync<P: AsRef<Path>>(
        &mut self,
        reference_audio_path: P,
        ref_text: &str,
        lang_id: LangId,
    ) -> Result<(), GSVError> {
        Self::run_async_in_context(self.process_reference(reference_audio_path, ref_text, lang_id))
    }

    /// Returns the cached SV embedding after [`Self::process_reference`].
    pub fn sv_embedding(&self) -> Option<ArrayD<f32>> {
        self.ref_data.as_ref().and_then(|data| data.sv_emb.clone())
    }

    /// Cached reference tensors after [`Self::process_reference`].
    pub fn reference_data(&self) -> Option<Arc<ReferenceData>> {
        self.ref_data.clone()
    }

    /// Phoneme ids and BERT features for synthesis text.
    pub fn phones_and_bert_for_text(
        &mut self,
        text: &str,
        lang_id: LangId,
    ) -> Result<Vec<(String, Vec<i64>, Array2<f32>)>, GSVError> {
        let text = ensure_punctuation(text);
        Ok(self.text_processor.get_phone_and_bert(&text, lang_id)?)
    }

    /// Debug snapshot of T2S inputs and first-stage logits (for parity checks).
    pub fn debug_t2s_snapshot(
        &mut self,
        text: &str,
        lang_id: LangId,
    ) -> Result<serde_json::Value, GSVError> {
        let phones = self.phones_and_bert_for_text(text, lang_id)?;
        let ref_data = self
            .ref_data
            .as_ref()
            .ok_or_else(|| GSVError::from("Reference data not initialized"))?
            .clone();
        let text_seq: Vec<i64> = phones.iter().fold(Vec::new(), |mut seq, p| {
            seq.extend(p.1.clone());
            seq
        });
        let text_bert_parts: Vec<_> = phones.iter().map(|p| p.2.view()).collect();
        let text_bert = concatenate(Axis(0), &text_bert_parts)?;
        let text_seq_arr = Array2::from_shape_vec((1, text_seq.len()), text_seq)?;
        let x = concatenate(Axis(1), &[ref_data.ref_seq.view(), text_seq_arr.view()])?;
        let bert = concatenate(Axis(1), &[ref_data.ref_bert.t(), text_bert.t()])?;
        let bert = bert.insert_axis(Axis(0)).to_owned();

        let prompts = &ref_data.prompts;

        let fs_decoder_output = self.t2s_fs_decoder.run(inputs![
            "x" => TensorRef::from_array_view(&x.as_standard_layout())?,
            "prompts" => TensorRef::from_array_view(prompts.view())?,
            "bert" => TensorRef::from_array_view(&bert.as_standard_layout())?,
        ])?;
        let logits = fs_decoder_output["logits"]
            .try_extract_array::<f32>()?
            .into_owned();
        let logits_vec = logits.as_slice().unwrap();
        let masked = &logits_vec[..logits_vec.len().saturating_sub(1)];
        let argmax = logits_sampler::argmax(masked);

        Ok(serde_json::json!({
            "ref_seq_len": ref_data.ref_seq.shape()[1],
            "ref_seq_head": ref_data.ref_seq.slice(s![0, ..8.min(ref_data.ref_seq.shape()[1])]).iter().copied().collect::<Vec<_>>(),
            "x_len": x.shape()[1],
            "x_head": x.slice(s![0, ..8.min(x.shape()[1])]).iter().copied().collect::<Vec<_>>(),
            "bert_shape": bert.shape().to_vec(),
            "bert_mean": bert.mean().unwrap_or(0.0),
            "prompts_len": prompts.shape()[1],
            "prompts_head": prompts.slice(s![0, ..5]).iter().copied().collect::<Vec<_>>(),
            "ssl_shape": ref_data.ssl_shape,
            "fs_argmax_masked": argmax,
        }))
    }

    /// Run either the legacy growing decoder or the head-major shared-cache export.
    fn run_t2s_s_decoder_loop(
        &mut self,
        sampler: &mut Sampler,
        sampling_param: SamplingParams,
        mut y_vec: Vec<i64>,
        mut caches: Vec<LayerCache>,
        prefix_len: usize,
        mut progress: impl FnMut(&[i64], bool) -> bool,
    ) -> Result<Option<ArrayBase<OwnedRepr<i64>, IxDyn>>, GSVError> {
        let mut idx = 0;
        y_vec.reserve(2048);

        let y_len_arr = Array1::from_elem(1, prefix_len as i64);
        let mut idx_arr = Array1::from_elem(1, 0i64);
        let mut logits_scratch = Vec::with_capacity(VOCAB_SIZE + 2);
        let mut prefill_length = 0;
        let mut shared = if let Some(layout) = self.t2s_shared_layout {
            let key = &caches
                .first()
                .ok_or_else(|| GSVError::from("missing T2S prefill caches"))?
                .0;
            prefill_length = layout.prefill_length(key)?;
            let cache = SharedKvCache::new(
                &self.t2s_s_decoder,
                layout,
                &caches,
                prefill_length,
                prefill_length + CACHE_RESERVE,
                VOCAB_SIZE,
            )?;
            caches.clear();
            Some(cache)
        } else {
            None
        };

        loop {
            let old_capacity = shared.as_ref().map(|cache| cache.capacity);
            let filled = prefill_length + idx + 1;
            let mut output = if let Some(cache) = &mut shared {
                cache.run(
                    &mut self.t2s_s_decoder,
                    *y_vec.last().unwrap(),
                    (prefix_len + idx) as i64,
                    filled - 1,
                )?
            } else {
                idx_arr[0] = idx as i64;
                let iy =
                    TensorRef::from_array_view(ArrayView2::from_shape((1, y_vec.len()), &y_vec)?)?;
                let mut run_inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
                    Vec::with_capacity(3 + 2 * caches.len());
                run_inputs.extend([
                    (Cow::Borrowed("iy"), iy.into()),
                    (
                        Cow::Borrowed("y_len"),
                        TensorRef::from_array_view(y_len_arr.view())?.into(),
                    ),
                    (
                        Cow::Borrowed("idx"),
                        TensorRef::from_array_view(idx_arr.view())?.into(),
                    ),
                ]);
                for (i, (key, value)) in caches.iter().enumerate() {
                    run_inputs.push((Cow::Borrowed(self.t2s_dec_ik[i].as_str()), key.into()));
                    run_inputs.push((Cow::Borrowed(self.t2s_dec_iv[i].as_str()), value.into()));
                }
                self.t2s_s_decoder.run(run_inputs)?
            };

            {
                let logits = output["logits"].try_extract_array::<f32>()?;
                let src = logits.as_slice().unwrap();
                logits_scratch.clear();
                if idx < 11 {
                    let keep = src.len().saturating_sub(1);
                    logits_scratch.extend_from_slice(&src[..keep]);
                } else {
                    logits_scratch.extend_from_slice(src);
                }
            }

            let sampled = sampler.sample(&mut logits_scratch, &y_vec, &sampling_param);
            y_vec.push(sampled);

            let argmax = logits_sampler::argmax(&logits_scratch);

            if idx + 1 >= MAX_DECODER_STEPS
                || sampled == T2S_DECODER_EOS
                || argmax == T2S_DECODER_EOS
            {
                let (stable, sliced) = semantic_at_stop(&y_vec, prefix_len, idx);
                debug!(
                    "t2s final len: {}, prefix_len: {}, stop_idx: {}",
                    sliced.len(),
                    prefix_len,
                    idx
                );
                if !progress(stable, true) {
                    return Ok(None);
                }
                let y = ArrayD::from_shape_vec(IxDyn(&[1, 1, sliced.len()]), sliced)?;
                return Ok(Some(y));
            }

            if !progress(&y_vec[prefix_len + 1..], false) {
                return Ok(None);
            }
            if old_capacity.is_none() {
                for (i, cache) in caches.iter_mut().enumerate() {
                    *cache = take_kv_pair(
                        &mut output,
                        &self.t2s_k_cache_out[i],
                        &self.t2s_v_cache_out[i],
                    )?;
                }
            } else if old_capacity == Some(filled) {
                let next = (0..self.num_layers)
                    .map(|i| {
                        take_kv_pair(
                            &mut output,
                            &self.t2s_k_cache_out[i],
                            &self.t2s_v_cache_out[i],
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                drop(output);
                let capacity =
                    prefill_length + (2 * (filled - prefill_length)).min(MAX_DECODER_STEPS);
                shared = Some(SharedKvCache::new(
                    &self.t2s_s_decoder,
                    self.t2s_shared_layout.unwrap(),
                    &next,
                    filled,
                    capacity,
                    VOCAB_SIZE,
                )?);
            }
            idx += 1;
        }
    }

    fn synthesize_semantic(
        &mut self,
        text_bert: &Array2<f32>,
        text_seq_vec: &[i64],
        ref_data: &ReferenceData,
        sampling_param: SamplingParams,
    ) -> Result<ArrayD<i64>, GSVError> {
        Ok(self
            .synthesize_semantic_with_progress(
                text_bert,
                text_seq_vec,
                ref_data,
                sampling_param,
                |_, _| true,
            )?
            .expect("semantic synthesis continues through EOS"))
    }

    fn synthesize_semantic_with_progress(
        &mut self,
        text_bert: &Array2<f32>,
        text_seq_vec: &[i64],
        ref_data: &ReferenceData,
        sampling_param: SamplingParams,
        progress: impl FnMut(&[i64], bool) -> bool,
    ) -> Result<Option<ArrayD<i64>>, GSVError> {
        let text_seq: ArrayView2<'_, i64> =
            ArrayView2::from_shape((1, text_seq_vec.len()), text_seq_vec)
                .map_err(|_| GSVError::from("invalid text sequence layout"))?;
        let mut sampler = match sampling_param.seed {
            Some(seed) => Sampler::with_seed(seed),
            None => Sampler::new(VOCAB_SIZE),
        };

        let prompts = &ref_data.prompts;

        let mut y_vec: Vec<i64> = prompts.iter().copied().collect();
        let prefix_len = y_vec.len();

        let x = concatenate(Axis(1), &[ref_data.ref_seq.view(), text_seq])?.into_owned();
        let bert = concatenate(Axis(1), &[ref_data.ref_bert.t(), text_bert.t()])?;

        let bert = bert.insert_axis(Axis(0)).to_owned();

        let caches = {
            let time = SystemTime::now();
            let mut fs_decoder_output = self.t2s_fs_decoder.run(inputs![
                "x" => TensorRef::from_array_view(&x.as_standard_layout())?,
                "prompts" => TensorRef::from_array_view(prompts)?,
                "bert" => TensorRef::from_array_view(&bert.as_standard_layout())?,
            ])?;
            debug!("T2S FS Decoder time: {:?}", time.elapsed()?);

            let logits = fs_decoder_output["logits"]
                .try_extract_array::<f32>()?
                .into_owned();

            let (logits, _) = logits.into_raw_vec_and_offset();
            let mut logits_scratch = Vec::with_capacity(logits.len());
            // First fs_decoder sample matches s_decoder idx=0 (mask EOS logit).
            let keep = logits.len().saturating_sub(1);
            logits_scratch.extend_from_slice(&logits[..keep]);
            let sampling_rst = sampler.sample(&mut logits_scratch, &y_vec, &sampling_param);
            y_vec.push(sampling_rst);
            (0..self.num_layers)
                .map(|i| {
                    take_kv_pair(
                        &mut fs_decoder_output,
                        &self.t2s_k_cache_out[i],
                        &self.t2s_v_cache_out[i],
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        let time = SystemTime::now();
        let pred_semantic = self.run_t2s_s_decoder_loop(
            &mut sampler,
            sampling_param,
            y_vec,
            caches,
            prefix_len,
            progress,
        )?;
        debug!("T2S S Decoder all time: {:?}", time.elapsed()?);

        Ok(pred_semantic)
    }

    /// Synthesize the complete waveform using an ordinary CPU VITS graph.
    /// Returns the output format and all PCM samples after postprocessing.
    pub fn synthesize_sync(
        &mut self,
        text: &str,
        sampling_param: SamplingParams,
        lang_id: LangId,
        postprocess_params: PostprocessParams,
    ) -> Result<(WavSpec, Vec<f32>), GSVError> {
        if self.has_streaming_vits() {
            return Err(GSVError::from(
                "Non-streaming synthesis requires an ordinary VITS graph",
            ));
        }
        let reference = self
            .ref_data
            .as_ref()
            .ok_or_else(|| GSVError::from("Reference data not initialized"))?
            .clone();
        let sentences = self.text_processor.get_phone_and_bert(text, lang_id)?;
        let mut fragments = Vec::with_capacity(sentences.len());
        for (_, phones, bert) in sentences {
            let semantic = self.synthesize_semantic(&bert, &phones, &reference, sampling_param)?;
            let samples = self
                .sovits
                .as_mut()
                .expect("vocoder is available outside the pipeline")
                .render_from_frame(&phones, &semantic, &reference, 0)?;
            fragments.push(samples);
        }
        let spec = self.output_spec;
        let samples = audio_postprocess(vec![fragments], spec.sample_rate, &postprocess_params, None);
        Ok((spec, samples))
    }
}

#[cfg(test)]
mod reference_tests {
    use super::*;

    #[test]
    #[ignore = "requires a local v2Pro bundle with split reference VITS and two reference WAVs"]
    fn switching_reference_refreshes_text_and_audio_conditions() {
        let bundle = std::env::var_os("GSV_TEST_BUNDLE").unwrap();
        let first_wav = std::env::var_os("GSV_TEST_REF_WAV").unwrap();
        let second_wav = std::env::var_os("GSV_TEST_ALT_REF_WAV").unwrap();
        let text = std::env::var("GSV_TEST_REF_TEXT").unwrap();
        let mut model = TTSModel::from_bundle(Path::new(&bundle)).unwrap();

        let load = |model: &mut TTSModel, wav: &std::ffi::OsStr, transcript: &str| {
            model
                .process_reference_sync(Path::new(wav), transcript, LangId::Auto)
                .unwrap();
            model.reference_data().unwrap()
        };
        fn ge(reference: &ReferenceData) -> &ArrayD<f32> {
            let VitsReference::Features(features) = &reference.vits else {
                panic!("expected cached VITS reference features");
            };
            features
        }

        let first = load(&mut model, &first_wav, &text);
        let different_text = load(&mut model, &first_wav, "这是用来核对转录的另一段文字。");
        assert!(first.ref_seq != different_text.ref_seq);
        assert!(first.ref_bert != different_text.ref_bert);
        assert!(first.prompts == different_text.prompts);
        assert!(ge(&first) == ge(&different_text));

        let different_audio = load(&mut model, &second_wav, &text);
        assert!(first.ref_seq == different_audio.ref_seq);
        assert!(first.prompts != different_audio.prompts);
        assert!(first.sv_emb != different_audio.sv_emb);
        assert!(ge(&first) != ge(&different_audio));

        let restored = load(&mut model, &first_wav, &text);
        assert!(first.ref_seq == restored.ref_seq);
        assert!(first.ref_bert == restored.ref_bert);
        assert!(first.prompts == restored.prompts);
        assert!(first.sv_emb == restored.sv_emb);
        assert!(ge(&first) == ge(&restored));

        assert!(model.process_reference_sync(Path::new(&bundle), &text, LangId::Auto).is_err());
        assert!(Arc::ptr_eq(&restored, &model.reference_data().unwrap()));
    }
}

#[cfg(test)]
mod semantic_tests {
    use super::*;

    #[test]
    fn semantic_stop_preserves_stream_prefix_and_legacy_length() {
        let (stable, legacy) = semantic_at_stop(&[900, 11, 12, 13, T2S_DECODER_EOS], 1, 2);
        assert_eq!(stable, [12, 13]);
        assert_eq!(stable, legacy);

        let (stable, legacy) = semantic_at_stop(&[900, 11, 12, 13, 14], 1, 2);
        assert_eq!(stable, [12, 13]); // EOS selected by argmax, sampled token differs.
        assert_eq!(legacy, [13, 14]);

        let mut full = vec![900, 11];
        full.extend((0..MAX_DECODER_STEPS as i64).map(|step| step % 1000));
        let (stable, legacy) = semantic_at_stop(&full, 1, MAX_DECODER_STEPS - 1);
        assert_eq!(stable, &full[2..full.len() - 1]);
        assert_eq!(stable.len(), legacy.len());
    }
}

fn ensure_punctuation(text: &str) -> String {
    if !text.ends_with(['。', '！', '？', '；', '.', '!', '?', ';']) {
        text.to_string() + "。"
    } else {
        text.to_string()
    }
}

fn resample_audio(input: &[f32], in_rate: u32, out_rate: u32) -> Result<Vec<f32>, GSVError> {
    if in_rate == out_rate {
        return Ok(input.to_vec());
    }
    let mut resampler = SincFixedIn::new(
        out_rate as f64 / in_rate as f64,
        1.0,
        SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 16,
            window: WindowFunction::BlackmanHarris2,
        },
        input.len(),
        1,
    )
    .map_err(|e| GSVError::from(format!("Resampler creation failed: {}", e)))?;
    let output = resampler
        .process(&[input], None)
        .map_err(|e| GSVError::from(format!("Resampling failed: {}", e)))?;
    output
        .into_iter()
        .next()
        .ok_or_else(|| GSVError::from("resampler returned no channel"))
}

fn read_and_resample_audio<P: AsRef<Path>>(
    path: P,
) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>), GSVError> {
    let file = File::open(&path)
        .map_err(|e| GSVError::from(format!("Failed to open reference audio: {}", e)))?;
    let wav_reader = WavReader::new(file)?;
    let spec = wav_reader.spec();
    debug!("Reference audio spec: {:?}", spec);

    // Validate input audio format
    if spec.channels != 1 || spec.sample_format != hound::SampleFormat::Int {
        return Err(GSVError::from("Reference audio must be mono 16-bit PCM"));
    }

    let mut audio_samples: Vec<f32> = wav_reader
        .into_samples::<i16>()
        .collect::<Result<Vec<i16>, _>>()?
        .into_iter()
        .map(|s| s as f32 / i16::MAX as f32)
        .collect();

    if let Some(&max_abs) = audio_samples.iter().max_by(|a, b| {
        a.abs()
            .partial_cmp(&b.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        if max_abs > 1.0 {
            let scale = max_abs.min(2.0);
            for sample in &mut audio_samples {
                *sample /= scale;
            }
        }
    }

    // Ensure audio is not too short (0.5s) or too long (10s at 16kHz, matching Python TTS.py)
    if audio_samples.len() < spec.sample_rate as usize / 2 {
        return Err(GSVError::from(
            "Reference audio too short, must be at least 0.5 seconds",
        ));
    }
    let max_samples_16k = 16000 * 10;
    let samples_16k_len = audio_samples.len() * 16000 / spec.sample_rate as usize;
    if samples_16k_len > max_samples_16k {
        log::warn!(
            "Reference audio exceeds 10s at 16kHz ({} samples); trimming may improve quality",
            samples_16k_len
        );
    }

    // Resample to 16kHz and 32kHz (single PCM buffer, no full clone between passes).
    let ref_audio_16k_raw = resample_audio(&audio_samples, spec.sample_rate, 16000)?;
    let ref_audio_32k = resample_audio(&audio_samples, spec.sample_rate, 32000)?;

    // Append trailing silence for SSL (matches Python TTS._set_prompt_semantic:
    // `np.zeros(int(configs.sampling_rate * 0.3))` appended to 16 kHz HuBERT input).
    let silence_16k = vec![0.0; (32000_f32 * 0.3) as usize];
    let mut ref_audio_16k_ssl = ref_audio_16k_raw.clone();
    ref_audio_16k_ssl.extend(silence_16k);

    Ok((
        Array2::from_shape_vec((1, ref_audio_16k_ssl.len()), ref_audio_16k_ssl)?,
        Array2::from_shape_vec((1, ref_audio_16k_raw.len()), ref_audio_16k_raw)?,
        Array2::from_shape_vec((1, ref_audio_32k.len()), ref_audio_32k)?,
    ))
}
