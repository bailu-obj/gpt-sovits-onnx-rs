use async_stream::stream;
use futures::{Stream, StreamExt};
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
mod postprocess;
mod preprocessor;
mod sv;
mod t2s_cache;

use onnx_builder::create_onnx_cpu_session;
pub use postprocess::{PostprocessParams, audio_postprocess, recovery_order};
pub use preprocessor::LangId;
pub use preprocessor::lang::Lang;
pub use preprocessor::{TextProcessor, bert, en, phoneme_finalize, text_normalize, zh};

use logits_sampler::Sampler;
use preprocessor::{bert::BertModel, en::g2p_en::G2pEn, zh::g2pw::G2PW};
use t2s_cache::{CACHE_RESERVE, CacheLayout, LayerCache, SharedKvCache};

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

/// Expected VITS waveform length before postprocess, matching Python
/// `pred_semantic.shape[0] * 2 * upsample_rate`.
fn vits_output_samples(semantic_token_count: usize) -> usize {
    semantic_token_count * 2 * VITS_UPSAMPLE_RATE
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
    ref_audio_32k: Array2<f32>,
    ssl_content: ArrayBase<OwnedRepr<f32>, IxDyn>,
    sv_emb: Option<ArrayD<f32>>,
}

pub struct TTSModel {
    text_processor: TextProcessor,
    sovits: Session,
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
    /// create new tts instance
    /// bert_path, g2pw_path and g2p_en_path can be None
    /// if bert path is none, the speech speed in chinese may become worse
    /// if g2pw path is none, the chinese speech quality may be worse
    /// g2p_en is still experimental, english speak quality may not be better because of bugs
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

        Ok(TTSModel {
            text_processor: TextProcessor::new(
                G2PW::new(g2pw_path)?,
                G2pEn::new(g2p_en_path)?,
                BertModel::new(bert_path)?,
            )?,
            sovits: create_onnx_cpu_session(sovits_path)?,
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

        self.ref_data = Some(Arc::new(ReferenceData {
            ref_seq,
            ref_bert,
            sv_emb,
            ref_audio_32k,
            ssl_content,
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

        let encoder_output = self.t2s_encoder.run(inputs![
            "ssl_content" => TensorRef::from_array_view(&ref_data.ssl_content)?
        ])?;
        let prompts = encoder_output["prompts"]
            .try_extract_array::<i64>()?
            .into_owned();

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
            "ssl_shape": ref_data.ssl_content.shape().to_vec(),
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
    ) -> Result<ArrayBase<OwnedRepr<i64>, IxDyn>, GSVError> {
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
                let sliced = logits_sampler::extract_semantic_tokens(&y_vec, prefix_len, idx);
                debug!(
                    "t2s final len: {}, prefix_len: {}, stop_idx: {}",
                    sliced.len(),
                    prefix_len,
                    idx
                );
                let y = ArrayD::from_shape_vec(IxDyn(&[1, 1, sliced.len()]), sliced)?;
                return Ok(y);
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

    /// synthesize async
    ///
    /// `text` is input text for run
    ///
    /// `lang_id` can be LangId::Auto(Mandarin) or LangId::AutoYue（cantonese）
    ///
    pub async fn synthesize(
        &mut self,
        text: &str,
        sampling_param: SamplingParams,
        lang_id: LangId,
        postprocess_params: PostprocessParams,
    ) -> Result<
        (
            WavSpec,
            impl Stream<Item = Result<f32, GSVError>> + Send + Unpin,
        ),
        GSVError,
    > {
        debug!("g2pw synth start");
        let ref_data = self
            .ref_data
            .as_ref()
            .ok_or(GSVError::from("Reference data not initialized"))?;
        let spec = self.output_spec;
        let time = SystemTime::now();
        let texts_and_seqs = self.text_processor.get_phone_and_bert(&text, lang_id)?;
        debug!("g2pw and preprocess time: {:?}", time.elapsed()?);
        let ref_data = ref_data.clone();
        let sample_rate = spec.sample_rate;

        let stream = stream! {
            let mut fragments = Vec::new();
            for (text, seq, bert) in texts_and_seqs {
                debug!("process: {:?}", text);
                match self.in_stream_once_gen(&text, &bert, &seq, &ref_data, sampling_param).await {
                    Ok(samples) => fragments.push(samples),
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }

            let final_audio = audio_postprocess(
                vec![fragments],
                sample_rate,
                &postprocess_params,
                None,
            );
            for sample in final_audio {
                yield Ok(sample);
            }
        };

        Ok((spec, Box::pin(stream)))
    }

    /// Sentence-level streaming: each item is a complete, postprocessed waveform.
    /// VITS does not stream within a sentence. Polling the next item starts the
    /// next sentence; dropping the stream stops subsequent inference.
    pub async fn synthesize_sentence_chunks(
        &mut self,
        text: &str,
        sampling_param: SamplingParams,
        lang_id: LangId,
        postprocess_params: PostprocessParams,
    ) -> Result<
        (
            WavSpec,
            impl Stream<Item = Result<Vec<f32>, GSVError>> + Send + Unpin,
        ),
        GSVError,
    > {
        let ref_data = self
            .ref_data
            .as_ref()
            .ok_or_else(|| GSVError::from("Reference data not initialized"))?
            .clone();
        let spec = self.output_spec;
        let fragments = self.text_processor.get_phone_and_bert(text, lang_id)?;
        let stream = stream! {
            for (text, seq, bert) in fragments {
                match self.in_stream_once_gen(&text, &bert, &seq, &ref_data, sampling_param).await {
                    Ok(samples) => yield Ok(audio_postprocess(
                        vec![vec![samples]], spec.sample_rate, &postprocess_params, None,
                    )),
                    Err(error) => { yield Err(error); break; }
                }
            }
        };
        Ok((spec, Box::pin(stream)))
    }

    /// Blocking bridge with backpressure and cancellation between sentences.
    /// Return false from `emit` to stop before another sentence is synthesized.
    pub fn synthesize_sentences_sync(
        &mut self,
        text: &str,
        sampling_param: SamplingParams,
        lang_id: LangId,
        postprocess_params: PostprocessParams,
        emit: impl FnMut(WavSpec, Vec<f32>) -> bool,
    ) -> Result<(), GSVError> {
        Self::run_async_in_context(async {
            let (spec, chunks) = self
                .synthesize_sentence_chunks(text, sampling_param, lang_id, postprocess_params)
                .await?;
            emit_sentence_chunks(spec, chunks, emit).await
        })
    }

    async fn in_stream_once_gen(
        &mut self,
        _text: &str,
        text_bert: &Array2<f32>,
        text_seq_vec: &[i64],
        ref_data: &ReferenceData,
        sampling_param: SamplingParams,
    ) -> Result<Vec<f32>, GSVError> {
        let text_seq: ArrayView2<'_, i64> =
            ArrayView2::from_shape((1, text_seq_vec.len()), text_seq_vec)
                .map_err(|_| GSVError::from("invalid text sequence layout"))?;
        let mut sampler = match sampling_param.seed {
            Some(seed) => Sampler::with_seed(seed),
            None => Sampler::new(VOCAB_SIZE),
        };

        let prompts = {
            let time = SystemTime::now();
            let encoder_output = self.t2s_encoder.run(inputs![
                "ssl_content" => TensorRef::from_array_view(&ref_data.ssl_content)?
            ])?;
            debug!("T2S Encoder time: {:?}", time.elapsed()?);
            encoder_output["prompts"]
                .try_extract_array::<i64>()?
                .into_owned()
        };

        let mut y_vec: Vec<i64> = prompts.iter().copied().collect();
        let prefix_len = y_vec.len();

        let x = concatenate(Axis(1), &[ref_data.ref_seq.view(), text_seq])?.into_owned();
        let bert = concatenate(Axis(1), &[ref_data.ref_bert.t(), text_bert.t()])?;

        let bert = bert.insert_axis(Axis(0)).to_owned();

        let caches = {
            let time = SystemTime::now();
            let mut fs_decoder_output = self.t2s_fs_decoder.run(inputs![
                "x" => TensorRef::from_array_view(&x.as_standard_layout())?,
                "prompts" => TensorRef::from_array_view(&prompts)?,
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
        let pred_semantic =
            self.run_t2s_s_decoder_loop(&mut sampler, sampling_param, y_vec, caches, prefix_len)?;
        debug!("T2S S Decoder all time: {:?}", time.elapsed()?);

        let time = SystemTime::now();
        // use sv_emb if have
        let outputs = match &ref_data.sv_emb {
            Some(sv_emb) => self.sovits.run(inputs![
                "text_seq" => TensorRef::from_array_view(text_seq)?,
                "pred_semantic" => TensorRef::from_array_view(&pred_semantic)?,
                "ref_audio" => TensorRef::from_array_view(&ref_data.ref_audio_32k)?,
                "sv_emb" => TensorRef::from_array_view(sv_emb)?,
            ])?,
            None => self.sovits.run(inputs![
                "text_seq" => TensorRef::from_array_view(text_seq)?,
                "pred_semantic" => TensorRef::from_array_view(&pred_semantic)?,
                "ref_audio" => TensorRef::from_array_view(&ref_data.ref_audio_32k)?,
            ])?,
        };
        debug!("SoVITS all time: {:?}", time.elapsed()?);
        let semantic_len = pred_semantic.shape()[2];
        let expected_samples = vits_output_samples(semantic_len);
        if let Some(slice) = pred_semantic.as_slice() {
            let preview: Vec<_> = slice.iter().take(8).copied().collect();
            debug!("pred_semantic len={}, head={:?}", semantic_len, preview);
        }
        let output_audio = outputs["audio"].try_extract_array::<f32>()?;
        let (mut audio, _) = output_audio.into_owned().into_raw_vec_and_offset();
        // ONNX VITS returns dynamic-length audio after export fix; only trim excess padding.
        if audio.len() > expected_samples {
            audio.truncate(expected_samples);
        }
        debug!(
            "SoVITS output cropped to {} samples (semantic_len={})",
            audio.len(),
            semantic_len
        );
        for sample in &mut audio {
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }

        Ok(audio)
    }

    /// synthesize
    ///
    /// `text` is input text for run
    ///
    /// `lang_id` can be LangId::Auto(Mandarin) or LangId::AutoYue（cantonese）
    ///
    pub fn synthesize_sync(
        &mut self,
        text: &str,
        sampling_param: SamplingParams,
        lang_id: LangId,
        postprocess_params: PostprocessParams,
    ) -> Result<(WavSpec, Vec<f32>), GSVError> {
        Self::run_async_in_context(async {
            let (spec, stream) = self
                .synthesize(text, sampling_param, lang_id, postprocess_params)
                .await?;
            let mut samples = Vec::new();
            futures::pin_mut!(stream);
            while let Some(sample) = stream.next().await {
                samples.push(sample?);
            }
            Ok((spec, samples))
        })
    }
}

async fn emit_sentence_chunks(
    spec: WavSpec,
    mut chunks: impl Stream<Item = Result<Vec<f32>, GSVError>> + Unpin,
    mut emit: impl FnMut(WavSpec, Vec<f32>) -> bool,
) -> Result<(), GSVError> {
    while let Some(chunk) = chunks.next().await {
        if !emit(spec, chunk?) {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod sentence_stream_tests {
    use super::*;
    #[tokio::test]
    async fn stopping_after_first_sentence_does_not_poll_next_inference() {
        let produced = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = produced.clone();
        let spec = WavSpec {
            channels: 1,
            sample_rate: 32000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let chunks = stream! {
            for n in 1..=3 { counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed); yield Ok(vec![n as f32]); }
        };
        emit_sentence_chunks(spec, Box::pin(chunks), |_, samples| {
            assert_eq!(samples, [1.0]);
            false
        })
        .await
        .unwrap();
        assert_eq!(produced.load(std::sync::atomic::Ordering::Relaxed), 1);
    }
    #[tokio::test]
    async fn sentence_order_and_errors_are_preserved() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 32000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let chunks = futures::stream::iter(vec![
            Ok(vec![1.0, 2.0]),
            Ok(vec![3.0]),
            Err(GSVError::from("failed sentence")),
            Ok(vec![4.0]),
        ]);
        let mut actual = Vec::new();
        assert!(
            emit_sentence_chunks(spec, chunks, |s, samples| {
                assert_eq!(s.sample_rate, 32000);
                actual.push(samples);
                true
            })
            .await
            .is_err()
        );
        assert_eq!(actual, [vec![1.0, 2.0], vec![3.0]]);
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
