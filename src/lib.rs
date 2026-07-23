use async_stream::stream;
use futures::{Stream, StreamExt};
use hound::{WavReader, WavSpec};
use log::{debug, info};
use ndarray::{
    Array1, Array2, ArrayBase, ArrayD, ArrayView2, Axis, Dimension, IxDyn, OwnedRepr, concatenate, s,
};
use ort::{
    inputs,
    session::{Session, SessionInputValue},
    value::TensorRef,
};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;
use std::{fs::File, path::Path};
use tokio::task::block_in_place;

mod cpu_info;
mod error;
mod infer_params;
mod kv_workspace;
mod logits_sampler;
mod onnx_builder;
mod ort_dtype;
mod postprocess;
mod preprocessor;
mod sv;
mod t2s_batch;

use onnx_builder::create_onnx_cpu_session;
pub use onnx_builder::{OrtConfig, OrtRuntimeProfile, configure_ort_runtime, ort_config};
pub use postprocess::{PostprocessParams, audio_postprocess, process_single_fragment, recovery_order};
pub use preprocessor::LangId;
pub use preprocessor::lang::Lang;
pub use preprocessor::{TextProcessor, bert, en, phoneme_finalize, text_normalize, zh};

use kv_workspace::{KvDType, KvWorkspace};
use logits_sampler::Sampler;
use preprocessor::{bert::BertModel, en::g2p_en::G2pEn, zh::g2pw::G2PW};

pub use error::GSVError;
pub use infer_params::InferParams;
pub use logits_sampler::{SamplingParams, SamplingParamsBuilder};
pub use t2s_batch::{
    ActiveBatch, CompactRemap, FragmentSlot, assign_length_buckets, default_bucket_edges,
    estimate_batched_ar_upper_bound, estimate_batched_fs_upper_bound, remap_kv_layer,
};

use crate::{onnx_builder::BIG_CORES, sv::SvModel};

const T2S_DECODER_EOS: i64 = 1024;
const VOCAB_SIZE: usize = 1025;
const DEFAULT_NUM_LAYERS: usize = 24;
/// Product of VITS `upsample_rates` for v2 / v2Pro / v2ProPlus (`[10, 8, 2, 2, 2]`).
const VITS_UPSAMPLE_RATE: usize = 640;
const DEFAULT_VITS_NOISE_SCALE: f32 = 0.5;
const DEFAULT_VITS_SPEED: f32 = 1.0;

/// Expected VITS waveform length before postprocess, matching Python
/// `pred_semantic.shape[0] * 2 * upsample_rate`.
fn vits_output_samples(semantic_token_count: usize) -> usize {
    semantic_token_count * 2 * VITS_UPSAMPLE_RATE
}

static STANDALONE_TOKIO_RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

/// Cooperative cancellation handle shared with [`TTSModel::synthesize`].
#[derive(Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.flag.store(false, Ordering::SeqCst);
    }
}

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

/// Detect FS-decoder BERT layout from ONNX input metadata.
///
/// - Legacy BFT: `[batch, 1024, time]` (dim1 fixed at 1024)
/// - Native BTF: `[batch, time, 1024]` (dim2 fixed at 1024)
fn bert_input_is_bft(fs_decoder: &Session) -> bool {
    use ort::value::ValueType;
    for input in fs_decoder.inputs() {
        if input.name() != "bert" {
            continue;
        }
        if let ValueType::Tensor { shape, .. } = input.dtype() {
            let dims: Vec<i64> = shape.iter().copied().collect();
            if dims.len() >= 3 {
                if dims[1] == 1024 {
                    return true;
                }
                if dims[2] == 1024 {
                    return false;
                }
            }
        }
    }
    // Historical default before native-layout exports.
    true
}

/// True when stage-decoder K/V outputs are single-row deltas.
///
/// Supports legacy `[B,1,H]` and head-major `[B,H,1,D]` delta exports.
fn kv_outputs_are_delta(s_decoder: &Session) -> bool {
    use ort::value::ValueType;
    for output in s_decoder.outputs() {
        if output.name() != "k_cache_0" {
            continue;
        }
        if let ValueType::Tensor { shape, .. } = output.dtype() {
            let dims: Vec<i64> = shape.iter().copied().collect();
            if dims.len() >= 4 {
                return dims[2] == 1;
            }
            if dims.len() >= 2 {
                return dims[1] == 1;
            }
        }
    }
    false
}

/// Detect whether a VITS session exposes native FP16 floating I/O.
fn vits_session_is_fp16(session: &Session) -> bool {
    ort_dtype::session_input_is_f16(session, "ge")
        || ort_dtype::session_input_is_f16(session, "noise_scale")
        || ort_dtype::session_input_is_f16(session, "ref_audio")
        || ort_dtype::session_output_is_f16(session, "audio")
        || ort_dtype::session_output_is_f16(session, "ge")
}

fn detect_vits_fp16(
    sovits: &Option<Session>,
    sovits_ref: &Option<Session>,
    sovits_decode: &Option<Session>,
) -> bool {
    if let Some(s) = sovits_decode {
        return vits_session_is_fp16(s);
    }
    if let Some(s) = sovits_ref {
        return vits_session_is_fp16(s);
    }
    if let Some(s) = sovits {
        return vits_session_is_fp16(s);
    }
    false
}

#[derive(Clone)]
pub struct ReferenceData {
    ref_seq: Array2<i64>,
    ref_bert: Array2<f32>,
    ref_audio_32k: Array2<f32>,
    ssl_content: ArrayBase<OwnedRepr<f32>, IxDyn>,
    /// Cached T2S encoder prompts derived from `ssl_content` (reference-only).
    prompts: ArrayBase<OwnedRepr<i64>, IxDyn>,
    sv_emb: Option<ArrayD<f32>>,
    /// Cached VITS style vector from `{prefix}_vits_ref.onnx` when split graphs are loaded.
    ge: Option<ArrayD<f32>>,
}

pub struct TTSModel {
    text_processor: TextProcessor,
    /// Monolithic VITS (`{prefix}_vits.onnx`); `None` when split graphs are owned instead.
    sovits: Option<Session>,
    /// Optional `{prefix}_vits_ref.onnx` (ref_audio [+ sv_emb] → ge).
    sovits_ref: Option<Session>,
    /// Optional `{prefix}_vits_decode.onnx` (text_seq + pred_semantic + ge → audio).
    sovits_decode: Option<Session>,
    /// SSL / CNHubert session. Dropped after reference when `release_ssl_after_reference` is used.
    ssl: Option<Session>,
    t2s_encoder: Session,
    t2s_fs_decoder: Session,
    t2s_s_decoder: Session,
    sv: Option<SvModel>,
    ref_data: Option<Arc<ReferenceData>>,
    t2s_dec_ik: Vec<String>,
    t2s_dec_iv: Vec<String>,
    t2s_k_cache_out: Vec<String>,
    t2s_v_cache_out: Vec<String>,
    num_layers: usize,
    output_spec: WavSpec,
    /// Reusable KV buffers across fragments / synthesize calls.
    kv_workspace: KvWorkspace,
    cancel: CancelToken,
    /// Reused VITS scalar inputs (avoid per-fragment allocation).
    vits_noise_scale: Array1<f32>,
    vits_speed: Array1<f32>,
    /// FS `bert` input is legacy `[B,1024,T]` when true; native `[B,T,1024]` when false.
    bert_layout_bft: bool,
    /// Stage-decoder K/V outputs are single-row deltas (new export) vs full caches.
    kv_out_delta: bool,
    /// VITS graphs use native FP16 public I/O (`ge` / `noise_scale` / `speed` / `audio`).
    vits_fp16: bool,
    /// Reusable contiguous phoneme id buffer for FS `x` input.
    x_scratch: Vec<i64>,
    /// Reusable contiguous BERT feature buffer (layout depends on `bert_layout_bft`).
    bert_scratch: Vec<f32>,
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

        let output_spec = WavSpec {
            channels: 1,
            sample_rate: 32000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let t2s_s_decoder = create_onnx_cpu_session(t2s_s_decoder_path)?;
        let num_layers = t2s_num_layers_from_session(&t2s_s_decoder);
        info!("T2S decoder num_layers: {}", num_layers);
        let kv_out_delta = kv_outputs_are_delta(&t2s_s_decoder);
        info!("T2S stage KV outputs are delta rows: {}", kv_out_delta);

        let (t2s_dec_ik, t2s_dec_iv, t2s_k_cache_out, t2s_v_cache_out) =
            t2s_kv_io_names(num_layers);

        // Prefer split VITS when both graphs exist beside the monolithic path so we never
        // keep three VITS-related sessions resident at once.
        let mono_path = sovits_path.as_ref();
        let (sovits, sovits_ref, sovits_decode) = load_vits_sessions(mono_path)?;

        let t2s_fs_decoder = create_onnx_cpu_session(t2s_fs_decoder_path)?;
        let bert_layout_bft = bert_input_is_bft(&t2s_fs_decoder);
        info!(
            "T2S FS bert layout: {}",
            if bert_layout_bft {
                "[B,1024,T] legacy"
            } else {
                "[B,T,1024] native"
            }
        );

        let vits_fp16 = detect_vits_fp16(&sovits, &sovits_ref, &sovits_decode);
        info!("VITS native FP16 I/O: {}", vits_fp16);

        Ok(TTSModel {
            text_processor: TextProcessor::new(
                G2PW::new(g2pw_path)?,
                G2pEn::new(g2p_en_path)?,
                BertModel::new(bert_path)?,
            )?,
            sovits,
            sovits_ref,
            sovits_decode,
            ssl: Some(create_onnx_cpu_session(ssl_path)?),
            t2s_encoder: create_onnx_cpu_session(t2s_encoder_path)?,
            t2s_fs_decoder,
            t2s_s_decoder,
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
            kv_workspace: KvWorkspace::new(num_layers),
            cancel: CancelToken::new(),
            vits_noise_scale: Array1::from_elem(1, DEFAULT_VITS_NOISE_SCALE),
            vits_speed: Array1::from_elem(1, DEFAULT_VITS_SPEED),
            bert_layout_bft,
            kv_out_delta,
            vits_fp16,
            x_scratch: Vec::with_capacity(256),
            bert_scratch: Vec::with_capacity(256 * 1024),
        })
    }

    /// Load optional split VITS graphs if both files exist beside a monolithic `*_vits.onnx`.
    ///
    /// Looks for `{stem}_ref.onnx` and `{stem}_decode.onnx` where `stem` is the monolithic
    /// path without `.onnx` (e.g. `custom_v2_vits` → `custom_v2_vits_ref.onnx`).
    /// On success, drops the monolithic session so only split graphs remain resident.
    /// Returns `true` when split sessions were loaded.
    pub fn try_load_split_vits_beside(
        &mut self,
        monolithic_vits_path: impl AsRef<Path>,
    ) -> Result<bool, GSVError> {
        if self.uses_split_vits() {
            return Ok(true);
        }
        let mono = monolithic_vits_path.as_ref();
        let Some((ref_path, decode_path)) = split_vits_paths(mono) else {
            return Ok(false);
        };
        if !(ref_path.is_file() && decode_path.is_file()) {
            return Ok(false);
        }
        self.sovits_ref = Some(create_onnx_cpu_session(&ref_path)?);
        self.sovits_decode = Some(create_onnx_cpu_session(&decode_path)?);
        // Exclusive ownership: free monolithic weights once split is ready.
        self.sovits = None;
        self.vits_fp16 = detect_vits_fp16(&self.sovits, &self.sovits_ref, &self.sovits_decode);
        info!(
            "Loaded split VITS (dropped monolithic): {} + {} (fp16_io={})",
            ref_path.display(),
            decode_path.display(),
            self.vits_fp16
        );
        Ok(true)
    }

    pub fn uses_split_vits(&self) -> bool {
        self.sovits_ref.is_some() && self.sovits_decode.is_some()
    }

    /// Pack phoneme ids + BERT features into reusable contiguous scratch buffers.
    ///
    /// Returns `(x_len, bert_shape)`. Callers should read `x_scratch` / `bert_scratch`
    /// pointers immediately and form ORT views from those raw pointers so the session
    /// borrow does not conflict with buffer borrows.
    fn pack_fs_inputs(
        &mut self,
        ref_seq: &Array2<i64>,
        text_seq: ArrayView2<'_, i64>,
        ref_bert: &Array2<f32>,
        text_bert: &Array2<f32>,
    ) -> Result<(usize, IxDyn), GSVError> {
        let ref_len = ref_seq.shape()[1];
        let text_len = text_seq.shape()[1];
        let total = ref_len + text_len;
        self.x_scratch.clear();
        self.x_scratch.reserve(total);
        self.x_scratch.extend(ref_seq.row(0).iter().copied());
        self.x_scratch.extend(text_seq.row(0).iter().copied());

        if ref_bert.shape()[1] != 1024 || text_bert.shape()[1] != 1024 {
            return Err(GSVError::from("bert features must have width 1024"));
        }
        if ref_bert.shape()[0] != ref_len || text_bert.shape()[0] != text_len {
            return Err(GSVError::from("bert time dim must match phoneme length"));
        }

        self.bert_scratch.clear();
        self.bert_scratch.reserve(total * 1024);
        if self.bert_layout_bft {
            for c in 0..1024 {
                for t in 0..ref_len {
                    self.bert_scratch.push(ref_bert[[t, c]]);
                }
                for t in 0..text_len {
                    self.bert_scratch.push(text_bert[[t, c]]);
                }
            }
        } else if let (Some(r), Some(t)) = (ref_bert.as_slice(), text_bert.as_slice()) {
            self.bert_scratch.extend_from_slice(r);
            self.bert_scratch.extend_from_slice(t);
        } else {
            for row in ref_bert.rows() {
                self.bert_scratch.extend(row.iter().copied());
            }
            for row in text_bert.rows() {
                self.bert_scratch.extend(row.iter().copied());
            }
        }

        let bert_shape = if self.bert_layout_bft {
            IxDyn(&[1, 1024, total])
        } else {
            IxDyn(&[1, total, 1024])
        };
        Ok((total, bert_shape))
    }

    /// Shared cancellation token for in-flight synthesis.
    pub fn cancel_token(&self) -> CancelToken {
        self.cancel.clone()
    }

    /// Request cooperative cancellation of the current synthesize loop.
    pub fn request_cancel(&self) {
        self.cancel.cancel();
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

        let prompts = {
            let time = SystemTime::now();
            let encoder_output = self.t2s_encoder.run(inputs![
                "ssl_content" => TensorRef::from_array_view(&ssl_content)?
            ])?;
            debug!("T2S Encoder (reference) time: {:?}", time.elapsed()?);
            encoder_output["prompts"]
                .try_extract_array::<i64>()?
                .into_owned()
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

        let ge = self.compute_vits_ge(&ref_audio_32k, sv_emb.as_ref())?;

        self.ref_data = Some(Arc::new(ReferenceData {
            ref_seq,
            ref_bert,
            sv_emb,
            ref_audio_32k,
            ssl_content,
            prompts,
            ge,
        }));

        Ok(())
    }

    fn compute_vits_ge(
        &mut self,
        ref_audio_32k: &Array2<f32>,
        sv_emb: Option<&ArrayD<f32>>,
    ) -> Result<Option<ArrayD<f32>>, GSVError> {
        let Some(ref_sess) = self.sovits_ref.as_mut() else {
            return Ok(None);
        };
        if self.sovits_decode.is_none() {
            return Ok(None);
        }
        let time = SystemTime::now();
        let outputs = match sv_emb {
            Some(sv_emb) => {
                if self.vits_fp16 {
                    let ref_f16 = ort_dtype::array_to_f16(ref_audio_32k);
                    let sv_f16 = ort_dtype::array_to_f16(sv_emb);
                    ref_sess.run(inputs![
                        "ref_audio" => TensorRef::from_array_view(ref_f16.view())?,
                        "sv_emb" => TensorRef::from_array_view(sv_f16.view())?,
                    ])?
                } else {
                    ref_sess.run(inputs![
                        "ref_audio" => TensorRef::from_array_view(ref_audio_32k)?,
                        "sv_emb" => TensorRef::from_array_view(sv_emb)?,
                    ])?
                }
            }
            None => {
                if self.vits_fp16 {
                    let ref_f16 = ort_dtype::array_to_f16(ref_audio_32k);
                    ref_sess.run(inputs![
                        "ref_audio" => TensorRef::from_array_view(ref_f16.view())?,
                    ])?
                } else {
                    ref_sess.run(inputs![
                        "ref_audio" => TensorRef::from_array_view(ref_audio_32k)?,
                    ])?
                }
            }
        };
        let ge = ort_dtype::extract_array_f32(&outputs["ge"])?;
        debug!(
            "VITS ref conditioning (ge) time: {:?}, shape={:?}",
            time.elapsed()?,
            ge.shape()
        );
        Ok(Some(ge))
    }

    fn process_ssl(
        &mut self,
        ref_audio_16k: &Array2<f32>,
    ) -> Result<ArrayBase<OwnedRepr<f32>, IxDyn>, GSVError> {
        let time = SystemTime::now();
        let ssl = self
            .ssl
            .as_mut()
            .ok_or_else(|| GSVError::from("SSL session was released; cannot reprocess reference"))?;
        let ssl_output = ssl
            .run(inputs!["ref_audio_16k" => TensorRef::from_array_view(ref_audio_16k).unwrap()])?;
        debug!("SSL processing time: {:?}", time.elapsed()?);
        Ok(ssl_output["ssl_content"]
            .try_extract_array::<f32>()?
            .into_owned())
    }

    /// Drop the SSL/CNHubert session after reference is cached to reclaim RSS.
    ///
    /// Safe once `process_reference` has completed: synthesize uses cached
    /// `ssl_content` / prompts / ge. Calling again before a new reference is fine;
    /// a subsequent `process_reference` will fail until a new model is constructed.
    pub fn release_ssl_after_reference(&mut self) {
        if self.ssl.take().is_some() {
            info!("Released SSL session after reference (RSS reclaim)");
        }
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
        let (x_len, bert_shape) = self.pack_fs_inputs(
            &ref_data.ref_seq,
            text_seq_arr.view(),
            &ref_data.ref_bert,
            &text_bert,
        )?;
        let x_ptr = self.x_scratch.as_ptr();
        let bert_ptr = self.bert_scratch.as_ptr();
        let bert_shape_vec: Vec<usize> = (0..bert_shape.ndim()).map(|i| bert_shape[i]).collect();
        let x_head: Vec<i64> = self.x_scratch.iter().take(8).copied().collect();

        let encoder_output = self.t2s_encoder.run(inputs![
            "ssl_content" => TensorRef::from_array_view(&ref_data.ssl_content)?
        ])?;
        let prompts = encoder_output["prompts"]
            .try_extract_array::<i64>()?
            .into_owned();

        let x_view = unsafe { ArrayView2::from_shape_ptr((1, x_len), x_ptr) };
        let bert_view = unsafe { ndarray::ArrayView::from_shape_ptr(bert_shape, bert_ptr) };
        let bert_mean = bert_view.mean().unwrap_or(0.0);
        let fs_decoder_output = self.t2s_fs_decoder.run(inputs![
            "x" => TensorRef::from_array_view(x_view)?,
            "prompts" => TensorRef::from_array_view(prompts.view())?,
            "bert" => TensorRef::from_array_view(bert_view)?,
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
            "x_len": x_len,
            "x_head": x_head,
            "bert_shape": bert_shape_vec,
            "bert_mean": bert_mean,
            "bert_layout_bft": self.bert_layout_bft,
            "kv_out_delta": self.kv_out_delta,
            "prompts_len": prompts.shape()[1],
            "prompts_head": prompts.slice(s![0, ..5]).iter().copied().collect::<Vec<_>>(),
            "ssl_shape": ref_data.ssl_content.shape().to_vec(),
            "fs_argmax_masked": argmax,
        }))
    }

    /// Efficiently runs the streaming decoder loop with a reusable KV workspace.
    fn run_t2s_s_decoder_loop(
        &mut self,
        sampler: &mut Sampler,
        sampling_param: SamplingParams,
        mut y_vec: Vec<i64>,
        prefix_len: usize,
        initial_valid_len: usize,
    ) -> Result<ArrayBase<OwnedRepr<i64>, IxDyn>, GSVError> {
        let mut idx = 0;
        let mut valid_len = initial_valid_len;
        y_vec.reserve(512);

        let y_len_arr = Array1::from_elem(1, prefix_len as i64);
        let mut idx_arr = Array1::from_elem(1, 0i64);
        let mut logits_scratch = Vec::with_capacity(VOCAB_SIZE + 2);
        let input_cap = 3 + 2 * self.num_layers;

        loop {
            if self.cancel.is_cancelled() {
                return Err(GSVError::Cancelled);
            }
            idx_arr[0] = idx as i64;

            let iy = TensorRef::from_array_view(unsafe {
                ArrayView2::from_shape_ptr((1, y_vec.len()), y_vec.as_ptr())
            })
            .unwrap();

            // Head-major caches need a contiguous [B,H,T,D] feed for ORT.
            self.kv_workspace.compact_for_ort(valid_len);

            let mut run_inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
                Vec::with_capacity(input_cap);
            run_inputs.push((Cow::Borrowed("iy"), iy.into()));
            run_inputs.push((
                Cow::Borrowed("y_len"),
                TensorRef::from_array_view(y_len_arr.view()).unwrap().into(),
            ));
            run_inputs.push((
                Cow::Borrowed("idx"),
                TensorRef::from_array_view(idx_arr.view()).unwrap().into(),
            ));

            for i in 0..self.num_layers {
                let k_view = self.kv_workspace.k_view(i, valid_len);
                let v_view = self.kv_workspace.v_view(i, valid_len);

                run_inputs.push((
                    Cow::Borrowed(self.t2s_dec_ik[i].as_str()),
                    TensorRef::from_array_view(k_view)?.into(),
                ));
                run_inputs.push((
                    Cow::Borrowed(self.t2s_dec_iv[i].as_str()),
                    TensorRef::from_array_view(v_view)?.into(),
                ));
            }

            let mut output = self.t2s_s_decoder.run(run_inputs)?;

            // Prefer in-place sampling on ORT logits when EOS masking is not required
            // (idx >= 11). Early steps still copy into scratch with EOS stripped.
            let (sampled, stop_by_argmax) = {
                let mut logits_arr = output["logits"].try_extract_array_mut::<f32>()?;
                let src = logits_arr.as_slice_mut().unwrap();
                if idx < 11 {
                    logits_scratch.clear();
                    let keep = src.len().saturating_sub(1);
                    logits_scratch.extend_from_slice(&src[..keep]);
                    let sampled = sampler.sample(&mut logits_scratch, &y_vec, &sampling_param);
                    let stop = sampled == T2S_DECODER_EOS
                        || logits_sampler::argmax(&logits_scratch) == T2S_DECODER_EOS;
                    (sampled, stop)
                } else {
                    let sampled = sampler.sample(src, &y_vec, &sampling_param);
                    let stop = sampled == T2S_DECODER_EOS
                        || logits_sampler::argmax(src) == T2S_DECODER_EOS;
                    (sampled, stop)
                }
            };
            y_vec.push(sampled);

            let new_valid_len = valid_len + 1;
            if new_valid_len > self.kv_workspace.capacity() {
                self.kv_workspace.grow_to(new_valid_len, valid_len)?;
            }

            for i in 0..self.num_layers {
                let inc_k_cache =
                    output[self.t2s_k_cache_out[i].as_str()].try_extract_array::<KvDType>()?;
                let inc_v_cache =
                    output[self.t2s_v_cache_out[i].as_str()].try_extract_array::<KvDType>()?;
                self.kv_workspace.write_slice_from_inc(
                    i,
                    valid_len,
                    &inc_k_cache.view(),
                    &inc_v_cache.view(),
                );
            }

            valid_len = new_valid_len;

            if idx >= 1500 || sampled == T2S_DECODER_EOS || stop_by_argmax {
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
            idx += 1;
        }
    }

    /// synthesize async — yields each sentence fragment as soon as it is decoded.
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
        self.cancel.reset();
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
        let fragment_interval = postprocess_params.fragment_interval;

        let stream = stream! {
            for (text, seq, bert) in texts_and_seqs {
                if self.cancel.is_cancelled() {
                    yield Err(GSVError::Cancelled);
                    return;
                }
                debug!("process: {:?}", text);
                match self.in_stream_once_gen(&text, &bert, &seq, &ref_data, sampling_param).await {
                    Ok(samples) => {
                        // Normalize + trailing silence per fragment, then yield immediately
                        // so callers observe time-to-first-audio after the first VITS decode.
                        let processed = postprocess::process_single_fragment(
                            samples,
                            sample_rate,
                            fragment_interval,
                        );
                        for sample in processed {
                            yield Ok(sample);
                        }
                    }
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }
        };

        Ok((spec, Box::pin(stream)))
    }

    async fn in_stream_once_gen(
        &mut self,
        _text: &str,
        text_bert: &Array2<f32>,
        text_seq_vec: &[i64],
        ref_data: &ReferenceData,
        sampling_param: SamplingParams,
    ) -> Result<Vec<f32>, GSVError> {
        if self.cancel.is_cancelled() {
            return Err(GSVError::Cancelled);
        }
        let text_seq: ArrayView2<'_, i64> =
            ArrayView2::from_shape((1, text_seq_vec.len()), text_seq_vec)
                .map_err(|_| GSVError::from("invalid text sequence layout"))?;
        let mut sampler = match sampling_param.seed {
            Some(seed) => Sampler::with_seed(seed),
            None => Sampler::new(VOCAB_SIZE),
        };

        let prompts = &ref_data.prompts;
        debug!("T2S Encoder time: cached prompts (reference)");

        let mut y_vec: Vec<i64> = prompts.iter().copied().collect();
        let prefix_len = y_vec.len();

        let (y_vec, initial_seq_len) = {
            let (x_len, bert_shape) = self.pack_fs_inputs(
                &ref_data.ref_seq,
                text_seq,
                &ref_data.ref_bert,
                text_bert,
            )?;
            let x_ptr = self.x_scratch.as_ptr();
            let bert_ptr = self.bert_scratch.as_ptr();
            let x_view = unsafe { ArrayView2::from_shape_ptr((1, x_len), x_ptr) };
            let bert_view = unsafe { ndarray::ArrayView::from_shape_ptr(bert_shape, bert_ptr) };
            let time = SystemTime::now();
            let fs_decoder_output = self.t2s_fs_decoder.run(inputs![
                "x" => TensorRef::from_array_view(x_view)?,
                "prompts" => TensorRef::from_array_view(prompts)?,
                "bert" => TensorRef::from_array_view(bert_view)?,
            ])?;
            debug!("T2S FS Decoder time: {:?}", time.elapsed()?);

            let logits = fs_decoder_output["logits"]
                .try_extract_array::<f32>()?
                .into_owned();

        let initial_seq_len = {
                let k0 = fs_decoder_output[self.t2s_k_cache_out[0].as_str()]
                    .try_extract_array::<KvDType>()?;
                let shape = k0.shape();
                let seq = if shape.len() >= 4 { shape[2] } else { shape[1] };
                self.kv_workspace
                    .ensure_capacity(KvWorkspace::suggested_capacity(seq), &k0.view())?;
                seq
            };
            // Single copy: ORT output views → workspace (no intermediate owned tensors).
            for i in 0..self.num_layers {
                let k = fs_decoder_output[self.t2s_k_cache_out[i].as_str()]
                    .try_extract_array::<KvDType>()?;
                let v = fs_decoder_output[self.t2s_v_cache_out[i].as_str()]
                    .try_extract_array::<KvDType>()?;
                self.kv_workspace
                    .write_prefix_layer(i, &k.view(), &v.view())?;
            }

            let (logits, _) = logits.into_raw_vec_and_offset();
            let mut logits_scratch = Vec::with_capacity(logits.len());
            let keep = logits.len().saturating_sub(1);
            logits_scratch.extend_from_slice(&logits[..keep]);
            let sampling_rst = sampler.sample(&mut logits_scratch, &y_vec, &sampling_param);
            y_vec.push(sampling_rst);
            (y_vec, initial_seq_len)
        };

        let time = SystemTime::now();
        let pred_semantic = self.run_t2s_s_decoder_loop(
            &mut sampler,
            sampling_param,
            y_vec,
            prefix_len,
            initial_seq_len,
        )?;
        debug!("T2S S Decoder all time: {:?}", time.elapsed()?);

        let time = SystemTime::now();
        let outputs = if let (Some(ge), Some(decode)) =
            (ref_data.ge.as_ref(), self.sovits_decode.as_mut())
        {
            if self.vits_fp16 {
                let ge_f16 = ort_dtype::array_to_f16(ge);
                let noise_f16 = ort_dtype::array_to_f16(&self.vits_noise_scale);
                let speed_f16 = ort_dtype::array_to_f16(&self.vits_speed);
                decode.run(inputs![
                    "text_seq" => TensorRef::from_array_view(text_seq)?,
                    "pred_semantic" => TensorRef::from_array_view(&pred_semantic)?,
                    "ge" => TensorRef::from_array_view(ge_f16.view())?,
                    "noise_scale" => TensorRef::from_array_view(noise_f16.view())?,
                    "speed" => TensorRef::from_array_view(speed_f16.view())?,
                ])?
            } else {
                decode.run(inputs![
                    "text_seq" => TensorRef::from_array_view(text_seq)?,
                    "pred_semantic" => TensorRef::from_array_view(&pred_semantic)?,
                    "ge" => TensorRef::from_array_view(ge)?,
                    "noise_scale" => TensorRef::from_array_view(self.vits_noise_scale.view())?,
                    "speed" => TensorRef::from_array_view(self.vits_speed.view())?,
                ])?
            }
        } else {
            let sovits = self
                .sovits
                .as_mut()
                .ok_or_else(|| GSVError::from("VITS session not loaded"))?;
            match &ref_data.sv_emb {
                Some(sv_emb) => {
                    if self.vits_fp16 {
                        let ref_f16 = ort_dtype::array_to_f16(&ref_data.ref_audio_32k);
                        let sv_f16 = ort_dtype::array_to_f16(sv_emb);
                        sovits.run(inputs![
                            "text_seq" => TensorRef::from_array_view(text_seq)?,
                            "pred_semantic" => TensorRef::from_array_view(&pred_semantic)?,
                            "ref_audio" => TensorRef::from_array_view(ref_f16.view())?,
                            "sv_emb" => TensorRef::from_array_view(sv_f16.view())?,
                        ])?
                    } else {
                        sovits.run(inputs![
                            "text_seq" => TensorRef::from_array_view(text_seq)?,
                            "pred_semantic" => TensorRef::from_array_view(&pred_semantic)?,
                            "ref_audio" => TensorRef::from_array_view(&ref_data.ref_audio_32k)?,
                            "sv_emb" => TensorRef::from_array_view(sv_emb)?,
                        ])?
                    }
                }
                None => {
                    if self.vits_fp16 {
                        let ref_f16 = ort_dtype::array_to_f16(&ref_data.ref_audio_32k);
                        sovits.run(inputs![
                            "text_seq" => TensorRef::from_array_view(text_seq)?,
                            "pred_semantic" => TensorRef::from_array_view(&pred_semantic)?,
                            "ref_audio" => TensorRef::from_array_view(ref_f16.view())?,
                        ])?
                    } else {
                        sovits.run(inputs![
                            "text_seq" => TensorRef::from_array_view(text_seq)?,
                            "pred_semantic" => TensorRef::from_array_view(&pred_semantic)?,
                            "ref_audio" => TensorRef::from_array_view(&ref_data.ref_audio_32k)?,
                        ])?
                    }
                }
            }
        };
        debug!("SoVITS all time: {:?}", time.elapsed()?);
        let semantic_len = pred_semantic.shape()[2];
        let expected_samples = vits_output_samples(semantic_len);
        if let Some(slice) = pred_semantic.as_slice() {
            let preview: Vec<_> = slice.iter().take(8).copied().collect();
            debug!("pred_semantic len={}, head={:?}", semantic_len, preview);
        }
        let output_audio = ort_dtype::extract_array_f32(&outputs["audio"])?;
        let (mut audio, _) = output_audio.into_raw_vec_and_offset();
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

fn ensure_punctuation(text: &str) -> String {
    if !text.ends_with(['。', '！', '？', '；', '.', '!', '?', ';']) {
        text.to_string() + "。"
    } else {
        text.to_string()
    }
}

fn split_vits_paths(mono: &Path) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let stem = mono.file_stem()?.to_str()?;
    let parent = mono.parent().unwrap_or_else(|| Path::new("."));
    Some((
        parent.join(format!("{stem}_ref.onnx")),
        parent.join(format!("{stem}_decode.onnx")),
    ))
}

/// Load either split VITS (exclusive) or monolithic — never both.
fn load_vits_sessions(
    mono_path: &Path,
) -> Result<(Option<Session>, Option<Session>, Option<Session>), GSVError> {
    if let Some((ref_path, decode_path)) = split_vits_paths(mono_path) {
        if ref_path.is_file() && decode_path.is_file() {
            info!(
                "Loading split VITS exclusively: {} + {}",
                ref_path.display(),
                decode_path.display()
            );
            return Ok((
                None,
                Some(create_onnx_cpu_session(&ref_path)?),
                Some(create_onnx_cpu_session(&decode_path)?),
            ));
        }
    }
    Ok((Some(create_onnx_cpu_session(mono_path)?), None, None))
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
