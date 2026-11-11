use crate::{
    GSVError, ReferenceData, VITS_FRAMES_PER_TOKEN, VITS_UPSAMPLE_RATE, VitsReference,
    vits_output_samples,
};
use ndarray::{Array3, ArrayD, ArrayView2, Ix3, arr1};
use ort::{inputs, session::Session, value::TensorRef};

pub(crate) struct Vocoder {
    cpu: Session,
    stream_tail: bool,
    latent_history: bool,
    #[cfg(target_os = "android")]
    pub(crate) htp: Option<qnn::QnnVocoder>,
}

pub(crate) const GENERATOR_HALO_FRAMES: usize = 11;

pub(crate) fn empty_flow_history() -> Array3<f32> {
    Array3::zeros((1, 192, 0))
}

impl Vocoder {
    pub(crate) fn new(cpu: Session) -> Self {
        let stream_tail = cpu
            .inputs()
            .iter()
            .any(|input| input.name() == "stream_start_frame");
        let latent_history = cpu
            .inputs()
            .iter()
            .any(|input| input.name() == "stream_latent_history");
        Self {
            cpu,
            stream_tail,
            latent_history,
            #[cfg(target_os = "android")]
            htp: None,
        }
    }

    #[cfg(target_os = "android")]
    pub(crate) fn render_qnn(
        &mut self,
        text_seq: &[i64],
        pred_semantic: &ArrayD<i64>,
        reference: &ReferenceData,
        sample_rate: u32,
        mut emit: impl FnMut(Vec<f32>) -> bool,
    ) -> Result<bool, GSVError> {
        let postprocess = crate::PostprocessParams::default();
        if !self.latent_history
            && pred_semantic.shape()[2] * VITS_FRAMES_PER_TOKEN >= qnn::WINDOW
            && let (Some(htp), VitsReference::Features(ge)) = (&mut self.htp, &reference.vits)
        {
            let text_seq = ArrayView2::from_shape((1, text_seq.len()), text_seq)?;
            let silence = (sample_rate as f32 * postprocess.fragment_interval) as usize;
            let mut emitted = false;
            match htp.render_windows(text_seq, pred_semantic, ge, |samples, last| {
                // The W8A16 generator's final Tanh/QDQ already bounds its PCM.
                let mut audio = samples.to_vec();
                Self::finish_pcm(&mut audio, samples.len());
                if last {
                    audio.resize(audio.len() + silence, 0.0);
                }
                emitted = true;
                emit(audio)
            }) {
                Ok(completed) => return Ok(completed),
                Err(error) => {
                    qnn::report(log::Level::Warn, format_args!("htp_render_failed {error}"));
                    self.htp = None;
                    if emitted {
                        return Err(error);
                    }
                }
            }
        }
        let audio = self.render_from_frame(text_seq, pred_semantic, reference, 0)?;
        Ok(emit(crate::audio_postprocess(
            vec![vec![audio]],
            sample_rate,
            &postprocess,
            None,
        )))
    }

    pub(crate) fn supports_stream_tail(&self) -> bool {
        self.stream_tail
    }

    pub(crate) fn supports_latent_history(&self) -> bool {
        self.latent_history
    }

    pub(crate) fn render_with_history(
        &mut self,
        text_seq: &[i64],
        pred_semantic: &ArrayD<i64>,
        reference: &ReferenceData,
        encoded_left_frames: usize,
        commit_frames: usize,
        history: &mut Array3<f32>,
    ) -> Result<Vec<f32>, GSVError> {
        let VitsReference::Features(ge) = &reference.vits else {
            return Err(GSVError::from(
                "VITS latent history requires cached reference features",
            ));
        };
        let text_seq = ArrayView2::from_shape((1, text_seq.len()), text_seq)?;
        let start = arr1(&[encoded_left_frames as i64]);
        let commit = arr1(&[commit_frames as i64]);
        let outputs = self.cpu.run(inputs![
            "text_seq" => TensorRef::from_array_view(text_seq)?,
            "pred_semantic" => TensorRef::from_array_view(pred_semantic)?,
            "ge" => TensorRef::from_array_view(ge)?,
            "stream_start_frame" => TensorRef::from_array_view(start.view())?,
            "stream_commit_frames" => TensorRef::from_array_view(commit.view())?,
            "stream_latent_history" => TensorRef::from_array_view(history.view())?,
        ])?;
        let generator_left = history.shape()[2].min(GENERATOR_HALO_FRAMES);
        let mut audio = outputs["audio"].try_extract_tensor::<f32>()?.1.to_vec();
        let frames = pred_semantic.shape()[2] * VITS_FRAMES_PER_TOKEN - encoded_left_frames;
        let expected = (frames + generator_left) * VITS_UPSAMPLE_RATE;
        if audio.len() != expected {
            return Err(GSVError::from("unexpected VITS latent-history PCM length"));
        }
        *history = outputs["stream_next_history"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix3>()?
            .into_owned();
        Self::finish_pcm(&mut audio, expected);
        Ok(audio)
    }

    pub(crate) fn render_from_frame(
        &mut self,
        text_seq: &[i64],
        pred_semantic: &ArrayD<i64>,
        reference: &ReferenceData,
        start_frame: usize,
    ) -> Result<Vec<f32>, GSVError> {
        if self.latent_history {
            if start_frame != 0 {
                return Err(GSVError::from(
                    "cropped VITS rendering requires a latent history",
                ));
            }
            return self.render_with_history(
                text_seq,
                pred_semantic,
                reference,
                0,
                0,
                &mut empty_flow_history(),
            );
        }
        let text_seq = ArrayView2::from_shape((1, text_seq.len()), text_seq)?;
        let semantic_len = pred_semantic.shape()[2];
        let started = std::time::Instant::now();
        let frame = arr1(&[start_frame as i64]);
        let mut feeds = match (&reference.vits, &reference.sv_emb) {
            (VitsReference::Features(ge), _) => inputs![
                "text_seq" => TensorRef::from_array_view(text_seq)?,
                "pred_semantic" => TensorRef::from_array_view(pred_semantic)?,
                "ge" => TensorRef::from_array_view(ge)?,
            ],
            (VitsReference::Audio(audio), Some(sv_emb)) => inputs![
                "text_seq" => TensorRef::from_array_view(text_seq)?,
                "pred_semantic" => TensorRef::from_array_view(pred_semantic)?,
                "ref_audio" => TensorRef::from_array_view(audio)?,
                "sv_emb" => TensorRef::from_array_view(sv_emb)?,
            ],
            (VitsReference::Audio(audio), None) => inputs![
                "text_seq" => TensorRef::from_array_view(text_seq)?,
                "pred_semantic" => TensorRef::from_array_view(pred_semantic)?,
                "ref_audio" => TensorRef::from_array_view(audio)?,
            ],
        };
        if self.stream_tail {
            feeds.push((
                "stream_start_frame".into(),
                TensorRef::from_array_view(frame.view())?.into(),
            ));
        }
        let outputs = self.cpu.run(feeds)?;
        log::debug!("SoVITS all time: {:?}", started.elapsed());
        let mut audio = outputs["audio"]
            .try_extract_array::<f32>()?
            .into_owned()
            .into_raw_vec_and_offset()
            .0;
        Self::finish_pcm(
            &mut audio,
            vits_output_samples(semantic_len) - start_frame * crate::VITS_UPSAMPLE_RATE,
        );
        Ok(audio)
    }

    fn finish_pcm(audio: &mut Vec<f32>, sample_count: usize) {
        audio.truncate(sample_count);
        for sample in audio.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
    }
}

#[cfg(target_os = "android")]
pub(crate) mod qnn {
    use crate::{GSVError, VITS_FRAMES_PER_TOKEN, VITS_UPSAMPLE_RATE, create_onnx_cpu_session};
    use log::Log;
    use ndarray::{ArrayD, ArrayView2, Ix3, s};
    use ort::{
        environment::Environment, ep::ExecutionProviderLibrary, inputs, memory::DeviceType,
        session::Session, value::TensorRef,
    };
    use std::{
        path::Path,
        sync::{LazyLock, Mutex},
        time::Instant,
    };

    const PROVIDER: &str = "QNNExecutionProvider";
    pub(super) const WINDOW: usize = 128;
    const HALO: usize = super::GENERATOR_HALO_FRAMES;
    // Keep the provider registered across model reloads.
    static REGISTRATION: Mutex<Option<ExecutionProviderLibrary>> = Mutex::new(None);

    static LOGGER: LazyLock<android_logger::AndroidLogger> = LazyLock::new(|| {
        android_logger::AndroidLogger::new(
            android_logger::Config::default()
                .with_tag("GptSovitsQnn")
                .with_max_level(log::LevelFilter::Info)
                .format(|output, record| write!(output, "{}", record.args())),
        )
    });

    pub(crate) fn report(level: log::Level, message: std::fmt::Arguments<'_>) {
        LOGGER.log(
            &log::Record::builder()
                .level(level)
                .target("gpt_sovits_qnn")
                .args(message)
                .build(),
        );
    }

    pub(crate) struct QnnVocoder {
        prefix: Session,
        generator: Session,
    }

    impl QnnVocoder {
        pub(crate) fn load(
            root: &Path,
            libraries: &Path,
            context_output: Option<&Path>,
        ) -> Result<Self, GSVError> {
            let env = Environment::current()?;
            {
                let mut registration = REGISTRATION
                    .lock()
                    .map_err(|_| GSVError::from("QNN registration lock"))?;
                if registration.is_none() {
                    *registration = Some(env.register_ep_library(
                        PROVIDER,
                        libraries.join("libonnxruntime_providers_qnn.so"),
                    )?);
                }
            }
            let options = [
                (
                    format!("{PROVIDER}.backend_path"),
                    libraries
                        .join("libQnnHtp.so")
                        .to_string_lossy()
                        .into_owned(),
                ),
                (
                    format!("{PROVIDER}.htp_performance_mode"),
                    "burst".to_owned(),
                ),
                (
                    format!("{PROVIDER}.offload_graph_io_quantization"),
                    "0".to_owned(),
                ),
            ];
            let build = |path: &Path, output: Option<&Path>| -> Result<Session, GSVError> {
                let device = env
                    .devices()
                    .find(|device| {
                        device.ep().is_ok_and(|name| name == PROVIDER)
                            && device.hardware_device().ty() == DeviceType::NPU
                    })
                    .ok_or_else(|| GSVError::from("QNN HTP device unavailable"))?;
                let builder = Session::builder()?
                    .with_devices([device], Some(&options))?
                    .with_config_entry("session.disable_cpu_ep_fallback", "1")?;
                let mut builder = if let Some(output) = output {
                    if output.symlink_metadata().is_ok() {
                        return Err(GSVError::from("QNN context output already exists"));
                    }
                    let output = output
                        .to_str()
                        .ok_or_else(|| GSVError::from("QNN context path must be UTF-8"))?;
                    builder
                        .with_config_entry("ep.context_enable", "1")?
                        .with_config_entry("ep.context_embed_mode", "0")?
                        .with_config_entry("ep.context_file_path", output)?
                } else {
                    builder
                };
                Ok(builder.commit_from_file(path)?)
            };
            let graph = root.join("vits_generator_htp.onnx");
            let context = root.join("vits_generator_context.onnx");
            let (generator, cached) = if let Some(output) = context_output {
                (build(&graph, Some(output))?, false)
            } else if context.is_file() {
                match build(&context, None) {
                    Ok(generator) => (generator, true),
                    Err(error) => {
                        report(log::Level::Warn, format_args!("htp_context_failed {error}"));
                        (build(&graph, None)?, false)
                    }
                }
            } else {
                (build(&graph, None)?, false)
            };
            let prefix = create_onnx_cpu_session(root.join("vits_prefix.onnx"))?;
            report(
                log::Level::Info,
                format_args!("htp_ready window={WINDOW} context={cached}"),
            );
            Ok(Self { prefix, generator })
        }

        pub(crate) fn render_windows(
            &mut self,
            text_seq: ArrayView2<'_, i64>,
            tokens: &ArrayD<i64>,
            ge: &ArrayD<f32>,
            mut emit: impl FnMut(&[f32], bool) -> bool,
        ) -> Result<bool, GSVError> {
            let started = Instant::now();
            let latent = self.prefix.run(inputs![
                "text_seq" => TensorRef::from_array_view(text_seq)?,
                "pred_semantic" => TensorRef::from_array_view(tokens)?,
                "ge" => TensorRef::from_array_view(ge)?,
            ])?;
            let z = latent["z"]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix3>()?;
            let frames = z.shape()[2];
            if frames < WINDOW || frames != tokens.shape()[2] * VITS_FRAMES_PER_TOKEN {
                return Err(GSVError::from("unexpected VITS latent length"));
            }
            let mut cursor = 0usize;
            let mut calls = 0;
            while cursor < frames {
                let offset = cursor.saturating_sub(HALO).min(frames - WINDOW);
                let end = offset + WINDOW;
                let stop = if end == frames { frames } else { end - HALO };
                let tile = z
                    .slice(s![.., .., offset..end])
                    .as_standard_layout()
                    .into_owned();
                let output = self.generator.run(inputs![
                    "z" => TensorRef::from_array_view(&tile)?,
                    "ge" => TensorRef::from_array_view(ge)?,
                ])?;
                let (_, samples) = output["audio"].try_extract_tensor::<f32>()?;
                if samples.len() != WINDOW * VITS_UPSAMPLE_RATE {
                    return Err(GSVError::from("unexpected HTP VITS PCM length"));
                }
                let pcm = &samples[(cursor - offset) * VITS_UPSAMPLE_RATE
                    ..(stop - offset) * VITS_UPSAMPLE_RATE];
                if !emit(pcm, stop == frames) {
                    return Ok(false);
                }
                cursor = stop;
                calls += 1;
            }
            report(
                log::Level::Info,
                format_args!(
                    "htp_sentence tokens={} tiles={calls} render_ms={}",
                    tokens.shape()[2],
                    started.elapsed().as_millis()
                ),
            );
            Ok(true)
        }
    }
}
