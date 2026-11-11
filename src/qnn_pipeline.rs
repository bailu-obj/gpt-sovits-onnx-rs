use crate::vocoder::qnn::{QnnVocoder, report};
use crate::{GSVError, LangId, SamplingParams, TTSModel};
use hound::WavSpec;
use ndarray::ArrayD;
use std::{path::Path, sync::mpsc::sync_channel};

impl TTSModel {
    /// Load the QNN generator alongside the ordinary CPU VITS fallback graph.
    pub fn enable_qnn_pipeline(&mut self, models: &Path, libraries: &Path) -> Result<(), GSVError> {
        if self.has_streaming_vits() {
            return Err(GSVError::from(
                "QNN pipeline requires an ordinary VITS graph",
            ));
        }
        if self.sovits_reference.is_none() {
            return Err(GSVError::from(
                "HTP VITS requires cached reference features",
            ));
        }
        let vocoder = self
            .sovits
            .as_mut()
            .expect("vocoder is available outside the pipeline");
        vocoder.htp = match QnnVocoder::load(models, libraries, None) {
            Ok(htp) => Some(htp),
            Err(error) => {
                report(log::Level::Warn, format_args!("htp_load_failed {error}"));
                return Err(error);
            }
        };
        Ok(())
    }

    /// Compile a device-specific EPContext and its external binary.
    pub fn compile_qnn_context(
        models: &Path,
        libraries: &Path,
        output: &Path,
    ) -> Result<(), GSVError> {
        QnnVocoder::load(models, libraries, Some(output))?;
        Ok(())
    }

    pub fn has_qnn_pipeline(&self) -> bool {
        self.sovits
            .as_ref()
            .is_some_and(|vocoder| vocoder.htp.is_some())
    }

    /// Run T2S for the next sentence while the vocoder renders the previous one.
    /// Load the generator with [`Self::enable_qnn_pipeline`] before calling this.
    /// Emit each completed HTP window in order after trimming its overlap.
    /// CPU VITS fallback emits a complete sentence. Returning false stops synthesis.
    pub fn synthesize_qnn_pipeline(
        &mut self,
        text: &str,
        sampling: SamplingParams,
        lang: LangId,
        cancelled: impl Fn() -> bool + Sync,
        mut emit: impl FnMut(WavSpec, Vec<f32>) -> bool + Send,
    ) -> Result<(), GSVError> {
        if !self.has_qnn_pipeline() {
            return Err(GSVError::from(
                "QNN synthesis requires enable_qnn_pipeline before inference",
            ));
        }
        let reference = self
            .ref_data
            .as_ref()
            .ok_or_else(|| GSVError::from("Reference data not initialized"))?
            .clone();
        let started = std::time::Instant::now();
        let sentences = self.text_processor.get_phone_and_bert(text, lang)?;
        let spec = self.output_spec;
        let mut vocoder = self
            .sovits
            .take()
            .expect("vocoder is available outside the pipeline");
        let result = std::thread::scope(|scope| {
            let (sender, receiver) = sync_channel::<(Vec<i64>, ArrayD<i64>)>(1);
            let reference = &reference;
            let cancelled = &cancelled;
            let renderer = &mut vocoder;
            let worker = scope.spawn(move || {
                let mut first_pcm = true;
                for (phones, tokens) in receiver {
                    if cancelled() {
                        break;
                    }
                    let completed = renderer.render_qnn(
                        &phones,
                        &tokens,
                        reference,
                        spec.sample_rate,
                        |samples| {
                            if cancelled() {
                                return false;
                            }
                            if first_pcm {
                                first_pcm = false;
                                report(
                                    log::Level::Info,
                                    format_args!("pipeline_first_pcm_ms={}", started.elapsed().as_millis()),
                                );
                            }
                            emit(spec, samples)
                        },
                    )?;
                    if !completed {
                        break;
                    }
                }
                Ok(())
            });
            let produce = || -> Result<(), GSVError> {
                for (_, phones, bert) in sentences {
                    if cancelled() {
                        break;
                    }
                    let tokens = self.synthesize_semantic(&bert, &phones, reference, sampling)?;
                    if cancelled() || sender.send((phones, tokens)).is_err() {
                        break;
                    }
                }
                Ok(())
            };
            let produced = produce();
            drop(sender);
            let rendered = worker
                .join()
                .map_err(|_| GSVError::from("VITS worker panicked"))?;
            rendered.and(produced)
        });
        self.sovits = Some(vocoder);
        result
    }
}
