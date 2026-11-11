use crate::vocoder::{GENERATOR_HALO_FRAMES, empty_flow_history};
use crate::{
    GSVError, LangId, PostprocessParams, SamplingParams, TTSModel, VITS_FRAMES_PER_TOKEN,
    VITS_UPSAMPLE_RATE, vits_output_samples,
};
use hound::WavSpec;
use ndarray::{ArrayD, IxDyn};
use std::sync::{Arc, Mutex, mpsc::sync_channel};
use std::time::{Duration, Instant};

const LOOKAHEAD_TOKENS: usize = 6;
const OVERLAP_TOKENS: usize = 2;
const MIN_STREAM_PHONES: usize = 24;
const MAX_BLOCK_TOKENS: usize = 128;

struct Render {
    phones: Arc<[i64]>,
    tokens: Vec<i64>,
    start_token: usize,
    new_tokens: usize,
    final_chunk: bool,
}

struct PlaybackProgress {
    playback_end: Option<Instant>,
    last_output: Instant,
    seconds_per_token: Option<f64>,
    completed_tokens: usize,
}

impl PlaybackProgress {
    fn new(now: Instant) -> Self {
        Self {
            playback_end: None,
            last_output: now,
            seconds_per_token: None,
            completed_tokens: 0,
        }
    }

    fn next_block(&self, now: Instant, minimum: usize, previous: usize, submitted: usize) -> usize {
        let (Some(end), Some(cost)) = (self.playback_end, self.seconds_per_token) else {
            return minimum;
        };
        // Estimate immediate playback from the PCM handed to the sink.
        let slack = end.saturating_duration_since(now).as_secs_f64();
        let budget = ((slack - 0.1).max(0.0) / (1.25 * cost)) as usize;
        let pending = submitted.saturating_sub(self.completed_tokens);
        let maximum = (previous * 3).div_ceil(2).min(MAX_BLOCK_TOKENS);
        budget.saturating_sub(pending).clamp(minimum, maximum)
    }

    fn observe(
        &mut self,
        now: Instant,
        samples: usize,
        sample_rate: u32,
        new_tokens: usize,
        final_chunk: bool,
    ) {
        let duration = Duration::from_secs_f64(samples as f64 / f64::from(sample_rate));
        let end = self.playback_end.map_or(now, |end| end.max(now));
        self.playback_end = Some(end + duration);
        // A final chunk can be a short tail or a whole short sentence.
        if !final_chunk {
            let cost = (now - self.last_output).as_secs_f64() / new_tokens as f64;
            self.seconds_per_token = Some(cost.max(self.seconds_per_token.unwrap_or(0.0) * 0.8));
        }
        self.last_output = now;
        self.completed_tokens += new_tokens;
    }
}

#[derive(Default)]
struct PrefixAudio {
    emitted: usize,
    overlap: Vec<f32>,
}

impl PrefixAudio {
    fn append(
        &mut self,
        audio: &[f32],
        start_frame: usize,
        final_chunk: bool,
    ) -> Result<Vec<f32>, GSVError> {
        let offset = start_frame * VITS_UPSAMPLE_RATE;
        let lookahead_samples = vits_output_samples(LOOKAHEAD_TOKENS);
        let end = if final_chunk {
            audio.len()
        } else {
            audio.len().saturating_sub(lookahead_samples)
        };
        let mut chunk = audio
            .get(self.emitted.saturating_sub(offset)..end)
            .ok_or_else(|| GSVError::from("VITS prefix ended before the emitted PCM"))?
            .to_vec();
        let overlap = self.overlap.len().min(chunk.len());
        for (index, (sample, previous)) in chunk
            .iter_mut()
            .zip(&self.overlap)
            .take(overlap)
            .enumerate()
        {
            let phase =
                std::f32::consts::PI * index as f32 / overlap.saturating_sub(1).max(1) as f32;
            let weight = 0.5 - 0.5 * phase.cos();
            *sample = previous * (1.0 - weight) + *sample * weight;
        }
        self.emitted = offset + end;
        self.overlap.clear();
        let overlap_end = (end + vits_output_samples(OVERLAP_TOKENS)).min(audio.len());
        self.overlap.extend_from_slice(&audio[end..overlap_end]);
        Ok(chunk)
    }
}

/// Options for adaptive semantic-token streaming.
#[derive(Clone, Copy)]
pub struct TokenStreamOptions {
    /// Number of semantic tokens in the first block and the minimum subsequent block.
    pub block_tokens: usize,
    /// Bound the VITS encoder history on each chunk. The lookahead is separate.
    pub vits_left_context_tokens: Option<usize>,
}

impl TTSModel {
    /// Whether the loaded VITS graph exposes the streaming tail input.
    pub fn has_streaming_vits(&self) -> bool {
        self.sovits
            .as_ref()
            .is_some_and(|vocoder| vocoder.supports_stream_tail())
    }

    /// Synthesize with adaptive token blocks using a streaming CPU VITS graph.
    /// Returns an error if the loaded VITS graph lacks `stream_start_frame`.
    pub fn synthesize_token_chunks_sync(
        &mut self,
        text: &str,
        sampling: SamplingParams,
        lang: LangId,
        postprocess: PostprocessParams,
        options: TokenStreamOptions,
        cancelled: impl Fn() -> bool + Sync,
        mut emit: impl FnMut(WavSpec, Vec<f32>) -> bool + Send,
    ) -> Result<(), GSVError> {
        use crate::preprocessor::{clean::infer_lang_en, seg::pre_seg_text};

        if !self.has_streaming_vits() {
            return Err(GSVError::from(
                "Token streaming requires a streaming VITS graph (vits.onnx with stream_start_frame)",
            ));
        }
        if !(8..=MAX_BLOCK_TOKENS).contains(&options.block_tokens) {
            return Err(GSVError::from(
                "token stream block must be between 8 and 128 tokens",
            ));
        }
        let reference = self
            .ref_data
            .as_ref()
            .ok_or_else(|| GSVError::from("Reference data not initialized"))?
            .clone();
        let sentences = pre_seg_text(text, infer_lang_en(text));
        let spec = self.output_spec;
        let mut vocoder = self
            .sovits
            .take()
            .expect("vocoder is available outside the pipeline");
        let playback = Mutex::new(PlaybackProgress::new(Instant::now()));
        let playback = &playback;
        let result = std::thread::scope(|scope| {
            let (sender, receiver) = sync_channel::<Render>(1);
            let reference = &reference;
            let cancelled = &cancelled;
            let renderer = &mut vocoder;
            let worker = scope.spawn(move || {
                let mut output = PrefixAudio::default();
                let mut history = renderer.supports_latent_history().then(empty_flow_history);
                for Render {
                    phones,
                    tokens,
                    start_token,
                    new_tokens,
                    final_chunk,
                } in receiver
                {
                    if cancelled() {
                        break;
                    }
                    let length = tokens.len();
                    let semantic = ArrayD::from_shape_vec(IxDyn(&[1, 1, length]), tokens)?;
                    let window_start = start_token * VITS_FRAMES_PER_TOKEN;
                    let emitted_frames = output.emitted / VITS_UPSAMPLE_RATE;
                    let (audio, audio_start) = if let Some(history) = &mut history {
                        let encoded_left =
                            emitted_frames.checked_sub(window_start).ok_or_else(|| {
                                GSVError::from("VITS window starts after emitted PCM")
                            })?;
                        let lookahead = if final_chunk { 0 } else { LOOKAHEAD_TOKENS };
                        let committed = (length.saturating_sub(lookahead) * VITS_FRAMES_PER_TOKEN)
                            .checked_sub(encoded_left)
                            .ok_or_else(|| GSVError::from("VITS window has no committed frames"))?;
                        let audio_start = emitted_frames
                            .saturating_sub(history.shape()[2].min(GENERATOR_HALO_FRAMES));
                        let audio = renderer.render_with_history(
                            &phones,
                            &semantic,
                            reference,
                            encoded_left,
                            committed,
                            history,
                        )?;
                        (audio, audio_start)
                    } else {
                        let start_frame =
                            emitted_frames.saturating_sub(GENERATOR_HALO_FRAMES + window_start);
                        let audio = renderer.render_from_frame(
                            &phones,
                            &semantic,
                            reference,
                            start_frame,
                        )?;
                        (audio, window_start + start_frame)
                    };
                    let mut chunk = output.append(&audio, audio_start, final_chunk)?;
                    if final_chunk {
                        let silence = (spec.sample_rate as f32
                            * postprocess.fragment_interval.max(0.0))
                            as usize;
                        chunk.resize(chunk.len() + silence, 0.0);
                        output = PrefixAudio::default();
                        if let Some(history) = &mut history {
                            *history = empty_flow_history();
                        }
                    }
                    let samples = chunk.len();
                    if cancelled() || (samples != 0 && !emit(spec, chunk)) {
                        break;
                    }
                    playback.lock().unwrap().observe(
                        Instant::now(),
                        samples,
                        spec.sample_rate,
                        new_tokens,
                        final_chunk,
                    );
                }
                Ok(())
            });
            let produce = || -> Result<(), GSVError> {
                let mut prepared = false;
                let mut submitted_tokens = 0;
                for sentence in sentences {
                    if cancelled() {
                        return Ok(());
                    }
                    let Ok((_, phones, bert)) = self
                        .text_processor
                        .get_phone_and_bert_whole(&sentence, lang)
                    else {
                        continue;
                    };
                    let phones: Arc<[i64]> = phones.into();
                    let stream_sentence = phones.len() >= MIN_STREAM_PHONES;
                    prepared = true;
                    let mut block_tokens = options.block_tokens;
                    let mut next_preview = block_tokens + LOOKAHEAD_TOKENS;
                    let mut sent_preview = false;
                    let mut committed_tokens = 0usize;
                    let result = self.synthesize_semantic_with_progress(
                        &bert,
                        &phones,
                        reference,
                        sampling,
                        |tokens, final_chunk| {
                            if cancelled() {
                                return false;
                            }
                            if final_chunk && !sent_preview {
                                return true;
                            }
                            let mut token_end = tokens.len();
                            if !final_chunk {
                                if !stream_sentence || tokens.len() < next_preview {
                                    return true;
                                }
                                let target = if sent_preview {
                                    playback.lock().unwrap().next_block(
                                        Instant::now(),
                                        options.block_tokens,
                                        block_tokens,
                                        submitted_tokens,
                                    )
                                } else {
                                    options.block_tokens
                                };
                                token_end = committed_tokens + target + LOOKAHEAD_TOKENS;
                                if tokens.len() < token_end {
                                    return true;
                                }
                                block_tokens = target;
                                next_preview = token_end + options.block_tokens;
                                sent_preview = true;
                            }
                            let start_token = options
                                .vits_left_context_tokens
                                .map_or(0, |left| committed_tokens.saturating_sub(left));
                            let previous = committed_tokens;
                            committed_tokens =
                                token_end - if final_chunk { 0 } else { LOOKAHEAD_TOKENS };
                            let new_tokens = committed_tokens - previous;
                            submitted_tokens += new_tokens;
                            sender
                                .send(Render {
                                    phones: phones.clone(),
                                    tokens: tokens[start_token..token_end].to_vec(),
                                    start_token,
                                    new_tokens,
                                    final_chunk,
                                })
                                .is_ok()
                        },
                    )?;
                    let Some(semantic) = result else {
                        return Ok(());
                    };
                    if !sent_preview {
                        let new_tokens = semantic.len();
                        submitted_tokens += new_tokens;
                        if sender
                            .send(Render {
                                phones,
                                tokens: semantic.into_raw_vec_and_offset().0,
                                start_token: 0,
                                new_tokens,
                                final_chunk: true,
                            })
                            .is_err()
                        {
                            return Ok(());
                        }
                    }
                }
                if !prepared {
                    return Err(GSVError::from(
                        "No phonemes or BERT features could be generated for the text",
                    ));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_block_respects_playback_slack_and_pending_work() {
        let start = Instant::now();
        let mut progress = PlaybackProgress::new(start);
        assert_eq!(progress.next_block(start, 8, 8, 0), 8);

        let first = start + Duration::from_millis(80);
        progress.observe(first, vits_output_samples(8), 32_000, 8, false);
        assert_eq!(progress.next_block(first, 8, 8, 8), 12);
        assert_eq!(progress.next_block(first, 8, 8, 20), 8);
        let almost_empty = first + Duration::from_millis(250);
        assert_eq!(progress.next_block(almost_empty, 8, 8, 8), 8);

        let delayed = first + Duration::from_millis(420);
        progress.observe(delayed, vits_output_samples(8), 32_000, 8, false);
        assert_eq!(progress.next_block(delayed, 8, 12, 16), 8);
        assert_eq!(
            progress.playback_end,
            Some(delayed + Duration::from_millis(320))
        );
    }

    #[test]
    fn prefix_join_preserves_length_and_crossfades_the_retained_audio() {
        let overlap = vits_output_samples(OVERLAP_TOKENS);
        let lookahead = vits_output_samples(LOOKAHEAD_TOKENS);
        let mut stream = PrefixAudio::default();
        assert_eq!(
            stream
                .append(&vec![0.25; overlap + lookahead], 0, false)
                .unwrap(),
            vec![0.25; overlap]
        );
        let tail = stream
            .append(&vec![0.75; overlap * 3 - VITS_UPSAMPLE_RATE], 1, true)
            .unwrap();
        assert_eq!(tail.len(), overlap * 2);
        assert_eq!(tail[0], 0.25);
        assert_eq!(tail[overlap - 1], 0.75);
        assert_eq!(tail[overlap], 0.75);
    }
}
