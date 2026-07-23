mod merge;
mod normalize;

pub use merge::recovery_order;

/// Parameters mirroring Python `TTS.audio_postprocess` options.
#[derive(Debug, Clone, Copy)]
pub struct PostprocessParams {
    /// Silence gap appended after each fragment, in seconds (default 0.3).
    pub fragment_interval: f32,
    /// Whether to reorder batched fragments via `batch_index_list`.
    pub split_bucket: bool,
}

impl Default for PostprocessParams {
    fn default() -> Self {
        Self {
            fragment_interval: 0.3,
            split_bucket: false,
        }
    }
}

/// Port of Python `TTS.audio_postprocess` (without super-sampling / int16 conversion).
///
/// `audio` is `[batch][fragment]` matching Python `List[List[Tensor]]`.
pub fn audio_postprocess(
    audio: Vec<Vec<Vec<f32>>>,
    sample_rate: u32,
    params: &PostprocessParams,
    batch_index_list: Option<&[Vec<usize>]>,
) -> Vec<f32> {
    let processed = merge::process_fragments(audio, sample_rate, params.fragment_interval);
    merge::flatten_and_concat(processed, params.split_bucket, batch_index_list)
}

/// Normalize one fragment and append trailing silence — used for real streaming.
pub fn process_single_fragment(
    mut samples: Vec<f32>,
    sample_rate: u32,
    fragment_interval: f32,
) -> Vec<f32> {
    normalize::normalize_fragment(&mut samples);
    if fragment_interval > 0.0 {
        let pad = (sample_rate as f32 * fragment_interval) as usize;
        samples.extend(std::iter::repeat_n(0.0f32, pad));
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params_match_python() {
        let p = PostprocessParams::default();
        assert!((p.fragment_interval - 0.3).abs() < f32::EPSILON);
        assert!(!p.split_bucket);
    }

    #[test]
    fn audio_postprocess_end_to_end() {
        let audio = vec![vec![vec![2.0, -2.0], vec![0.5; 10]]];
        let out = audio_postprocess(audio, 32000, &PostprocessParams::default(), None);
        // 2 normalized samples + 9600 silence + 10 samples + 9600 silence
        assert_eq!(out.len(), 2 + 9600 + 10 + 9600);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn process_single_fragment_matches_batch_path() {
        let single = process_single_fragment(vec![2.0, -2.0], 32000, 0.3);
        let batch = audio_postprocess(
            vec![vec![vec![2.0, -2.0]]],
            32000,
            &PostprocessParams::default(),
            None,
        );
        assert_eq!(single, batch);
    }
}
