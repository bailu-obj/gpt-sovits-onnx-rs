/// Peak-normalize a single audio fragment if max amplitude exceeds 1.0.
pub fn normalize_fragment(audio: &mut [f32]) {
    let max_audio = audio
        .iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);

    if max_audio > 1.0 {
        let inv = 1.0 / max_audio;
        for sample in audio.iter_mut() {
            *sample *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_fragment_clamps_above_one() {
        let mut audio = vec![2.0, -2.0];
        normalize_fragment(&mut audio);
        assert!((audio[0] - 1.0).abs() < 1e-6);
        assert!((audio[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_fragment_leaves_below_one() {
        let mut audio = vec![0.5, -0.25];
        normalize_fragment(&mut audio);
        assert_eq!(audio, vec![0.5, -0.25]);
    }
}
