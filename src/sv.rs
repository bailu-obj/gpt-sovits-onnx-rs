use log::debug;
use ndarray::Array2;
use ndarray::ArrayD;
use ort::{inputs, session::Session, value::TensorRef};

use crate::GSVError;

pub struct SvModel {
    sv_session: Session,
}

/// Kaldi-compatible fbank without per-feature mean normalization.
/// `knf_rs::compute_fbank` subtracts temporal mean per mel bin, which diverges
/// from Python `torchaudio.compliance.kaldi.fbank` used during ONNX export.
fn compute_kaldi_fbank(samples: &[f32]) -> Result<Array2<f32>, GSVError> {
    if samples.is_empty() {
        return Err(GSVError::from("SV fbank input is empty"));
    }

    let mut result = unsafe {
        knf_rs_sys::ComputeFbank(
            samples.as_ptr(),
            samples
                .len()
                .try_into()
                .map_err(|_| GSVError::from("SV fbank sample length overflow"))?,
        )
    };

    let frames = unsafe {
        std::slice::from_raw_parts(
            result.frames,
            (result.num_frames * result.num_bins) as usize,
        )
        .to_vec()
    };

    let frames_array = Array2::from_shape_vec(
        (result.num_frames as usize, result.num_bins as usize),
        frames,
    )
    .map_err(|e| GSVError::from(format!("SV fbank reshape failed: {e}")))?;

    unsafe {
        knf_rs_sys::DestroyFbankResult(&mut result as *mut _);
    }

    if frames_array.is_empty() {
        return Err(GSVError::from("SV fbank produced no frames"));
    }

    Ok(frames_array)
}

impl SvModel {
    pub fn new(sv_session: Session) -> Self {
        Self { sv_session }
    }

    pub fn infer(&mut self, audio_16k: &[f32]) -> Result<ArrayD<f32>, GSVError> {
        let features = compute_kaldi_fbank(audio_16k)?;
        debug!("SV features shape: {:?}", features.shape());
        let input_features = TensorRef::from_array_view(features.view());

        let outputs = self.sv_session.run(inputs![
            "audio_feature" => input_features?,
        ])?;

        let output_tensor = outputs["sv_emb"].try_extract_array::<f32>()?.into_owned();

        Ok(output_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kaldi_fbank_frame_count_matches_python_reference() {
        let wav_path =
            std::path::Path::new("gpt-sovits-upstream/onnx-patched/custom_v2proplus/ref.wav");
        if !wav_path.exists() {
            return;
        }

        let file = std::fs::File::open(wav_path).expect("open ref wav");
        let mut reader = hound::WavReader::new(file).expect("read ref wav");
        let spec = reader.spec();
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect();

        let audio_16k = if spec.sample_rate == 16_000 {
            samples
        } else {
            // Reference bundle wav is 16 kHz; skip if a different fixture is used.
            return;
        };

        let features = compute_kaldi_fbank(&audio_16k).expect("compute fbank");
        // Python Kaldi fbank on the same ref.wav yields 458 frames.
        assert_eq!(features.shape(), &[458, 80]);
    }
}
