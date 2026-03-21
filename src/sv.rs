use knf_rs::compute_fbank;
use log::debug;
use ndarray::ArrayD;
use ort::{
    inputs,
    session::Session,
    value::{TensorRef},
};

use crate::GSVError;

pub struct SvModel {
    sv_session: Session,
}

impl SvModel {
    pub fn new(sv_session: Session) -> Self {
        Self { sv_session }
    }

    pub fn infer(&mut self, audio_16k: &[f32]) -> Result<ArrayD<f32>, GSVError> {
        let features = compute_fbank(audio_16k)
            .map_err(|e| GSVError::from(format!("SV fbank failed: {}", e)))?;
        debug!("SV features shape: {:?}", features.shape());
        let input_features = TensorRef::from_array_view(features.view());

        let outputs = self.sv_session.run(inputs![
            "audio_feature" => input_features?,
        ])?;

        let output_tensor = outputs["sv_emb"].try_extract_array::<f32>()?.into_owned();

        Ok(output_tensor)
    }
}
