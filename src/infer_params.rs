use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::GSVError;
use crate::logits_sampler::{SamplingParams, SamplingParamsBuilder};

/// Built-in sampling defaults for inference.
#[derive(Clone, Debug, Deserialize)]
pub struct InferParams {
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_top_k() -> usize {
    4
}

fn default_top_p() -> f32 {
    0.9
}

fn default_temperature() -> f32 {
    1.0
}

fn default_repetition_penalty() -> f32 {
    1.35
}

impl Default for InferParams {
    fn default() -> Self {
        Self {
            top_k: default_top_k(),
            top_p: default_top_p(),
            temperature: default_temperature(),
            repetition_penalty: default_repetition_penalty(),
            seed: None,
        }
    }
}

impl InferParams {
    pub fn from_json_str(s: &str) -> Result<Self, GSVError> {
        serde_json::from_str(s)
            .map_err(|e| GSVError::from(format!("invalid infer params JSON: {e}")))
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, GSVError> {
        let content = fs::read_to_string(path.as_ref()).map_err(|e| {
            GSVError::from(format!(
                "failed to read infer params {:?}: {e}",
                path.as_ref()
            ))
        })?;
        Self::from_json_str(&content)
    }

    /// Load optional `infer_params.json` override from a model bundle directory.
    pub fn from_bundle<P: AsRef<Path>>(model_dir: P) -> Result<Self, GSVError> {
        let path = model_dir.as_ref().join("infer_params.json");
        if path.is_file() {
            Self::from_file(path)
        } else {
            Ok(Self::default())
        }
    }

    pub fn to_sampling_params(&self) -> SamplingParams {
        let mut builder = SamplingParamsBuilder::new()
            .top_k(self.top_k)
            .top_p(self.top_p)
            .temperature(self.temperature)
            .repetition_penalty(self.repetition_penalty);
        if let Some(seed) = self.seed {
            builder = builder.seed(seed);
        }
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_json() {
        let p = InferParams::from_json_str(r#"{"top_k":5,"temperature":0.8}"#).unwrap();
        assert_eq!(p.top_k, 5);
        assert!((p.temperature - 0.8).abs() < 1e-6);
        assert_eq!(p.top_p, default_top_p());
    }

    #[test]
    fn sampling_params_roundtrip() {
        let p = InferParams::default();
        let s = p.to_sampling_params();
        assert_eq!(s.top_k, Some(4));
        assert!((s.top_p.unwrap() - 0.9).abs() < 1e-6);
        assert_eq!(s.seed, None);
    }
}
