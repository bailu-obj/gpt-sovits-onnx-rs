use ndarray::{Array, Axis, Ix2};
use ort::value::Tensor;
use std::{fmt::Debug, path::Path, str::FromStr, sync::Arc, time::Instant};
use tokenizers::Tokenizer;

use crate::{onnx_builder::create_onnx_cpu_session, preprocessor::utils::*};
pub static LABELS: &str = include_str!("../../../resource/g2pw/dict_poly_index_list.json");

lazy_static::lazy_static! {
    pub static ref POLY_LABLES: Vec<String> = serde_json::from_str(LABELS).unwrap();
    static ref SIMPLIFIED_TO_TRADITIONAL: std::collections::HashMap<char, char> =
        include_str!("../../../resource/g2pw/bert-base-chinese_s2t_dict.txt")
            .lines()
            .filter_map(|line| {
                let (from, to) = line.split_once('\t')?;
                Some((from.chars().next()?, to.chars().next()?))
            })
            .collect();
}

#[derive(Clone)]
pub enum G2PWOut {
    Pinyin(String),
    Yue(String),
    RawChar(char),
}

impl Debug for G2PWOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pinyin(s) => write!(f, "\"{}\"", s),
            Self::Yue(s) => write!(f, "\"{}\"", s),
            Self::RawChar(s) => write!(f, "\"{}\"", s),
        }
    }
}

#[derive(Debug)]
pub struct G2PW {
    model: Option<ort::session::Session>,
    tokenizers: Option<Arc<tokenizers::Tokenizer>>,
    shared_context: bool,
}

impl G2PW {
    pub fn new<P: AsRef<Path>>(g2pw_path: Option<P>) -> anyhow::Result<Self> {
        if let Some(g2pw_path) = g2pw_path {
            let model = create_onnx_cpu_session(g2pw_path)?;
            let shared_context =
                model.metadata()?.custom("g2pw.shared_context").as_deref() == Some("1");
            Ok(Self {
                model: Some(model),
                tokenizers: Some(Arc::new(Tokenizer::from_str(BERT_TOKENIZER).unwrap())),
                shared_context,
            })
        } else {
            Ok(Self {
                model: None,
                tokenizers: None,
                shared_context: false,
            })
        }
    }

    pub fn g2p<'s>(&mut self, text: &'s str) -> Vec<G2PWOut> {
        if self.model.is_some() && self.tokenizers.is_some() {
            self.get_pinyin_ml(text).unwrap_or_else(|_| {
                log::warn!("G2PW inference failed; using dictionary pronunciation");
                self.simple_get_pinyin(text)
            })
        } else {
            self.simple_get_pinyin(text)
        }
    }

    /// Batch G2PW for multiple segments (Python g2pw._g2pw batch path).
    pub fn g2p_batch(&mut self, texts: &[&str]) -> Vec<Vec<String>> {
        texts
            .iter()
            .map(|&text| {
                self.g2p(text)
                    .into_iter()
                    .map(|out| match out {
                        G2PWOut::Pinyin(p) => p,
                        G2PWOut::RawChar(c) => c.to_string(),
                        G2PWOut::Yue(p) => p,
                    })
                    .collect()
            })
            .collect()
    }

    pub fn simple_get_pinyin(&self, text: &str) -> Vec<G2PWOut> {
        let mut pre_data = vec![];
        for (_, c) in text.chars().enumerate() {
            if let Some(mono) = DICT_MONO_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin(mono.phone.clone()));
            } else if let Some(poly) = DICT_POLY_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin(poly.phones[0].0.clone()));
            } else {
                pre_data.push(G2PWOut::RawChar(c));
            }
        }
        pre_data
    }

    fn get_pinyin_ml<'s>(&mut self, text: &'s str) -> anyhow::Result<Vec<G2PWOut>> {
        // Upstream G2PW uses traditional context. Keep source text for word
        // segmentation and preserve one output slot per original character.
        let context: String = text
            .chars()
            // The generic map folds 幺 into 么, which changes phone-number 一 to me5.
            .map(|c| {
                if c == '幺' {
                    c
                } else {
                    SIMPLIFIED_TO_TRADITIONAL.get(&c).copied().unwrap_or(c)
                }
            })
            .collect();
        let c = self
            .tokenizers
            .as_ref()
            .unwrap()
            .encode(context.as_str(), true)
            .map_err(|e| anyhow::anyhow!("encode error: {}", e))?;
        let input_ids = c.get_ids().iter().map(|x| *x as i64).collect::<Vec<i64>>();
        let token_type_ids = vec![0i64; input_ids.len()];
        let attention_mask = vec![1i64; input_ids.len()];

        let mut phoneme_masks = vec![];
        let mut pre_data = vec![];
        let mut query_id = vec![];
        let mut chars_id = vec![];

        for (i, (original, c)) in text.chars().zip(context.chars()).enumerate() {
            if let Some(mono) = DICT_MONO_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin(mono.phone.clone()));
            } else if let Some(poly) = DICT_POLY_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin("".to_owned()));
                // 这个位置是 tokens 的位置，它的前后添加了 '[CLS]' 和 '[SEP]' 两个特殊字符
                query_id.push((i + 1) as i64);
                chars_id.push(poly.index as i64);
                let mut phoneme_mask = vec![0f32; POLY_LABLES.len()];
                for (_, i) in &poly.phones {
                    phoneme_mask[*i] = 1.0;
                }
                phoneme_masks.extend(phoneme_mask);
            } else {
                pre_data.push(G2PWOut::RawChar(original));
            }
        }
        if query_id.is_empty() {
            return Ok(pre_data);
        }
        let input_ids =
            Tensor::from_array(Array::from_shape_vec((1, input_ids.len()), input_ids).unwrap())
                .unwrap();
        let token_type_ids = Tensor::from_array(
            Array::from_shape_vec((1, token_type_ids.len()), token_type_ids).unwrap(),
        )
        .unwrap();
        let attention_mask = Tensor::from_array(
            Array::from_shape_vec((1, attention_mask.len()), attention_mask).unwrap(),
        )
        .unwrap();

        let query_count = query_id.len();
        let chunk_size = if self.shared_context { query_count } else { 1 };
        let label_count = POLY_LABLES.len();
        for start in (0..query_count).step_by(chunk_size) {
            let end = (start + chunk_size).min(query_count);
            let positions = &query_id[start..end];
            let phoneme_mask = Tensor::from_array(Array::from_shape_vec(
                (positions.len(), label_count),
                phoneme_masks[start * label_count..end * label_count].to_vec(),
            )?)?;
            let position_ids = Tensor::from_array(Array::from_vec(positions.to_vec()))?;
            let char_ids = Tensor::from_array(Array::from_vec(chars_id[start..end].to_vec()))?;

            let started = Instant::now();
            let model_output = self.model.as_mut().unwrap().run(ort::inputs![
                "input_ids" => &input_ids,
                "token_type_ids" => &token_type_ids,
                "attention_mask" => &attention_mask,
                "phoneme_mask"=> phoneme_mask,
                "char_ids" => char_ids,
                "position_ids" => position_ids,
            ])?;
            log::debug!("G2PW inference time: {:?}", started.elapsed());

            let probs = model_output["probs"]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix2>()?;
            anyhow::ensure!(
                probs.dim() == (positions.len(), label_count),
                "unexpected G2PW output shape"
            );
            for (&position, row) in positions.iter().zip(probs.rows()) {
                let (_, label) = argmax(&row.insert_axis(Axis(0)).into_dyn());
                pre_data[position as usize - 1] = G2PWOut::Pinyin(POLY_LABLES[label].clone());
            }
        }

        Ok(pre_data)
    }
}
