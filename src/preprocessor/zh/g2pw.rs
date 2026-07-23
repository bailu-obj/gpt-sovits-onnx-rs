use ndarray::{Array1, Array2};
use ort::value::Tensor;
use std::collections::HashMap;
use std::{fmt::Debug, path::Path, str::FromStr, sync::Arc};
use tokenizers::Tokenizer;

use crate::{onnx_builder::create_onnx_cpu_session, preprocessor::utils::*};
pub static LABELS: &str = include_str!("../../../resource/g2pw/dict_poly_index_list.json");

lazy_static::lazy_static! {
    pub static ref POLY_LABLES: Vec<String> = serde_json::from_str(LABELS).unwrap();
}

/// Characters of context kept around polyphonic spans for long sentences
/// (matches CPUFast `g2pw_polyphonic_context_chars`, default 16).
const POLYPHONIC_CONTEXT_CHARS: usize = 16;

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
}

struct PolyQuery {
    text_idx: usize,
    model_char_idx: usize,
    result_char_idx: usize,
    char_id: usize,
    phoneme_mask: Vec<f32>,
    window_text: String,
}

impl G2PW {
    pub fn new<P: AsRef<Path>>(g2pw_path: Option<P>) -> anyhow::Result<Self> {
        if let Some(g2pw_path) = g2pw_path {
            Ok(Self {
                model: Some(create_onnx_cpu_session(g2pw_path)?),
                tokenizers: Some(Arc::new(Tokenizer::from_str(BERT_TOKENIZER).unwrap())),
            })
        } else {
            Ok(Self {
                model: None,
                tokenizers: None,
            })
        }
    }

    pub fn g2p(&mut self, text: &str) -> Vec<G2PWOut> {
        if self.model.is_some() && self.tokenizers.is_some() {
            match self.g2p_batch_ml(&[text]) {
                Ok(mut batch) => batch.pop().unwrap_or_else(|| self.simple_get_pinyin(text)),
                Err(_) => self.simple_get_pinyin(text),
            }
        } else {
            self.simple_get_pinyin(text)
        }
    }

    /// True batch G2PW for multiple segments — one ONNX run for all polyphonic queries.
    pub fn g2p_batch(&mut self, texts: &[&str]) -> Vec<Vec<String>> {
        if self.model.is_none() || self.tokenizers.is_none() || texts.is_empty() {
            return texts
                .iter()
                .map(|t| {
                    self.simple_get_pinyin(t)
                        .into_iter()
                        .map(out_to_string)
                        .collect()
                })
                .collect();
        }
        match self.g2p_batch_ml(texts) {
            Ok(batch) => batch
                .into_iter()
                .map(|outs| outs.into_iter().map(out_to_string).collect())
                .collect(),
            Err(_) => texts
                .iter()
                .map(|t| {
                    self.simple_get_pinyin(t)
                        .into_iter()
                        .map(out_to_string)
                        .collect()
                })
                .collect(),
        }
    }

    pub fn simple_get_pinyin(&self, text: &str) -> Vec<G2PWOut> {
        let mut pre_data = vec![];
        for c in text.chars() {
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

    fn g2p_batch_ml(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<G2PWOut>>> {
        let mut results: Vec<Vec<G2PWOut>> =
            texts.iter().map(|t| self.simple_get_pinyin(t)).collect();
        let mut queries: Vec<PolyQuery> = Vec::new();

        for (text_idx, text) in texts.iter().enumerate() {
            let chars: Vec<char> = text.chars().collect();
            let mut poly_indices = Vec::new();
            for (i, &c) in chars.iter().enumerate() {
                if DICT_POLY_CHARS.contains_key(&c) {
                    results[text_idx][i] = G2PWOut::Pinyin(String::new());
                    poly_indices.push(i);
                }
            }
            if poly_indices.is_empty() {
                continue;
            }

            let (window_text, offset) = if POLYPHONIC_CONTEXT_CHARS > 0 {
                let left = poly_indices[0].saturating_sub(POLYPHONIC_CONTEXT_CHARS);
                let right = (poly_indices[poly_indices.len() - 1] + POLYPHONIC_CONTEXT_CHARS + 1)
                    .min(chars.len());
                (chars[left..right].iter().collect::<String>(), left)
            } else {
                (text.to_string(), 0)
            };

            for &idx in &poly_indices {
                let c = chars[idx];
                let poly = DICT_POLY_CHARS.get(&c).unwrap();
                let mut phoneme_mask = vec![0f32; POLY_LABLES.len()];
                for (_, li) in &poly.phones {
                    phoneme_mask[*li] = 1.0;
                }
                queries.push(PolyQuery {
                    text_idx,
                    model_char_idx: idx - offset,
                    result_char_idx: idx,
                    char_id: poly.index,
                    phoneme_mask,
                    window_text: window_text.clone(),
                });
            }
        }

        if queries.is_empty() {
            return Ok(results);
        }

        let tokenizer = self.tokenizers.as_ref().unwrap().clone();
        // Cache: window_text -> (input_ids, char_index -> token_index including CLS offset)
        let mut encoded_cache: HashMap<String, (Vec<i64>, Vec<usize>)> = HashMap::new();
        for q in &queries {
            if encoded_cache.contains_key(&q.window_text) {
                continue;
            }
            let encoding = tokenizer
                .encode(q.window_text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("encode error: {}", e))?;
            let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
            let offsets = encoding.get_offsets();
            let chars: Vec<char> = q.window_text.chars().collect();
            let mut char_to_token = vec![0usize; chars.len()];
            let mut byte_starts = Vec::with_capacity(chars.len());
            let mut b = 0usize;
            for ch in &chars {
                byte_starts.push(b);
                b += ch.len_utf8();
            }
            for (ti, &(start, end)) in offsets.iter().enumerate() {
                if end <= start {
                    continue;
                }
                for (ci, &bs) in byte_starts.iter().enumerate() {
                    if bs >= start && bs < end {
                        char_to_token[ci] = ti;
                    }
                }
            }
            for (ci, slot) in char_to_token.iter_mut().enumerate() {
                if *slot == 0 {
                    *slot = ci + 1; // CLS at 0
                }
            }
            encoded_cache.insert(q.window_text.clone(), (ids, char_to_token));
        }

        let n = queries.len();
        let max_len = queries
            .iter()
            .map(|q| encoded_cache[&q.window_text].0.len())
            .max()
            .unwrap_or(1);

        let mut input_ids = Array2::<i64>::zeros((n, max_len));
        let token_type_ids = Array2::<i64>::zeros((n, max_len));
        let mut attention_mask = Array2::<i64>::zeros((n, max_len));
        let mut phoneme_masks = Array2::<f32>::zeros((n, POLY_LABLES.len()));
        let mut char_ids = Array1::<i64>::zeros(n);
        let mut position_ids = Array1::<i64>::zeros(n);

        for (qi, q) in queries.iter().enumerate() {
            let (ids, char_to_token) = &encoded_cache[&q.window_text];
            for (t, &id) in ids.iter().enumerate() {
                input_ids[[qi, t]] = id;
                attention_mask[[qi, t]] = 1;
            }
            for (j, &m) in q.phoneme_mask.iter().enumerate() {
                phoneme_masks[[qi, j]] = m;
            }
            char_ids[qi] = q.char_id as i64;
            position_ids[qi] = char_to_token
                .get(q.model_char_idx)
                .copied()
                .unwrap_or(q.model_char_idx + 1) as i64;
        }

        let model_output = self.model.as_mut().unwrap().run(ort::inputs![
            "input_ids" => Tensor::from_array(input_ids)?,
            "token_type_ids" => Tensor::from_array(token_type_ids)?,
            "attention_mask" => Tensor::from_array(attention_mask)?,
            "phoneme_mask" => Tensor::from_array(phoneme_masks)?,
            "char_ids" => Tensor::from_array(char_ids)?,
            "position_ids" => Tensor::from_array(position_ids)?,
        ])?;

        let probs = model_output["probs"].try_extract_array::<f32>()?;
        for (qi, q) in queries.iter().enumerate() {
            let row = probs.slice(ndarray::s![qi, ..]);
            let best = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            results[q.text_idx][q.result_char_idx] = G2PWOut::Pinyin(POLY_LABLES[best].clone());
        }

        Ok(results)
    }
}

fn out_to_string(out: G2PWOut) -> String {
    match out {
        G2PWOut::Pinyin(p) | G2PWOut::Yue(p) => p,
        G2PWOut::RawChar(c) => c.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_get_pinyin_covers_ascii() {
        let g = G2PW {
            model: None,
            tokenizers: None,
        };
        let out = g.simple_get_pinyin("啊");
        assert!(!out.is_empty());
    }
}
