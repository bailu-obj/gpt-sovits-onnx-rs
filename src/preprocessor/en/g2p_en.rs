// text/en/g2p_en.rs
use std::{collections::HashMap, path::Path, str::FromStr};

use anyhow::{Ok, Result};
use arpabet::Arpabet;
use log::debug;
use ndarray::{Array, s};
use once_cell::sync::Lazy;
use ort::{inputs, session::Session, value::Tensor};
use regex::Regex;
use tokenizers::Tokenizer;

use crate::{onnx_builder::create_onnx_cpu_session, preprocessor::dict};

static MINI_BART_G2P_TOKENIZER: &str =
    include_str!("../../../resource/tokenizer.mini-bart-g2p.json");

static DECODER_START_TOKEN_ID: u32 = 2;

#[allow(unused)]
static BOS_TOKEN: &str = "<s>";
#[allow(unused)]
static EOS_TOKEN: &str = "</s>";

#[allow(unused)]
static BOS_TOKEN_ID: u32 = 0;
static EOS_TOKEN_ID: u32 = 2;

static TOKEN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+(?:\.\d+)?|[^\s\w]").unwrap());

static HOMOGRAPHS: Lazy<HashMap<&'static str, Vec<&'static str>>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("read", vec!["R", "IY1", "D"]);
    m.insert("complex", vec!["K", "AH0", "M", "P", "L", "EH1", "K", "S"]);
    m
});

pub struct G2PEnModel {
    encoder_model: Session,
    decoder_model: Session,
    tokenizer: Tokenizer,
}

impl G2PEnModel {
    pub fn new<P: AsRef<Path>>(encoder_path: P, decoder_path: P) -> Result<Self> {
        let encoder_model = create_onnx_cpu_session(encoder_path)?;
        let decoder_model = create_onnx_cpu_session(decoder_path)?;
        let tokenizer = Tokenizer::from_str(MINI_BART_G2P_TOKENIZER)
            .map_err(|e| anyhow::anyhow!("load g2p_en tokenizer error: {}", e))?;

        Ok(Self {
            encoder_model,
            decoder_model,
            tokenizer,
        })
    }

    pub fn get_phoneme(&mut self, text: &str) -> Result<Vec<String>> {
        debug!("processing {:?}", text);
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("encode error: {}", e))?;
        let input_ids = encoding
            .get_ids()
            .iter()
            .map(|x| *x as i64)
            .collect::<Vec<i64>>();
        let mut decoder_input_ids = vec![DECODER_START_TOKEN_ID as i64];

        let input_id_len = input_ids.len();
        let input_ids_tensor =
            Tensor::from_array(Array::from_shape_vec((1, input_id_len), input_ids.clone())?)?;
        let attention_mask_tensor =
            Tensor::from_array(Array::from_elem((1, input_id_len), 1 as i64))?;
        let encoder_outputs = self.encoder_model.run(inputs![
            "input_ids" => input_ids_tensor.clone(),
            "attention_mask" => attention_mask_tensor.clone()
        ])?;

        for _ in 0..50 {
            let encoder_output = encoder_outputs["last_hidden_state"].view();

            let decoder_input_ids_tensor = Tensor::from_array(Array::from_shape_vec(
                (1, decoder_input_ids.len()),
                decoder_input_ids.clone(),
            )?)?;

            let outputs = self.decoder_model.run(inputs![
                "input_ids" => decoder_input_ids_tensor,
                "encoder_attention_mask" => attention_mask_tensor.clone(),
                "encoder_hidden_states" => encoder_output,
            ])?;

            let output_array = outputs["logits"].try_extract_array::<f32>()?;

            let last_token_logits = &output_array.slice(s![0, output_array.shape()[1] - 1, ..]);

            let next_token_id = last_token_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .ok_or_else(|| anyhow::anyhow!("failed to compute argmax"))?;

            decoder_input_ids.push(next_token_id);
            if next_token_id == EOS_TOKEN_ID as i64 {
                break;
            }
        }

        let decoder_input_ids = decoder_input_ids
            .iter()
            .map(|x| *x as u32)
            .collect::<Vec<u32>>();
        Ok(self
            .tokenizer
            .decode(&decoder_input_ids, true)
            .map_err(|e| anyhow::anyhow!("g2p_en decode error: {}", e))?
            .split(" ")
            .map(|v| v.to_owned())
            .collect::<Vec<String>>())
    }
}

pub struct G2pEn {
    model: Option<G2PEnModel>,
    arpabet: Arpabet,
}

impl G2pEn {
    pub fn new<P: AsRef<Path>>(path: Option<P>) -> Result<Self> {
        let arpabet = arpabet::load_cmudict().clone();
        if let Some(path) = path {
            let path = path.as_ref();
            Ok(G2pEn {
                model: Some(G2PEnModel::new(
                    path.join("encoder_model.onnx"),
                    path.join("decoder_model.onnx"),
                )?),
                arpabet: arpabet,
            })
        } else {
            Ok(G2pEn {
                model: None,
                arpabet: arpabet,
            })
        }
    }

    pub fn g2p(&mut self, text: &str) -> Result<Vec<String>> {
        if let Some(v) = dict::en_word_dict(text) {
            return Ok(v.to_owned());
        }

        let tokens = tokenize_en(text);
        let mut phonemes = Vec::new();
        for token in tokens {
            phonemes.extend(self.g2p_token(&token)?);
        }
        Ok(phonemes)
    }

    fn g2p_token(&mut self, token: &str) -> Result<Vec<String>> {
        if token.chars().all(|c| !c.is_ascii_alphabetic()) {
            return Ok(vec![token.to_string()]);
        }

        let lower = token.to_lowercase();

        // Possessive 's (Python qryword)
        if let Some(stem) = lower.strip_suffix("'s") {
            if !stem.is_empty() {
                let mut phones = self.g2p_token(stem)?;
                if let Some(last) = phones.last().map(|s| s.as_str()) {
                    if matches!(last, "P" | "T" | "K" | "F" | "TH" | "HH") {
                        phones.push("S".to_string());
                    } else if matches!(last, "S" | "Z" | "SH" | "ZH" | "CH" | "JH") {
                        phones.extend(["AH0".to_string(), "Z".to_string()]);
                    } else {
                        phones.push("Z".to_string());
                    }
                }
                return Ok(phones);
            }
        }

        if let Some(v) = dict::en_word_dict(&lower) {
            return Ok(v.to_owned());
        }
        if let Some(v) = dict::en_word_dict(token) {
            return Ok(v.to_owned());
        }

        if let Some(phones) = homograph_phones(&lower) {
            return Ok(phones);
        }

        if token.len() == 1 {
            if token == "A" {
                return Ok(vec!["EY1".to_string()]);
            }
            if let Some(phones) = self.arpabet.get_polyphone_str(&lower) {
                return Ok(phones.iter().map(|&p| p.to_string()).collect());
            }
        }

        if lower.len() > 1 {
            if let Some(phones) = self.arpabet.get_polyphone_str(&lower) {
                return Ok(phones.iter().map(|&p| p.to_string()).collect());
            }
        }

        // Hyphenated compound split
        if lower.contains('-') {
            let mut phones = Vec::new();
            for part in lower.split('-') {
                if part.is_empty() {
                    continue;
                }
                phones.extend(self.g2p_token(part)?);
            }
            if !phones.is_empty() {
                return Ok(phones);
            }
        }

        if let Some(model) = &mut self.model {
            return model.get_phoneme(&lower);
        }

        if lower.len() <= 3 {
            let mut phones = Vec::new();
            for c in lower.chars() {
                let c_str = c.to_string();
                if c == 'a' {
                    phones.push("EY1".to_string());
                } else if let Some(p) = self.arpabet.get_polyphone_str(&c_str) {
                    phones.extend(p.iter().map(|&s| s.to_string()));
                } else {
                    phones.push(c_str);
                }
            }
            return Ok(phones);
        }

        Ok(vec![token.to_string()])
    }
}

fn tokenize_en(text: &str) -> Vec<String> {
    TOKEN_RE
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .filter(|t| !t.chars().all(|c| c.is_whitespace()))
        .collect()
}

fn homograph_phones(word: &str) -> Option<Vec<String>> {
    HOMOGRAPHS
        .get(word)
        .map(|phones| phones.iter().map(|p| p.to_string()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_en() {
        let tokens = tokenize_en("hello, world");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&",".to_string()));
    }

    #[test]
    fn test_homograph_read() {
        let phones = homograph_phones("read").unwrap();
        assert_eq!(phones[0], "R");
    }

    #[test]
    fn test_possessive() {
        let mut g2p = G2pEn::new(None::<&str>).unwrap();
        let phones = g2p.g2p("cat's").unwrap();
        assert!(!phones.is_empty());
    }
}
