use std::{path::Path, str::FromStr, sync::Arc};

use anyhow::Ok;
use log::{debug, warn};
use ndarray::Array2;
use ort::{inputs, value::Tensor};
use tokenizers::Tokenizer;

use crate::{
    onnx_builder::create_onnx_cpu_session,
    preprocessor::{lang, utils::BERT_TOKENIZER},
};

#[derive(Debug)]
pub struct BertModel {
    model: Option<ort::session::Session>,
    tokenizers: Option<Arc<tokenizers::Tokenizer>>,
}

impl BertModel {
    pub fn new<P: AsRef<Path>>(path: Option<P>) -> anyhow::Result<Self> {
        let mut model = None;
        if let Some(path) = path {
            model = Some(create_onnx_cpu_session(path)?);
        }
        Ok(Self {
            model: model,
            tokenizers: Some(Arc::new(Tokenizer::from_str(BERT_TOKENIZER).unwrap())),
        })
    }

    pub fn get_bert(
        &mut self,
        text: &str,
        word2ph: &[i32],
        total_phones: usize,
        lang: lang::Lang,
    ) -> anyhow::Result<Array2<f32>> {
        if self.model.is_some() && self.tokenizers.is_some() && lang == lang::Lang::Zh {
            let text = &text.replace(",", ".");
            let tmp = self.get_real_bert(text, word2ph)?;
            debug!("use real bert, {}", text);
            if tmp.shape()[0] != total_phones {
                let error_msg = format!(
                    "BERT output length mismatch for text '{}': expected {}, got {}",
                    text,
                    total_phones,
                    tmp.shape()[0]
                );
                warn!("{}", error_msg);
                if cfg!(debug_assertions) {
                    return Err(anyhow::anyhow!(error_msg));
                }
                return Ok(self.get_fake_bert(total_phones));
            }
            Ok(tmp)
        } else {
            debug!("use empty bert, {}", text);
            Ok(self.get_fake_bert(total_phones))
        }
    }

    fn get_real_bert(&mut self, text: &str, word2ph: &[i32]) -> anyhow::Result<Array2<f32>> {
        let tokenizer = self.tokenizers.as_ref().unwrap();
        let session = self.model.as_mut().unwrap();

        let encoding = tokenizer.encode(text, true).unwrap();
        let (input_ids, attention_mask, token_type_ids): (Vec<i64>, Vec<i64>, Vec<i64>) = (
            encoding.get_ids().iter().map(|&id| id as i64).collect(),
            encoding
                .get_attention_mask()
                .iter()
                .map(|&m| m as i64)
                .collect(),
            encoding.get_type_ids().iter().map(|&t| t as i64).collect(),
        );

        let inputs = inputs![
            "input_ids" => Tensor::from_array(Array2::from_shape_vec((1, input_ids.len()), input_ids).unwrap()).unwrap(),
            "attention_mask" => Tensor::from_array(Array2::from_shape_vec((1, attention_mask.len()), attention_mask).unwrap()).unwrap(),
            "token_type_ids" => Tensor::from_array(Array2::from_shape_vec((1, token_type_ids.len()), token_type_ids).unwrap()).unwrap()
        ];

        let bert_out = session.run(inputs)?;
        let bert_feature = bert_out["bert_feature"]
            .try_extract_array::<f32>()?
            .to_owned();

        let bert_feature_2d: Array2<f32> = bert_feature.into_dimensionality()?;

        Ok(build_phone_level_feature(bert_feature_2d, word2ph))
    }

    fn get_fake_bert(&self, total_phones: usize) -> Array2<f32> {
        // The BERT model outputs features of size 1024
        Array2::<f32>::zeros((total_phones, 1024))
    }
}

// Expands word-level BERT features to phone-level features in one allocation.
fn build_phone_level_feature(res: Array2<f32>, word2ph: &[i32]) -> Array2<f32> {
    let ncols = res.ncols();
    let n_words = res.dim().0;
    let last_word = n_words.saturating_sub(1);
    let total_rows: usize = word2ph.iter().map(|&c| c.max(0) as usize).sum();
    let mut out = Array2::<f32>::zeros((total_rows, ncols));
    let mut row = 0usize;
    for (i, &count_raw) in word2ph.iter().enumerate() {
        let count = count_raw.max(0) as usize;
        if count == 0 {
            continue;
        }
        let src_row = if i < n_words {
            res.row(i)
        } else {
            res.row(last_word)
        };
        for r in 0..count {
            out.row_mut(row + r).assign(&src_row);
        }
        row += count;
    }
    out
}
