// preprocessor/mod.rs — TextProcessor orchestration
use anyhow::Result;
use log::{debug, warn};
use ndarray::{Array2, ArrayBase, Dim, OwnedRepr};

pub mod bert;
pub mod clean;
pub mod dict;
pub mod en;
pub mod g2p;
pub mod lang;
pub mod lang_segment;
pub mod num;
pub mod phone_symbol;
pub mod phoneme_finalize;
pub mod seg;
pub mod text_normalize;
pub mod tokenize;
pub mod utils;
pub mod zh;

pub use lang::{Lang, LangId};
pub use text_normalize::text_normalize;

use crate::preprocessor::{
    bert::BertModel,
    clean::{infer_lang_en, normalize_input, silence_tags_for_chunk, CleanedInput},
    en::g2p_en::G2pEn,
    g2p::{G2pDeps, g2p_spans},
    phoneme_finalize::{needs_short_retry, ChunkInput, ChunkOutput},
    seg::pre_seg_text,
    zh::g2pw::G2PW,
};
use jieba_rs::Jieba;
use std::sync::Arc;

pub struct TextProcessor {
    pub jieba: Arc<Jieba>,
    pub g2pw: G2PW,
    pub g2p_en: G2pEn,
    pub bert: BertModel,
}

impl TextProcessor {
    pub fn new(g2pw: G2PW, g2p_en: G2pEn, bert: BertModel) -> Result<Self> {
        Ok(Self {
            jieba: Arc::new(Jieba::new()),
            g2pw,
            g2p_en,
            bert,
        })
    }

    pub fn get_phone_and_bert(
        &mut self,
        text: &str,
        lang_id: LangId,
    ) -> Result<Vec<(String, Vec<i64>, Array2<f32>)>> {
        let cleaned = normalize_input(text)?;
        let lang_en = infer_lang_en(&cleaned.text);
        debug!("Cleaned text: {}", cleaned.text);
        let chunks = pre_seg_text(&cleaned.text, lang_en);
        let mut result: Vec<(String, Vec<i64>, ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>)> =
            Vec::with_capacity(chunks.len());

        for chunk in chunks {
            debug!("Processing chunk: {}", chunk);
            if chunk.trim().is_empty() {
                continue;
            }

            match self.process_chunk(&cleaned, &chunk, lang_id, false) {
                Ok(Some(output)) => result.push((output.norm_text, output.phone_ids, output.bert)),
                Ok(None) => {}
                Err(e) => warn!("Failed to process chunk '{}': {}", chunk, e),
            }
        }

        debug!("RESULT (total sentences: {})", result.len());
        if result.is_empty() {
            return Err(anyhow::anyhow!(
                "No phonemes or BERT features could be generated for the text: {}",
                text
            ));
        }
        Ok(result)
    }

    fn process_chunk(
        &mut self,
        cleaned: &CleanedInput,
        chunk: &str,
        lang_id: LangId,
        is_final: bool,
    ) -> Result<Option<ChunkOutput>> {
        let spans = lang_segment::get_spans(chunk, &self.jieba);
        if spans.is_empty() {
            return Ok(None);
        }

        let silence_tags =
            silence_tags_for_chunk(&cleaned.silence_markers, &cleaned.text, chunk);

        let mut deps = G2pDeps {
            g2pw: &mut self.g2pw,
            g2p_en: &mut self.g2p_en,
            jieba: &self.jieba,
        };
        let span_results = g2p_spans(&spans, lang_id, &mut deps)?;
        if span_results.is_empty() {
            return Ok(None);
        }

        let output = phoneme_finalize::finalize_chunk(
            &mut self.bert,
            ChunkInput {
                spans: span_results,
                silence_tags,
            },
        )?;

        if needs_short_retry(output.phone_ids.len(), is_final) {
            let retry_chunk = format!(".{}", chunk);
            return self.process_chunk(cleaned, &retry_chunk, lang_id, true);
        }

        Ok(Some(output))
    }
}
