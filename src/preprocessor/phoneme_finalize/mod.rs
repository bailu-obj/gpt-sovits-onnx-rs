// Phoneme finalization after G2P: symbol mapping, validation, merge, retry.

use anyhow::Result;
use ndarray::Array2;

use crate::preprocessor::{
    bert::BertModel,
    clean::SilenceTag,
    g2p::SpanResult,
};

pub mod english;
pub mod merge;
pub mod retry;
pub mod silence;
pub mod symbols;
pub mod validate;

pub use english::{filter_english_phonemes, finalize_span_en};
pub use merge::merge_span_results;
pub use retry::{needs_short_retry, MIN_PHONES};
pub use symbols::{pad_short_english_phones, phone_to_id, phones_to_ids};
pub use validate::validate_word2ph;

pub struct ChunkInput {
    pub spans: Vec<SpanResult>,
    pub silence_tags: Vec<SilenceTag>,
}

pub struct ChunkOutput {
    pub norm_text: String,
    pub phone_ids: Vec<i64>,
    pub bert: Array2<f32>,
}

/// Merge spans, inject silence tokens, validate alignment — single chunk entry point.
pub fn finalize_chunk(bert: &mut BertModel, input: ChunkInput) -> Result<ChunkOutput> {
    let (norm_text, phone_ids, bert_features) =
        merge_span_results(bert, input.spans, &input.silence_tags)?;
    Ok(ChunkOutput {
        norm_text,
        phone_ids,
        bert: bert_features,
    })
}

/// Mandarin span: symbol strings → integer IDs.
pub fn finalize_span_zh(phones: Vec<String>, context: &str) -> Vec<i64> {
    phones_to_ids(&phones, context)
}
