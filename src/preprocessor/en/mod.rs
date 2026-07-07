use crate::preprocessor::en::{g2p_en::G2pEn, normalize::text_normalize_en};
use anyhow::Result;
use log::debug;
use std::borrow::Cow;

pub mod g2p_en;
pub mod normalize;

#[derive(PartialEq, Eq, Clone)]
pub enum EnWord {
    Word(String),
    Punctuation(&'static str),
}

impl std::fmt::Debug for EnWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnWord::Word(w) => write!(f, "\"{}\"", w),
            EnWord::Punctuation(p) => write!(f, "\"{}\"", p),
        }
    }
}

#[derive(Debug, Default)]
pub struct EnSentence {
    pub phone_ids: Vec<i64>,
    pub phones: Vec<Cow<'static, str>>,
    pub word2ph: Vec<i32>,
    pub text: Vec<EnWord>,
}

impl EnSentence {
    pub fn g2p(&mut self, g2p_en: &mut G2pEn) -> Result<()> {
        let raw = self.get_text_string();
        let normalized = text_normalize_en(&raw);
        self.g2p_from_normalized(g2p_en, &normalized)
    }

    /// Phonemize already-normalized English text; phoneme_finalize happens via phoneme_finalize::finalize_span_en.
    pub fn g2p_from_normalized(&mut self, g2p_en: &mut G2pEn, normalized: &str) -> Result<()> {
        self.phones.clear();
        self.phone_ids.clear();
        self.word2ph.clear();

        let phonemes = g2p_en.g2p(normalized)?;
        let (phone_ids, word2ph) =
            crate::preprocessor::phoneme_finalize::finalize_span_en(phonemes);

        self.phone_ids = phone_ids;
        self.word2ph = word2ph;
        for id in &self.phone_ids {
            self.phones.push(Cow::Owned(id.to_string()));
        }

        debug!("EnSentence phone_ids: {:?}", self.phone_ids);
        Ok(())
    }

    pub fn get_text_string(&self) -> String {
        let mut result = String::with_capacity(self.text.len() * 5);
        for w in &self.text {
            match w {
                EnWord::Word(s) => {
                    if !result.is_empty() && !result.ends_with(' ') {
                        result.push(' ');
                    }
                    result.push_str(s);
                }
                EnWord::Punctuation(p) => result.push_str(p),
            }
        }
        result
    }
}
