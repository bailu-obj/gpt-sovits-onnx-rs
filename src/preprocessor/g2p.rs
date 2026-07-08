// Per-span G2P for mixed-language preprocessing.

use anyhow::Result;
use log::warn;

use crate::preprocessor::{
    en::{EnSentence, EnWord, g2p_en::G2pEn, normalize::text_normalize_en},
    lang::{Lang, LangId},
    lang_segment::LangSpan,
    phoneme_finalize::finalize_span_zh,
    text_normalize::text_normalize_zh,
    zh::{ZhMode, ZhSentence, g2pw::G2PW},
};
use jieba_rs::Jieba;

#[derive(Debug)]
pub struct SpanResult {
    pub text: String,
    pub word2ph: Vec<i32>,
    pub phone_ids: Vec<i64>,
    pub lang: Lang,
}

pub struct G2pDeps<'a> {
    pub g2pw: &'a mut G2PW,
    pub g2p_en: &'a mut G2pEn,
    pub jieba: &'a Jieba,
}

pub fn g2p_spans(
    spans: &[LangSpan],
    lang_id: LangId,
    deps: &mut G2pDeps<'_>,
) -> Result<Vec<SpanResult>> {
    let mut results = Vec::new();
    for span in spans {
        let normalized = match span.lang {
            Lang::Zh => text_normalize_zh(&span.text),
            Lang::En => text_normalize_en(&span.text),
        };
        if normalized.trim().is_empty() {
            continue;
        }

        match span.lang {
            Lang::Zh => {
                let mode = if matches!(lang_id, LangId::AutoYue) {
                    ZhMode::Cantonese
                } else {
                    ZhMode::Mandarin
                };
                let mut zh = ZhSentence {
                    text: normalized.clone(),
                    ..Default::default()
                };
                zh.g2p(deps.g2pw, deps.jieba, mode);
                if zh.phones.is_empty() {
                    continue;
                }
                let phone_ids = finalize_span_zh(zh.phones, &zh.text);
                results.push(SpanResult {
                    text: zh.text,
                    word2ph: zh.word2ph,
                    phone_ids,
                    lang: Lang::Zh,
                });
            }
            Lang::En => {
                let mut en = EnSentence {
                    text: vec![EnWord::Word(normalized.clone())],
                    ..Default::default()
                };
                if let Err(e) = en.g2p_from_normalized(deps.g2p_en, &normalized) {
                    warn!("English G2P failed for '{}': {}", span.text, e);
                    continue;
                }
                if en.phone_ids.is_empty() {
                    continue;
                }
                results.push(SpanResult {
                    text: en.get_text_string(),
                    word2ph: en.word2ph,
                    phone_ids: en.phone_ids,
                    lang: Lang::En,
                });
            }
        }
    }
    Ok(results)
}
