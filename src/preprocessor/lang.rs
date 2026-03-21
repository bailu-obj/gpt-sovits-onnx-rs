// preprocessor/lang.rs
use crate::preprocessor::processor;
use crate::preprocessor::sentence::Sentence;
use crate::preprocessor::utils::is_numeric_or_punctuation;
use jieba_rs::Jieba;
use log::debug;
use once_cell::sync::Lazy;
use regex::Regex;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lang {
    Zh,
    En,
}

impl Default for Lang {
    fn default() -> Self {
        Lang::Zh
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LangId {
    Auto,    // Mandarin
    AutoYue, // Cantonese
}

/// One `TOKEN_REGEX` scan over `text`: Jieba refines Han spans; other spans use the same regex
/// tokens. Digits / punctuation after Chinese stay `Lang::Zh` (e.g. `中文123`); leading `2024`
/// stays English. Whitespace-only matches do not advance that “previous language” state so a
/// space between `中文` and `123` does not flip context to English.
pub fn lang_split(text: &str, jieba: &Jieba) -> Vec<Sentence> {
    let mut sentences = Vec::with_capacity(16);
    let mut prev_lang: Option<Lang> = None;
    debug!("Lang split: {}", text);

    for m in TOKEN_REGEX.find_iter(text) {
        let token = m.as_str();
        if token.trim().is_empty() {
            continue;
        }

        let has_han = HAN_ONLY.is_match(token);
        let content_lang = if has_han { Lang::Zh } else { Lang::En };
        let attaches_after_zh = is_numeric_or_punctuation(token)
            || processor::parse_punctuation(token).is_some();
        let token_lang = if prev_lang == Some(Lang::Zh) && attaches_after_zh {
            Lang::Zh
        } else {
            content_lang
        };

        let advances_prev = !token.chars().all(|c| c.is_whitespace());

        if has_han {
            for word in jieba.cut(token, true) {
                if word.trim().is_empty() {
                    continue;
                }
                processor::lang_process_token(&mut sentences, word, Lang::Zh);
            }
            if advances_prev {
                prev_lang = Some(Lang::Zh);
            }
        } else {
            processor::lang_process_token(&mut sentences, token, token_lang);
            if advances_prev {
                prev_lang = Some(token_lang);
            }
        }
    }

    sentences
}

pub(crate) static TOKEN_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?x)
        [\p{Han}]+ |              # Chinese characters
        [a-zA-Z]+(?:['-][a-zA-Z]+)* | # English words with optional apostrophes/hyphens
        \d+(?:\.\d+)? |          # Numbers (including decimals)
        [.,!?;:()\[\]<>\-\"$/\u{3001}\u{3002}\u{FF01}\u{FF1F}\u{FF1B}\u{FF1A}\u{FF0C}\u{2018}\u{2019}\u{201C}\u{201D}] | # Punctuation
        \s+                      # Whitespace
        "#,
    )
    .unwrap()
});

static HAN_ONLY: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\p{Han}+$").unwrap());
