// Language segmentation for mixed zh/en/digit input (port of LangSegmenter.getTexts)

use crate::preprocessor::lang::Lang;
use jieba_rs::Jieba;
use once_cell::sync::Lazy;
use regex::Regex;

use super::tokenize::{HAN_ONLY, TOKEN_REGEX};

#[derive(Debug, Clone)]
pub struct LangSpan {
    pub lang: Lang,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawLang {
    Zh,
    En,
    Digit,
    Punct,
    Unknown,
}

static DIGIT_ONLY: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\d+(?:\.\d+)?%?$").unwrap());
static ASCII_WORD: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[a-zA-Z]+(?:['-][a-zA-Z]+)*$").unwrap());

const PUNCT_CHARS: &[char] = &['，', '。', '！', '？', ',', '.', '!', '?'];

/// Segment text into language spans (Python `LangSegmenter.getTexts` for auto mode).
pub fn get_spans(text: &str, jieba: &Jieba) -> Vec<LangSpan> {
    let raw = tokenize_raw(text, jieba);
    let resolved = assign_digit_lang_v2(raw);
    merge_adjacent(resolved)
}

fn tokenize_raw(text: &str, jieba: &Jieba) -> Vec<(RawLang, String)> {
    let mut spans: Vec<(RawLang, String)> = Vec::new();
    for m in TOKEN_REGEX.find_iter(text) {
        let token = m.as_str();
        if token.chars().all(|c| c.is_whitespace()) {
            if let Some((_, last_text)) = spans.last_mut() {
                if !last_text.ends_with(' ') {
                    last_text.push(' ');
                }
            }
            continue;
        }
        if token.trim().is_empty() {
            continue;
        }
        if HAN_ONLY.is_match(token) {
            for word in jieba.cut(token, true) {
                if word.trim().is_empty() {
                    continue;
                }
                push_raw(&mut spans, RawLang::Zh, word);
            }
        } else if DIGIT_ONLY.is_match(token) {
            push_raw(&mut spans, RawLang::Digit, token);
        } else if ASCII_WORD.is_match(token) {
            push_raw(&mut spans, RawLang::En, token);
        } else if is_punct_token(token) {
            push_raw(&mut spans, RawLang::Punct, token);
        } else if token.chars().all(|c| c.is_ascii_digit() || c == '.') {
            push_raw(&mut spans, RawLang::Digit, token);
        } else if token.chars().any(|c| c.is_ascii_alphabetic()) {
            push_raw(&mut spans, RawLang::En, token);
        } else if token.chars().any(|c| is_han(c)) {
            push_raw(&mut spans, RawLang::Zh, token);
        } else {
            push_raw(&mut spans, RawLang::Unknown, token);
        }
    }
    spans
}

fn push_raw(spans: &mut Vec<(RawLang, String)>, lang: RawLang, text: &str) {
    if lang == RawLang::Punct {
        if let Some((_, last_text)) = spans.last_mut() {
            last_text.push_str(text);
            return;
        }
    }
    if let Some((last_lang, last_text)) = spans.last_mut() {
        if *last_lang == lang && lang != RawLang::Punct {
            if lang == RawLang::En
                && !last_text.ends_with(' ')
                && !starts_with_punct(text)
                && !ends_with_punct(last_text)
            {
                last_text.push(' ');
            }
            last_text.push_str(text);
            return;
        }
    }
    spans.push((lang, text.to_string()));
}

fn assign_digit_lang_v2(spans: Vec<(RawLang, String)>) -> Vec<(Lang, String)> {
    let n = spans.len();
    if n == 0 {
        return Vec::new();
    }

    let mut langs: Vec<Lang> = spans
        .iter()
        .map(|(l, _)| match l {
            RawLang::En => Lang::En,
            RawLang::Zh | RawLang::Punct => Lang::Zh,
            RawLang::Digit | RawLang::Unknown => Lang::Zh,
        })
        .collect();

    if spans.iter().any(|(l, _)| *l == RawLang::Digit) {
        for i in 0..n {
            if spans[i].0 != RawLang::Digit {
                continue;
            }
            langs[i] = if i > 0 && spans[i - 1].0 == RawLang::Zh {
                Lang::Zh
            } else if i + 1 < n && spans[i + 1].0 == RawLang::Zh {
                Lang::Zh
            } else if i > 0 && spans[i - 1].0 == RawLang::En {
                Lang::En
            } else if i + 1 < n && spans[i + 1].0 == RawLang::En {
                Lang::En
            } else if i > 0 {
                langs[i - 1]
            } else if i + 1 < n {
                langs[i + 1]
            } else {
                Lang::Zh
            };

            if i > 0 && i + 1 < n {
                let prev_text = &spans[i - 1].1;
                let next_text = &spans[i + 1].1;
                if langs[i - 1] == langs[i + 1] {
                    langs[i] = langs[i - 1];
                } else if prev_text
                    .chars()
                    .last()
                    .map_or(false, |c| PUNCT_CHARS.contains(&c))
                {
                    langs[i] = langs[i + 1];
                } else if next_text
                    .chars()
                    .next()
                    .map_or(false, |c| PUNCT_CHARS.contains(&c))
                {
                    langs[i] = langs[i - 1];
                } else if prev_text.chars().count() >= next_text.chars().count() {
                    langs[i] = langs[i - 1];
                } else {
                    langs[i] = langs[i + 1];
                }
            }
        }
    }

    for i in 0..n {
        if full_en(&spans[i].1) {
            langs[i] = Lang::En;
        }
    }

    for i in 0..n {
        if spans[i].0 == RawLang::Unknown {
            let cjk = full_cjk(&spans[i].1);
            if !cjk.is_empty() {
                langs[i] = Lang::Zh;
            } else if i > 0 {
                langs[i] = langs[i - 1];
            } else if i + 1 < n {
                langs[i] = langs[i + 1];
            } else {
                langs[i] = Lang::Zh;
            }
        }
    }

    let mut result: Vec<(Lang, String)> = Vec::new();
    for i in 0..n {
        let text = if spans[i].0 == RawLang::Unknown {
            let cjk = full_cjk(&spans[i].1);
            if cjk.is_empty() {
                spans[i].1.clone()
            } else {
                cjk
            }
        } else {
            spans[i].1.clone()
        };
        merge_lang(&mut result, langs[i], text);
    }
    result
}

fn merge_lang(result: &mut Vec<(Lang, String)>, lang: Lang, text: String) {
    if let Some((last_lang, last_text)) = result.last_mut() {
        if *last_lang == lang {
            if lang == Lang::En
                && !last_text.ends_with(' ')
                && !starts_with_punct(&text)
                && !ends_with_punct(last_text)
            {
                last_text.push(' ');
            }
            last_text.push_str(&text);
            return;
        }
    }
    result.push((lang, text));
}

fn merge_adjacent(spans: Vec<(Lang, String)>) -> Vec<LangSpan> {
    spans
        .into_iter()
        .map(|(lang, text)| LangSpan { lang, text })
        .filter(|s| !s.text.trim().is_empty())
        .collect()
}

/// Python `full_en`: ASCII-alphanumeric span with optional punctuation.
pub fn full_en(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    let has_alpha = text.chars().any(|c| c.is_ascii_alphabetic());
    if !has_alpha {
        return false;
    }
    text.chars().all(|c| {
        c.is_ascii_alphanumeric()
            || c.is_ascii_punctuation()
            || c.is_ascii_whitespace()
            || "，。！？,.!?;:()[]{}'\"".contains(c)
    })
}

/// Extract CJK characters from unknown span.
pub fn full_cjk(text: &str) -> String {
    text.chars()
        .filter(|c| is_han(*c) || PUNCT_CHARS.contains(c) || c.is_ascii_digit())
        .collect()
}

fn is_han(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF)
}

fn is_punct_token(token: &str) -> bool {
    token.chars().all(|c| {
        PUNCT_CHARS.contains(&c)
            || matches!(c, ';' | ':' | '(' | ')' | '[' | ']' | '<' | '>' | '-' | '~' | '$' | '/')
    })
}

fn starts_with_punct(text: &str) -> bool {
    text.chars()
        .next()
        .map_or(false, |c| c.is_ascii_punctuation() || PUNCT_CHARS.contains(&c))
}

fn ends_with_punct(text: &str) -> bool {
    text.chars()
        .last()
        .map_or(false, |c| c.is_ascii_punctuation() || PUNCT_CHARS.contains(&c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_en() {
        assert!(full_en("hello"));
        assert!(full_en("GPT-SoVITS"));
        assert!(!full_en("你好"));
    }

    #[test]
    fn test_digit_before_zh_unit() {
        let jieba = Jieba::new();
        let spans = get_spans("2024年", &jieba);
        assert!(!spans.is_empty());
        assert_eq!(spans[0].lang, Lang::Zh);
    }

    #[test]
    fn test_preserves_space_after_comma() {
        let jieba = Jieba::new();
        let spans = get_spans("了, 这是一个Test", &jieba);
        let combined: String = spans.iter().map(|s| s.text.as_str()).collect();
        assert!(
            combined.contains(", 这") || spans.iter().any(|s| s.text.contains(", 这")),
            "expected space after comma, got spans: {:?}",
            spans
        );
    }

    #[test]
    fn test_mixed_zh_en() {
        let jieba = Jieba::new();
        let spans = get_spans("你好hello世界", &jieba);
        assert!(spans.len() >= 2);
    }

    #[test]
    fn test_preserves_english_word_spaces() {
        let jieba = Jieba::new();
        let spans = get_spans("Change, Do you like it?", &jieba);
        let en_text: String = spans
            .iter()
            .filter(|s| s.lang == Lang::En)
            .map(|s| s.text.as_str())
            .collect();
        assert!(
            en_text.contains("Do you like it"),
            "expected spaced English words, got {:?}",
            en_text
        );
    }
}
