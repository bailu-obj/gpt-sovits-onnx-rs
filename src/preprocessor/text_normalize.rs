// preprocessor/text_normalize.rs
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

use crate::preprocessor::{
    seg::replace_consecutive_punctuation,
    zh::number_normalization::{normalize_numbers, spoken_symbol},
};

fn normalize_width(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '！'..='～' => char::from_u32(c as u32 - 0xFEE0).unwrap(),
            '　' => ' ',
            _ => c,
        })
        .collect()
}

/// Normalize punctuation without losing the distinction between a question,
/// an exclamation, an ellipsis and a period.
pub fn text_normalize(text: &str) -> String {
    let text = normalize_width(text);
    let text = CLEANUP_REGEX.replace_all(&text, " ");
    let text = text
        .replace("...", "…")
        .replace('。', ".")
        .replace('、', ",");
    replace_consecutive_punctuation(&text)
}

/// Chinese punctuation replacement after TextNormalizer (Python `replace_punctuation`).
pub fn replace_punctuation_zh(text: &str) -> String {
    let mut result = text.replace('嗯', "恩").replace('呣', "母");
    for (from, to) in REP_MAP_ZH {
        result = result.replace(from, to);
    }
    ZH_UNSPOKEN.replace_all(&result, "").into_owned()
}

static ZH_UNSPOKEN: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^\u{4e00}-\u{9fa5}!?,….]+").unwrap());

static TRADITIONAL_TO_SIMPLIFIED: Lazy<HashMap<char, char>> = Lazy::new(|| {
    include_str!("../../resource/zh_traditional_to_simplified.tsv")
        .lines()
        .filter_map(|line| {
            let (from, to) = line.split_once('\t')?;
            Some((from.chars().next()?, to.chars().next()?))
        })
        .collect()
});

/// Mandarin normalization before Jieba, G2PW and BERT.
pub fn text_normalize_zh(text: &str) -> String {
    let mut mandarin = String::with_capacity(text.len());
    for c in normalize_width(text).chars() {
        if (c.is_whitespace() && c != '\n') || "《》【】<>{}()[]#&@“”^_|\\".contains(c) {
            continue;
        }
        if let Some(spoken) = spoken_symbol(c) {
            mandarin.push_str(spoken);
        } else {
            mandarin.push(TRADITIONAL_TO_SIMPLIFIED.get(&c).copied().unwrap_or(c));
        }
    }
    let expanded = normalize_numbers(&mandarin);
    replace_consecutive_punctuation(&replace_punctuation_zh(&expanded))
}

const REP_MAP_ZH: &[(&str, &str)] = &[
    ("：", ","),
    ("；", ","),
    ("，", ","),
    ("。", "."),
    ("！", "!"),
    ("？", "?"),
    ("\n", "."),
    ("·", ","),
    ("、", ","),
    ("...", "…"),
    ("$", "."),
    ("/", ","),
    ("—", "-"),
    ("~", "…"),
    ("～", "…"),
];

// Regex to handle emojis and symbols
static CLEANUP_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F900}-\u{1F9FF}\u{2600}-\u{27BF}\u{200B}-\u{200F}\u{202A}-\u{202E}\u{2060}-\u{206F}\u{2300}-\u{23FF}]+",
    )
    .unwrap()
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_marks_keep_sentence_boundaries() {
        let text = text_normalize("风凉吗？午饭吃了吗？我们走走吧。");
        assert_eq!(
            super::super::seg::pre_seg_text(&text, false),
            vec!["。风凉吗?", "午饭吃了吗?", "我们走走吧."]
        );
    }

    #[test]
    fn mandarin_normalization_matches_python_spacing() {
        assert_eq!(text_normalize_zh("问题, 请修复它。"), "问题,请修复它.");
    }
}
