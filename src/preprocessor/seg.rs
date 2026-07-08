// Python-compatible text segmentation (TextPreprocessor.pre_seg_text + text_segmentation_method)

use once_cell::sync::Lazy;
use regex::Regex;

pub const SPLITS: &[char] = &[
    '，', '。', '？', '！', ',', '.', '?', '!', '~', ':', '：', '—', '…',
];

static PURE_SYMBOL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\W+$").unwrap());

fn is_split(c: char) -> bool {
    SPLITS.contains(&c)
}

/// Collapse consecutive punctuation (Python `replace_consecutive_punctuation`).
pub fn replace_consecutive_punctuation(text: &str) -> String {
    let punct: String = SPLITS
        .iter()
        .chain(&['!', '?', '…', ',', '.', '-'])
        .map(|c| regex::escape(&c.to_string()))
        .collect();
    let pattern = format!(r"([{punct}])([{punct}])+");
    let re = Regex::new(&pattern).unwrap();
    re.replace_all(text, "$1").into_owned()
}

/// Python `split_big_text(max_len=510)`
pub fn split_big_text(text: &str, max_len: usize) -> Vec<String> {
    let punctuation: String = SPLITS.iter().collect();
    let re = Regex::new(&format!("([{punctuation}])")).unwrap();
    let segments: Vec<&str> = re.split(text).collect();

    let mut result = Vec::new();
    let mut current = String::new();

    for segment in segments {
        if current.chars().count() + segment.chars().count() > max_len {
            if !current.is_empty() {
                result.push(current.clone());
            }
            current = segment.to_string();
        } else {
            current.push_str(segment);
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

/// Python `merge_short_text_in_array(texts, threshold)`
pub fn merge_short_text_in_array(texts: &[String], threshold: usize) -> Vec<String> {
    if texts.len() < 2 {
        return texts.to_vec();
    }
    let mut result = Vec::new();
    let mut text = String::new();
    for ele in texts {
        text.push_str(ele);
        if text.chars().count() >= threshold {
            result.push(text.clone());
            text.clear();
        }
    }
    if !text.is_empty() {
        if result.is_empty() {
            result.push(text);
        } else {
            let last = result.len() - 1;
            result[last].push_str(&text);
        }
    }
    result
}

fn get_first_segment(text: &str) -> String {
    let pattern: String = SPLITS
        .iter()
        .map(|c| regex::escape(&c.to_string()))
        .collect();
    let re = Regex::new(&format!("[{pattern}]")).unwrap();
    re.split(text).next().unwrap_or("").trim().to_string()
}

fn filter_text(texts: &[String]) -> Vec<String> {
    texts
        .iter()
        .filter(|t| !t.trim().is_empty())
        .cloned()
        .collect()
}

fn is_pure_symbols(text: &str) -> bool {
    PURE_SYMBOL_RE.is_match(text)
}

fn has_content(text: &str) -> bool {
    text.chars()
        .any(|c| c.is_alphanumeric() || matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF))
}

fn is_decimal_point(chars: &[char], idx: usize) -> bool {
    chars[idx] == '.'
        && idx > 0
        && idx + 1 < chars.len()
        && chars[idx - 1].is_ascii_digit()
        && chars[idx + 1].is_ascii_digit()
}

/// Split on sentence-ending punctuation without treating decimal points or
/// short-text prefix punctuation as standalone sentences.
pub fn split_sentence_chunks(line: &str) -> Vec<String> {
    let chars: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let mut current = String::new();

    for (idx, &c) in chars.iter().enumerate() {
        current.push(c);
        if !matches!(c, '。' | '！' | '？' | '.' | '!' | '?') || is_decimal_point(&chars, idx) {
            continue;
        }
        if !has_content(&current) {
            continue;
        }

        let trimmed = current.trim();
        if !trimmed.is_empty() && !is_pure_symbols(trimmed) {
            out.push(trimmed.to_string());
        }
        current.clear();
    }

    let tail = current.trim();
    if !tail.is_empty() && !is_pure_symbols(tail) {
        out.push(tail.to_string());
    }
    if out.is_empty() {
        let whole = line.trim();
        if !whole.is_empty() && !is_pure_symbols(whole) {
            out.push(whole.to_string());
        }
    }
    out
}

/// Python `TextPreprocessor.pre_seg_text` (default cut0-style: newline split + merge)
pub fn pre_seg_text(text: &str, lang_en: bool) -> Vec<String> {
    let mut text = text.trim().to_string();
    if text.is_empty() {
        return vec![];
    }

    if !text.chars().next().map_or(false, |c| is_split(c))
        && get_first_segment(&text).chars().count() < 4
    {
        if lang_en {
            text = format!(".{text}");
        } else {
            text = format!("。{text}");
        }
    }

    while text.contains("\n\n") {
        text = text.replace("\n\n", "\n");
    }

    let lines: Vec<String> = text.split('\n').map(|s| s.to_string()).collect();
    let lines = filter_text(&lines);
    let lines = merge_short_text_in_array(&lines, 5);

    let mut texts = Vec::new();
    for line in lines {
        if line.trim().is_empty() || is_pure_symbols(&line) {
            continue;
        }
        for mut chunk in split_sentence_chunks(&line) {
            if !chunk.chars().last().map_or(false, is_split) {
                if lang_en {
                    chunk.push('.');
                } else {
                    chunk.push('。');
                }
            }
            if chunk.chars().count() > 510 {
                texts.extend(split_big_text(&chunk, 510));
            } else {
                texts.push(chunk);
            }
        }
    }
    texts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collapse_punctuation() {
        assert_eq!(
            replace_consecutive_punctuation("你好，，世界"),
            "你好，世界"
        );
    }

    #[test]
    fn test_split_sentence_chunks_ignores_decimal() {
        assert_eq!(
            split_sentence_chunks("价格是99.5元。"),
            vec!["价格是99.5元。"]
        );
    }

    #[test]
    fn test_pre_seg_keeps_short_prefix_with_text() {
        assert_eq!(pre_seg_text("嗯", false), vec!["。嗯。"]);
    }

    #[test]
    fn test_pre_seg_splits_mixed_tail() {
        let chunks = pre_seg_text(
            "你好啊,这是一个测试.吃葡萄不吐葡萄皮,不吃葡萄倒吐葡萄皮.This demo is only for test  usage. If you find any 问题, 请修复它.",
            false,
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.contains("This demo is only for test"))
        );
        assert!(chunks.iter().any(|c| c.contains("If you find any 问题")));
    }
}
