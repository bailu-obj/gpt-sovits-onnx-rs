// Python-compatible text segmentation (TextPreprocessor.pre_seg_text + text_segmentation_method)

use once_cell::sync::Lazy;
use regex::Regex;

pub const SPLITS: &[char] = &[
    '，', '。', '？', '！', ',', '.', '?', '!', '~', ':', '：', '—', '…',
];

static PURE_SYMBOL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\W+$").unwrap());

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
    let pattern: String = SPLITS.iter().map(|c| regex::escape(&c.to_string())).collect();
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
    for mut line in lines {
        if line.trim().is_empty() || is_pure_symbols(&line) {
            continue;
        }
        if !line.chars().last().map_or(false, is_split) {
            if lang_en {
                line.push('.');
            } else {
                line.push('。');
            }
        }
        if line.chars().count() > 510 {
            texts.extend(split_big_text(&line, 510));
        } else {
            texts.push(line);
        }
    }
    texts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collapse_punctuation() {
        assert_eq!(replace_consecutive_punctuation("你好，，世界"), "你好，世界");
    }
}
