// Sentence boundaries for inference and punctuation cleanup for phonemization.

use regex::Regex;

pub const SPLITS: &[char] = &[
    '，', '。', '？', '！', ',', '.', '?', '!', '~', ':', '：', '—', '…',
];

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

const MAX_CHARS: usize = 510;

fn sentence_end(c: char) -> bool {
    matches!(c, '。' | '！' | '？' | '．' | '.' | '!' | '?' | '…')
}

fn closer(c: char) -> bool {
    matches!(
        c,
        '”' | '’' | '"' | '\'' | '」' | '』' | '》' | '）' | ')' | '】' | ']'
    )
}

fn ascii_word(chars: impl Iterator<Item = char>) -> String {
    chars.take_while(char::is_ascii_alphabetic).collect()
}

fn decimal_digit(c: char) -> bool {
    c.is_ascii_digit() || matches!(c, '０'..='９')
}

fn internal_period(chars: &[char], index: usize, following: &[char]) -> bool {
    if chars.get(index + 1).is_some_and(|&c| decimal_digit(c))
        && (index == 0
            || !chars[index - 1].is_ascii_alphabetic()
                && !matches!(chars[index - 1], '。' | '.' | '．'))
    {
        return true;
    }
    let Some(&previous) = chars[..index].last() else {
        return false;
    };
    let Some(&next) = following.first() else {
        return false;
    };
    if chars.get(index + 1).is_some_and(|&immediate| {
        previous.is_ascii_alphanumeric() && immediate.is_ascii_lowercase()
    }) {
        return true;
    }

    let word: String = ascii_word(chars[..index].iter().rev().copied())
        .chars()
        .rev()
        .collect();
    let abbreviation = word.to_ascii_lowercase();
    if matches!(
        abbreviation.as_str(),
        "mr" | "mrs" | "ms" | "dr" | "prof" | "sr" | "jr" | "st" | "vs" | "no" | "fig"
    ) {
        return next.is_ascii_alphanumeric();
    }

    let dotted = word.len() == 1
        && index >= 3
        && chars[index - 2] == '.'
        && chars[index - 3].is_ascii_alphabetic();
    if dotted {
        if chars.get(index + 1).is_some_and(char::is_ascii_alphabetic) || next.is_ascii_lowercase()
        {
            return true;
        }
        return matches!(
            (
                chars[index - 3].to_ascii_lowercase(),
                previous.to_ascii_lowercase()
            ),
            ('e', 'g') | ('i', 'e')
        ) && next.is_ascii_alphanumeric();
    }
    word.len() == 1 && previous.is_ascii_uppercase() && next.is_ascii_uppercase()
}

fn inside_url(chars: &[char], index: usize) -> bool {
    if !matches!(chars[index], '.' | '?' | '!')
        || !chars
            .get(index + 1)
            .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '%' | '#' | '_'))
    {
        return false;
    }
    let start = chars[..index]
        .iter()
        .rposition(|c| c.is_whitespace() || matches!(c, '(' | '"' | '“' | '「'))
        .map_or(0, |position| position + 1);
    let prefix: String = chars[start..index].iter().take(8).collect();
    prefix.starts_with("https://") || prefix.starts_with("http://") || prefix.starts_with("www")
}

fn push_sentence(out: &mut Vec<String>, chars: &[char]) {
    if chars
        .iter()
        .any(|c| c.is_alphanumeric() || matches!(c, '￥' | '^'))
    {
        out.push(chars.iter().collect::<String>().trim().to_owned());
    }
}

fn split_sentence_chunks(line: &str) -> Vec<String> {
    let chars: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let (mut start, mut index, mut parentheses) = (0, 0, 0usize);
    while index < chars.len() {
        let c = chars[index];
        match c {
            '(' | '（' | '[' | '【' => parentheses += 1,
            ')' | '）' | ']' | '】' => parentheses = parentheses.saturating_sub(1),
            _ => {}
        }
        if parentheses > 0 || !sentence_end(c) || inside_url(&chars, index) {
            index += 1;
            continue;
        }

        let mut end = index + 1;
        while end < chars.len() && sentence_end(chars[end]) {
            end += 1;
        }
        let terminal_end = end;
        while end < chars.len() && closer(chars[end]) {
            end += 1;
        }
        let following_start = chars[end..]
            .iter()
            .position(|c| !c.is_whitespace())
            .map_or(chars.len(), |position| end + position);
        let following = &chars[following_start..];
        let next = following.first().copied();
        let internal = (matches!(c, '.' | '．')
            && terminal_end == index + 1
            && internal_period(&chars, index, following))
            || ((c == '…' || (c == '.' && terminal_end - index >= 3))
                && next.is_some_and(|next| next.is_alphabetic() && !next.is_uppercase()))
            || (end > terminal_end
                && next.is_some_and(|next| {
                    matches!(next, ',' | '，' | ';' | '；' | '—') || next.is_ascii_lowercase()
                }));
        if !internal && chars[start..index].iter().any(|c| c.is_alphanumeric()) {
            push_sentence(&mut out, &chars[start..end]);
            start = end;
        }
        index = if internal { index + 1 } else { end };
    }
    push_sentence(&mut out, &chars[start..]);
    out
}

fn bounded_chunks(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut remaining = chars.as_slice();
    let mut chunks = Vec::new();
    while remaining.len() > MAX_CHARS {
        // Reserve one character for punctuation on a forced break.
        let cap = MAX_CHARS - 1;
        let tail = &remaining[cap / 2..cap];
        let boundary = tail
            .iter()
            .rposition(|c| matches!(c, ',' | '，' | ';' | '；' | ':' | '：'))
            .or_else(|| tail.iter().rposition(|c| c.is_whitespace()))
            .map_or(cap, |position| cap / 2 + position + 1);
        push_sentence(&mut chunks, &remaining[..boundary]);
        remaining = &remaining[boundary..];
    }
    push_sentence(&mut chunks, remaining);
    chunks
}

/// Preserve complete sentences before punctuation and language normalization.
pub fn pre_seg_text(text: &str, lang_en: bool) -> Vec<String> {
    let terminator = if lang_en { '.' } else { '。' };
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut chunks = Vec::new();
    for paragraph in text.split("\n\n") {
        for sentence in split_sentence_chunks(&paragraph.replace('\n', " ")) {
            for mut chunk in bounded_chunks(&sentence) {
                if !chunk
                    .chars()
                    .rev()
                    .find(|&c| !closer(c))
                    .is_some_and(sentence_end)
                {
                    chunk.push(terminator);
                }
                if chunks.is_empty()
                    && chunk
                        .chars()
                        .take_while(|c| !SPLITS.contains(c) && !matches!(c, '￥' | '^'))
                        .count()
                        < 4
                    && !chunk.chars().next().is_some_and(|c| SPLITS.contains(&c))
                    && chunk.chars().count() < MAX_CHARS
                {
                    chunk.insert(0, terminator);
                }
                chunks.push(chunk);
            }
        }
    }
    chunks
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

    #[test]
    fn english_abbreviations_decimals_and_urls_stay_inside_the_sentence() {
        assert_eq!(
            split_sentence_chunks(
                "Dr. J. Smith paid 3.14 in the U.S. last week. Try e.g. https://example.com?q=1. Next sentence."
            ),
            [
                "Dr. J. Smith paid 3.14 in the U.S. last week.",
                "Try e.g. https://example.com?q=1.",
                "Next sentence."
            ]
        );
        assert_eq!(
            split_sentence_chunks("We met at 9 a.m. Then we left."),
            ["We met at 9 a.m.", "Then we left."]
        );
    }

    #[test]
    fn quotes_pauses_and_line_wrapping_keep_their_context() {
        assert_eq!(
            pre_seg_text(
                "他说：“价格是99.5元，你要买吗？”\n我想……先等等！\n\nShe asked “Why?” quietly. “Ready?!” Then we left.",
                false
            ),
            [
                "。他说：“价格是99.5元，你要买吗？”",
                "我想……先等等！",
                "She asked “Why?” quietly.",
                "“Ready?!”",
                "Then we left."
            ]
        );
        assert_eq!(
            pre_seg_text("This sentence\nwraps on screen.", true),
            ["This sentence wraps on screen."]
        );
    }

    #[test]
    fn splits_long_sentences_at_clauses_without_discarding_punctuation() {
        let left = "甲".repeat(300);
        let right = "乙".repeat(260);
        assert_eq!(
            pre_seg_text(&format!("{left}，{right}。"), false),
            [format!("{left}，。"), format!("{right}。")]
        );
    }
}
