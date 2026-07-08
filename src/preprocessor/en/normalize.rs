// English text normalization (Python text/english.py + expend.py subset)

use once_cell::sync::Lazy;
use regex::Regex;

use crate::preprocessor::seg::replace_consecutive_punctuation;

static REP_MAP: &[(&str, &str)] = &[
    ("[;:：，；]", ","),
    ("[\"'']", "'"),
    ("。", "."),
    ("！", "!"),
    ("？", "?"),
];

static REP_PATTERN: Lazy<Regex> = Lazy::new(|| {
    let parts: Vec<String> = REP_MAP
        .iter()
        .map(|(from, _)| format!("({from})"))
        .collect();
    Regex::new(&parts.join("|")).unwrap()
});

static RE_ORDINAL: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(\d+)(st|nd|rd|th)\b").unwrap());
static RE_DECIMAL: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(\d+)\.(\d+)\b").unwrap());
static RE_DOLLAR: Lazy<Regex> = Lazy::new(|| Regex::new(r"\$(\d+(?:\.\d+)?)").unwrap());
static RE_INTEGER: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(\d+)\b").unwrap());

/// Normalize English text for G2P (punctuation + number expansion).
pub fn text_normalize_en(text: &str) -> String {
    let replaced = REP_PATTERN.replace_all(text, |caps: &regex::Captures| {
        for (i, (_, to)) in REP_MAP.iter().enumerate() {
            if caps.get(i + 1).is_some() {
                return to.to_string();
            }
        }
        caps[0].to_string()
    });
    let result = expand_numbers_en(&replaced);
    let result = expand_basic_en(&result);
    replace_consecutive_punctuation(&result)
}

fn expand_numbers_en(text: &str) -> String {
    let mut result = text.to_string();
    result = RE_ORDINAL
        .replace_all(&result, |caps: &regex::Captures| {
            let n = caps.get(1).map(|m| m.as_str()).unwrap_or("0");
            ordinal_to_words(n, caps.get(2).map(|m| m.as_str()).unwrap_or("th"))
        })
        .into_owned();
    result = RE_DOLLAR
        .replace_all(&result, |caps: &regex::Captures| {
            let n = caps.get(1).map(|m| m.as_str()).unwrap_or("0");
            format!("{} dollars", num_to_words(n))
        })
        .into_owned();
    result = RE_DECIMAL
        .replace_all(&result, |caps: &regex::Captures| {
            let int_part = caps.get(1).map(|m| m.as_str()).unwrap_or("0");
            let frac_part = caps.get(2).map(|m| m.as_str()).unwrap_or("0");
            format!(
                "{} point {}",
                num_to_words(int_part),
                frac_digits_to_words(frac_part)
            )
        })
        .into_owned();
    result = RE_INTEGER
        .replace_all(&result, |caps: &regex::Captures| num_to_words(&caps[1]))
        .into_owned();
    result = result.replace('%', " percent");
    result
}

fn num_to_words(num: &str) -> String {
    num2en::str_to_words(num).unwrap_or_else(|_| num.to_string())
}

fn frac_digits_to_words(digits: &str) -> String {
    digits
        .chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| num_to_words(&c.to_string()))
        .collect::<Vec<_>>()
        .join(" ")
}

fn ordinal_to_words(num: &str, suffix: &str) -> String {
    let _ = suffix;
    match num2en::str_to_words(num) {
        Ok(words) => format!("{} {}", words, ordinal_suffix_word(suffix)),
        Err(_) => format!("{}{}", num, suffix),
    }
}

fn ordinal_suffix_word(suffix: &str) -> &'static str {
    match suffix {
        "st" => "first",
        "nd" => "second",
        "rd" => "third",
        _ => "th",
    }
}

/// Lightweight expansion of common English abbreviations.
fn expand_basic_en(text: &str) -> String {
    text.replace("e.g.", "for example")
        .replace("i.e.", "that is")
        .replace("etc.", "etc")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_punctuation() {
        assert!(text_normalize_en("Hello!").contains('!'));
    }

    #[test]
    fn test_normalize_integer() {
        let n = text_normalize_en("we propose 1 DSPGAN");
        assert!(n.contains("one"));
    }
}
