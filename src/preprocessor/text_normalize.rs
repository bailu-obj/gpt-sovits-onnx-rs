// preprocessor/text_normalize.rs
use once_cell::sync::Lazy;
use regex::Regex;

use crate::preprocessor::seg::replace_consecutive_punctuation;

/// Filters out emojis and normalizes punctuation for TTS preprocessing.
pub fn text_normalize(text: &str) -> String {
    let temp = CLEANUP_REGEX.replace_all(text, " ").into_owned();
    let temp = temp.replace('嗯', "恩").replace('呣', "母");
    let temp = PUNCTUATION_COMMAS_REGEX.replace_all(&temp, ",");
    let temp = PUNCTUATION_PERIODS_REGEX
        .replace_all(&temp, ".")
        .into_owned();
    let temp = COLLAPSE_COMMAS_REGEX.replace_all(&temp, ",").into_owned();
    let temp = COLLAPSE_PERIODS_REGEX.replace_all(&temp, ".").into_owned();
    replace_consecutive_punctuation(&temp)
}

/// Chinese punctuation replacement after TextNormalizer (Python `replace_punctuation`).
pub fn replace_punctuation_zh(text: &str) -> String {
    let mut result = text.to_string();
    for (from, to) in REP_MAP_ZH {
        result = result.replace(from, to);
    }
    let keep = format!(r"[^\u{{4e00}}-\u{{9fa5}}{}\s]+", regex::escape("!?,….-"));
    let re = Regex::new(&keep).unwrap();
    re.replace_all(&result, "").into_owned()
}

/// Lightweight Chinese text normalization (subset of PaddleSpeech TextNormalizer).
pub fn text_normalize_zh(text: &str) -> String {
    let mut result = text.to_string();
    result = result.replace('％', "%");
    result = result.replace('《', "").replace('》', "");
    result = result.replace('（', "").replace('）', "");
    result = result.replace('【', "").replace('】', "");
    result = expand_zh_numbers(&result);
    replace_punctuation_zh(&result)
}

/// Expand numbers with Chinese unit suffixes (年/月/日/元/%) and bare numerics.
fn expand_zh_numbers(text: &str) -> String {
    let mut result = text.to_string();
    result = RE_ZH_PERCENT
        .replace_all(&result, |caps: &regex::Captures| {
            let num = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            expand_zh_percent(num)
        })
        .into_owned();
    result = RE_ZH_YEAR
        .replace_all(&result, |caps: &regex::Captures| {
            let num = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            format!("{}年", digits_to_zh_year(num))
        })
        .into_owned();
    result = RE_ZH_YUAN
        .replace_all(&result, |caps: &regex::Captures| {
            let num = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            format!("{}元", expand_zh_integer(num))
        })
        .into_owned();
    result = RE_ZH_MONTH
        .replace_all(&result, |caps: &regex::Captures| {
            let num = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            format!("{}月", expand_zh_integer(num))
        })
        .into_owned();
    result = RE_ZH_DAY
        .replace_all(&result, |caps: &regex::Captures| {
            let num = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            format!("{}日", expand_zh_integer(num))
        })
        .into_owned();
    result = RE_ZH_DECIMAL_YUAN
        .replace_all(&result, |caps: &regex::Captures| {
            let int_part = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let frac_part = caps.get(2).map(|m| m.as_str()).unwrap_or("");
            format!(
                "{}点{}元",
                expand_zh_integer(int_part),
                frac_digits_to_zh(frac_part)
            )
        })
        .into_owned();
    expand_bare_zh_numbers(&result)
}

fn expand_bare_zh_numbers(text: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let start = i;
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            let unit = chars.get(i).copied();
            if matches!(unit, Some('年' | '月' | '日' | '元' | '%')) {
                result.push_str(&chars[start..i].iter().collect::<String>());
            } else {
                let num: String = chars[start..i].iter().collect();
                result.push_str(&expand_zh_number_token(&num));
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    result
}

fn expand_zh_number_token(token: &str) -> String {
    crate::preprocessor::num::expand_zh(token)
}

fn expand_zh_percent(num: &str) -> String {
    format!("百分之{}", expand_zh_number_token(num))
}

fn expand_zh_integer(num: &str) -> String {
    expand_zh_number_token(num)
}

fn digits_to_zh_year(num: &str) -> String {
    const DIGITS: [&str; 10] = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];
    num.chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| DIGITS[(c as u8 - b'0') as usize])
        .collect()
}

fn frac_digits_to_zh(num: &str) -> String {
    digits_to_zh_year(num)
}

static RE_ZH_PERCENT: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+(?:\.\d+)?)%").unwrap());
static RE_ZH_YEAR: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)年").unwrap());
static RE_ZH_YUAN: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)元").unwrap());
static RE_ZH_MONTH: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)月").unwrap());
static RE_ZH_DAY: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)日").unwrap());
static RE_ZH_DECIMAL_YUAN: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)\.(\d+)元").unwrap());

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
        r"[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F900}-\u{1F9FF}\u{2600}-\u{27BF}\u{2000}-\u{206F}\u{2300}-\u{23FF}]+",
    )
    .unwrap()
});

static PUNCTUATION_PERIODS_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[\u{2026}\u{003F}\u{0021}\u{002E}\u{FF01}\u{FF1F}\u{3002}\u{FF0E}]+").unwrap()
});

static PUNCTUATION_COMMAS_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"[\u{002C}\u{2018}\u{2019}\u{201C}\u{201D}\u{2022}\u{FF0C}\u{FF1A}\u{FF1B}\u{FF0B}\u{FF1D}\u{FF5E}\u{2014}\u{2013}\u{FF3B}\u{FF3D}\u{FF08}\u{FF09}\u{3001}\u{FF5F}\u{FF1C}\u{FF1E}\u{300A}\u{300B}\u{300C}\u{300D}\u{FF3F}\u{002A}\u{003D}\u{00A9}\u{2212}\u{2021}\u{203B}\u{2047}\u{3008}\u{3009}\u{300E}\u{300F}\u{FF0F}\u{0023}]+",
    )
    .unwrap()
});

static COLLAPSE_COMMAS_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r",{2,}").unwrap());

static COLLAPSE_PERIODS_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\.{2,}").unwrap());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_marks_keep_sentence_boundaries() {
        let text = text_normalize("风凉吗？午饭吃了吗？我们走走吧。");
        assert_eq!(
            super::super::seg::pre_seg_text(&text, false),
            vec!["。风凉吗.", "午饭吃了吗.", "我们走走吧."]
        );
    }

    #[test]
    fn test_zh_normalize_preserves_comma_space() {
        assert_eq!(text_normalize_zh("问题, 请修复它。"), "问题, 请修复它.");
    }
}
