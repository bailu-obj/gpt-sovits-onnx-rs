// Port of GPT-SoVITS text/chinese2.py _g2p pipeline

use std::collections::HashMap;

use jieba_rs::Jieba;
use log::warn;
use regex::Regex;

use crate::preprocessor::{
    phone_symbol::PUNCTUATION,
    zh::{
        g2pw::G2PW,
        tone_sandhi::{self, ToneSandhi},
    },
};

lazy_static::lazy_static! {
    static ref PINYIN_TO_SYMBOL: HashMap<String, String> = load_opencpop();
    static ref PP_DICT: HashMap<String, Vec<String>> = load_polyphonic_dict();
    static ref STRIP_EN_RE: Regex = Regex::new(r"[a-zA-Z]+").unwrap();
}

fn split_on_punctuation(text: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if PUNCTUATION.contains(&c.to_string().as_str()) {
            if !current.trim().is_empty() {
                segments.push(current.clone());
            }
            current.clear();
            segments.push(c.to_string());
        } else if c.is_whitespace() {
            continue;
        } else {
            current.push(c);
        }
    }
    if !current.trim().is_empty() {
        segments.push(current);
    }
    segments
}

static MUST_ERHUA: &[&str] = &[
    "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿",
];

static NOT_ERHUA: &[&str] = &[
    "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿", "妻儿",
    "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿", "脑瘫儿",
    "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿", "侄儿",
    "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿", "猪儿", "猫儿",
    "狗儿", "少儿",
];

fn load_opencpop() -> HashMap<String, String> {
    include_str!("../../../resource/opencpop-strict.txt")
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                Some((parts[0].to_string(), parts[1].to_string()))
            } else {
                None
            }
        })
        .collect()
}

fn load_polyphonic_dict() -> HashMap<String, Vec<String>> {
    let mut dict = HashMap::new();
    for content in [
        include_str!("../../../resource/polyphonic.rep"),
        include_str!("../../../resource/polyphonic-fix.rep"),
    ] {
        for line in content.lines() {
            if let Some((key, value_str)) = line.split_once(':') {
                if let Ok(value) = serde_json::from_str::<Vec<String>>(value_str.trim()) {
                    dict.insert(key.trim().to_string(), value);
                }
            }
        }
    }
    dict
}

pub struct G2pResult {
    pub phones: Vec<String>,
    pub word2ph: Vec<i32>,
    pub norm_text: String,
}

/// Full Mandarin G2P matching Python chinese2.g2p
pub fn g2p_mandarin(text: &str, g2pw: &mut G2PW, jieba: &Jieba) -> G2pResult {
    let segments: Vec<String> = split_on_punctuation(text)
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .collect();

    let tone_modifier = ToneSandhi::default();
    let mut phones_list: Vec<String> = Vec::new();
    let mut word2ph: Vec<i32> = Vec::new();

    let processed: Vec<String> = segments
        .iter()
        .map(|seg| STRIP_EN_RE.replace_all(seg, "").to_string())
        .collect();

    let batch_inputs: Vec<&str> = processed.iter().map(|s| s.as_str()).filter(|s| !s.is_empty()).collect();
    let g2pw_batch: Vec<Vec<String>> = if batch_inputs.is_empty() {
        vec![]
    } else {
        g2pw.g2p_batch(&batch_inputs)
    };
    let mut batch_cursor = 0usize;

    for seg in processed {
        if seg.is_empty() {
            continue;
        }
        let pinyins = g2pw_batch[batch_cursor].clone();
        batch_cursor += 1;

        let mut seg_cut = tone_sandhi::tag_segment(jieba, &seg);
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut);

        let mut initials: Vec<Vec<String>> = Vec::new();
        let mut finals: Vec<Vec<String>> = Vec::new();
        let mut pre_word_length = 0usize;

        for (word, pos) in seg_cut {
            let now_word_length = pre_word_length + word.chars().count();
            if pos == "eng" {
                pre_word_length = now_word_length;
                continue;
            }

            let mut word_pinyins: Vec<String> = pinyins[pre_word_length..now_word_length].to_vec();
            word_pinyins = correct_pronunciation(&word, word_pinyins);

            let mut sub_initials = Vec::new();
            let mut sub_finals = Vec::new();
            for pinyin in word_pinyins {
                if pinyin.chars().next().map_or(false, |c| c.is_ascii_alphabetic()) {
                    sub_initials.push(tone_sandhi::to_initials(&pinyin));
                    sub_finals.push(tone_sandhi::to_finals_tone3(&pinyin));
                } else {
                    sub_initials.push(pinyin.clone());
                    sub_finals.push(pinyin);
                }
            }

            pre_word_length = now_word_length;
            let sub_finals = tone_modifier.modified_tone(&word, &pos, sub_finals);
            let (sub_initials, sub_finals) = merge_erhua(sub_initials, sub_finals, &word, &pos);
            initials.push(sub_initials);
            finals.push(sub_finals);
        }

        let initials: Vec<String> = initials.into_iter().flatten().collect();
        let finals: Vec<String> = finals.into_iter().flatten().collect();

        for (c, v) in initials.iter().zip(finals.iter()) {
            let raw_pinyin = format!("{c}{v}");
            if c == v {
                if !PUNCTUATION.contains(&c.as_str()) {
                    warn!("Unexpected punctuation phoneme: {}", c);
                }
                phones_list.push(c.clone());
                word2ph.push(1);
            } else {
                let v_without_tone = &v[..v.len().saturating_sub(1)];
                let tone = v.chars().last().unwrap_or('5');

                let mut pinyin = format!("{c}{v_without_tone}");

                if !c.is_empty() {
                    let v_rep = match v_without_tone {
                        "uei" => "ui",
                        "iou" => "iu",
                        "uen" => "un",
                        _ => v_without_tone,
                    };
                    if v_without_tone != v_rep {
                        pinyin = format!("{c}{v_rep}");
                    }
                } else {
                    let pinyin_rep = match pinyin.as_str() {
                        "ing" => "ying",
                        "i" => "yi",
                        "in" => "yin",
                        "u" => "wu",
                        other => other,
                    };
                    if pinyin_rep != pinyin.as_str() {
                        pinyin = pinyin_rep.to_string();
                    } else if let Some(first) = pinyin.chars().next() {
                        let single_rep = match first {
                            'v' => Some("yu"),
                            'e' => Some("e"),
                            'i' => Some("y"),
                            'u' => Some("w"),
                            _ => None,
                        };
                        if let Some(rep) = single_rep {
                            pinyin = format!("{}{}", rep, &pinyin[1..]);
                        }
                    }
                }

                let mapped = PINYIN_TO_SYMBOL.get(&pinyin).cloned().unwrap_or_else(|| {
                    warn!("Unknown pinyin mapping: {} (raw: {})", pinyin, raw_pinyin);
                    format!("{c} {v_without_tone}")
                });
                let parts: Vec<&str> = mapped.split(' ').collect();
                let new_c = parts[0];
                let new_v = format!("{}{}", parts.get(1).unwrap_or(&""), tone);
                phones_list.push(new_c.to_string());
                phones_list.push(new_v);
                word2ph.push(2);
            }
        }
    }

    G2pResult {
        phones: phones_list,
        word2ph,
        norm_text: text.to_string(),
    }
}

fn correct_pronunciation(word: &str, mut word_pinyins: Vec<String>) -> Vec<String> {
    if let Some(fixed) = PP_DICT.get(word) {
        return fixed.clone();
    }
    let chars: Vec<char> = word.chars().collect();
    for (idx, c) in chars.iter().enumerate() {
        let key = c.to_string();
        if let Some(w_pinyin) = PP_DICT.get(&key) {
            if let Some(first) = w_pinyin.first() {
                if idx < word_pinyins.len() {
                    word_pinyins[idx] = first.clone();
                }
            }
        }
    }
    word_pinyins
}

fn merge_erhua(
    initials: Vec<String>,
    mut finals: Vec<String>,
    word: &str,
    pos: &str,
) -> (Vec<String>, Vec<String>) {
    let chars: Vec<char> = word.chars().collect();
    if let Some(i) = chars.iter().rposition(|&c| c == '儿') {
        if i == chars.len() - 1 && finals.get(i).map_or(false, |f| f == "er1") {
            finals[i] = "er2".to_string();
        }
    }

    if !MUST_ERHUA.contains(&word)
        && (NOT_ERHUA.contains(&word) || matches!(pos, "a" | "j" | "nr"))
    {
        return (initials, finals);
    }

    if finals.len() != chars.len() {
        return (initials, finals);
    }

    let mut new_initials = Vec::new();
    let mut new_finals: Vec<String> = Vec::new();
    for i in 0..finals.len() {
        let mut phn = finals[i].clone();
        if i == finals.len() - 1
            && chars.get(i) == Some(&'儿')
            && (phn == "er2" || phn == "er5")
        {
            let suffix = if chars.len() >= 2 {
                chars[chars.len() - 2..].iter().collect::<String>()
            } else {
                String::new()
            };
            if !NOT_ERHUA.contains(&suffix.as_str()) && !NOT_ERHUA.contains(&word) {
                if let Some(prev) = new_finals.last() {
                    if let Some(tone) = prev.chars().last() {
                        phn = format!("er{tone}");
                    }
                }
            }
        }
        new_initials.push(initials[i].clone());
        new_finals.push(phn);
    }
    (new_initials, new_finals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correct_pronunciation_dict() {
        let result = correct_pronunciation("行", vec!["xing2".to_string()]);
        assert!(!result.is_empty());
    }
}
