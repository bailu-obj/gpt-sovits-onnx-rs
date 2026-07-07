// Port of GPT-SoVITS text/tone_sandhi.py

use std::collections::HashSet;

use jieba_rs::Jieba;

use crate::preprocessor::utils::{DICT_MONO_CHARS, DICT_POLY_CHARS};

lazy_static::lazy_static! {
    static ref MUST_NEURAL: HashSet<String> = load_word_set(include_str!(
        "../../../resource/must_neural_tone_words.txt"
    ));
    static ref MUST_NOT_NEURAL: HashSet<String> = load_word_set(include_str!(
        "../../../resource/must_not_neural_tone_words.txt"
    ));
}

fn load_word_set(data: &str) -> HashSet<String> {
    data.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect()
}

pub type Seg = Vec<(String, String)>;

pub struct ToneSandhi {
    punc: String,
}

impl Default for ToneSandhi {
    fn default() -> Self {
        Self {
            punc: "：，；。？！\"\"''':,;.?!".to_string(),
        }
    }
}

impl ToneSandhi {
    pub fn pre_merge_for_modify(&self, seg: Seg) -> Seg {
        let seg = self.merge_bu(seg);
        let seg = self.merge_yi(seg);
        let seg = self.merge_reduplication(seg);
        let seg = self.merge_continuous_three_tones(seg);
        let seg = self.merge_continuous_three_tones_2(seg);
        self.merge_er(seg)
    }

    pub fn modified_tone(&self, word: &str, pos: &str, finals: Vec<String>) -> Vec<String> {
        let finals = self.bu_sandhi(word, finals);
        let finals = self.yi_sandhi(word, finals);
        let finals = self.neural_sandhi(word, pos, finals);
        self.three_sandhi(word, finals)
    }

    fn neural_sandhi(&self, word: &str, pos: &str, mut finals: Vec<String>) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        for j in 0..chars.len() {
            if j > 0
                && chars[j] == chars[j - 1]
                && pos.chars().next().map_or(false, |c| matches!(c, 'n' | 'v' | 'a'))
                && !MUST_NOT_NEURAL.contains(word)
            {
                if let Some(f) = finals.get_mut(j) {
                    if f.len() >= 2 {
                        f.truncate(f.len() - 1);
                        f.push('5');
                    }
                }
            }
        }

        if let Some(last) = finals.last_mut() {
            if word.ends_with('吧')
                || word.ends_with('呢')
                || word.ends_with('哈')
                || word.ends_with('啊')
                || word.ends_with('呐')
                || word.ends_with('噻')
                || word.ends_with('嘛')
                || word.ends_with('吖')
                || word.ends_with('嗨')
                || word.ends_with('呐')
                || word.ends_with('哦')
                || word.ends_with('哒')
                || word.ends_with('额')
                || word.ends_with('滴')
                || word.ends_with('哩')
                || word.ends_with('哟')
                || word.ends_with('喽')
                || word.ends_with('啰')
                || word.ends_with('耶')
                || word.ends_with('喔')
                || word.ends_with('诶')
            {
                if last.len() >= 2 {
                    last.truncate(last.len() - 1);
                    last.push('5');
                }
            } else if word.ends_with('的') || word.ends_with('地') || word.ends_with('得') {
                if last.len() >= 2 {
                    last.truncate(last.len() - 1);
                    last.push('5');
                }
            }
        }

        if word.chars().count() == 1
            && matches!(word, "了" | "着" | "过")
            && matches!(pos, "ul" | "uz" | "ug")
        {
            if let Some(last) = finals.last_mut() {
                if last.len() >= 2 {
                    last.truncate(last.len() - 1);
                    last.push('5');
                }
            }
        }

        if MUST_NEURAL.contains(word) || MUST_NEURAL.iter().any(|w| word.ends_with(w)) {
            if let Some(last) = finals.last_mut() {
                if last.len() >= 2 {
                    last.truncate(last.len() - 1);
                    last.push('5');
                }
            }
        }

        finals
    }

    fn bu_sandhi(&self, word: &str, mut finals: Vec<String>) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() == 3 && chars[1] == '不' {
            if let Some(f) = finals.get_mut(1) {
                if f.len() >= 2 {
                    f.truncate(f.len() - 1);
                    f.push('5');
                }
            }
        } else {
            for i in 0..chars.len() {
                if chars[i] == '不' && i + 1 < chars.len() {
                    if let Some(next) = finals.get(i + 1) {
                        if next.ends_with('4') {
                            if let Some(f) = finals.get_mut(i) {
                                if f.len() >= 2 {
                                    f.truncate(f.len() - 1);
                                    f.push('2');
                                }
                            }
                        }
                    }
                }
            }
        }
        finals
    }

    fn yi_sandhi(&self, word: &str, mut finals: Vec<String>) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        if word.contains('一')
            && chars
                .iter()
                .filter(|c| **c != '一')
                .all(|c| c.is_numeric())
        {
            return finals;
        }
        if chars.len() == 3 && chars[1] == '一' && chars[0] == chars[2] {
            if let Some(f) = finals.get_mut(1) {
                if f.len() >= 2 {
                    f.truncate(f.len() - 1);
                    f.push('5');
                }
            }
        } else if word.starts_with("第一") {
            if let Some(f) = finals.get_mut(1) {
                if f.len() >= 2 {
                    f.truncate(f.len() - 1);
                    f.push('1');
                }
            }
        } else {
            for i in 0..chars.len() {
                if chars[i] == '一' && i + 1 < chars.len() {
                    if let Some(next) = finals.get(i + 1) {
                        let next_ch = chars[i + 1];
                        if next.ends_with('4') {
                            if let Some(f) = finals.get_mut(i) {
                                if f.len() >= 2 {
                                    f.truncate(f.len() - 1);
                                    f.push('2');
                                }
                            }
                        } else if !self.punc.contains(next_ch) {
                            if let Some(f) = finals.get_mut(i) {
                                if f.len() >= 2 {
                                    f.truncate(f.len() - 1);
                                    f.push('4');
                                }
                            }
                        }
                    }
                }
            }
        }
        finals
    }

    fn three_sandhi(&self, word: &str, mut finals: Vec<String>) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() == 2 && all_tone_three(&finals) {
            if let Some(f) = finals.get_mut(0) {
                if f.len() >= 2 {
                    f.truncate(f.len() - 1);
                    f.push('2');
                }
            }
        }
        finals
    }

    fn merge_bu(&self, seg: Seg) -> Seg {
        let mut new_seg: Seg = Vec::new();
        let mut pending_bu = false;
        for (word, pos) in seg {
            if pending_bu {
                new_seg.push((format!("不{word}"), pos));
                pending_bu = false;
            } else if word == "不" {
                pending_bu = true;
            } else {
                new_seg.push((word, pos));
            }
        }
        if pending_bu {
            new_seg.push(("不".to_string(), "d".to_string()));
        }
        new_seg
    }

    fn merge_yi(&self, mut seg: Seg) -> Seg {
        let mut new_seg: Seg = Vec::new();
        let mut i = 0;
        while i < seg.len() {
            let (word, pos) = seg[i].clone();
            let mut merged = false;
            if i > 0 && word == "一" && i + 1 < seg.len() {
                if let Some(last) = new_seg.last() {
                    if last.0 == seg[i + 1].0 && last.1 == "v" && seg[i + 1].1 == "v" {
                        let combined = format!("{}{}一{}", last.0, "", seg[i + 1].0);
                        new_seg.last_mut().unwrap().0 = combined;
                        i += 2;
                        merged = true;
                    }
                }
            }
            if !merged {
                new_seg.push((word, pos));
                i += 1;
            }
        }
        seg = new_seg;
        let mut new_seg: Seg = Vec::new();
        for (word, pos) in seg {
            if let Some(last) = new_seg.last_mut() {
                if last.0 == "一" {
                    last.0.push_str(&word);
                    continue;
                }
            }
            new_seg.push((word, pos));
        }
        new_seg
    }

    fn merge_continuous_three_tones(&self, seg: Seg) -> Seg {
        let sub_finals: Vec<Vec<String>> = seg
            .iter()
            .map(|(w, _)| lazy_finals_tone3(w))
            .collect();
        let mut new_seg: Seg = Vec::new();
        let mut merge_last = vec![false; seg.len()];
        for i in 0..seg.len() {
            if i > 0
                && all_tone_three(&sub_finals[i - 1])
                && all_tone_three(&sub_finals[i])
                && !merge_last[i - 1]
            {
                if !is_reduplication(&seg[i - 1].0)
                    && seg[i - 1].0.chars().count() + seg[i].0.chars().count() <= 3
                {
                    new_seg.last_mut().unwrap().0.push_str(&seg[i].0);
                    merge_last[i] = true;
                } else {
                    new_seg.push(seg[i].clone());
                }
            } else {
                new_seg.push(seg[i].clone());
            }
        }
        new_seg
    }

    fn merge_continuous_three_tones_2(&self, seg: Seg) -> Seg {
        let sub_finals: Vec<Vec<String>> = seg
            .iter()
            .map(|(w, _)| lazy_finals_tone3(w))
            .collect();
        let mut new_seg: Seg = Vec::new();
        let mut merge_last = vec![false; seg.len()];
        for i in 0..seg.len() {
            if i > 0
                && sub_finals[i - 1].last().map_or(false, |f| f.ends_with('3'))
                && sub_finals[i].first().map_or(false, |f| f.ends_with('3'))
                && !merge_last[i - 1]
            {
                if !is_reduplication(&seg[i - 1].0)
                    && seg[i - 1].0.chars().count() + seg[i].0.chars().count() <= 3
                {
                    new_seg.last_mut().unwrap().0.push_str(&seg[i].0);
                    merge_last[i] = true;
                } else {
                    new_seg.push(seg[i].clone());
                }
            } else {
                new_seg.push(seg[i].clone());
            }
        }
        new_seg
    }

    fn merge_er(&self, seg: Seg) -> Seg {
        let mut new_seg: Seg = Vec::new();
        for (i, (word, pos)) in seg.into_iter().enumerate() {
            if i > 0 && word == "儿" && new_seg.last().map_or(false, |(w, _)| w != "#") {
                new_seg.last_mut().unwrap().0.push_str(&word);
            } else {
                new_seg.push((word, pos));
            }
        }
        new_seg
    }

    fn merge_reduplication(&self, seg: Seg) -> Seg {
        let mut new_seg: Seg = Vec::new();
        for (word, pos) in seg {
            if let Some(last) = new_seg.last_mut() {
                if last.0 == word {
                    last.0.push_str(&word);
                    continue;
                }
            }
            new_seg.push((word, pos));
        }
        new_seg
    }
}

fn all_tone_three(finals: &[String]) -> bool {
    finals.iter().all(|f| f.ends_with('3'))
}

fn is_reduplication(word: &str) -> bool {
    let chars: Vec<char> = word.chars().collect();
    chars.len() == 2 && chars[0] == chars[1]
}

/// Approximate pypinyin finals for tone-sandhi merge heuristics.
pub fn lazy_finals_tone3(word: &str) -> Vec<String> {
    word.chars()
        .map(|c| {
            if let Some(m) = DICT_MONO_CHARS.get(&c) {
                to_finals_tone3(&m.phone)
            } else if let Some(p) = DICT_POLY_CHARS.get(&c) {
                to_finals_tone3(&p.phones[0].0)
            } else {
                String::new()
            }
        })
        .collect()
}

pub fn to_initials(pinyin: &str) -> String {
    static INITIALS: &[&str] = &[
        "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x",
        "r", "z", "c", "s", "y", "w",
    ];
    for ini in INITIALS {
        if pinyin.starts_with(ini) {
            let rest = &pinyin[ini.len()..];
            if rest.is_empty() || rest.chars().next().map_or(false, |c| c.is_ascii_alphabetic()) {
                return ini.to_string();
            }
        }
    }
    String::new()
}

pub fn to_finals_tone3(pinyin: &str) -> String {
    let ini = to_initials(pinyin);
    let mut final_part = if ini.is_empty() {
        pinyin.to_string()
    } else {
        pinyin[ini.len()..].to_string()
    };
    // Python pypinyin neutral_tone_with_five=True
    if !final_part.is_empty() && !final_part.chars().last().map_or(false, |c| c.is_ascii_digit()) {
        final_part.push('5');
    }
    final_part
}

pub fn tag_segment(jieba: &Jieba, text: &str) -> Seg {
    jieba
        .tag(text, true)
        .into_iter()
        .map(|t| (t.word.to_string(), t.tag.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_initials() {
        assert_eq!(to_initials("zhang1"), "zh");
        assert_eq!(to_initials("ai2"), "");
    }
}
