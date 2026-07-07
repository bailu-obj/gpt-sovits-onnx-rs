use log::debug;
use std::collections::HashSet;

use crate::preprocessor::zh::jyutping_list::get_jyutping_list;

const INITIALS: &[&str] = &[
    "aa", "aai", "aak", "aap", "aat", "aau", "ai", "au", "ap", "at", "ak", "a", "p", "b", "e",
    "ts", "t", "dz", "d", "kw", "k", "gw", "g", "f", "h", "l", "m", "ng", "n", "s", "y", "w", "c",
    "z", "j", "ong", "on", "ou", "oi", "ok", "o", "uk", "ung", "sp", "spl", "spn", "sil",
];

lazy_static::lazy_static! {
    static ref PUNCTUATION_SET: HashSet<char> = {
        let punctuation = ",.!?;:()[]{}'\"-…";
        punctuation.chars().collect()
    };
}

pub fn g2p(text: &str) -> (Vec<String>, Vec<i32>) {
    let jyutping_list = get_jyutping_list(text);
    debug!("jyutping_list: {:?}", jyutping_list);

    let mut phones = Vec::new();
    let mut word2ph = Vec::new();

    for (word, jyutping) in jyutping_list {
        let chars: Vec<char> = word.chars().collect();
        let jyutping_parts: Vec<&str> = jyutping.split_whitespace().collect();

        if chars.len() == jyutping_parts.len() {
            for (c, jp) in chars.iter().zip(jyutping_parts.iter()) {
                let (initial, final_) = split_jyutping(jp);
                if !initial.is_empty() {
                    phones.push(format!("Y{initial}"));
                }
                if !final_.is_empty() {
                    phones.push(format!("Y{final_}"));
                    word2ph.push(2);
                } else if PUNCTUATION_SET.contains(c) {
                    phones.push(c.to_string());
                    word2ph.push(1);
                }
            }
        } else {
            for c in chars {
                if PUNCTUATION_SET.contains(&c) {
                    phones.push(c.to_string());
                    word2ph.push(1);
                }
            }
        }
    }

    (phones, word2ph)
}

fn split_jyutping(jyutping: &str) -> (String, String) {
    for ini in INITIALS.iter().rev() {
        if jyutping.starts_with(ini) {
            let rest = &jyutping[ini.len()..];
            return (ini.to_string(), rest.to_string());
        }
    }
    (String::new(), jyutping.to_string())
}
