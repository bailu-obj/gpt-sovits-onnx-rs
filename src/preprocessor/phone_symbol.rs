use std::collections::HashMap;

use lazy_static::lazy_static;
use log::warn;

static SYMBOLS_V2: &str = include_str!("../../resource/symbols_v2.json");

pub const PUNCTUATION: &[&str] = &["!", "?", "…", ",", ".", "-"];

lazy_static! {
    pub static ref SYMBOLS: HashMap<String, i64> = {
        let mut symbols: HashMap<String, i64> = serde_json::from_str(SYMBOLS_V2).unwrap();
        symbols.insert(" ".to_string(), symbols["\u{7a7a}"]);
        symbols.insert("'".to_string(), symbols["-"]);
        symbols
    };
}

#[inline]
pub fn get_phone_symbol(ph: &str) -> i64 {
    get_phone_symbol_logged(ph, "")
}

pub fn get_phone_symbol_logged(ph: &str, context: &str) -> i64 {
    match SYMBOLS.get(ph) {
        Some(&id) => id,
        None => {
            if ph != "UNK" && !ph.is_empty() {
                warn!(
                    "Unknown phoneme '{}' in context '{}', using UNK",
                    ph, context
                );
            }
            SYMBOLS.get("UNK").copied().unwrap_or(86)
        }
    }
}
