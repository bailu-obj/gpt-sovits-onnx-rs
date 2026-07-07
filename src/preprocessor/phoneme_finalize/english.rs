// English post-G2P filter (Python english.py replace_phs + g2p meta-token drop).

use crate::preprocessor::{
    phone_symbol::SYMBOLS,
    phoneme_finalize::symbols::{pad_short_english_phones, phones_to_ids},
};

const DROP_TOKENS: &[&str] = &[" ", "<pad>", "UW", "</s>", "<s>"];

/// Filter raw ARPABET tokens, map to symbol IDs, apply short-phone padding.
pub fn finalize_span_en(phonemes: Vec<String>) -> (Vec<i64>, Vec<i32>) {
    let filtered = filter_english_phonemes(phonemes);
    let mut word2ph = vec![1i32; filtered.len()];
    let mut phone_ids = phones_to_ids(&filtered, "en");
    pad_short_english_phones(&mut phone_ids);
    if phone_ids.len() > word2ph.len() {
        word2ph.insert(0, 1);
    }
    (phone_ids, word2ph)
}

/// Python english.g2p + replace_phs.
pub fn filter_english_phonemes(phonemes: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    for ph in phonemes {
        if DROP_TOKENS.contains(&ph.as_str()) {
            continue;
        }
        let ph = if ph == "<unk>" { "UNK".to_string() } else { ph };
        if ph == "'" {
            out.push("-".to_string());
        } else if SYMBOLS.contains_key(&ph) {
            out.push(ph);
        }
        // drop unknown symbols (Python replace_phs warns and skips)
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_drops_meta_tokens() {
        let filtered = filter_english_phonemes(vec![
            "HH".to_string(),
            " ".to_string(),
            "<pad>".to_string(),
            "AH0".to_string(),
        ]);
        assert_eq!(filtered, vec!["HH", "AH0"]);
    }

    #[test]
    fn test_filter_unk_mapping() {
        let filtered = filter_english_phonemes(vec!["<unk>".to_string()]);
        assert_eq!(filtered, vec!["UNK"]);
    }
}
