// Symbol-string → phone ID mapping and padding (Python cleaner.py + cleaned_text_to_sequence).

use crate::preprocessor::phone_symbol::{SYMBOLS, get_phone_symbol_logged};

/// Map phoneme symbol strings to integer IDs; unknown → UNK (Python cleaner.py).
pub fn phones_to_ids(phones: &[String], context: &str) -> Vec<i64> {
    phones.iter().map(|ph| phone_to_id(ph, context)).collect()
}

pub fn phone_to_id(ph: &str, context: &str) -> i64 {
    if SYMBOLS.contains_key(ph) {
        get_phone_symbol_logged(ph, context)
    } else {
        get_phone_symbol_logged("UNK", context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unknown_maps_to_unk() {
        let unk_id = phone_to_id("NOTAPHONEME", "test");
        let explicit = phone_to_id("UNK", "test");
        assert_eq!(unk_id, explicit);
    }
}

/// Python cleaner.py: pad with comma when English phones < 4.
pub fn pad_short_english_phones(phone_ids: &mut Vec<i64>) {
    if phone_ids.len() < 4 {
        let comma = get_phone_symbol_logged(",", "");
        phone_ids.insert(0, comma);
    }
}
