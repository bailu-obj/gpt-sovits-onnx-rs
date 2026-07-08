// SP2/SP3 silence token injection after special-symbol commas.

use crate::preprocessor::{clean::SilenceTag, phone_symbol::get_phone_symbol};

/// Insert SP2/SP3 after comma phonemes that replaced ￥ or ^ during cleaning.
pub fn inject_silence_tokens(phone_ids: &mut Vec<i64>, tags: &[SilenceTag]) {
    if tags.is_empty() {
        return;
    }
    let comma_id = get_phone_symbol(",");
    let mut tag_idx = 0;
    let mut i = 0;
    while i < phone_ids.len() && tag_idx < tags.len() {
        if phone_ids[i] == comma_id {
            let sp_id = match tags[tag_idx] {
                SilenceTag::Sp2 => get_phone_symbol("SP2"),
                SilenceTag::Sp3 => get_phone_symbol("SP3"),
            };
            phone_ids.insert(i + 1, sp_id);
            i += 2;
            tag_idx += 1;
        } else {
            i += 1;
        }
    }
}
