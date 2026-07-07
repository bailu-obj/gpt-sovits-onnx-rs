// Short-sequence retry (Python TextPreprocessor.get_phones_and_bert).

pub const MIN_PHONES: usize = 6;

pub fn needs_short_retry(phone_count: usize, is_final: bool) -> bool {
    !is_final && phone_count < MIN_PHONES
}
