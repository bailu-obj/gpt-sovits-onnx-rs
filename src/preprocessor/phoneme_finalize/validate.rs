// word2ph alignment checks (Python cleaner.py assertions).

use anyhow::{Result, bail};

/// Python cleaner.py: `len(phones) == sum(word2ph)` and `len(norm_text) == len(word2ph)`.
pub fn validate_word2ph(norm_text: &str, word2ph: &[i32], phone_count: usize) -> Result<()> {
    let char_count = norm_text.chars().count();
    let w2p_sum: usize = word2ph.iter().map(|&n| n.max(0) as usize).sum();

    if w2p_sum != phone_count {
        bail!("word2ph sum {w2p_sum} != phone count {phone_count} for text {norm_text:?}");
    }

    if word2ph.len() != char_count {
        bail!(
            "word2ph len {} != norm_text char count {char_count} for {norm_text:?}",
            word2ph.len()
        );
    }

    Ok(())
}
