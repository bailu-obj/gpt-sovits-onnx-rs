// Shared tokenization regex used by language segmentation.

use once_cell::sync::Lazy;
use regex::Regex;

pub(crate) static TOKEN_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?x)
        [\p{Han}]+ |              # Chinese characters
        [a-zA-Z]+(?:['-][a-zA-Z]+)* | # English words with optional apostrophes/hyphens
        \d+(?:\.\d+)? |          # Numbers (including decimals)
        [.,!?;:()\[\]<>\-\"$/\u{3001}\u{3002}\u{FF01}\u{FF1F}\u{FF1B}\u{FF1A}\u{FF0C}\u{2018}\u{2019}\u{201C}\u{201D}] | # Punctuation
        \s+                      # Whitespace
        "#,
    )
    .unwrap()
});

pub(crate) static HAN_ONLY: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\p{Han}+$").unwrap());
