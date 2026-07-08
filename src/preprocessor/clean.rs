// Global text cleaning before chunking and language segmentation.

use anyhow::{Result, bail};

use crate::preprocessor::text_normalize;

/// Silence phoneme to inject after a comma that replaced a special symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilenceTag {
    Sp2,
    Sp3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SilenceMarker {
    /// Character index in the cleaned text where the replacement comma appears.
    pub char_index: usize,
    pub tag: SilenceTag,
}

pub struct CleanedInput {
    pub text: String,
    pub silence_markers: Vec<SilenceMarker>,
}

/// Apply GPT-SoVITS global normalization (emoji strip, punct unify, special symbols).
pub fn normalize_input(text: &str) -> Result<CleanedInput> {
    if text.trim().is_empty() {
        bail!("Input text is empty");
    }
    let normalized = text_normalize::text_normalize(text);
    let (text, silence_markers) = apply_special_symbols(&normalized);
    Ok(CleanedInput {
        text,
        silence_markers,
    })
}

/// Markers whose replacement comma falls within `chunk` at `chunk_start` in the full cleaned text.
pub fn silence_tags_for_chunk(
    markers: &[SilenceMarker],
    cleaned: &str,
    chunk: &str,
) -> Vec<SilenceTag> {
    let chunk_start = chunk_offset_in_cleaned(cleaned, chunk);
    let chunk_end = chunk_start + chunk.chars().count();
    markers
        .iter()
        .filter(|m| m.char_index >= chunk_start && m.char_index < chunk_end)
        .map(|m| m.tag)
        .collect()
}

/// Locate the character offset of `chunk` within the pre-chunking cleaned text.
fn chunk_offset_in_cleaned(cleaned: &str, chunk: &str) -> usize {
    let trimmed = chunk.trim_end_matches(['。', '.']);
    if let Some(pos) = cleaned.find(trimmed) {
        return cleaned[..pos].chars().count();
    }
    if let Some(rest) = trimmed.strip_prefix('。') {
        if cleaned.starts_with(rest) {
            return 0;
        }
        if let Some(pos) = cleaned.find(rest) {
            return cleaned[..pos].chars().count();
        }
    }
    if let Some(pos) = cleaned.find(trimmed.trim_start_matches('。')) {
        return cleaned[..pos].chars().count();
    }
    0
}

/// Handle GPT-SoVITS special silence symbols (Python cleaner.py special list).
fn apply_special_symbols(text: &str) -> (String, Vec<SilenceMarker>) {
    let mut out = String::with_capacity(text.len());
    let mut markers = Vec::new();
    let mut char_index = 0usize;
    for c in text.chars() {
        match c {
            '￥' => {
                out.push(',');
                markers.push(SilenceMarker {
                    char_index,
                    tag: SilenceTag::Sp2,
                });
                char_index += 1;
            }
            '^' => {
                out.push(',');
                markers.push(SilenceMarker {
                    char_index,
                    tag: SilenceTag::Sp3,
                });
                char_index += 1;
            }
            other => {
                out.push(other);
                char_index += 1;
            }
        }
    }
    (out, markers)
}

/// True when input is English-only (no CJK) and should use English punctuation in chunking.
pub fn infer_lang_en(text: &str) -> bool {
    let t = text.trim();
    t.chars().any(|c| c.is_ascii_alphabetic())
        && !t
            .chars()
            .any(|c| matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF))
}
