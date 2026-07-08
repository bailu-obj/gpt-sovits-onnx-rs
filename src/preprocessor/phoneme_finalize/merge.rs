// Merge per-span G2P + BERT features into one chunk output.

use anyhow::{Result, bail};
use log::warn;
use ndarray::{Array2, Axis, concatenate};

use crate::preprocessor::{
    bert::BertModel,
    clean::SilenceTag,
    g2p::SpanResult,
    lang::Lang,
    phoneme_finalize::{silence::inject_silence_tokens, validate::validate_word2ph},
};

pub fn merge_span_results(
    bert: &mut BertModel,
    spans: Vec<SpanResult>,
    silence_tags: &[SilenceTag],
) -> Result<(String, Vec<i64>, Array2<f32>)> {
    let mut norm_text = String::new();
    let mut phone_ids = Vec::new();
    let mut bert_parts: Vec<Array2<f32>> = Vec::new();

    for span in spans {
        let phone_len = span.phone_ids.len();
        if span.lang == Lang::Zh {
            validate_word2ph(&span.text, &span.word2ph, phone_len)?;
        }

        let bert_feat = match span.lang {
            Lang::Zh => bert.get_bert(&span.text, &span.word2ph, phone_len, Lang::Zh)?,
            Lang::En => bert.get_bert(&span.text, &span.word2ph, phone_len, Lang::En)?,
        };
        if bert_feat.shape()[0] != phone_len {
            let error_msg = format!(
                "BERT length mismatch for '{}': expected {}, got {}",
                span.text,
                phone_len,
                bert_feat.shape()[0]
            );
            warn!("{}", error_msg);
            bail!(error_msg);
        }

        norm_text.push_str(&span.text);
        phone_ids.extend_from_slice(&span.phone_ids);
        bert_parts.push(bert_feat);
    }

    inject_silence_tokens(&mut phone_ids, silence_tags);

    let padding = phone_ids
        .len()
        .saturating_sub(bert_parts.iter().map(|b| b.shape()[0]).sum::<usize>());
    if padding > 0 {
        bert_parts.push(bert.get_bert("", &[], padding, Lang::En)?);
    }

    let bert_merged = if bert_parts.len() == 1 {
        bert_parts.remove(0)
    } else if bert_parts.is_empty() {
        Array2::<f32>::zeros((0, 1024))
    } else {
        concatenate(
            Axis(0),
            &bert_parts.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )?
    };

    if bert_merged.shape()[0] != phone_ids.len() {
        bail!(
            "Merged BERT length {} != phone count {}",
            bert_merged.shape()[0],
            phone_ids.len()
        );
    }

    Ok((norm_text, phone_ids, bert_merged))
}
