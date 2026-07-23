//! Preprocess parity tests against Python GPT-SoVITS reference output.
//! Run Python exporter first:
//!   python3 scripts/preprocess_parity.py > /tmp/python_preprocess.json
//! Then:
//!   PYTHON_PREPROCESS_JSON=/tmp/python_preprocess.json cargo test preprocess_parity

use gpt_sovits_onnx_rs::{
    LangId, TextProcessor,
    bert::BertModel,
    en::g2p_en::G2pEn,
    text_normalize::text_normalize,
    zh::{g2pw::G2PW, mandarin_g2p},
};
use jieba_rs::Jieba;
use serde::Deserialize;
use std::{env, fs, path::{Path, PathBuf}};

#[derive(Debug, Deserialize)]
struct CorpusItem {
    text: String,
    lang: String,
}

#[derive(Debug, Deserialize)]
struct PythonResult {
    text: String,
    ok: bool,
    phone_ids: Option<Vec<i64>>,
    #[serde(default)]
    error: Option<String>,
}

#[test]
fn preprocess_corpus_runs_without_panic() {
    let corpus = load_corpus();
    let mut tp = make_processor();
    for item in &corpus {
        let lang = if item.lang.contains("yue") {
            LangId::AutoYue
        } else {
            LangId::Auto
        };
        let result = tp.get_phone_and_bert(&item.text, lang);
        assert!(
            result.is_ok(),
            "failed on '{}': {:?}",
            item.text,
            result.err()
        );
    }
}

#[test]
fn preprocess_text_normalize_smoke() {
    let n = text_normalize("你好，，世界！！");
    assert!(!n.contains("，"));
    assert!(n.contains(','));
}

#[test]
fn preprocess_mandarin_g2p_smoke() {
    let mut g2pw = G2PW::new(None::<&str>).unwrap();
    let jieba = Jieba::new();
    let result = mandarin_g2p::g2p_mandarin("你好世界。", &mut g2pw, &jieba);
    assert!(result.phones.len() >= 4);
}

#[test]
fn preprocess_mixed_single_chunk() {
    let mut tp = make_processor();
    let result = tp
        .get_phone_and_bert("你好hello世界", LangId::Auto)
        .expect("mixed input should succeed");
    assert_eq!(
        result.len(),
        1,
        "mixed zh/en should produce one merged chunk, got {}",
        result.len()
    );
    assert!(
        result[0].1.len() > 6,
        "merged phone sequence should be substantial"
    );
}

#[test]
fn preprocess_year_as_chinese() {
    let mut tp = make_processor();
    let result = tp
        .get_phone_and_bert("2024年", LangId::Auto)
        .expect("year input should succeed");
    assert_eq!(result.len(), 1);
    let (_text, phone_ids, _) = &result[0];
    assert!(
        !phone_ids.is_empty(),
        "2024年 should produce Chinese phonemes"
    );
}

#[test]
fn preprocess_yuan_as_chinese() {
    let mut tp = make_processor();
    let result = tp
        .get_phone_and_bert("123元", LangId::Auto)
        .expect("yuan input should succeed");
    assert_eq!(result.len(), 1);
}

#[test]
fn preprocess_percent_as_chinese() {
    let mut tp = make_processor();
    let result = tp
        .get_phone_and_bert("50%", LangId::Auto)
        .expect("percent input should succeed");
    assert_eq!(result.len(), 1);
}

#[test]
fn preprocess_unk_symbol_mapping() {
    use gpt_sovits_onnx_rs::phoneme_finalize::symbols::phone_to_id;
    let unk = phone_to_id("NOT_A_REAL_PHONE", "test");
    let dot = phone_to_id(".", "test");
    assert_ne!(unk, dot);
    assert_eq!(unk, phone_to_id("UNK", "test"));
}

#[test]
fn preprocess_word2ph_alignment_mandarin() {
    let mut g2pw = G2PW::new(None::<&str>).unwrap();
    let jieba = Jieba::new();
    let result = mandarin_g2p::g2p_mandarin("你好世界", &mut g2pw, &jieba);
    let phone_count = result.phones.len();
    let w2p_sum: i32 = result.word2ph.iter().sum();
    assert_eq!(w2p_sum as usize, phone_count);
    assert_eq!(result.word2ph.len(), result.norm_text.chars().count());
}

#[test]
fn preprocess_g2pw_batch_matches_single_dict_fallback() {
    let mut g2pw = G2PW::new(None::<&str>).unwrap();
    let texts = ["银行行长", "银行行长", "你好世界"];
    let batch = g2pw.g2p_batch(&texts);
    assert_eq!(batch.len(), texts.len());
    for (i, t) in texts.iter().enumerate() {
        let single = g2pw
            .simple_get_pinyin(t)
            .into_iter()
            .map(|o| match o {
                gpt_sovits_onnx_rs::zh::g2pw::G2PWOut::Pinyin(p)
                | gpt_sovits_onnx_rs::zh::g2pw::G2PWOut::Yue(p) => p,
                gpt_sovits_onnx_rs::zh::g2pw::G2PWOut::RawChar(c) => c.to_string(),
            })
            .collect::<Vec<_>>();
        assert_eq!(batch[i], single, "mismatch for '{}'", t);
    }
}

#[test]
fn preprocess_g2pw_onnx_batch_is_deterministic() {
    let path = env::var("G2PW_ONNX_PATH")
        .ok()
        .or_else(|| {
            let candidates = [
                "models/gpt-sovits-onnx-custom/quant/g2pW.onnx",
                "models/gpt-sovits-onnx-custom/unquant/g2pW.onnx",
            ];
            candidates
                .into_iter()
                .find(|p| Path::new(p).is_file())
                .map(|p| p.to_string())
        });
    let Some(path) = path else {
        eprintln!("skipping G2PW ONNX batch test: no g2pW.onnx found");
        return;
    };
    let mut g2pw = G2PW::new(Some(Path::new(&path))).expect("load g2pW.onnx");
    let texts = ["银行行长吃饭了", "你好世界"];
    let a = g2pw.g2p_batch(&texts);
    let b = g2pw.g2p_batch(&texts);
    assert_eq!(a, b, "batched G2PW should be deterministic");
    assert_eq!(a.len(), texts.len());
    for outs in &a {
        assert!(!outs.is_empty());
        assert!(outs.iter().all(|s| !s.is_empty()), "empty pinyin slot");
    }
}

#[test]
fn preprocess_neutral_tone_smoke() {
    let mut g2pw = G2PW::new(None::<&str>).unwrap();
    let jieba = Jieba::new();
    let result = mandarin_g2p::g2p_mandarin("妈妈", &mut g2pw, &jieba);
    assert!(result.phones.len() >= 4);
}

#[test]
fn preprocess_possessive_en_g2p() {
    let mut g2p = G2pEn::new(None::<&str>).unwrap();
    let phones = g2p.g2p("cat's").unwrap();
    assert!(!phones.is_empty());
}

#[test]
#[ignore = "requires PYTHON_PREPROCESS_JSON from scripts/preprocess_parity.py"]
fn preprocess_parity_zh_subset() {
    let path = env::var("PYTHON_PREPROCESS_JSON").expect("set PYTHON_PREPROCESS_JSON");
    let python: Vec<PythonResult> =
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let mut tp = make_processor();
    let mut matched = 0usize;
    let mut total = 0usize;
    for py in &python {
        if !py.ok {
            continue;
        }
        if py.text.chars().any(|c| c.is_ascii_alphabetic()) {
            continue;
        }
        total += 1;
        let rust = tp
            .get_phone_and_bert(&py.text, LangId::Auto)
            .unwrap_or_else(|e| panic!("rust failed on '{}': {}", py.text, e));
        let rust_ids: Vec<i64> = rust
            .iter()
            .flat_map(|(_, ids, _)| ids.iter().copied())
            .collect();
        let py_ids = py.phone_ids.clone().unwrap_or_default();
        if rust_ids == py_ids {
            matched += 1;
        }
    }
    if total == 0 {
        return;
    }
    let ratio = matched as f64 / total as f64;
    assert!(
        ratio >= 0.95,
        "zh-only parity {:.1}% below 95%",
        ratio * 100.0
    );
}

#[test]
#[ignore = "requires PYTHON_PREPROCESS_JSON from scripts/preprocess_parity.py"]
fn preprocess_parity_vs_python() {
    let path = env::var("PYTHON_PREPROCESS_JSON").expect("set PYTHON_PREPROCESS_JSON");
    let python: Vec<PythonResult> =
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let mut tp = make_processor();
    let mut matched = 0usize;
    let mut total = 0usize;
    for py in &python {
        if !py.ok {
            continue;
        }
        total += 1;
        let rust = tp
            .get_phone_and_bert(&py.text, LangId::Auto)
            .unwrap_or_else(|e| panic!("rust failed on '{}': {}", py.text, e));
        let rust_ids: Vec<i64> = rust
            .iter()
            .flat_map(|(_, ids, _)| ids.iter().copied())
            .collect();
        let py_ids = py.phone_ids.clone().unwrap_or_default();
        if rust_ids == py_ids {
            matched += 1;
        } else {
            eprintln!(
                "MISMATCH '{}'\n  python({}): {:?}\n  rust({}): {:?}",
                py.text,
                py_ids.len(),
                py_ids,
                rust_ids.len(),
                rust_ids
            );
        }
    }
    if total == 0 {
        eprintln!("SKIP: no successful Python reference outputs (install GPT-SoVITS deps)");
        return;
    }
    let ratio = matched as f64 / total.max(1) as f64;
    eprintln!("Parity: {}/{} = {:.1}%", matched, total, ratio * 100.0);
    assert!(ratio >= 0.8, "parity below 80%");
}

#[test]
fn preprocess_golden_short_mandarin() {
    assert_golden("你好", 1, &[3, 227, 167, 158, 119, 3]);
}

#[test]
fn preprocess_golden_short_en() {
    assert_golden("嗯", 1, &[3, 3, 33, 140, 3]);
}

#[test]
fn preprocess_golden_mixed() {
    assert_golden(
        "你好hello世界",
        1,
        &[227, 167, 158, 119, 51, 12, 62, 68, 251, 214, 221, 194, 3],
    );
}

#[test]
fn preprocess_golden_sp2_silence() {
    assert_golden(
        "你好￥世界",
        1,
        &[3, 227, 167, 158, 119, 1, 78, 251, 214, 221, 194, 3],
    );
}

#[test]
fn preprocess_golden_sp3_silence() {
    assert_golden(
        "你好^世界",
        1,
        &[3, 227, 167, 158, 119, 1, 79, 251, 214, 221, 194, 3],
    );
}

#[test]
fn preprocess_golden_year() {
    assert_golden(
        "2024年",
        1,
        &[33, 153, 224, 202, 33, 153, 250, 164, 227, 177, 3],
    );
}

#[test]
fn preprocess_golden_decimal_yuan() {
    assert_golden(
        "价格是99.5元",
        1,
        &[
            221, 174, 156, 131, 251, 214, 224, 202, 127, 178, 221, 217, 221, 218, 316, 257, 318,
            302, 3,
        ],
    );
}

fn assert_golden(text: &str, expected_chunks: usize, expected_ids: &[i64]) {
    let mut tp = make_processor();
    let result = tp
        .get_phone_and_bert(text, LangId::Auto)
        .unwrap_or_else(|e| panic!("golden case {:?} failed: {}", text, e));
    assert_eq!(
        result.len(),
        expected_chunks,
        "chunk count mismatch for {:?}",
        text
    );
    let ids: Vec<i64> = result
        .iter()
        .flat_map(|(_, phone_ids, _)| phone_ids.iter().copied())
        .collect();
    assert_eq!(ids, expected_ids, "phone_ids mismatch for {:?}", text);
    let total_phones: usize = result.iter().map(|(_, ids, _)| ids.len()).sum();
    let total_bert: usize = result.iter().map(|(_, _, bert)| bert.shape()[0]).sum();
    assert_eq!(
        total_bert, total_phones,
        "BERT/phone length mismatch for {:?}",
        text
    );
}

#[test]
fn preprocess_english_short_uses_dot_prefix() {
    let mut tp = make_processor();
    let result = tp
        .get_phone_and_bert("hi", LangId::Auto)
        .expect("english short input should succeed");
    assert_eq!(result.len(), 1);
    assert!(
        result[0].0.starts_with('.'),
        "english short text should get leading period, got {:?}",
        result[0].0
    );
}

fn load_corpus() -> Vec<CorpusItem> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resource/preprocess_corpus.json");
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn make_processor() -> TextProcessor {
    TextProcessor::new(
        G2PW::new(None::<&str>).unwrap(),
        G2pEn::new(None::<&str>).unwrap(),
        BertModel::new(None::<&str>).unwrap(),
    )
    .unwrap()
}

#[test]
fn preprocess_preserves_mixed_language_spaces() {
    let text = "你好啊，我最喜欢你了, 这是一个Test，可以带来巨大的Change, Do you like it?";
    let mut tp = make_processor();
    let result = tp
        .get_phone_and_bert(text, LangId::Auto)
        .expect("mixed sentence should succeed");
    let norm = &result[0].0;
    assert!(
        norm.contains(", 这是一个") || norm.contains(", 这"),
        "space after comma before Chinese should be preserved, got {:?}",
        norm
    );
    assert!(
        norm.contains("Do you like it"),
        "English word spaces should be preserved, got {:?}",
        norm
    );
    assert!(
        !norm.contains("这是一个 Test"),
        "should not insert space before embedded English, got {:?}",
        norm
    );
}
