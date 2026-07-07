#!/usr/bin/env python3
"""Export Python reference preprocess output for parity comparison with Rust."""
import json
import os
import sys

GPT_SOVITS_ROOT = os.environ.get(
    "GPT_SOVITS_ROOT", os.path.expanduser("~/projects/GPT-SoVITS")
)
sys.path.insert(0, GPT_SOVITS_ROOT)
sys.path.insert(0, os.path.join(GPT_SOVITS_ROOT, "GPT_SoVITS"))

from text import cleaned_text_to_sequence
from text import chinese2
from text.cleaner import clean_text


def get_phones_zh(text: str, version: str = "v2"):
    norm = chinese2.text_normalize(text)
    phones, word2ph = chinese2.g2p(norm)
    phone_ids = cleaned_text_to_sequence(phones, version)
    if len(phone_ids) < 6:
        phones2, w2p2 = chinese2.g2p("." + norm)
        phone_ids = cleaned_text_to_sequence(phones2, version)
        word2ph = w2p2
        norm = "." + norm
    return {
        "phone_ids": phone_ids,
        "word2ph": word2ph,
        "norm_text": norm,
        "phones": phones,
    }


def get_phones_auto(text: str, version: str = "v2"):
    """Mixed auto mode via LangSegmenter when available, else zh clean_text."""
    try:
        from text.LangSegmenter.langsegmenter import LangSegmenter

        spans = LangSegmenter.getTexts(text)
        phones_list = []
        word2ph_list = []
        norm_parts = []
        for span in spans:
            lang = span["lang"]
            if lang == "digit":
                lang = "zh"
            phones, word2ph, norm = clean_text(span["text"], lang, version)
            phones_list.extend(cleaned_text_to_sequence(phones, version))
            if word2ph:
                word2ph_list.extend(word2ph)
            norm_parts.append(norm)
        norm_text = "".join(norm_parts)
        if len(phones_list) < 6:
            return get_phones_auto("." + text, version)
        return {
            "phone_ids": phones_list,
            "word2ph": word2ph_list or None,
            "norm_text": norm_text,
            "phones": phones_list,
        }
    except Exception:
        try:
            phones, word2ph, norm_text = clean_text(text, "zh", version)
            phone_ids = cleaned_text_to_sequence(phones, version)
            if len(phone_ids) < 6:
                phones2, w2p2, norm2 = clean_text("." + text, "zh", version)
                phone_ids = cleaned_text_to_sequence(phones2, version)
                word2ph = w2p2
                norm_text = norm2
            return {
                "phone_ids": phone_ids,
                "word2ph": word2ph,
                "norm_text": norm_text,
                "phones": phones,
            }
        except Exception:
            return get_phones_zh(text, version)


def main():
    corpus_path = os.environ.get(
        "PREPROCESS_CORPUS",
        os.path.join(os.path.dirname(__file__), "..", "resource", "preprocess_corpus.json"),
    )
    with open(corpus_path) as f:
        corpus = json.load(f)

    results = []
    for item in corpus:
        text = item["text"]
        lang = item.get("lang", "auto")
        try:
            if lang == "auto":
                out = get_phones_auto(text)
            else:
                out = get_phones_zh(text)
            results.append({"text": text, "lang": lang, "ok": True, **out})
        except Exception as e:
            results.append({"text": text, "lang": lang, "ok": False, "error": str(e)})

    json.dump(results, sys.stdout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
