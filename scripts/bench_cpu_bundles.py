#!/usr/bin/env python3
"""Run gpt_sovits_demo --benchmark for several bundles and write JSON summaries.

Bundles can be labeled with a precision preset either via:
  --bundles path1 path2 ...          (precision inferred from directory name)
  --preset-bundles fast=path quality=path ...
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SHORT_TEXT = "你好，这是一个测试。"
MULTI_TEXT = "你好啊。这是一个测试。吃葡萄不吐葡萄皮。"
KNOWN_PRECISIONS = ("fp32", "fp16", "quality", "fast")


def infer_precision(bundle: Path) -> str | None:
    name = bundle.name.lower()
    for prec in KNOWN_PRECISIONS:
        if name == prec or name.endswith(f"_{prec}") or f"_{prec}_" in name:
            return prec
    # Legacy names
    if "unquant" in name or name.endswith("_fp32"):
        return "fp32"
    if "quant" in name or name.endswith("_layout"):
        return "fast"
    return None


def parse_demo_output(text: str) -> dict:
    out: dict = {"raw_tail": text[-2000:]}
    patterns = {
        "cold_init_ms": r"Cold init:\s*([\d.]+)\s*ms",
        "reference_ms": r"Reference:\s*([\d.]+)\s*ms",
        "ttfa_ms": r"TTFA \(first audio sample\):\s*([\d.]+)\s*ms",
        "stream_total_ms": r"stream total:\s*([\d.]+)\s*ms",
        "sync_median_ms": r"Median:\s*([\d.]+)\s*ms",
        "sync_p95_ms": r"P95:\s*([\d.]+)\s*ms",
        "sync_avg_ms": r"Average:\s*([\d.]+)\s*ms",
        "rss_after_init": r"RSS after init:\s*(.+)",
        "rss_after_ref": r"RSS after reference:\s*(.+)",
        "rss_final": r"RSS \(max/current\):\s*(.+)",
        "split_vits": r"split_vits=(\w+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if not m:
            continue
        val = m.group(1).strip()
        if key.endswith("_ms"):
            out[key] = float(val)
        elif key == "split_vits":
            out[key] = val == "true"
        else:
            out[key] = val
    return out


def bundle_size_mib(bundle: Path) -> float:
    total = 0
    for p in bundle.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return round(total / (1024 * 1024), 1)


def run_case(
    bundle: Path,
    text: str,
    runs: int,
    seed: int,
    ort_profile: str,
    precision: str | None,
) -> dict:
    out_wav = REPO / "target" / "bench" / f"bench_{bundle.name}.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cargo",
        "run",
        "--release",
        "--example",
        "gpt_sovits_demo",
        "--",
        "--model-path",
        str(bundle),
        "--text",
        text,
        "--seed",
        str(seed),
        "--run-count",
        str(runs),
        "--benchmark",
        "--ort-profile",
        ort_profile,
        "--output",
        str(out_wav),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - t0
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    parsed = parse_demo_output(combined)
    parsed.update(
        {
            "bundle": str(bundle),
            "precision": precision,
            "bundle_size_mib": bundle_size_mib(bundle),
            "text": text,
            "runs": runs,
            "seed": seed,
            "ort_profile": ort_profile,
            "exit_code": proc.returncode,
            "wall_s": round(wall, 3),
        }
    )
    if proc.returncode != 0:
        parsed["error"] = combined[-4000:]
    return parsed


def parse_preset_bundles(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(
                f"Invalid --preset-bundles entry '{item}' (expected precision=path)"
            )
        prec, path = item.split("=", 1)
        prec = prec.strip().lower()
        if prec not in KNOWN_PRECISIONS:
            raise SystemExit(
                f"Unknown precision '{prec}' (expected: {', '.join(KNOWN_PRECISIONS)})"
            )
        out.append((prec, Path(path)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark CPU bundles and label results by precision preset."
    )
    ap.add_argument(
        "--bundles",
        nargs="+",
        type=Path,
        help="Bundle directories (precision inferred from name when possible)",
    )
    ap.add_argument(
        "--preset-bundles",
        nargs="+",
        metavar="PRECISION=PATH",
        help="Explicit precision=path pairs, e.g. fast=models/cpu_opt_v2_fast",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ort-profile", default="latency")
    args = ap.parse_args()

    jobs: list[tuple[str | None, Path]] = []
    if args.preset_bundles:
        jobs.extend(parse_preset_bundles(args.preset_bundles))
    if args.bundles:
        for bundle in args.bundles:
            jobs.append((infer_precision(bundle), bundle))
    if not jobs:
        raise SystemExit("Provide --bundles and/or --preset-bundles")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": args.runs,
        "seed": args.seed,
        "ort_profile": args.ort_profile,
        "cases": [],
    }
    for precision, bundle in jobs:
        for label, text in (("short", SHORT_TEXT), ("multi", MULTI_TEXT)):
            print(f"=== {precision or '?'} / {bundle} / {label} ===", flush=True)
            result = run_case(
                bundle.resolve(),
                text,
                args.runs,
                args.seed,
                args.ort_profile,
                precision,
            )
            result["case"] = label
            report["cases"].append(result)
            print(
                json.dumps(
                    {
                        k: result.get(k)
                        for k in (
                            "precision",
                            "case",
                            "cold_init_ms",
                            "reference_ms",
                            "ttfa_ms",
                            "sync_median_ms",
                            "rss_final",
                            "bundle_size_mib",
                            "split_vits",
                            "exit_code",
                        )
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
