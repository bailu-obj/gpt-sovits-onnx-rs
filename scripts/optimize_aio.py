#!/usr/bin/env python3
"""Optimize ONNX models with named passes and an explicit precision preset.

Presets (default: fast):
  fp32     — optimize/slim only; keep FP32 weights
  fp16     — selective FP16 on VITS graphs with native FP16 I/O (BERT/T2S stay FP32)
  quality  — INT4 BERT/g2pW; T2S/SSL/VITS stay FP32 (parity-friendly)
  fast     — INT4 BERT/g2pW + INT8 T2S decoders; SSL/VITS FP32 (best CPU latency)

On Apple Silicon CPU EP, blanket FP16 (including BERT/T2S) is slower than FP32
even with native FP16 I/O — ORT FP16 MatMul loses to FP32. Default ``fp16``
therefore converts only VITS kinds. Pass ``--fp16-kinds all --fp16-keep-io`` for
the legacy weight-only (FP32 I/O) experiment.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

import onnx
from onnx import TensorProto, checker, shape_inference, version_converter
from onnxruntime.quantization import (
    QuantType,
    matmul_nbits_quantizer,
    quant_utils,
    quantize_dynamic,
)
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxslim import slim

# Typed ops whose dtype/to attribute must match output value_info after FP16 convert.
_FP16_TYPED_ATTRS = {
    "Cast": "to",
    "RandomNormalLike": "dtype",
    "RandomNormal": "dtype",
    "RandomUniform": "dtype",
    "RandomUniformLike": "dtype",
    "EyeLike": "dtype",
    "Multinomial": "dtype",
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PRECISIONS = ("fp32", "fp16", "quality", "fast")
MODEL_KINDS = (
    "vits",
    "vits_decode",
    "vits_ref",
    "bert",
    "decoder",
    "g2p",
    "g2p_en",
    "default",
)

# Apple Silicon CPU EP: only VITS graphs measured faster than FP32 in isolation.
# BERT / T2S decoders regress badly; keep them FP32 under the default fp16 preset.
DEFAULT_FP16_KINDS = ("vits", "vits_decode", "vits_ref")

# Base optimize passes (before precision-specific weight transforms).
PASS_SETS = {
    "vits": ("onnxslim", "opset21", "check"),
    "vits_decode": ("onnxslim", "opset21", "check"),
    "vits_ref": ("onnxslim", "opset21", "check"),
    "bert": ("ort_bert", "opset21", "check"),
    "decoder": ("ort_generic", "onnxslim", "opset21", "check"),
    "g2p": ("onnxslim", "opset21", "check"),
    "g2p_en": ("onnxslim", "opset21", "check"),
    "default": ("onnxslim", "opset21", "check"),
}


def parse_fp16_kinds(value: str) -> tuple[str, ...]:
    """Parse comma-separated model kinds, or 'all' for every kind."""
    text = value.strip().lower()
    if not text:
        raise argparse.ArgumentTypeError("--fp16-kinds must not be empty")
    if text == "all":
        return MODEL_KINDS
    kinds = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [k for k in kinds if k not in MODEL_KINDS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown kind(s) {unknown}; expected subset of {', '.join(MODEL_KINDS)} or 'all'"
        )
    return kinds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize ONNX models with a precision preset."
    )
    parser.add_argument("--input-dir", default="onnx/custom")
    parser.add_argument("--output-dir", default="onnx-patched/custom")
    parser.add_argument(
        "--precision",
        choices=PRECISIONS,
        default="fast",
        help="Weight precision preset (default: fast).",
    )
    parser.add_argument(
        "--fp16-kinds",
        type=parse_fp16_kinds,
        default=DEFAULT_FP16_KINDS,
        metavar="KINDS",
        help=(
            "Comma-separated model kinds to convert under --precision fp16 "
            f"(default: {','.join(DEFAULT_FP16_KINDS)}; use 'all' for legacy blanket FP16)."
        ),
    )
    parser.add_argument(
        "--fp16-keep-io",
        action="store_true",
        help=(
            "Keep public graph I/O as FP32 when converting to FP16 (legacy). "
            "Default is native FP16 I/O for converted kinds (required for VITS speedup)."
        ),
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip onnx.checker / shape_inference validation",
    )
    return parser.parse_args()


def validate_environment():
    for module, install_name in [
        ("onnxruntime", "onnxruntime"),
        ("onnxslim", "onnxslim"),
    ]:
        try:
            __import__(module)
        except ImportError:
            logger.error(
                f"{install_name} is not installed. Install it with 'pip install {install_name}'"
            )
            raise SystemExit(1)


def model_kind(path: str | Path) -> str:
    """Classify by filename; g2p_en/ must not be mistaken for a T2S decoder."""
    p = Path(path)
    parts_lower = [part.lower() for part in p.parts]
    name = p.name.lower()
    if "g2p_en" in parts_lower:
        return "g2p_en"
    if "vits_decode" in name:
        return "vits_decode"
    if "vits_ref" in name:
        return "vits_ref"
    if "vits" in name:
        return "vits"
    if "bert" in name:
        return "bert"
    if "decoder" in name:
        return "decoder"
    if "g2p" in name:
        return "g2p"
    return "default"


def validate_model(model: onnx.ModelProto, label: str) -> onnx.ModelProto:
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:
        logger.warning(f"shape_inference failed for {label}: {exc}")
    try:
        checker.check_model(model, full_check=False)
    except Exception as exc:
        logger.warning(f"onnx.checker failed for {label}: {exc}")
    return model


def apply_int4(model: onnx.ModelProto, output_path: str) -> None:
    quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=64,
        is_symmetric=True,
        accuracy_level=2,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul", "Gather", "Attention"),
        bits=4,
    )
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model,
        nodes_to_exclude=None,
        nodes_to_include=None,
        algo_config=quant_config,
    )
    quant.process()
    quant.model.save_model_to_file(output_path)


def repair_fp16_typed_attrs(model: onnx.ModelProto) -> int:
    """Sync typed-op attrs when value_info says FLOAT16 but attr still FLOAT.

    onnxconverter_common / some converters leave Cast.to=1 (float) while tagging
    the output as float16; ORT then rejects the model on load. Also clear stale
    intermediate value_info so ORT re-infers types (graph I/O kept).
    """
    vi = {
        v.name: v
        for v in list(model.graph.value_info)
        + list(model.graph.input)
        + list(model.graph.output)
    }
    fixed = 0
    for node in model.graph.node:
        attr_name = _FP16_TYPED_ATTRS.get(node.op_type)
        if not attr_name or not node.output:
            continue
        out = node.output[0]
        if out not in vi:
            continue
        et = vi[out].type.tensor_type.elem_type
        for attr in node.attribute:
            if (
                attr.name == attr_name
                and attr.i == TensorProto.FLOAT
                and et == TensorProto.FLOAT16
            ):
                attr.i = TensorProto.FLOAT16
                fixed += 1
    del model.graph.value_info[:]
    return fixed


def apply_fp16(
    model: onnx.ModelProto, *, keep_io_types: bool = False
) -> onnx.ModelProto:
    """Convert floating weights (and optionally I/O) to FP16.

    Default ``keep_io_types=False`` exposes native FP16 public I/O so ORT does
    not insert boundary Casts. Apple Silicon CPU EP still loses on BERT/T2S
    MatMul even with native I/O; VITS decode benefits.
    """
    model_fp16 = convert_float_to_float16(model, keep_io_types=keep_io_types)
    n_fixed = repair_fp16_typed_attrs(model_fp16)
    if n_fixed:
        logger.info(f"FP16 typed-attr repair fixed {n_fixed} node(s)")
    return model_fp16


def apply_optimize_passes(
    model: onnx.ModelProto, kind: str, output_path: str, skip_check: bool
) -> onnx.ModelProto:
    for pass_name in PASS_SETS[kind]:
        if pass_name == "onnxslim":
            model = slim(model)
            logger.info(f"onnxslim done for: {output_path}")
        elif pass_name == "ort_bert":
            optimized = optimizer.optimize_model(
                model,
                model_type="bert",
                num_heads=16,
                hidden_size=1024,
                only_onnxruntime=True,
            )
            model = optimized.model
            logger.info(f"ort_bert done for: {output_path}")
        elif pass_name == "ort_generic":
            optimized = optimizer.optimize_model(model, only_onnxruntime=True)
            model = optimized.model
            logger.info(f"ort_generic done for: {output_path}")
        elif pass_name == "opset21":
            model = version_converter.convert_version(model, 21)
            logger.info(f"opset21 done for: {output_path}")
        elif pass_name == "check":
            if not skip_check:
                model = validate_model(model, output_path)
        else:
            raise ValueError(f"Unknown pass: {pass_name}")
    return model


# Kinds where Apple Silicon CPU EP FP16 MatMul is slower than FP32 even with
# native FP16 I/O (measured 2026-07-23 on M4 Pro). Converting them requires an
# explicit override via --fp16-kinds.
_FP16_CPU_REGRESS_KINDS = frozenset({"bert", "decoder", "g2p", "g2p_en", "default"})


def apply_precision(
    model: onnx.ModelProto,
    kind: str,
    precision: str,
    output_path: str,
    skip_check: bool,
    fp16_kinds: tuple[str, ...] = DEFAULT_FP16_KINDS,
    fp16_keep_io: bool = False,
) -> bool:
    """Apply precision-specific weight transforms. Returns True if already saved."""
    if precision == "fp32":
        return False

    if precision == "fp16":
        if kind not in fp16_kinds:
            logger.info(
                f"Skipping FP16 for kind={kind} (not in {fp16_kinds}): {output_path}"
            )
            return False
        if kind in _FP16_CPU_REGRESS_KINDS:
            logger.warning(
                f"FP16 on kind={kind} regresses vs FP32 on Apple CPU EP "
                f"(native I/O still slower); converting only because listed in "
                f"--fp16-kinds: {output_path}"
            )
        model_fp16 = apply_fp16(model, keep_io_types=fp16_keep_io)
        if not skip_check:
            model_fp16 = validate_model(model_fp16, output_path)
        onnx.save(model_fp16, output_path)
        logger.info(
            f"FP16 conversion done for kind={kind} "
            f"(keep_io_types={fp16_keep_io}): {output_path}"
        )
        return True

    # quality / fast: INT4 on BERT and g2pW
    if kind in ("bert", "g2p") and precision in ("quality", "fast"):
        if not skip_check:
            model = validate_model(model, output_path)
        apply_int4(model, output_path)
        logger.info(f"INT4 quantization done for: {output_path}")
        return True

    # fast only: INT8 on T2S decoders
    if kind == "decoder" and precision == "fast":
        if not skip_check:
            model = validate_model(model, output_path)
        quantize_dynamic(
            model,
            output_path,
            op_types_to_quantize=["MatMul", "Attention", "Gather"],
            per_channel=True,
            reduce_range=True,
            weight_type=QuantType.QInt8,
        )
        logger.info(f"INT8 quantization done for: {output_path}")
        return True

    return False


def process_model(
    file_path: str,
    output_path: str,
    precision: str,
    skip_check: bool,
    fp16_kinds: tuple[str, ...] = DEFAULT_FP16_KINDS,
    fp16_keep_io: bool = False,
) -> str:
    logger.info(f"Processing model: {file_path}")
    model = onnx.load(file_path)
    kind = model_kind(output_path)
    logger.info(f"Pass set for {kind}: {PASS_SETS[kind]} (precision={precision})")

    model = apply_optimize_passes(model, kind, output_path, skip_check)
    saved = apply_precision(
        model,
        kind,
        precision,
        output_path,
        skip_check,
        fp16_kinds=fp16_kinds,
        fp16_keep_io=fp16_keep_io,
    )
    if not saved:
        onnx.save(model, output_path)
    return output_path


def collect_onnx_files(input_dir: str) -> list[str]:
    """Top-level *.onnx plus nested g2p_en/*.onnx."""
    files = sorted(glob.glob(os.path.join(input_dir, "*.onnx")))
    g2p_en = sorted(glob.glob(os.path.join(input_dir, "g2p_en", "*.onnx")))
    return files + g2p_en


def main():
    args = parse_args()
    validate_environment()
    os.makedirs(args.output_dir, exist_ok=True)

    onnx_files = collect_onnx_files(args.input_dir)
    top_level = [f for f in onnx_files if Path(f).parent.name.lower() != "g2p_en"]
    if not top_level:
        logger.error(f"No ONNX files found in {args.input_dir}")
        raise SystemExit(1)

    has_decoder = any(
        model_kind(f) == "decoder" for f in top_level
    )
    has_vits = any(model_kind(f).startswith("vits") for f in top_level)
    # Allow decode-only / single-file folders for A/B experiments.
    if len(top_level) > 1 and (not has_decoder or not has_vits):
        logger.error("Need both decoder and vits models in the folder")
        raise SystemExit(1)

    logger.info(f"Precision preset: {args.precision}")
    if args.precision == "fp16":
        logger.info(
            f"FP16 kinds: {args.fp16_kinds}; keep_io_types={args.fp16_keep_io}"
        )
    for file_path in onnx_files:
        rel = os.path.relpath(file_path, args.input_dir)
        output_path = os.path.join(args.output_dir, rel)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_path = process_model(
            file_path,
            output_path,
            args.precision,
            args.skip_check,
            fp16_kinds=tuple(args.fp16_kinds),
            fp16_keep_io=args.fp16_keep_io,
        )
        logger.info(f"Optimization complete for: {final_path}")

    logger.info("All models processed successfully!")


if __name__ == "__main__":
    main()
