"""Export a fixed-length VITS generator for accelerator experiments."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnxslim import slim

from vits_dynamic import _reference_graphs, validate_vits_masks


def expose_latent(model):
    graph = model.graph
    conv = next(
        (node for node in graph.node if node.name == "/vq_model/dec/conv_pre/Conv"),
        None,
    )
    if conv is None or conv.op_type != "Conv":
        raise ValueError("expected the exported VITS generator input convolution")
    latent, weight, *_ = conv.input
    channels = next(tensor.dims[1] for tensor in graph.initializer if tensor.name == weight)
    for node in graph.node:
        for names in (node.input, node.output):
            for index, name in enumerate(names):
                if name == latent:
                    names[index] = "z"
    info = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, channels, "latent_frames"])
    for value in graph.value_info:
        if value.name == latent:
            value.CopyFrom(info)
            return value
    graph.value_info.append(info)
    return graph.value_info[-1]


def quantize_generator(generator, calibration_dir, activation_bits):
    from onnxruntime.quantization import QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import (
        get_qnn_qdq_config,
        qnn_preprocess_model,
    )

    from optimize_aio import VitsCalibration

    preprocessed = generator.parent.parent / "generator-preprocessed.onnx"
    source = preprocessed if qnn_preprocess_model(generator, preprocessed) else generator
    config = get_qnn_qdq_config(
        source,
        VitsCalibration(calibration_dir, {"z", "ge"}),
        activation_type=QuantType.QUInt8 if activation_bits == 8 else QuantType.QUInt16,
        weight_type=QuantType.QUInt8,
        calibration_providers=["CPUExecutionProvider"],
    )
    target = generator.with_name("vits_generator_htp.onnx")
    quantize(source, target, config)
    onnx.checker.check_model(target)


def collect_calibration(prefix, inputs, output_dir, frames):
    import onnxruntime as ort

    session = ort.InferenceSession(str(prefix), providers=["CPUExecutionProvider"])
    count = 0
    for path in inputs:
        for entry in json.loads(path.read_text(encoding="utf-8")):
            ge = np.asarray(entry["ge"], dtype=np.float32).reshape(entry["ge_shape"])
            z, = session.run(["z"], {
                "text_seq": np.asarray(entry["text_seq"], dtype=np.int64).reshape(1, -1),
                "pred_semantic": np.asarray(entry["pred_semantic"], dtype=np.int64).reshape(1, 1, -1),
                "ge": ge,
            })
            if z.shape[-1] >= frames:
                start = (z.shape[-1] - frames) // 2
                np.savez(output_dir / f"window-{count:04}.npz", z=z[..., start:start + frames], ge=ge)
                count += 1
    if not count:
        raise ValueError(f"calibration inputs must include a sentence with at least {frames} latent frames")
    print(f"Prepared {count} representative {frames}-frame windows")


def export_generator(source, output_dir, frames, calibration_dir, activation_bits, calibration_inputs=()):
    model = onnx.load(source)
    if {value.name for value in model.graph.input} != {
        "text_seq", "pred_semantic", "ref_audio", "sv_emb"
    }:
        raise ValueError("expected a full v2Pro or v2ProPlus VITS export")
    validate_vits_masks(model)

    reference, decoder = _reference_graphs(model)
    latent = expose_latent(decoder)
    prefix = onnx.utils.Extractor(decoder).extract_model(["text_seq", "pred_semantic", "ge"], ["z"])
    latent.type.tensor_type.shape.dim[2].dim_value = frames
    generator = onnx.utils.Extractor(decoder).extract_model(["z", "ge"], ["audio"])
    if any(node.domain not in ("", "ai.onnx") for node in generator.graph.node):
        raise ValueError("expected a generator before CPU-specific graph optimization")
    del generator.graph.value_info[:]
    generator = slim(onnx.shape_inference.infer_shapes(generator))
    for value in (*generator.graph.input, *generator.graph.output):
        tensor = value.type.tensor_type
        if not tensor.HasField("shape") or any(
            dim.dim_value <= 0 for dim in tensor.shape.dim
        ):
            raise ValueError(f"expected a static generator tensor: {value.name}")

    with tempfile.TemporaryDirectory(dir=output_dir.parent) as directory:
        staged = Path(directory) / "models"
        staged.mkdir()
        for name, graph in (
            ("vits_ref.onnx", reference),
            ("vits_prefix.onnx", prefix),
            ("vits_generator.onnx", generator),
        ):
            onnx.checker.check_model(graph)
            onnx.save(graph, staged / name)
        if calibration_inputs:
            calibration_dir = Path(directory) / "calibration"
            calibration_dir.mkdir()
            collect_calibration(staged / "vits_prefix.onnx", calibration_inputs, calibration_dir, frames)
        if calibration_dir is not None:
            quantize_generator(staged / "vits_generator.onnx", calibration_dir, activation_bits)
        staged.rename(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--frames", type=int, required=True, help="Fixed latent length at the generator input")
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--htp-calibration-dir", type=Path, help="QNN calibration .npz files with fixed z and ge inputs")
    calibration.add_argument("--htp-calibration-inputs", type=Path, action="append", default=[],
                             help="JSON inputs from gpt_sovits_demo --dump-qnn-inputs; repeat for multiple files")
    parser.add_argument("--htp-activation-bits", type=int, choices=(8, 16))
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("output directory must be new")
    if args.htp_activation_bits is not None and not (args.htp_calibration_dir or args.htp_calibration_inputs):
        parser.error("--htp-activation-bits requires calibration inputs")
    export_generator(args.model, args.output_dir, args.frames, args.htp_calibration_dir,
                     args.htp_activation_bits or 8, args.htp_calibration_inputs)
