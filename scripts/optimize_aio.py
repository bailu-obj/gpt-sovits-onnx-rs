import os
import glob
import argparse
from pathlib import Path
import logging
import numpy as np
import onnx
from vits_dynamic import repair_vits_masks, validate_vits_masks
from step_decoder import quantize_step_decoder
from onnx import version_converter
from onnxsim import simplify
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static
from onnxoptimizer import optimize, get_fuse_and_elimination_passes
from onnxconverter_common import float16
from onnxslim import slim
from onnxruntime.transformers import optimizer

from onnxruntime.quantization import (
    matmul_nbits_quantizer,
    quant_utils,
    quantize
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize ONNX models.")
    parser.add_argument(
        "--input-dir",
        default="onnx/v2pro",
        help="Input directory containing ONNX files",
    )
    parser.add_argument(
        "--output-dir",
        default="onnx-patched/v2pro",
        help="Output directory for optimized models",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Disable INT8 quantization for applicable models",
    )
    parser.add_argument(
        "--profile", choices=("compact", "quality", "rtf", "mobile"), default="compact",
        help="compact: existing INT4/INT8 policy; quality: INT8 text weights; "
             "rtf: quality text weights plus INT4 T2S step projections; "
             "mobile: dynamic INT8 text/prefill plus INT4 T2S step projections",
    )
    parser.add_argument(
        "--vits-calibration-dir", type=Path,
        help="With --profile mobile, calibrate residual convolutions using named VITS input .npz files",
    )
    args = parser.parse_args()
    if args.vits_calibration_dir and (args.profile != "mobile" or args.no_quant):
        parser.error("--vits-calibration-dir requires quantized --profile mobile")
    return args


def validate_environment():
    """Validate required dependencies."""
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
            exit(1)


def quantize_text_quality(model, output_path):
    """Preserve float activations, embeddings and pronunciation heads."""
    excluded = [n.name for n in model.graph.node
                if any("classifier" in value for value in n.input)]
    config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=32, is_symmetric=True, accuracy_level=1,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",), bits=8,
    )
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model, nodes_to_exclude=excluded, algo_config=config,
    )
    quant.process()
    quant.model.save_model_to_file(output_path)
    logger.info(f"Quality INT8 weights / FP32 activations: {output_path}")
    return output_path


def share_g2pw_context(model):
    """Expose the official G2PW encoder's existing batch-one broadcasting."""
    graph = model.graph
    text_inputs = {"input_ids", "token_type_ids", "attention_mask"}
    query_inputs = {"phoneme_mask", "char_ids", "position_ids"}
    if {value.name for value in graph.input} != text_inputs | query_inputs:
        return
    nodes = {node.name: node for node in graph.node}
    # The official graph offsets each query by Range(text_batch) * text_length.
    # With one text row this broadcasts offset zero to every queried position.
    if not all(name in nodes and nodes[name].op_type == op for name, op in (
        ("Range_1163", "Range"), ("Add_1173", "Add"), ("Gather_1174", "Gather"),
    )):
        return
    for value in (*graph.input, *graph.output):
        dimension = value.type.tensor_type.shape.dim[0]
        dimension.dim_param = "text_batch" if value.name in text_inputs else "query_batch"
    del graph.value_info[:]
    onnx.helper.set_model_props(model, {
        **{prop.key: prop.value for prop in model.metadata_props},
        "g2pw.shared_context": "1",
    })


def quantize_mobile_matmuls(model, output_path):
    weights = {value.name: value for value in model.graph.initializer}
    projections = []
    for node in model.graph.node:
        if node.op_type not in ("MatMul", "Gemm") or len(node.input) < 2:
            continue
        weight = weights.get(node.input[1])
        if weight is not None and len(weight.dims) == 2 and min(weight.dims) >= 128:
            if not any("classifier" in value for value in node.input):
                projections.append(node.name)
    if not projections:
        raise ValueError(f"expected constant-weight projections in {output_path}")
    quantize_dynamic(
        model, output_path, op_types_to_quantize=["MatMul", "Gemm"],
        nodes_to_quantize=projections, weight_type=QuantType.QInt8, per_channel=True,
        extra_options={"MatMulConstBOnly": True},
    )


class VitsCalibration(CalibrationDataReader):
    def __init__(self, directory, inputs):
        self.paths = sorted(directory.glob("*.npz"))
        if not self.paths:
            raise ValueError(f"expected VITS input .npz files in {directory}")
        self.inputs = inputs
        self.rewind()

    def get_next(self):
        path = next(self.pending, None)
        if path is None:
            return None
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != self.inputs:
                raise ValueError(f"VITS calibration inputs differ from the graph: {path}")
            return {name: data[name] for name in self.inputs}

    def rewind(self):
        self.pending = iter(self.paths)


def quantize_mobile_vits(model, output_path, calibration_dir):
    # Keep the highest-rate vocoder stages in float. Quantizing them reduced
    # speaker similarity more than calibrating the first nine residual blocks.
    early_blocks = tuple(f"/dec/resblocks.{index}/" for index in range(9))
    residual_convs = [
        node.name for node in model.graph.node
        if node.op_type == "Conv" and any(block in node.name for block in early_blocks)
    ]
    if not residual_convs:
        raise ValueError("expected exported SoVITS generator residual convolutions")
    calibration = VitsCalibration(calibration_dir, {value.name for value in model.graph.input})
    quantize_static(
        model, output_path, calibration, quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv"], nodes_to_quantize=residual_convs,
        per_channel=True, calibration_providers=["CPUExecutionProvider"],
    )


def process_model(
    file_path: str, output_path: str, use_quant: bool, profile: str = "compact",
    vits_calibration_dir: Path | None = None,
) -> str:
    """Process and optimize an ONNX model."""
    logger.info(f"Processing model: {file_path}")
    model = onnx.load(file_path)
    output_lower = os.path.basename(output_path).lower()

    # vits model may change and have issue in simplify
    if "vits" in output_lower:
        logger.info("Repaired VITS masks: %s", repair_vits_masks(model))
        # model = optimize(model, passes=get_fuse_and_elimination_passes())
        # logger.info(f"ONNX optimization done for: {output_path}")
        model = slim(model)
        logger.info(f"ONNX simplification done for: {output_path}")
        model = version_converter.convert_version(model, 21)
        validate_vits_masks(model)
        onnx.checker.check_model(model)
        if use_quant and vits_calibration_dir is not None:
            quantize_mobile_vits(model, output_path, vits_calibration_dir)
        else:
            onnx.save(model, output_path)
        return output_path

    # Simplify non-BERT models
    if "bert" in output_lower:
        # BERT model optimization
        optimized_model = optimizer.optimize_model(
            model,
            model_type="bert",
            num_heads=16,
            hidden_size=1024,
            only_onnxruntime=True,
        )
        model = version_converter.convert_version(optimized_model.model, 21)
        # model = slim(model)
        if use_quant and profile == "mobile":
            quantize_mobile_matmuls(model, output_path)
            return output_path
        if use_quant and profile in ("quality", "rtf"):
            return quantize_text_quality(model, output_path)
        if use_quant:
            quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
                block_size=64,  # 2's exponential and >= 16
                # if true, quantize to Int4. otherwise, quantize to uint4.
                is_symmetric=True,
                accuracy_level=2,  # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35
                quant_format=quant_utils.QuantFormat.QOperator,
                # specify which op types to quantize
                op_types_to_quantize=("MatMul", "Gather", "Attention"),
                bits=4
            )
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                model,
                nodes_to_exclude=None,  # specify a list of nodes to exclude from quantization
                nodes_to_include=None,  # specify a list of nodes to force include from quantization
                algo_config=quant_config,)
            quant.process()
            quant.model.save_model_to_file(output_path)
            logger.info(f"INT4 quantization done for: {output_path}")
        else:
            onnx.save(model, output_path)
        return output_path

    # Serialize the shared step's original attention attributes. ORT's offline
    # optimizer materializes newer contrib defaults that older CPU runtimes reject.
    shared_step = output_lower == "t2s_s_decoder.onnx" and any(
        node.domain == "com.microsoft" and node.op_type == "GroupQueryAttention"
        for node in model.graph.node
    )
    if "decoder" in output_lower and not shared_step:
        optimized_model = optimizer.optimize_model(
            model,
            only_onnxruntime=True,
        )
        model = optimized_model.model

    # model = optimize(model, passes=get_fuse_and_elimination_passes())
    logger.info(f"ONNX optimization done for: {output_path}")

    model = slim(model)
    logger.info(f"ONNX slim optimization done for: {output_path}")

    model = version_converter.convert_version(model, 21)
    logger.info(f"Opset conversion done for: {output_path}")
    if output_lower == "g2pw.onnx":
        share_g2pw_context(model)
    if use_quant and profile == "mobile" and output_lower in ("g2pw.onnx", "t2s_fs_decoder.onnx"):
        quantize_mobile_matmuls(model, output_path)
        return output_path
    if use_quant and "g2p" in output_lower:
        if profile in ("quality", "rtf"):
            return quantize_text_quality(model, output_path)
        quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=64,  # 2's exponential and >= 16
            # if true, quantize to Int4. otherwise, quantize to uint4.
            is_symmetric=True,
            accuracy_level=2,  # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35
            quant_format=quant_utils.QuantFormat.QOperator,
            # specify which op types to quantize
            op_types_to_quantize=("MatMul", "Gather", "Attention"),
            bits=4
        )
        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model,
            nodes_to_exclude=None,  # specify a list of nodes to exclude from quantization
            nodes_to_include=None,  # specify a list of nodes to force include from quantization
            algo_config=quant_config,)
        quant.process()
        quant.model.save_model_to_file(output_path)
        logger.info(f"I4 quantization done for: {output_path}")
        return output_path

    if use_quant and profile in ("rtf", "mobile") and output_lower == "t2s_s_decoder.onnx":
        onnx.save(quantize_step_decoder(model), output_path)
        logger.info("INT4 T2S step projections: %s", output_path)
    elif use_quant and profile == "compact" and ("decoder" in output_lower):
        quantize_dynamic(
            model,
            output_path,
            op_types_to_quantize=["MatMul", "Attention", "Gather"],
            per_channel=True,
            reduce_range=True
        )
        logger.info(f"I8 quantization done for: {output_path}")
    else:
        onnx.save(model, output_path)

    return output_path


def main():
    args = parse_args()
    validate_environment()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Find ONNX files
    onnx_files = glob.glob(os.path.join(args.input_dir, "*.onnx"))
    if not onnx_files:
        logger.error(f"No ONNX files found in {args.input_dir}")
        exit(1)

    # Validate required models
    has_decoder = any("decoder" in f.lower() for f in onnx_files)
    has_vits = any("vits" in f.lower() for f in onnx_files)
    if not has_decoder or not has_vits:
        logger.error(
            f"{'No decoder model found' if not has_decoder else ''}{' and ' if not has_decoder and not has_vits else ''}{'No vits model found' if not has_vits else ''} in the folder"
        )
        exit(1)

    # Process each model
    for file_path in onnx_files:
        output_path = os.path.join(
            args.output_dir, os.path.basename(file_path))
        final_path = process_model(file_path, output_path, not args.no_quant, args.profile, args.vits_calibration_dir)
        logger.info(f"Optimization complete for: {final_path}")

    logger.info("All models processed successfully!")


if __name__ == "__main__":
    main()
