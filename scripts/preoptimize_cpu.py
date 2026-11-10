"""Pre-fuse selected ONNX graphs in a prepared GPT-SoVITS CPU export."""

import argparse
from pathlib import Path
import tempfile

import onnx
import onnxruntime as ort


GRAPHS = (
    "bert.onnx",
    "g2pW.onnx",
    "g2p_en/encoder_model.onnx",
    "g2p_en/decoder_model.onnx",
    "t2s_fs_decoder.onnx",
    "t2s_s_decoder.onnx",
)


def preoptimize_cpu(bundle: Path) -> None:
    bundle = bundle.resolve(strict=True)
    for name in GRAPHS:
        source = bundle / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.parent.resolve() != source.parent:
            raise ValueError(f"Graph must be in the output directory: {source}")

    with tempfile.TemporaryDirectory(prefix=".cpu-preopt-", dir=bundle) as directory:
        staged = Path(directory)
        completed = []
        for name in GRAPHS:
            source = bundle / name
            if name == "t2s_s_decoder.onnx":
                model = onnx.load(source, load_external_data=False)
                if any(
                    node.domain == "com.microsoft" and node.op_type == "GroupQueryAttention"
                    for node in model.graph.node
                ):
                    # Preserve the shared-KV export's original contrib attributes.
                    print(f"Kept shared-KV graph: {name}", flush=True)
                    continue

            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            options.optimized_model_filepath = str(target)
            ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
            # ORT may serialize fusions outside the original ONNX opset.
            options.optimized_model_filepath = ""
            ort.InferenceSession(str(target), options, providers=["CPUExecutionProvider"])
            completed.append((target, source))

        for target, source in completed:
            target.replace(source)
            print(f"Pre-fused CPU graph: {source.relative_to(bundle)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Prepared output bundle to update in place")
    preoptimize_cpu(parser.parse_args().bundle)
