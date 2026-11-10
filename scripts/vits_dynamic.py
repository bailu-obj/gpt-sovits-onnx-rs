"""Repair and verify full-length VITS masks in legacy ONNX exports.

Only full-length sequence masks (Unsqueeze(Range(0, dynamic_length, 1)))
are eligible. Attention masks and input-dependent valid-length masks stay intact.
"""
import argparse
from pathlib import Path
import tempfile
import onnx
from onnx import numpy_helper


def _masks(model):
    producers = {value: node for node in model.graph.node for value in node.output}
    constants = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attribute.t)

    def depends_on_shape(value, visited=None):
        visited = set() if visited is None else visited
        if value in visited:
            return False
        visited.add(value)
        node = producers.get(value)
        return node is not None and (node.op_type == "Shape" or any(
            depends_on_shape(x, visited) for x in node.input))

    masks = []
    for node in model.graph.node:
        scope = node.name.rpartition("/")[0]
        if node.op_type != "Less" or not (scope.endswith("/enc_p") or node.name in ("/vq_model/Less", "/Less")):
            continue
        unsqueeze = producers.get(node.input[0])
        sequence = producers.get(unsqueeze.input[0]) if unsqueeze is not None and unsqueeze.op_type == "Unsqueeze" else None
        if sequence is None or sequence.op_type != "Range":
            raise ValueError(f"unsupported VITS mask topology: {node.name}")
        start, limit, step = sequence.input
        if start not in constants or step not in constants or constants[start].size != 1 or constants[step].size != 1 or constants[start].item() != 0 or constants[step].item() != 1 or not depends_on_shape(limit):
            raise ValueError(f"VITS mask range is not dynamic: {node.name}")
        masks.append((node, limit, constants.get(node.input[1])))
    inputs = {x.name for x in model.graph.input}
    if any(prop.key == "vits.unpadded" and prop.value == "1" for prop in model.metadata_props):
        if masks or not {"pred_semantic", "text_seq", "ref_audio"} <= inputs:
            raise ValueError("unexpected masks or inputs in an unpadded VITS export")
        return masks
    expected = (2 if "pred_semantic" in inputs else 0) + (1 if "ref_audio" in inputs else 0)
    if not expected or len(masks) != expected:
        raise ValueError(f"expected {expected} dynamic VITS masks, found {len(masks)}")
    return masks


def repair_vits_masks(model):
    repaired = []
    for node, dynamic_limit, constant in _masks(model):
        if constant is not None:
            if constant.size != 1 or constant.item() <= 0:
                raise ValueError(f"invalid frozen VITS length: {node.name}")
            node.input[1] = dynamic_limit
            repaired.append(node.name)
        elif node.input[1] != dynamic_limit:
            # Fresh exports may reshape the Shape-derived scalar to [1,1].
            # Validate its ancestry independently below.
            pass
    validate_vits_masks(model)
    return repaired


def validate_vits_masks(model):
    masks = _masks(model)
    producers = {value: node for node in model.graph.node for value in node.output}
    def shape_dependent(value, visited=None):
        visited = set() if visited is None else visited
        if value in visited:
            return False
        visited.add(value)
        node = producers.get(value)
        return node is not None and (node.op_type == "Shape" or any(shape_dependent(x, visited) for x in node.input))
    for node, _, constant in masks:
        if constant is not None or not shape_dependent(node.input[1]):
            raise ValueError(f"frozen VITS mask length: {node.name}")
    return [node.name for node, _, _ in masks]


def validate_split_reference(reference, decoder):
    for model, inputs, outputs in (
        (reference, {"ref_audio", "sv_emb"}, {"ge"}),
        (decoder, {"text_seq", "pred_semantic", "ge"}, {"audio"}),
    ):
        if {value.name for value in model.graph.input} != inputs or {
            value.name for value in model.graph.output
        } != outputs:
            raise ValueError("unexpected split VITS graph interface")
        onnx.checker.check_model(model)
    if reference.graph.output[0].type != next(
        value.type for value in decoder.graph.input if value.name == "ge"
    ):
        raise ValueError("split VITS graphs disagree on the ge tensor type")


def _reference_graphs(model):
    boundary = [
        node for node in model.graph.node
        if node.op_type == "PRelu" and node.name == "/vq_model/prelu/PRelu"
    ]
    if len(boundary) != 1:
        raise ValueError("expected one exported Pro reference PReLU")
    ge_name = boundary[0].output[0]
    slope_name = boundary[0].input[1]
    slope = next((weight for weight in model.graph.initializer if weight.name == slope_name), None)
    if slope is None or len(slope.dims) != 2 or slope.dims[1] != 1:
        raise ValueError("expected one PReLU slope per reference feature channel")
    # Shape inference can leave the rank unknown after a dynamic STFT.
    # The Pro graph pools a single utterance to one feature frame.
    ge = onnx.helper.make_tensor_value_info("ge", onnx.TensorProto.FLOAT, [1, slope.dims[0], 1])
    for value in model.graph.value_info:
        if value.name == ge_name:
            value.CopyFrom(ge)
            break
    else:
        model.graph.value_info.append(ge)
    for node in model.graph.node:
        for names in (node.input, node.output):
            for index, name in enumerate(names):
                if name == ge_name:
                    names[index] = "ge"

    extractor = onnx.utils.Extractor(model)
    return (
        extractor.extract_model(["ref_audio", "sv_emb"], ["ge"]),
        extractor.extract_model(["text_seq", "pred_semantic", "ge"], ["audio"]),
    )


def split_reference(source, directory):
    import onnxruntime as ort

    original = onnx.load(source)
    if {value.name for value in original.graph.input} != {
        "text_seq", "pred_semantic", "ref_audio", "sv_emb"
    }:
        raise ValueError("expected a complete VITS v2Pro or v2ProPlus graph")
    validate_vits_masks(original)

    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        scratch = Path(temporary)
        optimized = scratch / "vits-optimized.onnx"
        options = ort.SessionOptions()
        # Preserve the original CPU fusions before introducing the split.
        # Optimizing the two halves separately can change INT8 rounding.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        options.optimized_model_filepath = str(optimized)
        ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
        reference, decoder = _reference_graphs(onnx.load(optimized))
        validate_split_reference(reference, decoder)
        staged = scratch / "models"
        staged.mkdir()
        onnx.save(reference, staged / "vits_ref.onnx")
        onnx.save(decoder, staged / "vits.onnx")
        staged.rename(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", type=Path, help="Repair into a separate output file")
    mode.add_argument("--split-reference-dir", type=Path, help="Write a split Pro vocoder to a new directory")
    args = parser.parse_args()
    if args.split_reference_dir:
        if args.split_reference_dir.exists() or args.split_reference_dir.is_symlink():
            parser.error("split output directory must be new")
        split_reference(args.model, args.split_reference_dir)
        print("Wrote split VITS graphs:", args.split_reference_dir)
        parser.exit()
    model = onnx.load(args.model)
    if args.output:
        if args.output.resolve() == args.model.resolve():
            parser.error("output must differ from source")
        print("Repaired:", repair_vits_masks(model))
        onnx.checker.check_model(model)
        onnx.save(model, args.output)
    elif any(value.name == "ge" for value in model.graph.input):
        validate_split_reference(onnx.load(args.model.with_name("vits_ref.onnx")), model)
        print("Verified split VITS reference and decoder")
    else:
        print("Verified dynamic masks:", validate_vits_masks(model))
