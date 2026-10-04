"""Repair and verify full-length VITS masks in legacy ONNX exports.

Only full-length sequence masks (Unsqueeze(Range(0, dynamic_length, 1)))
are eligible. Attention masks and input-dependent valid-length masks stay intact.
"""
import argparse
from pathlib import Path
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
        if node.op_type != "Less" or not ("/enc_p/" in node.name or node.name in ("/vq_model/Less", "/Less")):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, help="Repair into a separate output file; otherwise validate only")
    args = parser.parse_args()
    model = onnx.load(args.model)
    if args.output:
        if args.output.resolve() == args.model.resolve():
            parser.error("output must differ from source")
        print("Repaired:", repair_vits_masks(model))
        onnx.checker.check_model(model)
        onnx.save(model, args.output)
    else:
        print("Verified dynamic masks:", validate_vits_masks(model))
