"""Render only the requested v2Pro VITS tail and its convolution context."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def flow_left_context(nodes) -> int:
    """Conservatively sum left context across the flow's unit-stride convolutions."""
    supported = {
        "Add", "Concat", "Constant", "ConstantOfShape", "Conv", "Div", "Exp",
        "Mul", "Neg", "ReduceL2", "Shape", "Slice", "Split", "Sub",
    }
    left = 0
    for node in nodes:
        if node.domain not in ("", "ai.onnx") or node.op_type not in supported:
            raise ValueError(f"unsupported v2Pro flow operator: {node.name}")
        if node.op_type != "Conv":
            continue
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        pads = attrs.get("pads", [0, 0])
        if (
            attrs.get("strides", [1]) != [1]
            or attrs.get("auto_pad", b"NOTSET") != b"NOTSET"
            or len(pads) != 2
        ):
            raise ValueError(f"unsupported v2Pro flow convolution: {node.name}")
        left += pads[0]
    if not left:
        raise ValueError("expected temporal convolutions in the v2Pro flow")
    return left


def export(source: Path, target: Path, latent_history: bool = False) -> None:
    model = onnx.load(source)
    graph = model.graph
    start = "stream_start_frame"
    if any(value.name == start for value in graph.input):
        raise ValueError("VITS graph already accepts a streaming offset")
    nodes = {node.name: node for node in graph.node}
    flow = [node for node in graph.node if node.name.startswith("/vq_model/flow/")]
    flow_entry = nodes.get("/vq_model/flow/flows.7/Slice")
    generator = nodes.get("/vq_model/dec/conv_pre/Conv")
    if flow_entry is None or flow_entry.op_type != "Slice" or generator is None or generator.op_type != "Conv":
        raise ValueError("expected the v2Pro reverse flow and generator input")
    context = flow_left_context(flow)
    # Some exports multiply the flow output by the full-length sequence mask.
    latent = generator.input[0]
    flow_outputs = {name for node in flow for name in node.output}
    masked_latent = None
    if latent not in flow_outputs:
        masked_latent = next(node for node in graph.node if latent in node.output)
        if masked_latent.op_type != "Mul" or masked_latent.input[0] not in flow_outputs:
            raise ValueError("unexpected v2Pro flow-to-generator connection")

    axis = "stream_time_axis"
    end = "stream_end_frame"
    flow_start = "stream_flow_start_frame"
    generator_start = "stream_generator_start_frame"
    graph.input.append(helper.make_tensor_value_info(start, TensorProto.INT64, [1]))
    graph.initializer.extend([
        helper.make_tensor(axis, TensorProto.INT64, [1], [2]),
        helper.make_tensor(end, TensorProto.INT64, [1], [(1 << 63) - 1]),
        helper.make_tensor("stream_zero", TensorProto.INT64, [1], [0]),
    ])

    def crop(value, offset, output):
        return helper.make_node("Slice", [value, offset, end, axis], [output])

    # Keep the sampled z_p on the original timeline. The cropped flow has the
    # same right boundary as the full flow and enough history on the left.
    before_flow = [
        helper.make_node("Sub", [start, "stream_flow_left_context"], ["stream_flow_offset"]),
        helper.make_node("Max", ["stream_flow_offset", "stream_zero"], [flow_start]),
        helper.make_node("Sub", [start, flow_start], [generator_start]),
        crop(flow_entry.input[0], flow_start, "stream_flow_input"),
    ]
    if latent_history:
        if masked_latent is not None or {value.name for value in graph.input} != {
            "text_seq", "pred_semantic", "ge", start,
        }:
            raise ValueError("latent history requires the unpadded v2Pro graph with cached ge")
        weight = next(tensor for tensor in graph.initializer if tensor.name == generator.input[1])
        channels = weight.dims[1]
        if channels != 192:
            raise ValueError("expected 192 v2Pro latent channels")
        history = "stream_latent_history"
        commit = "stream_commit_frames"
        history_frames = "stream_history_frames"
        graph.input.extend([
            helper.make_tensor_value_info(history, TensorProto.FLOAT, [1, channels, "history_frames"]),
            helper.make_tensor_value_info(commit, TensorProto.INT64, [1]),
        ])
        graph.output.append(helper.make_tensor_value_info(
            "stream_next_history", TensorProto.FLOAT, [1, channels, "committed_history_frames"],
        ))
        graph.initializer.extend([
            helper.make_tensor("stream_generator_left_context", TensorProto.INT64, [1], [11]),
            helper.make_tensor("stream_history_capacity", TensorProto.INT64, [1], [context + 11]),
        ])
        before_flow = [
            crop(flow_entry.input[0], start, "stream_new_latent"),
            helper.make_node("Concat", [history, "stream_new_latent"], ["stream_flow_input"], axis=2),
            helper.make_node("Shape", [history], ["stream_history_shape"]),
            helper.make_node("Gather", ["stream_history_shape", axis], [history_frames], axis=0),
            helper.make_node("Sub", [history_frames, "stream_generator_left_context"], ["stream_generator_offset"]),
            helper.make_node("Max", ["stream_generator_offset", "stream_zero"], [generator_start]),
            helper.make_node("Add", [history_frames, commit], ["stream_history_end"]),
            helper.make_node("Sub", ["stream_history_end", "stream_history_capacity"], ["stream_history_offset"]),
            helper.make_node("Max", ["stream_history_offset", "stream_zero"], ["stream_history_start"]),
            helper.make_node("Slice", ["stream_flow_input", "stream_history_start", "stream_history_end", axis], ["stream_next_history"]),
        ]
    else:
        graph.initializer.append(helper.make_tensor(
            "stream_flow_left_context", TensorProto.INT64, [1], [context],
        ))
    flow_entry.input[0] = "stream_flow_input"
    if masked_latent is not None:
        mask = masked_latent.input[1]
        before_flow.append(crop(mask, flow_start, "stream_flow_mask"))
        for node in [*flow, masked_latent]:
            for index, value in enumerate(node.input):
                if value == mask:
                    node.input[index] = "stream_flow_mask"

    flow_index = next(i for i, node in enumerate(graph.node) if node.name == flow_entry.name)
    for index, node in enumerate(before_flow, start=flow_index):
        graph.node.insert(index, node)
    generator_index = next(i for i, node in enumerate(graph.node) if node.name == generator.name)
    generator.input[0] = "stream_latent_tail"
    graph.node.insert(generator_index, crop(latent, generator_start, "stream_latent_tail"))
    onnx.checker.check_model(model)
    onnx.save(model, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--latent-history", action="store_true")
    args = parser.parse_args()
    if args.target.exists():
        parser.error("target must be a new ONNX file")
    export(args.source, args.target, args.latent_history)
