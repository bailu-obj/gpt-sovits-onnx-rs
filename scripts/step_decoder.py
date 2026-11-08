"""Weight-only CPU export for the autoregressive T2S step decoder."""

import onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils


def quantize_step_decoder(model, *, bits=4, block_size=32, accuracy_level=4):
    graph = model.graph
    weights = {value.name: value for value in graph.initializer}
    nodes = []
    projections = set()
    replaced_weights = set()
    for index, node in enumerate(graph.node):
        if node.op_type not in ("MatMul", "Gemm") or node.input[1] not in weights:
            nodes.append(node)
            continue
        weight = weights[node.input[1]]
        # Large aligned projections dominate per-token reads. Keep the 1025-way
        # vocabulary head and dynamic attention in floating point.
        if len(weight.dims) != 2 or any(dim < 512 or dim % 32 for dim in weight.dims):
            nodes.append(node)
            continue
        if node.op_type == "MatMul":
            projections.add(node.name)
            nodes.append(node)
            continue

        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if (len(node.input) != 3 or attributes.get("transA", 0) != 0
                or attributes.get("transB", 0) != 1
                or attributes.get("alpha", 1.0) != 1.0
                or attributes.get("beta", 1.0) != 1.0):
            nodes.append(node)
            continue
        source, _, bias = node.input
        prefix = f"rtf_gemm_{index}"
        transposed = numpy_helper.to_array(weight).T.copy()
        graph.initializer.append(numpy_helper.from_array(transposed, f"{prefix}_weight"))
        nodes.append(helper.make_node(
            "MatMul", [source, f"{prefix}_weight"], [f"{prefix}_product"],
            name=f"{prefix}_matmul",
        ))
        nodes.append(helper.make_node("Add", [f"{prefix}_product", bias], node.output, name=node.name))
        projections.add(f"{prefix}_matmul")
        replaced_weights.add(weight.name)

    if not projections or "" in projections:
        raise ValueError("expected named, constant-weight T2S step decoder projections")
    del graph.node[:]
    graph.node.extend(nodes)
    used_weights = {name for node in nodes for name in node.input}
    used_weights.update(output.name for output in graph.output)
    for index in reversed(range(len(graph.initializer))):
        if graph.initializer[index].name in replaced_weights - used_weights:
            del graph.initializer[index]

    config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=block_size, is_symmetric=True, accuracy_level=accuracy_level,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",), bits=bits,
    )
    excluded = [node.name for node in graph.node if node.name not in projections]
    quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model, nodes_to_exclude=excluded, algo_config=config,
    )
    quantizer.process()
    optimized = quantizer.model.model
    onnx.checker.check_model(optimized)
    return optimized
