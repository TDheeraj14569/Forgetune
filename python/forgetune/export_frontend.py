"""
Frontend integration with PyTorch using torch.export.
Extracts computation graph, normalizes ATen operators into canonical ForgeTune IR,
handles parameter extraction, transposes linear weights to row-major [K, N],
and rejects unsupported operators with descriptive errors.
"""

from typing import Tuple, Dict, Any, List
import torch
import torch.export
import numpy as np

from .graph_adapter import Graph, Node
from .errors import UnsupportedOpError, InvalidGraphError, InvalidShapeError

def export_to_forgetune_ir(model: torch.nn.Module, example_inputs: Tuple[torch.Tensor, ...]) -> Graph:
    """
    Exports a PyTorch model via torch.export.export and converts it to canonical ForgeTune IR.
    """
    model.eval()

    # Perform ahead-of-time export
    try:
        exported_program = torch.export.export(model, example_inputs)
    except Exception as e:
        raise InvalidGraphError(f"torch.export.export failed: {e}") from e

    graph_signature = exported_program.graph_signature
    state_dict = exported_program.state_dict
    graph_module = exported_program.graph_module

    target_graph = Graph()
    node_mapping: Dict[str, int] = {}  # PyTorch node name -> ForgeTune NodeId

    # 1. Process inputs and parameters from graph signature
    # Map placeholder names to their signatures
    input_specs = {}
    for inp in graph_signature.input_specs:
        input_specs[inp.arg.name] = inp

    # Also extract constants from model state
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            name = node.name
            spec = input_specs.get(name)

            if spec and spec.kind.name == "USER_INPUT":
                # User-provided dynamic tensor input
                # Determine shape from example_inputs
                ex_tensor = example_inputs[0] # assuming single 2D input for v1
                shape = list(ex_tensor.shape)
                if len(shape) != 2:
                    raise InvalidShapeError(f"ForgeTune v1 only supports rank-2 inputs, got shape {shape}")

                nid = target_graph.allocate_id()
                ir_node = Node(
                    node_id=nid,
                    op="INPUT",
                    name=name,
                    shape=shape,
                    dtype="f32"
                )
                target_graph.add_node(ir_node)
                node_mapping[name] = nid
                target_graph.inputs.append(nid)

            elif spec and spec.kind.name in ("PARAMETER", "BUFFER"):
                # Lifted parameter or buffer
                target_name = spec.target
                param_tensor = state_dict.get(target_name)
                if param_tensor is None:
                    # Try accessing directly from model attributes
                    try:
                        param_tensor = model.get_parameter(target_name)
                    except Exception:
                        pass

                if param_tensor is None:
                    raise InvalidGraphError(f"Could not resolve parameter tensor for target '{target_name}'")

                param_np = param_tensor.detach().cpu().to(torch.float32).numpy()
                nid = target_graph.allocate_id()
                ir_node = Node(
                    node_id=nid,
                    op="CONST",
                    name=name,
                    shape=list(param_np.shape),
                    dtype="f32",
                    const_data=param_np.flatten().tolist()
                )
                target_graph.add_node(ir_node)
                node_mapping[name] = nid
            else:
                # Fallback: check if tensor exists in state_dict by node.target
                if node.target in state_dict:
                    param_np = state_dict[node.target].detach().cpu().to(torch.float32).numpy()
                    nid = target_graph.allocate_id()
                    ir_node = Node(
                        node_id=nid,
                        op="CONST",
                        name=name,
                        shape=list(param_np.shape),
                        dtype="f32",
                        const_data=param_np.flatten().tolist()
                    )
                    target_graph.add_node(ir_node)
                    node_mapping[name] = nid
                else:
                    # Treat as user input
                    ex_tensor = example_inputs[0]
                    nid = target_graph.allocate_id()
                    ir_node = Node(
                        node_id=nid,
                        op="INPUT",
                        name=name,
                        shape=list(ex_tensor.shape),
                        dtype="f32"
                    )
                    target_graph.add_node(ir_node)
                    node_mapping[name] = nid
                    target_graph.inputs.append(nid)

    # 2. Process computational nodes
    for node in graph_module.graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            continue

        if node.op == "call_function":
            target_str = str(node.target)

            if "aten.linear" in target_str:
                # aten.linear(x, weight, bias=None)
                # PyTorch linear computes: x @ weight.T + bias
                # x shape: [M, K], weight shape: [N, K]
                # Canonical IR:
                #   W_transposed = weight.T [K, N]
                #   MATMUL(x, W_transposed) -> [M, N]
                #   ADD(mm, bias) -> [M, N]
                x_node = node.args[0]
                w_node = node.args[1]
                b_node = node.args[2] if len(node.args) > 2 else None

                x_id = node_mapping[x_node.name]
                w_id = node_mapping[w_node.name]

                # We need to transpose the weight data into canonical layout [K, N]
                w_ir = target_graph.nodes[w_id]
                orig_shape = w_ir.shape # [N, K]
                if len(orig_shape) != 2:
                    raise InvalidShapeError(f"Linear weight must be rank-2, got {orig_shape}")
                N, K = orig_shape[0], orig_shape[1]

                # Transpose constant data
                w_arr = np.array(w_ir.const_data, dtype=np.float32).reshape((N, K))
                w_transposed = w_arr.T.copy() # shape [K, N]

                transposed_w_id = target_graph.allocate_id()
                transposed_w_node = Node(
                    node_id=transposed_w_id,
                    op="CONST",
                    name=f"{w_node.name}_transposed",
                    shape=[K, N],
                    dtype="f32",
                    const_data=w_transposed.flatten().tolist()
                )
                target_graph.add_node(transposed_w_node)

                # Determine M from x
                x_shape = target_graph.nodes[x_id].shape
                M = x_shape[0]

                # Create MATMUL node
                mm_id = target_graph.allocate_id()
                mm_node = Node(
                    node_id=mm_id,
                    op="MATMUL",
                    name=f"{node.name}_matmul",
                    inputs=[x_id, transposed_w_id],
                    shape=[M, N],
                    dtype="f32"
                )
                target_graph.add_node(mm_node)

                # If bias exists, create ADD node
                if b_node is not None:
                    b_id = node_mapping[b_node.name]
                    add_id = target_graph.allocate_id()
                    add_node = Node(
                        node_id=add_id,
                        op="ADD",
                        name=node.name,
                        inputs=[mm_id, b_id],
                        shape=[M, N],
                        dtype="f32"
                    )
                    target_graph.add_node(add_node)
                    node_mapping[node.name] = add_id
                else:
                    node_mapping[node.name] = mm_id

            elif "aten.relu" in target_str:
                in_node = node.args[0]
                in_id = node_mapping[in_node.name]
                in_shape = target_graph.nodes[in_id].shape

                relu_id = target_graph.allocate_id()
                relu_node = Node(
                    node_id=relu_id,
                    op="RELU",
                    name=node.name,
                    inputs=[in_id],
                    shape=list(in_shape),
                    dtype="f32"
                )
                target_graph.add_node(relu_node)
                node_mapping[node.name] = relu_id

            elif "aten.mm" in target_str or "aten.matmul" in target_str:
                a_node = node.args[0]
                b_node = node.args[1]
                a_id = node_mapping[a_node.name]
                b_id = node_mapping[b_node.name]
                a_shape = target_graph.nodes[a_id].shape
                b_shape = target_graph.nodes[b_id].shape
                if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[1] != b_shape[0]:
                    raise InvalidShapeError(f"Incompatible matmul dimensions: {a_shape} @ {b_shape}")

                mm_id = target_graph.allocate_id()
                mm_node = Node(
                    node_id=mm_id,
                    op="MATMUL",
                    name=node.name,
                    inputs=[a_id, b_id],
                    shape=[a_shape[0], b_shape[1]],
                    dtype="f32"
                )
                target_graph.add_node(mm_node)
                node_mapping[node.name] = mm_id

            elif "aten.add" in target_str:
                a_node = node.args[0]
                b_node = node.args[1]
                a_id = node_mapping[a_node.name]
                b_id = node_mapping[b_node.name]
                a_shape = target_graph.nodes[a_id].shape
                b_shape = target_graph.nodes[b_id].shape

                add_id = target_graph.allocate_id()
                add_node = Node(
                    node_id=add_id,
                    op="ADD",
                    name=node.name,
                    inputs=[a_id, b_id],
                    shape=list(a_shape), # broadcasted to a_shape
                    dtype="f32"
                )
                target_graph.add_node(add_node)
                node_mapping[node.name] = add_id

            elif "aten.addmm" in target_str:
                # addmm(bias, A, B) = beta * bias + alpha * (A @ B)
                # alpha and beta are typically 1.0
                bias_node = node.args[0]
                a_node = node.args[1]
                b_node = node.args[2]
                bias_id = node_mapping[bias_node.name]
                a_id = node_mapping[a_node.name]
                b_id = node_mapping[b_node.name]
                a_shape = target_graph.nodes[a_id].shape
                b_shape = target_graph.nodes[b_id].shape

                mm_id = target_graph.allocate_id()
                mm_node = Node(
                    node_id=mm_id,
                    op="MATMUL",
                    name=f"{node.name}_matmul",
                    inputs=[a_id, b_id],
                    shape=[a_shape[0], b_shape[1]],
                    dtype="f32"
                )
                target_graph.add_node(mm_node)

                add_id = target_graph.allocate_id()
                add_node = Node(
                    node_id=add_id,
                    op="ADD",
                    name=node.name,
                    inputs=[mm_id, bias_id],
                    shape=[a_shape[0], b_shape[1]],
                    dtype="f32"
                )
                target_graph.add_node(add_node)
                node_mapping[node.name] = add_id

            else:
                raise UnsupportedOpError(target_str, f"node '{node.name}' in graph")

        elif node.op == "output":
            # Output node: tuple of outputs
            out_args = node.args[0]
            if isinstance(out_args, (list, tuple)):
                for out_arg in out_args:
                    if hasattr(out_arg, "name") and out_arg.name in node_mapping:
                        target_graph.outputs.append(node_mapping[out_arg.name])
            elif hasattr(out_args, "name") and out_args.name in node_mapping:
                target_graph.outputs.append(node_mapping[out_args.name])

    if not target_graph.outputs:
        raise InvalidGraphError("Exported graph has no valid outputs.")

    return target_graph
