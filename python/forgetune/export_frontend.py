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

def export_to_forgetune_ir(
    model: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    dynamic_batch: bool = False,
    max_batch_size: int = 128
) -> Graph:
    """
    Exports a PyTorch model via torch.export.export and converts it to canonical ForgeTune IR.
    Supports dynamic batch dimension when dynamic_batch=True.
    """
    model.eval()

    if len(example_inputs) != 1:
        raise ValueError(f"ForgeTune v2.0 currently supports single-input models. Got {len(example_inputs)} example inputs.")

    # Configure dynamic shapes if requested
    dynamic_shapes = None
    if dynamic_batch:
        batch_dim = torch.export.Dim("batch", min=1, max=max_batch_size)
        dyn_list = []
        for inp in example_inputs:
            if isinstance(inp, torch.Tensor) and inp.ndim >= 2:
                dyn_list.append({0: batch_dim})
            else:
                dyn_list.append({})
        dynamic_shapes = tuple(dyn_list)

    # Perform ahead-of-time export
    try:
        exported_program = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    except Exception as e:
        raise InvalidGraphError(f"torch.export.export failed: {e}") from e

    graph_signature = exported_program.graph_signature
    state_dict = exported_program.state_dict
    graph_module = exported_program.graph_module

    target_graph = Graph()
    target_graph.max_batch_size = max_batch_size
    node_mapping: Dict[str, int] = {}  # PyTorch node name -> ForgeTune NodeId

    # 1. Process inputs and parameters from graph signature
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
                ex_tensor = example_inputs[0]
                shape = list(ex_tensor.shape)
                if len(shape) not in (2, 4):
                    raise InvalidShapeError(f"ForgeTune v2 only supports rank-2 and rank-4 inputs, got shape {shape}")
                if dynamic_batch:
                    shape[0] = -1

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
                    shape = list(ex_tensor.shape)
                    if len(shape) not in (2, 4):
                        raise InvalidShapeError(f"ForgeTune v2 only supports rank-2 and rank-4 inputs, got shape {shape}")
                    if dynamic_batch:
                        shape[0] = -1
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
                w_ir.deleted = True

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

            elif "aten.addmm" in target_str:
                # addmm(bias, A, B, beta=1.0, alpha=1.0) = beta * bias + alpha * (A @ B)
                bias_node = node.args[0]
                a_node = node.args[1]
                b_node = node.args[2]

                alpha = 1.0
                beta = 1.0
                if hasattr(node, "kwargs") and node.kwargs:
                    alpha = float(node.kwargs.get("alpha", 1.0))
                    beta = float(node.kwargs.get("beta", 1.0))
                if len(node.args) > 3 and isinstance(node.args[3], (int, float)):
                    beta = float(node.args[3])
                if len(node.args) > 4 and isinstance(node.args[4], (int, float)):
                    alpha = float(node.args[4])

                if alpha != 1.0 or beta != 1.0:
                    raise UnsupportedOpError("aten.addmm", f"addmm with non-default alpha={alpha}, beta={beta} is not supported in v2.0")

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

            elif "aten.add" in target_str:
                a_node = node.args[0]
                b_node = node.args[1]

                alpha = 1.0
                if hasattr(node, "kwargs") and node.kwargs:
                    alpha = float(node.kwargs.get("alpha", 1.0))
                if len(node.args) > 2 and isinstance(node.args[2], (int, float)):
                    alpha = float(node.args[2])

                if alpha != 1.0:
                    raise UnsupportedOpError("aten.add", f"aten.add with non-default alpha={alpha} is not supported in v2.0")

                a_id = node_mapping[a_node.name]
                b_id = node_mapping[b_node.name]
                a_shape = target_graph.nodes[a_id].shape
                b_shape = target_graph.nodes[b_id].shape

                # Commutative broadcasting: ensure higher-rank/dynamic input is primary
                inputs = [a_id, b_id]
                out_shape = list(a_shape)
                if len(a_shape) < len(b_shape) or (len(a_shape) == len(b_shape) and a_shape[0] == 1 and b_shape[0] != 1):
                    inputs = [b_id, a_id]
                    out_shape = list(b_shape)

                add_id = target_graph.allocate_id()
                add_node = Node(
                    node_id=add_id,
                    op="ADD",
                    name=node.name,
                    inputs=inputs,
                    shape=out_shape,
                    dtype="f32"
                )
                target_graph.add_node(add_node)
                node_mapping[node.name] = add_id

            elif "aten.mul" in target_str:
                a_node = node.args[0]
                b_node = node.args[1]
                a_id = node_mapping[a_node.name]
                b_id = node_mapping[b_node.name]
                a_shape = target_graph.nodes[a_id].shape
                b_shape = target_graph.nodes[b_id].shape

                # Commutative broadcasting
                inputs = [a_id, b_id]
                out_shape = list(a_shape)
                if len(a_shape) < len(b_shape) or (len(a_shape) == len(b_shape) and a_shape[0] == 1 and b_shape[0] != 1):
                    inputs = [b_id, a_id]
                    out_shape = list(b_shape)

                mul_id = target_graph.allocate_id()
                mul_node = Node(
                    node_id=mul_id,
                    op="MUL",
                    name=node.name,
                    inputs=inputs,
                    shape=out_shape,
                    dtype="f32"
                )
                target_graph.add_node(mul_node)
                node_mapping[node.name] = mul_id

            elif "aten.sigmoid" in target_str:
                in_node = node.args[0]
                in_id = node_mapping[in_node.name]
                in_shape = target_graph.nodes[in_id].shape

                sig_id = target_graph.allocate_id()
                sig_node = Node(
                    node_id=sig_id,
                    op="SIGMOID",
                    name=node.name,
                    inputs=[in_id],
                    shape=list(in_shape),
                    dtype="f32"
                )
                target_graph.add_node(sig_node)
                node_mapping[node.name] = sig_id

            elif "aten.tanh" in target_str:
                in_node = node.args[0]
                in_id = node_mapping[in_node.name]
                in_shape = target_graph.nodes[in_id].shape

                tanh_id = target_graph.allocate_id()
                tanh_node = Node(
                    node_id=tanh_id,
                    op="TANH",
                    name=node.name,
                    inputs=[in_id],
                    shape=list(in_shape),
                    dtype="f32"
                )
                target_graph.add_node(tanh_node)
                node_mapping[node.name] = tanh_id

            elif "aten.flatten" in target_str:
                in_node = node.args[0]
                in_id = node_mapping[in_node.name]
                in_shape = target_graph.nodes[in_id].shape
                if len(in_shape) == 4:
                    B = in_shape[0]
                    feats = in_shape[1] * in_shape[2] * in_shape[3]
                    out_shape = [B, feats]
                elif len(in_shape) == 2:
                    out_shape = list(in_shape)
                else:
                    raise InvalidShapeError(f"FLATTEN requires rank-2 or rank-4 input, got {in_shape}")

                flat_id = target_graph.allocate_id()
                flat_node = Node(
                    node_id=flat_id,
                    op="FLATTEN",
                    name=node.name,
                    inputs=[in_id],
                    shape=out_shape,
                    dtype="f32"
                )
                target_graph.add_node(flat_node)
                node_mapping[node.name] = flat_id

            elif "aten.view" in target_str or "aten.reshape" in target_str:
                in_node = node.args[0]
                in_id = node_mapping[in_node.name]
                in_shape = target_graph.nodes[in_id].shape
                target_shape = node.args[1] if len(node.args) > 1 else None

                if len(in_shape) == 4:
                    B = in_shape[0]
                    feats = in_shape[1] * in_shape[2] * in_shape[3]
                    if target_shape is not None:
                        t_list = list(target_shape)
                        if len(t_list) != 2:
                            raise UnsupportedOpError(target_str, f"arbitrary reshape of rank-4 tensor to non-rank-2 target shape {target_shape} is not supported in v2.0")
                    out_shape = [B, feats]
                    flat_id = target_graph.allocate_id()
                    flat_node = Node(
                        node_id=flat_id,
                        op="FLATTEN",
                        name=node.name,
                        inputs=[in_id],
                        shape=out_shape,
                        dtype="f32"
                    )
                    target_graph.add_node(flat_node)
                    node_mapping[node.name] = flat_id
                elif len(in_shape) == 2:
                    if target_shape is not None:
                        t_list = list(target_shape)
                        if len(t_list) != 2:
                            raise UnsupportedOpError(target_str, f"arbitrary reshape of rank-2 tensor to target shape {target_shape} is not supported in v2.0")
                    node_mapping[node.name] = in_id
                else:
                    raise UnsupportedOpError(target_str, f"arbitrary view/reshape for shape {in_shape} not supported")

            elif "conv2d" in target_str or "convolution" in target_str:
                x_node = node.args[0]
                w_node = node.args[1]
                b_node = node.args[2] if len(node.args) > 2 else None

                x_id = node_mapping[x_node.name]
                w_id = node_mapping[w_node.name]
                b_id = node_mapping[b_node.name] if (b_node is not None and hasattr(b_node, 'name') and b_node.name in node_mapping) else None

                if "convolution" in target_str:
                    stride = node.args[3] if len(node.args) > 3 else [1, 1]
                    padding = node.args[4] if len(node.args) > 4 else [0, 0]
                    dilation = node.args[5] if len(node.args) > 5 else [1, 1]
                    transposed = node.args[6] if len(node.args) > 6 else False
                    if transposed:
                        raise UnsupportedOpError("Transposed convolution is not supported")
                    groups = node.args[8] if len(node.args) > 8 else 1
                    if groups != 1:
                        raise UnsupportedOpError(f"Only groups=1 supported, got {groups}")
                else:  # conv2d
                    stride = node.args[3] if len(node.args) > 3 else [1, 1]
                    padding = node.args[4] if len(node.args) > 4 else [0, 0]
                    dilation = node.args[5] if len(node.args) > 5 else [1, 1]
                    groups = node.args[6] if len(node.args) > 6 else 1
                    if groups != 1:
                        raise UnsupportedOpError(f"Only groups=1 supported, got {groups}")

                sh = stride[0] if isinstance(stride, (list, tuple)) else int(stride)
                sw = stride[1] if isinstance(stride, (list, tuple)) and len(stride) > 1 else sh
                ph = padding[0] if isinstance(padding, (list, tuple)) else int(padding)
                pw = padding[1] if isinstance(padding, (list, tuple)) and len(padding) > 1 else ph
                dh = dilation[0] if isinstance(dilation, (list, tuple)) else int(dilation)
                dw = dilation[1] if isinstance(dilation, (list, tuple)) and len(dilation) > 1 else dh

                x_shape = target_graph.nodes[x_id].shape
                w_shape = target_graph.nodes[w_id].shape

                if len(x_shape) != 4 or len(w_shape) != 4:
                    raise InvalidShapeError(f"CONV2D requires rank-4 input and weight, got X:{x_shape}, W:{w_shape}")

                B, C_in, H_in, W_in = x_shape
                C_out, W_Cin, Kh, Kw = w_shape
                if C_in != W_Cin:
                    raise InvalidShapeError(f"CONV2D channel mismatch: X has {C_in}, W expects {W_Cin}")

                eff_kh = dh * (Kh - 1) + 1
                eff_kw = dw * (Kw - 1) + 1
                H_out = (H_in + 2 * ph - eff_kh) // sh + 1
                W_out = (W_in + 2 * pw - eff_kw) // sw + 1
                out_shape = [B, C_out, H_out, W_out]

                conv_inputs = [x_id, w_id]
                if b_id is not None:
                    conv_inputs.append(b_id)

                conv_attrs = {
                    "stride_h": sh,
                    "stride_w": sw,
                    "pad_h": ph,
                    "pad_w": pw,
                    "dilation_h": dh,
                    "dilation_w": dw,
                    "groups": 1
                }

                conv_id = target_graph.allocate_id()
                conv_node = Node(
                    node_id=conv_id,
                    op="CONV2D",
                    name=node.name,
                    inputs=conv_inputs,
                    shape=out_shape,
                    dtype="f32",
                    conv_attrs=conv_attrs
                )
                target_graph.add_node(conv_node)
                node_mapping[node.name] = conv_id

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
    if len(target_graph.outputs) > 1:
        raise ValueError(f"ForgeTune v2.0 currently supports single-output models. Got {len(target_graph.outputs)} graph outputs.")

    return target_graph
