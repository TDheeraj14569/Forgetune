#include "forgetune/graph_verifier.hpp"
#include <unordered_set>
#include <sstream>
#include <limits>

namespace forgetune {

VerificationResult verify_graph(const IRGraph& graph, bool check_throw) {
    VerificationResult res;

    // 1. Verify graph inputs
    const auto& inputs = graph.get_inputs();
    if (inputs.empty()) {
        res.add_error("Graph has no declared inputs.");
    } else if (inputs.size() > 1) {
        res.add_error("ForgeTune ABI strictly supports single-input graphs; got " + std::to_string(inputs.size()));
    }
    std::unordered_set<NodeId> declared_inputs(inputs.begin(), inputs.end());
    for (NodeId in_id : inputs) {
        if (!graph.has_node(in_id)) {
            res.add_error("Graph input %" + std::to_string(in_id) + " does not exist in graph.");
            continue;
        }
        const auto& node = graph.get_node(in_id);
        if (node.deleted) {
            res.add_error("Graph input %" + std::to_string(in_id) + " is marked deleted.");
        }
        if (node.op != OpKind::INPUT) {
            res.add_error("Graph input %" + std::to_string(in_id) + " has op kind " +
                          opkind_to_string(node.op) + ", expected INPUT.");
        }
    }

    // 2. Verify graph outputs
    const auto& outputs = graph.get_outputs();
    if (outputs.empty()) {
        res.add_error("Graph has no declared outputs.");
    } else if (outputs.size() > 1) {
        res.add_error("ForgeTune currently supports only single-output graphs.");
    }
    for (NodeId out_id : outputs) {
        if (!graph.has_node(out_id)) {
            res.add_error("Graph output %" + std::to_string(out_id) + " does not exist in graph.");
            continue;
        }
        const auto& node = graph.get_node(out_id);
        if (node.deleted) {
            res.add_error("Graph output %" + std::to_string(out_id) + " is marked deleted.");
        }
    }

    // 3. Verify acyclic dependencies
    try {
        auto topo = graph.topological_order();
    } catch (const std::exception& e) {
        res.add_error(std::string("Cycle detection failure: ") + e.what());
    }

    // 4. Node-level validations
    for (const auto& node : graph.nodes()) {
        if (node.deleted) continue;

        if (node.op == OpKind::INPUT && declared_inputs.find(node.id) == declared_inputs.end()) {
            res.add_error("Graph contains undeclared INPUT node %" + std::to_string(node.id));
        }

        // Check node ID
        if (node.id < 0) {
            res.add_error("Node has invalid negative ID: " + std::to_string(node.id));
        }

        // Check dtype
        if (node.output_meta.dtype != DType::FLOAT32) {
            res.add_error("Node %" + std::to_string(node.id) + " has unsupported dtype.");
        }

        // Check output shape validity
        const auto& out_shape = node.output_meta.shape;
        if (out_shape.rank() == 0) {
            res.add_error("Node %" + std::to_string(node.id) + " has scalar / empty shape.");
        }
        for (size_t di = 0; di < out_shape.dims.size(); ++di) {
            int64_t d = out_shape.dims[di];
            if (di == 0 && d == TensorShape::DYNAMIC_DIM && node.op != OpKind::CONST) {
                continue; // Valid leading dynamic batch dimension
            }
            if (d == TensorShape::DYNAMIC_DIM) {
                res.add_error("Node %" + std::to_string(node.id) + ": Only leading batch dimension (dim 0) may be dynamic, got dynamic dimension at index " + std::to_string(di));
            } else if (d <= 0) {
                res.add_error("Node %" + std::to_string(node.id) + " has non-positive dimension " +
                              std::to_string(d));
            }
        }

        // Check arity
        int expected_inputs = node.expected_input_count();
        if (expected_inputs >= 0 && static_cast<int>(node.inputs.size()) != expected_inputs) {
            res.add_error("Node %" + std::to_string(node.id) + " (" + opkind_to_string(node.op) +
                          ") expected " + std::to_string(expected_inputs) + " inputs but got " +
                          std::to_string(node.inputs.size()));
        } else if (expected_inputs == -1 && (node.op == OpKind::CONV2D || node.op == OpKind::FUSED_CONV2D_RELU)) {
            if (node.inputs.size() < 2 || node.inputs.size() > 3) {
                res.add_error("Node %" + std::to_string(node.id) + " (" + opkind_to_string(node.op) +
                              ") expected 2 or 3 inputs (X, W, [bias]) but got " +
                              std::to_string(node.inputs.size()));
            }
        }

        // Check operands exist and are not deleted
        for (size_t i = 0; i < node.inputs.size(); ++i) {
            NodeId op_id = node.inputs[i];
            if (!graph.has_node(op_id)) {
                res.add_error("Node %" + std::to_string(node.id) + " operand " +
                              std::to_string(i) + " refers to non-existent node %" +
                              std::to_string(op_id));
            } else {
                const auto& op_node = graph.get_node(op_id);
                if (op_node.deleted) {
                    res.add_error("Node %" + std::to_string(node.id) + " operand " +
                                  std::to_string(i) + " refers to deleted node %" +
                                  std::to_string(op_id));
                }
            }
        }

        // Operation-specific semantics and shape inference verification
        switch (node.op) {
            case OpKind::INPUT: {
                if (!node.inputs.empty()) {
                    res.add_error("INPUT node %" + std::to_string(node.id) + " must have no inputs.");
                }
                break;
            }
            case OpKind::CONST: {
                if (!node.inputs.empty()) {
                    res.add_error("CONST node %" + std::to_string(node.id) + " must have no inputs.");
                }
                if (node.output_meta.shape.is_dynamic()) {
                    res.add_error("CONST node %" + std::to_string(node.id) + " cannot have dynamic dimensions.");
                }
                int64_t expected_elements = 0;
                try {
                    expected_elements = node.output_meta.element_count();
                } catch (const std::exception& e) {
                    res.add_error("CONST node %" + std::to_string(node.id) + " element count overflow: " + e.what());
                }
                if (static_cast<int64_t>(node.const_data.size()) != expected_elements) {
                    res.add_error("CONST node %" + std::to_string(node.id) + " buffer size mismatch: data has " +
                                  std::to_string(node.const_data.size()) + " floats, shape expects " +
                                  std::to_string(expected_elements));
                }
                break;
            }
            case OpKind::MATMUL: {
                if (node.inputs.size() == 2 && graph.has_node(node.inputs[0]) && graph.has_node(node.inputs[1])) {
                    const auto& a = graph.get_node(node.inputs[0]);
                    const auto& b = graph.get_node(node.inputs[1]);
                    if (a.output_meta.shape.rank() != 2 || b.output_meta.shape.rank() != 2) {
                        res.add_error("MATMUL node %" + std::to_string(node.id) +
                                      " requires rank-2 inputs, got " +
                                      std::to_string(a.output_meta.shape.rank()) + " and " +
                                      std::to_string(b.output_meta.shape.rank()));
                    } else {
                        int64_t M = a.output_meta.shape.dims[0];
                        int64_t Ka = a.output_meta.shape.dims[1];
                        int64_t Kb = b.output_meta.shape.dims[0];
                        int64_t N = b.output_meta.shape.dims[1];
                        if (Ka != Kb) {
                            res.add_error("MATMUL node %" + std::to_string(node.id) +
                                          " dimension mismatch: A K-dim is " + std::to_string(Ka) +
                                          " but B K-dim is " + std::to_string(Kb));
                        }
                        TensorShape expected_out{M, N};
                        if (node.output_meta.shape != expected_out) {
                            res.add_error("MATMUL node %" + std::to_string(node.id) +
                                          " output shape mismatch: got " +
                                          node.output_meta.shape.to_string() + ", expected " +
                                          expected_out.to_string());
                        }
                    }
                }
                break;
            }
            case OpKind::ADD:
            case OpKind::MUL: {
                if (node.inputs.size() == 2 && graph.has_node(node.inputs[0]) && graph.has_node(node.inputs[1])) {
                    const auto& a = graph.get_node(node.inputs[0]);
                    const auto& b = graph.get_node(node.inputs[1]);
                    const auto& ashape = a.output_meta.shape;
                    const auto& bshape = b.output_meta.shape;

                    // Supported ADD/MUL broadcasting contract:
                    // 1. Equal shapes: A == B -> Out == A
                    // 2. Rank-1 bias over rank-2 last dim: A[B, N], B[N] -> Out[B, N]
                    // 3. Rank-2 broadcast row: A[B, N], B[1, N] -> Out[B, N]
                    // 4. Rank-4 exact broadcast: A[B, C, H, W], B[B, C, H, W] -> Out[B, C, H, W]
                    // 5. Rank-4 channel broadcast: A[B, C, H, W], B[C] or B[C, 1, 1] or B[1, C, 1, 1] -> Out[B, C, H, W]
                    bool valid_broadcast = false;
                    TensorShape expected_out;

                    if (ashape == bshape) {
                        valid_broadcast = true;
                        expected_out = ashape;
                    } else if (ashape.rank() == 2 && bshape.rank() == 1 && ashape.dims[1] == bshape.dims[0]) {
                        valid_broadcast = true;
                        expected_out = ashape;
                    } else if (ashape.rank() == 2 && bshape.rank() == 2 && bshape.dims[0] == 1 && ashape.dims[1] == bshape.dims[1]) {
                        valid_broadcast = true;
                        expected_out = ashape;
                    } else if (ashape.rank() == 4 && bshape.rank() == 1 && ashape.dims[1] == bshape.dims[0]) {
                        valid_broadcast = true;
                        expected_out = ashape;
                    } else if (ashape.rank() == 4 && bshape.rank() == 3 && ashape.dims[1] == bshape.dims[0] &&
                               bshape.dims[1] == 1 && bshape.dims[2] == 1) {
                        valid_broadcast = true;
                        expected_out = ashape;
                    } else if (ashape.rank() == 4 && bshape.rank() == 4 && bshape.dims[0] == 1 &&
                               ashape.dims[1] == bshape.dims[1] && bshape.dims[2] == 1 && bshape.dims[3] == 1) {
                        valid_broadcast = true;
                        expected_out = ashape;
                    }

                    if (!valid_broadcast) {
                        res.add_error(std::string(opkind_to_string(node.op)) + " node %" + std::to_string(node.id) +
                                      " unsupported broadcast shapes: A=" + ashape.to_string() +
                                      ", B=" + bshape.to_string());
                    } else if (node.output_meta.shape != expected_out) {
                        res.add_error(std::string(opkind_to_string(node.op)) + " node %" + std::to_string(node.id) +
                                      " output shape mismatch: got " +
                                      node.output_meta.shape.to_string() + ", expected " +
                                      expected_out.to_string());
                    }
                }
                break;
            }
            case OpKind::RELU:
            case OpKind::SIGMOID:
            case OpKind::TANH: {
                if (node.inputs.size() == 1 && graph.has_node(node.inputs[0])) {
                    const auto& in = graph.get_node(node.inputs[0]);
                    if (node.output_meta.shape != in.output_meta.shape) {
                        res.add_error(std::string(opkind_to_string(node.op)) + " node %" + std::to_string(node.id) +
                                      " output shape mismatch: got " +
                                      node.output_meta.shape.to_string() + ", expected " +
                                      in.output_meta.shape.to_string());
                    }
                }
                break;
            }
            case OpKind::FLATTEN: {
                if (node.inputs.size() == 1 && graph.has_node(node.inputs[0])) {
                    const auto& in = graph.get_node(node.inputs[0]);
                    const auto& in_shape = in.output_meta.shape;
                    TensorShape expected_out;
                    if (in_shape.rank() == 4) {
                        int64_t B = in_shape.dims[0];
                        int64_t feats = in_shape.dims[1] * in_shape.dims[2] * in_shape.dims[3];
                        expected_out = TensorShape{B, feats};
                    } else if (in_shape.rank() == 2) {
                        expected_out = in_shape;
                    } else {
                        res.add_error("FLATTEN node %" + std::to_string(node.id) +
                                      " requires rank-2 or rank-4 input, got rank " +
                                      std::to_string(in_shape.rank()));
                    }
                    if (node.output_meta.shape != expected_out) {
                        res.add_error("FLATTEN node %" + std::to_string(node.id) +
                                      " output shape mismatch: got " +
                                      node.output_meta.shape.to_string() + ", expected " +
                                      expected_out.to_string());
                    }
                }
                break;
            }
            case OpKind::CONV2D:
            case OpKind::FUSED_CONV2D_RELU: {
                if (node.inputs.size() >= 2 && graph.has_node(node.inputs[0]) && graph.has_node(node.inputs[1])) {
                    const auto& x = graph.get_node(node.inputs[0]);
                    const auto& w = graph.get_node(node.inputs[1]);
                    const auto& x_shape = x.output_meta.shape;
                    const auto& w_shape = w.output_meta.shape;

                    if (x_shape.rank() != 4) {
                        res.add_error("CONV2D node %" + std::to_string(node.id) + " input X must be rank-4 (NCHW).");
                    }
                    if (w_shape.rank() != 4) {
                        res.add_error("CONV2D node %" + std::to_string(node.id) + " weight W must be rank-4 (OIHW).");
                    }
                    if (w_shape.is_dynamic()) {
                        res.add_error("CONV2D node %" + std::to_string(node.id) + " weight W cannot have dynamic dimensions.");
                    }

                    if (x_shape.rank() == 4 && w_shape.rank() == 4) {
                        int64_t B = x_shape.dims[0];
                        int64_t C_in = x_shape.dims[1];
                        int64_t H_in = x_shape.dims[2];
                        int64_t W_in = x_shape.dims[3];

                        int64_t C_out = w_shape.dims[0];
                        int64_t W_Cin = w_shape.dims[1];
                        int64_t K_h = w_shape.dims[2];
                        int64_t K_w = w_shape.dims[3];

                        if (C_in != W_Cin) {
                            res.add_error("CONV2D node %" + std::to_string(node.id) +
                                          " channel mismatch: X has " + std::to_string(C_in) +
                                          " channels, W expects " + std::to_string(W_Cin));
                        }

                        // Validate bias if present
                        if (node.inputs.size() == 3 && graph.has_node(node.inputs[2])) {
                            const auto& b = graph.get_node(node.inputs[2]);
                            const auto& b_shape = b.output_meta.shape;
                            if (b_shape.rank() != 1 || b_shape.dims[0] != C_out) {
                                res.add_error("CONV2D node %" + std::to_string(node.id) +
                                              " bias shape mismatch: expected [" + std::to_string(C_out) +
                                              "], got " + b_shape.to_string());
                            }
                        }

                        const auto& attrs = node.conv_attrs;
                        bool attrs_valid = true;
                        if (K_h <= 0 || K_w <= 0) {
                            res.add_error("CONV2D node %" + std::to_string(node.id) + " kernel dimensions must be positive, got " +
                                          std::to_string(K_h) + "x" + std::to_string(K_w));
                            attrs_valid = false;
                        }
                        if (H_in <= 0 || W_in <= 0 || C_in <= 0 || C_out <= 0) {
                            res.add_error("CONV2D node %" + std::to_string(node.id) + " input/weight spatial and channel dimensions must be positive.");
                            attrs_valid = false;
                        }
                        if (attrs.stride_h <= 0 || attrs.stride_w <= 0) {
                            res.add_error("CONV2D node %" + std::to_string(node.id) + " stride must be positive.");
                            attrs_valid = false;
                        }
                        if (attrs.pad_h < 0 || attrs.pad_w < 0) {
                            res.add_error("CONV2D node %" + std::to_string(node.id) + " padding cannot be negative.");
                            attrs_valid = false;
                        }
                        if (attrs.dilation_h <= 0 || attrs.dilation_w <= 0) {
                            res.add_error("CONV2D node %" + std::to_string(node.id) + " dilation must be positive.");
                            attrs_valid = false;
                        }
                        if (attrs.groups != 1) {
                            res.add_error("CONV2D node %" + std::to_string(node.id) + " only groups=1 supported.");
                            attrs_valid = false;
                        }

                        if (attrs_valid) {
                            auto safe_mul = [](int64_t a, int64_t b, int64_t& out) -> bool {
                                if (a <= 0 || b <= 0) {
                                    if (a < 0 || b < 0) return false;
                                    out = 0;
                                    return true;
                                }
                                if (a > std::numeric_limits<int64_t>::max() / b) return false;
                                out = a * b;
                                return true;
                            };

                            auto safe_add = [](int64_t a, int64_t b, int64_t& out) -> bool {
                                if (a < 0 || b < 0) return false;
                                if (a > std::numeric_limits<int64_t>::max() - b) return false;
                                out = a + b;
                                return true;
                            };

                            int64_t eff_kh_minus_1 = 0, eff_kh = 0;
                            int64_t eff_kw_minus_1 = 0, eff_kw = 0;
                            int64_t two_pad_h = 0, padded_h = 0;
                            int64_t two_pad_w = 0, padded_w = 0;

                            bool math_ok = true;
                            math_ok = math_ok && safe_mul(attrs.dilation_h, K_h - 1, eff_kh_minus_1);
                            math_ok = math_ok && safe_add(eff_kh_minus_1, 1, eff_kh);
                            math_ok = math_ok && safe_mul(attrs.dilation_w, K_w - 1, eff_kw_minus_1);
                            math_ok = math_ok && safe_add(eff_kw_minus_1, 1, eff_kw);
                            math_ok = math_ok && safe_mul(2, attrs.pad_h, two_pad_h);
                            math_ok = math_ok && safe_add(H_in, two_pad_h, padded_h);
                            math_ok = math_ok && safe_mul(2, attrs.pad_w, two_pad_w);
                            math_ok = math_ok && safe_add(W_in, two_pad_w, padded_w);

                            if (!math_ok) {
                                res.add_error("CONV2D node %" + std::to_string(node.id) + " spatial dimension arithmetic overflowed.");
                            } else {
                                int64_t num_h = padded_h - eff_kh;
                                int64_t num_w = padded_w - eff_kw;

                                if (num_h < 0 || num_w < 0) {
                                    res.add_error("CONV2D node %" + std::to_string(node.id) +
                                                  " effective kernel size exceeds padded input dimension: eff_k=(" +
                                                  std::to_string(eff_kh) + ", " + std::to_string(eff_kw) +
                                                  ") > padded_in=(" + std::to_string(padded_h) + ", " +
                                                  std::to_string(padded_w) + ")");
                                } else {
                                    int64_t div_h = num_h / attrs.stride_h;
                                    int64_t div_w = num_w / attrs.stride_w;
                                    int64_t H_out = 0, W_out = 0;

                                    if (!safe_add(div_h, 1, H_out) || !safe_add(div_w, 1, W_out) || H_out <= 0 || W_out <= 0) {
                                        res.add_error("CONV2D node %" + std::to_string(node.id) +
                                                      " non-positive spatial output dimensions or overflow.");
                                    } else {
                                        TensorShape expected_out{B, C_out, H_out, W_out};
                                        if (node.output_meta.shape != expected_out) {
                                            res.add_error("CONV2D node %" + std::to_string(node.id) +
                                                          " output shape mismatch: got " +
                                                          node.output_meta.shape.to_string() + ", expected " +
                                                          expected_out.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
            case OpKind::OUTPUT: {
                if (node.inputs.size() == 1 && graph.has_node(node.inputs[0])) {
                    const auto& in = graph.get_node(node.inputs[0]);
                    if (node.output_meta.shape != in.output_meta.shape) {
                        res.add_error("OUTPUT node %" + std::to_string(node.id) +
                                      " output shape mismatch.");
                    }
                }
                break;
            }
            case OpKind::FUSED_MATMUL_ADD_RELU: {
                if (node.inputs.size() == 3 &&
                    graph.has_node(node.inputs[0]) &&
                    graph.has_node(node.inputs[1]) &&
                    graph.has_node(node.inputs[2])) {
                    const auto& x = graph.get_node(node.inputs[0]);
                    const auto& w = graph.get_node(node.inputs[1]);
                    const auto& b = graph.get_node(node.inputs[2]);
                    if (x.output_meta.shape.rank() != 2 || w.output_meta.shape.rank() != 2) {
                        res.add_error("FUSED_MATMUL_ADD_RELU node %" + std::to_string(node.id) +
                                      " requires rank-2 input and weight.");
                    } else {
                        int64_t M = x.output_meta.shape.dims[0];
                        int64_t K = x.output_meta.shape.dims[1];
                        int64_t Kw = w.output_meta.shape.dims[0];
                        int64_t N = w.output_meta.shape.dims[1];
                        if (K != Kw) {
                            res.add_error("FUSED_MATMUL_ADD_RELU node %" + std::to_string(node.id) +
                                          " K-dimension mismatch: X is " + std::to_string(K) +
                                          ", W is " + std::to_string(Kw));
                        }
                        // Validate bias compatibility
                        bool bias_ok = false;
                        if (b.output_meta.shape.rank() == 1 && b.output_meta.shape.dims[0] == N) {
                            bias_ok = true;
                        } else if (b.output_meta.shape.rank() == 2 &&
                                   ((b.output_meta.shape.dims[0] == 1 && b.output_meta.shape.dims[1] == N) ||
                                    (b.output_meta.shape.dims[0] == M && b.output_meta.shape.dims[1] == N))) {
                            bias_ok = true;
                        }
                        if (!bias_ok) {
                            res.add_error("FUSED_MATMUL_ADD_RELU node %" + std::to_string(node.id) +
                                          " incompatible bias shape: " + b.output_meta.shape.to_string());
                        }
                        TensorShape expected_out{M, N};
                        if (node.output_meta.shape != expected_out) {
                            res.add_error("FUSED_MATMUL_ADD_RELU node %" + std::to_string(node.id) +
                                          " output shape mismatch: got " +
                                          node.output_meta.shape.to_string() + ", expected " +
                                          expected_out.to_string());
                        }
                    }
                }
                break;
            }
        }
    }

    if (!res.passed && check_throw) {
        throw InvalidGraphError(res.to_string());
    }

    return res;
}

} // namespace forgetune
