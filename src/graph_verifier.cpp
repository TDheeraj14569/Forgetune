#include "forgetune/graph_verifier.hpp"
#include <unordered_set>
#include <sstream>

namespace forgetune {

VerificationResult verify_graph(const IRGraph& graph, bool check_throw) {
    VerificationResult res;

    // 1. Verify graph inputs
    const auto& inputs = graph.get_inputs();
    if (inputs.empty()) {
        res.add_error("Graph has no declared inputs.");
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
        for (int64_t d : out_shape.dims) {
            if (d <= 0) {
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
            case OpKind::ADD: {
                if (node.inputs.size() == 2 && graph.has_node(node.inputs[0]) && graph.has_node(node.inputs[1])) {
                    const auto& a = graph.get_node(node.inputs[0]);
                    const auto& b = graph.get_node(node.inputs[1]);
                    const auto& ashape = a.output_meta.shape;
                    const auto& bshape = b.output_meta.shape;

                    // Supported ADD broadcasting contract:
                    // 1. Equal shapes: A == B -> Out == A
                    // 2. Rank-1 bias over rank-2 last dim: A[M, N], B[N] -> Out[M, N]
                    // 3. Rank-2 broadcast row: A[M, N], B[1, N] -> Out[M, N]
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
                    }

                    if (!valid_broadcast) {
                        res.add_error("ADD node %" + std::to_string(node.id) +
                                      " unsupported broadcast shapes: A=" + ashape.to_string() +
                                      ", B=" + bshape.to_string());
                    } else if (node.output_meta.shape != expected_out) {
                        res.add_error("ADD node %" + std::to_string(node.id) +
                                      " output shape mismatch: got " +
                                      node.output_meta.shape.to_string() + ", expected " +
                                      expected_out.to_string());
                    }
                }
                break;
            }
            case OpKind::RELU: {
                if (node.inputs.size() == 1 && graph.has_node(node.inputs[0])) {
                    const auto& in = graph.get_node(node.inputs[0]);
                    if (node.output_meta.shape != in.output_meta.shape) {
                        res.add_error("RELU node %" + std::to_string(node.id) +
                                      " output shape mismatch: got " +
                                      node.output_meta.shape.to_string() + ", expected " +
                                      in.output_meta.shape.to_string());
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
