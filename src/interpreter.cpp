#include "forgetune/interpreter.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace forgetune {

std::vector<float> Interpreter::eval_matmul(
    const std::vector<float>& a, const TensorShape& a_shape,
    const std::vector<float>& b, const TensorShape& b_shape,
    const TensorShape& out_shape) const {

    int64_t M = a_shape.dims[0];
    int64_t K = a_shape.dims[1];
    int64_t N = b_shape.dims[1];

    std::vector<float> out(static_cast<size_t>(M * N), 0.0f);

    // Auditable naive row-major accumulation
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                float a_val = a[static_cast<size_t>(i * K + k)];
                float b_val = b[static_cast<size_t>(k * N + j)];
                sum += a_val * b_val;
            }
            out[static_cast<size_t>(i * N + j)] = sum;
        }
    }
    return out;
}

std::vector<float> Interpreter::eval_add(
    const std::vector<float>& a, const TensorShape& a_shape,
    const std::vector<float>& b, const TensorShape& b_shape,
    const TensorShape& out_shape) const {

    size_t out_elements = static_cast<size_t>(out_shape.element_count());
    std::vector<float> out(out_elements, 0.0f);

    if (a_shape == b_shape) {
        // Elementwise exact
        for (size_t i = 0; i < out_elements; ++i) {
            out[i] = a[i] + b[i];
        }
    } else if (a_shape.rank() == 2 && b_shape.rank() == 1 && a_shape.dims[1] == b_shape.dims[0]) {
        // Bias vector [N] broadcast over [M, N]
        int64_t M = a_shape.dims[0];
        int64_t N = a_shape.dims[1];
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                out[static_cast<size_t>(i * N + j)] =
                    a[static_cast<size_t>(i * N + j)] + b[static_cast<size_t>(j)];
            }
        }
    } else if (a_shape.rank() == 2 && b_shape.rank() == 2 && b_shape.dims[0] == 1 && a_shape.dims[1] == b_shape.dims[1]) {
        // Bias row [1, N] broadcast over [M, N]
        int64_t M = a_shape.dims[0];
        int64_t N = a_shape.dims[1];
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                out[static_cast<size_t>(i * N + j)] =
                    a[static_cast<size_t>(i * N + j)] + b[static_cast<size_t>(j)];
            }
        }
    } else {
        throw InvalidShapeError("Interpreter::eval_add: unsupported broadcasting");
    }

    return out;
}

std::vector<float> Interpreter::eval_relu(const std::vector<float>& input) const {
    std::vector<float> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        // Explicit IEEE-754 semantics: negative zero preserved or 0.0f, positive preserved
        float val = input[i];
        out[i] = (val > 0.0f) ? val : 0.0f;
    }
    return out;
}

std::unordered_map<NodeId, std::vector<float>> Interpreter::evaluate(
    const IRGraph& graph,
    const std::unordered_map<NodeId, std::vector<float>>& user_inputs) const {

    // 1. Verify graph input arguments
    for (NodeId in_id : graph.get_inputs()) {
        auto it = user_inputs.find(in_id);
        if (it == user_inputs.end()) {
            throw std::runtime_error("Interpreter: missing input for node %" + std::to_string(in_id));
        }
        const auto& node = graph.get_node(in_id);
        int64_t expected = node.output_meta.element_count();
        if (static_cast<int64_t>(it->second.size()) != expected) {
            throw InvalidShapeError("Interpreter: input buffer size mismatch for node %" +
                                    std::to_string(in_id) + ": expected " +
                                    std::to_string(expected) + ", got " +
                                    std::to_string(it->second.size()));
        }
    }

    // 2. Storage for intermediate and final tensor buffers
    std::unordered_map<NodeId, std::vector<float>> node_values;

    // Load inputs
    for (const auto& kv : user_inputs) {
        node_values[kv.first] = kv.second;
    }

    // 3. Execute in topological order
    auto order = graph.topological_order();
    for (NodeId id : order) {
        const auto& node = graph.get_node(id);
        if (node.deleted) continue;

        switch (node.op) {
            case OpKind::INPUT: {
                // Already populated
                if (node_values.find(id) == node_values.end()) {
                    throw std::runtime_error("Interpreter: uninitialized INPUT node %" + std::to_string(id));
                }
                break;
            }
            case OpKind::CONST: {
                node_values[id] = node.const_data;
                break;
            }
            case OpKind::MATMUL: {
                const auto& a = node_values.at(node.inputs[0]);
                const auto& b = node_values.at(node.inputs[1]);
                const auto& a_shape = graph.get_node(node.inputs[0]).output_meta.shape;
                const auto& b_shape = graph.get_node(node.inputs[1]).output_meta.shape;
                node_values[id] = eval_matmul(a, a_shape, b, b_shape, node.output_meta.shape);
                break;
            }
            case OpKind::ADD: {
                const auto& a = node_values.at(node.inputs[0]);
                const auto& b = node_values.at(node.inputs[1]);
                const auto& a_shape = graph.get_node(node.inputs[0]).output_meta.shape;
                const auto& b_shape = graph.get_node(node.inputs[1]).output_meta.shape;
                node_values[id] = eval_add(a, a_shape, b, b_shape, node.output_meta.shape);
                break;
            }
            case OpKind::RELU: {
                const auto& in = node_values.at(node.inputs[0]);
                node_values[id] = eval_relu(in);
                break;
            }
            case OpKind::OUTPUT: {
                node_values[id] = node_values.at(node.inputs[0]);
                break;
            }
            case OpKind::FUSED_MATMUL_ADD_RELU: {
                const auto& x = node_values.at(node.inputs[0]);
                const auto& w = node_values.at(node.inputs[1]);
                const auto& b = node_values.at(node.inputs[2]);
                const auto& x_shape = graph.get_node(node.inputs[0]).output_meta.shape;
                const auto& w_shape = graph.get_node(node.inputs[1]).output_meta.shape;
                const auto& b_shape = graph.get_node(node.inputs[2]).output_meta.shape;

                auto mm = eval_matmul(x, x_shape, w, w_shape, node.output_meta.shape);
                auto added = eval_add(mm, node.output_meta.shape, b, b_shape, node.output_meta.shape);
                node_values[id] = eval_relu(added);
                break;
            }
        }
    }

    // 4. Collect requested outputs
    std::unordered_map<NodeId, std::vector<float>> outputs;
    for (NodeId out_id : graph.get_outputs()) {
        auto it = node_values.find(out_id);
        if (it == node_values.end()) {
            throw std::runtime_error("Interpreter: failed to produce output node %" + std::to_string(out_id));
        }
        outputs[out_id] = it->second;
    }

    return outputs;
}

} // namespace forgetune
