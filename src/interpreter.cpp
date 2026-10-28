#include "forgetune/interpreter.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace forgetune {

std::vector<float> Interpreter::eval_matmul(
    const std::vector<float>& a, const TensorShape& a_shape,
    const std::vector<float>& b, const TensorShape& b_shape,
    const TensorShape& out_shape) const {
    (void)out_shape;

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
    } else if (a_shape.rank() == 4 && b_shape.rank() == 4 && b_shape.dims[0] == 1 &&
               b_shape.dims[2] == 1 && b_shape.dims[3] == 1 && a_shape.dims[1] == b_shape.dims[1]) {
        // Channel broadcast [B, C, H, W] + [1, C, 1, 1]
        int64_t B = a_shape.dims[0];
        int64_t C = a_shape.dims[1];
        int64_t H = a_shape.dims[2];
        int64_t W = a_shape.dims[3];
        for (int64_t bi = 0; bi < B; ++bi) {
            for (int64_t c = 0; c < C; ++c) {
                float b_val = b[static_cast<size_t>(c)];
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        size_t idx = static_cast<size_t>(((bi * C + c) * H + h) * W + w);
                        out[idx] = a[idx] + b_val;
                    }
                }
            }
        }
    } else if (a_shape.rank() == 4 && b_shape.rank() == 1 && a_shape.dims[1] == b_shape.dims[0]) {
        // Channel broadcast [B, C, H, W] + [C]
        int64_t B = a_shape.dims[0];
        int64_t C = a_shape.dims[1];
        int64_t H = a_shape.dims[2];
        int64_t W = a_shape.dims[3];
        for (int64_t bi = 0; bi < B; ++bi) {
            for (int64_t c = 0; c < C; ++c) {
                float b_val = b[static_cast<size_t>(c)];
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        size_t idx = static_cast<size_t>(((bi * C + c) * H + h) * W + w);
                        out[idx] = a[idx] + b_val;
                    }
                }
            }
        }
    } else if (a_shape.rank() == 4 && b_shape.rank() == 3 && a_shape.dims[1] == b_shape.dims[0] &&
               b_shape.dims[1] == 1 && b_shape.dims[2] == 1) {
        // Channel broadcast [B, C, H, W] + [C, 1, 1]
        int64_t B = a_shape.dims[0];
        int64_t C = a_shape.dims[1];
        int64_t H = a_shape.dims[2];
        int64_t W = a_shape.dims[3];
        for (int64_t bi = 0; bi < B; ++bi) {
            for (int64_t c = 0; c < C; ++c) {
                float b_val = b[static_cast<size_t>(c)];
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        size_t idx = static_cast<size_t>(((bi * C + c) * H + h) * W + w);
                        out[idx] = a[idx] + b_val;
                    }
                }
            }
        }
    } else {
        throw InvalidShapeError("Interpreter::eval_add: unsupported broadcasting");
    }

    return out;
}

std::vector<float> Interpreter::eval_mul(
    const std::vector<float>& a, const TensorShape& a_shape,
    const std::vector<float>& b, const TensorShape& b_shape,
    const TensorShape& out_shape) const {

    size_t out_elements = static_cast<size_t>(out_shape.element_count());
    std::vector<float> out(out_elements, 0.0f);

    if (a_shape == b_shape) {
        // Elementwise exact
        for (size_t i = 0; i < out_elements; ++i) {
            out[i] = a[i] * b[i];
        }
    } else if (a_shape.rank() == 2 && b_shape.rank() == 1 && a_shape.dims[1] == b_shape.dims[0]) {
        // Vector [N] broadcast over [M, N]
        int64_t M = a_shape.dims[0];
        int64_t N = a_shape.dims[1];
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                out[static_cast<size_t>(i * N + j)] =
                    a[static_cast<size_t>(i * N + j)] * b[static_cast<size_t>(j)];
            }
        }
    } else if (a_shape.rank() == 2 && b_shape.rank() == 2 && b_shape.dims[0] == 1 && a_shape.dims[1] == b_shape.dims[1]) {
        // Row [1, N] broadcast over [M, N]
        int64_t M = a_shape.dims[0];
        int64_t N = a_shape.dims[1];
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                out[static_cast<size_t>(i * N + j)] =
                    a[static_cast<size_t>(i * N + j)] * b[static_cast<size_t>(j)];
            }
        }
    } else if (a_shape.rank() == 4 && b_shape.rank() == 4 && b_shape.dims[0] == 1 &&
               b_shape.dims[2] == 1 && b_shape.dims[3] == 1 && a_shape.dims[1] == b_shape.dims[1]) {
        // Channel broadcast [B, C, H, W] * [1, C, 1, 1]
        int64_t B = a_shape.dims[0];
        int64_t C = a_shape.dims[1];
        int64_t H = a_shape.dims[2];
        int64_t W = a_shape.dims[3];
        for (int64_t bi = 0; bi < B; ++bi) {
            for (int64_t c = 0; c < C; ++c) {
                float b_val = b[static_cast<size_t>(c)];
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        size_t idx = static_cast<size_t>(((bi * C + c) * H + h) * W + w);
                        out[idx] = a[idx] * b_val;
                    }
                }
            }
        }
    } else if (a_shape.rank() == 4 && b_shape.rank() == 1 && a_shape.dims[1] == b_shape.dims[0]) {
        // Channel broadcast [B, C, H, W] * [C]
        int64_t B = a_shape.dims[0];
        int64_t C = a_shape.dims[1];
        int64_t H = a_shape.dims[2];
        int64_t W = a_shape.dims[3];
        for (int64_t bi = 0; bi < B; ++bi) {
            for (int64_t c = 0; c < C; ++c) {
                float b_val = b[static_cast<size_t>(c)];
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        size_t idx = static_cast<size_t>(((bi * C + c) * H + h) * W + w);
                        out[idx] = a[idx] * b_val;
                    }
                }
            }
        }
    } else if (a_shape.rank() == 4 && b_shape.rank() == 3 && a_shape.dims[1] == b_shape.dims[0] &&
               b_shape.dims[1] == 1 && b_shape.dims[2] == 1) {
        // Channel broadcast [B, C, H, W] * [C, 1, 1]
        int64_t B = a_shape.dims[0];
        int64_t C = a_shape.dims[1];
        int64_t H = a_shape.dims[2];
        int64_t W = a_shape.dims[3];
        for (int64_t bi = 0; bi < B; ++bi) {
            for (int64_t c = 0; c < C; ++c) {
                float b_val = b[static_cast<size_t>(c)];
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        size_t idx = static_cast<size_t>(((bi * C + c) * H + h) * W + w);
                        out[idx] = a[idx] * b_val;
                    }
                }
            }
        }
    } else {
        throw InvalidShapeError("Interpreter::eval_mul: unsupported broadcasting");
    }

    return out;
}

std::vector<float> Interpreter::eval_relu(const std::vector<float>& input) const {
    std::vector<float> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        float val = input[i];
        out[i] = (std::isnan(val) || val > 0.0f) ? val : 0.0f;
    }
    return out;
}

std::vector<float> Interpreter::eval_sigmoid(const std::vector<float>& input) const {
    std::vector<float> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
    return out;
}

std::vector<float> Interpreter::eval_tanh(const std::vector<float>& input) const {
    std::vector<float> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = std::tanh(input[i]);
    }
    return out;
}

std::vector<float> Interpreter::eval_flatten(const std::vector<float>& input) const {
    return input;
}

std::vector<float> Interpreter::eval_conv2d(
    const std::vector<float>& x, const TensorShape& x_shape,
    const std::vector<float>& w, const TensorShape& w_shape,
    const std::vector<float>* bias, const TensorShape* /*b_shape*/,
    const Conv2DAttrs& attrs,
    const TensorShape& out_shape) const {

    int64_t B = x_shape.dims[0];
    int64_t C_in = x_shape.dims[1];
    int64_t H_in = x_shape.dims[2];
    int64_t W_in = x_shape.dims[3];

    int64_t C_out = w_shape.dims[0];
    int64_t K_h = w_shape.dims[2];
    int64_t K_w = w_shape.dims[3];

    int64_t H_out = out_shape.dims[2];
    int64_t W_out = out_shape.dims[3];

    std::vector<float> out(static_cast<size_t>(B * C_out * H_out * W_out), 0.0f);

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t co = 0; co < C_out; ++co) {
            float b_val = (bias != nullptr) ? (*bias)[static_cast<size_t>(co)] : 0.0f;
            for (int64_t ho = 0; ho < H_out; ++ho) {
                for (int64_t wo = 0; wo < W_out; ++wo) {
                    float sum = b_val;
                    for (int64_t ci = 0; ci < C_in; ++ci) {
                        for (int64_t kh = 0; kh < K_h; ++kh) {
                            int64_t hi = ho * attrs.stride_h - attrs.pad_h + kh * attrs.dilation_h;
                            if (hi < 0 || hi >= H_in) continue;
                            for (int64_t kw = 0; kw < K_w; ++kw) {
                                int64_t wi = wo * attrs.stride_w - attrs.pad_w + kw * attrs.dilation_w;
                                if (wi < 0 || wi >= W_in) continue;
                                size_t x_idx = static_cast<size_t>(((b * C_in + ci) * H_in + hi) * W_in + wi);
                                size_t w_idx = static_cast<size_t>(((co * C_in + ci) * K_h + kh) * K_w + kw);
                                sum += x[x_idx] * w[w_idx];
                            }
                        }
                    }
                    size_t out_idx = static_cast<size_t>(((b * C_out + co) * H_out + ho) * W_out + wo);
                    out[out_idx] = sum;
                }
            }
        }
    }
    return out;
}

std::unordered_map<NodeId, std::vector<float>> Interpreter::evaluate(
    const IRGraph& graph,
    const std::unordered_map<NodeId, std::vector<float>>& user_inputs) const {

    int64_t runtime_batch = -1;

    // 1. Verify graph input arguments and deduce runtime batch
    for (NodeId in_id : graph.get_inputs()) {
        auto it = user_inputs.find(in_id);
        if (it == user_inputs.end()) {
            throw std::runtime_error("Interpreter: missing input for node %" + std::to_string(in_id));
        }
        const auto& node = graph.get_node(in_id);
        const auto& shape = node.output_meta.shape;
        if (shape.has_dynamic_batch()) {
            int64_t static_elems = shape.static_element_count();
            if (static_elems <= 0 || (it->second.size() % static_elems != 0)) {
                throw InvalidShapeError("Interpreter: input buffer size not a multiple of static element count for node %" +
                                        std::to_string(in_id));
            }
            int64_t b = static_cast<int64_t>(it->second.size()) / static_elems;
            int64_t max_b = graph.max_batch_size() > 0 ? graph.max_batch_size() : TensorShape::DEFAULT_MAX_BATCH;
            if (b <= 0 || b > max_b) {
                throw InvalidShapeError("Interpreter: input batch size " + std::to_string(b) +
                                        " out of valid bounds [1, " +
                                        std::to_string(max_b) + "]");
            }
            if (runtime_batch == -1) {
                runtime_batch = b;
            } else if (runtime_batch != b) {
                throw InvalidShapeError("Interpreter: batch size mismatch across inputs: " +
                                        std::to_string(runtime_batch) + " vs " + std::to_string(b));
            }
        } else {
            int64_t expected = shape.element_count();
            if (static_cast<int64_t>(it->second.size()) != expected) {
                throw InvalidShapeError("Interpreter: input buffer size mismatch for node %" +
                                        std::to_string(in_id) + ": expected " +
                                        std::to_string(expected) + ", got " +
                                        std::to_string(it->second.size()));
            }
            if (shape.rank() > 0 && runtime_batch == -1) {
                runtime_batch = shape.dims[0];
            }
        }
    }
    if (runtime_batch == -1) runtime_batch = 1;

    auto get_concrete_shape = [&](const IRNode& n) -> TensorShape {
        TensorShape cs = n.output_meta.shape;
        if (!cs.dims.empty() && cs.dims[0] == TensorShape::DYNAMIC_DIM) {
            cs.dims[0] = runtime_batch;
        }
        return cs;
    };

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

        TensorShape out_shape = get_concrete_shape(node);

        switch (node.op) {
            case OpKind::INPUT: {
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
                TensorShape a_shape = get_concrete_shape(graph.get_node(node.inputs[0]));
                TensorShape b_shape = get_concrete_shape(graph.get_node(node.inputs[1]));
                node_values[id] = eval_matmul(a, a_shape, b, b_shape, out_shape);
                break;
            }
            case OpKind::ADD: {
                const auto& a = node_values.at(node.inputs[0]);
                const auto& b = node_values.at(node.inputs[1]);
                TensorShape a_shape = get_concrete_shape(graph.get_node(node.inputs[0]));
                TensorShape b_shape = get_concrete_shape(graph.get_node(node.inputs[1]));
                node_values[id] = eval_add(a, a_shape, b, b_shape, out_shape);
                break;
            }
            case OpKind::MUL: {
                const auto& a = node_values.at(node.inputs[0]);
                const auto& b = node_values.at(node.inputs[1]);
                TensorShape a_shape = get_concrete_shape(graph.get_node(node.inputs[0]));
                TensorShape b_shape = get_concrete_shape(graph.get_node(node.inputs[1]));
                node_values[id] = eval_mul(a, a_shape, b, b_shape, out_shape);
                break;
            }
            case OpKind::RELU: {
                const auto& in = node_values.at(node.inputs[0]);
                node_values[id] = eval_relu(in);
                break;
            }
            case OpKind::SIGMOID: {
                const auto& in = node_values.at(node.inputs[0]);
                node_values[id] = eval_sigmoid(in);
                break;
            }
            case OpKind::TANH: {
                const auto& in = node_values.at(node.inputs[0]);
                node_values[id] = eval_tanh(in);
                break;
            }
            case OpKind::FLATTEN: {
                const auto& in = node_values.at(node.inputs[0]);
                node_values[id] = eval_flatten(in);
                break;
            }
            case OpKind::CONV2D: {
                const auto& x = node_values.at(node.inputs[0]);
                const auto& w = node_values.at(node.inputs[1]);
                const std::vector<float>* bias_ptr = nullptr;
                const TensorShape* b_shape_ptr = nullptr;
                TensorShape b_shape;
                if (node.inputs.size() == 3) {
                    bias_ptr = &node_values.at(node.inputs[2]);
                    b_shape = get_concrete_shape(graph.get_node(node.inputs[2]));
                    b_shape_ptr = &b_shape;
                }
                TensorShape x_shape = get_concrete_shape(graph.get_node(node.inputs[0]));
                TensorShape w_shape = get_concrete_shape(graph.get_node(node.inputs[1]));
                node_values[id] = eval_conv2d(x, x_shape, w, w_shape, bias_ptr, b_shape_ptr, node.conv_attrs, out_shape);
                break;
            }
            case OpKind::FUSED_CONV2D_RELU: {
                const auto& x = node_values.at(node.inputs[0]);
                const auto& w = node_values.at(node.inputs[1]);
                const std::vector<float>* bias_ptr = nullptr;
                const TensorShape* b_shape_ptr = nullptr;
                TensorShape b_shape;
                if (node.inputs.size() == 3) {
                    bias_ptr = &node_values.at(node.inputs[2]);
                    b_shape = get_concrete_shape(graph.get_node(node.inputs[2]));
                    b_shape_ptr = &b_shape;
                }
                TensorShape x_shape = get_concrete_shape(graph.get_node(node.inputs[0]));
                TensorShape w_shape = get_concrete_shape(graph.get_node(node.inputs[1]));
                auto conv_out = eval_conv2d(x, x_shape, w, w_shape, bias_ptr, b_shape_ptr, node.conv_attrs, out_shape);
                node_values[id] = eval_relu(conv_out);
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
                TensorShape x_shape = get_concrete_shape(graph.get_node(node.inputs[0]));
                TensorShape w_shape = get_concrete_shape(graph.get_node(node.inputs[1]));
                TensorShape b_shape = get_concrete_shape(graph.get_node(node.inputs[2]));

                auto mm = eval_matmul(x, x_shape, w, w_shape, out_shape);
                auto added = eval_add(mm, out_shape, b, b_shape, out_shape);
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
