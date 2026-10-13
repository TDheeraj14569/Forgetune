#ifndef FORGETUNE_INTERPRETER_HPP
#define FORGETUNE_INTERPRETER_HPP

#include "forgetune/ir.hpp"
#include <vector>
#include <unordered_map>

namespace forgetune {

// Reference interpreter: evaluates an IRGraph using simple, auditable C++ arithmetic.
// This serves as the ground truth for correctness validation.
class Interpreter {
public:
    // Evaluate the graph with given input tensors.
    // inputs: map from graph input node IDs to flat float32 data.
    // Returns: map from graph output node IDs to flat float32 data.
    std::unordered_map<NodeId, std::vector<float>> evaluate(
        const IRGraph& graph,
        const std::unordered_map<NodeId, std::vector<float>>& inputs) const;

private:
    // Individual operation implementations
    std::vector<float> eval_matmul(const std::vector<float>& a, const TensorShape& a_shape,
                                   const std::vector<float>& b, const TensorShape& b_shape,
                                   const TensorShape& out_shape) const;

    std::vector<float> eval_add(const std::vector<float>& a, const TensorShape& a_shape,
                                const std::vector<float>& b, const TensorShape& b_shape,
                                const TensorShape& out_shape) const;

    std::vector<float> eval_relu(const std::vector<float>& input) const;
};

} // namespace forgetune

#endif // FORGETUNE_INTERPRETER_HPP
