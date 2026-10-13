#ifndef FORGETUNE_FUSION_HPP
#define FORGETUNE_FUSION_HPP

#include "forgetune/ir.hpp"
#include "forgetune/pass_manager.hpp"

namespace forgetune {

// Operator fusion pass: recognizes MATMUL -> ADD -> RELU
// Replaces the sequence with a single FUSED_MATMUL_ADD_RELU node.
// Preconditions:
// 1. A MATMUL node %M producing rank-2 tensor [M, N].
// 2. An ADD node %A consuming %M and a bias tensor %B [N] or [1, N] or [M, N].
// 3. A RELU node %R consuming %A.
// 4. %M has exactly 1 consumer (%A) and is NOT a graph output.
// 5. %A has exactly 1 consumer (%R) and is NOT a graph output.
// Result:
// - Fused node %F = FUSED_MATMUL_ADD_RELU(%input, %weight, %bias) : f32[M, N]
// - Replaces %R with %F across all consumers.
// - Marks %M, %A, and %R as deleted.
class OperatorFusionPass : public Pass {
public:
    std::string name() const override { return "OperatorFusion"; }
    bool run(IRGraph& graph) override;
};

} // namespace forgetune

#endif // FORGETUNE_FUSION_HPP
