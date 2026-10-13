#ifndef FORGETUNE_OPTIMIZER_HPP
#define FORGETUNE_OPTIMIZER_HPP

#include "forgetune/ir.hpp"
#include "forgetune/pass_manager.hpp"
#include <memory>
#include <vector>

namespace forgetune {

// Optimization report capturing pass execution details
struct OptimizationReport {
    std::vector<PassResult> pass_results;
    size_t initial_node_count = 0;
    size_t final_node_count = 0;
    bool any_changed = false;
    std::string summary;
};

// High-level optimizer that configures standard pipelines
class Optimizer {
public:
    Optimizer() = default;

    // Build standard optimization pipeline:
    // 1. Constant folding
    // 2. Algebraic simplification
    // 3. Dead-code elimination
    // (Fusion is optionally added or controlled via configuration)
    static std::unique_ptr<PassManager> create_standard_pipeline(bool enable_fusion = false);

    // Run standard optimization on the graph
    OptimizationReport optimize(IRGraph& graph, bool enable_fusion = false);
};

} // namespace forgetune

#endif // FORGETUNE_OPTIMIZER_HPP
