#include "forgetune/optimizer.hpp"
#include "forgetune/fusion.hpp"
#include <sstream>

namespace forgetune {

std::unique_ptr<PassManager> Optimizer::create_standard_pipeline(bool enable_fusion) {
    auto pm = std::make_unique<PassManager>();
    pm->add_pass(std::make_unique<ConstantFoldingPass>());
    pm->add_pass(std::make_unique<AlgebraicSimplificationPass>());
    pm->add_pass(std::make_unique<DeadCodeEliminationPass>());
    if (enable_fusion) {
        pm->add_pass(std::make_unique<OperatorFusionPass>());
        // Re-run DCE after fusion to clean up any leftover definitions
        pm->add_pass(std::make_unique<DeadCodeEliminationPass>());
    }
    return pm;
}

OptimizationReport Optimizer::optimize(IRGraph& graph, bool enable_fusion) {
    OptimizationReport report;
    report.initial_node_count = graph.live_node_count();

    auto pm = create_standard_pipeline(enable_fusion);
    report.pass_results = pm->run(graph);

    report.final_node_count = graph.live_node_count();
    report.any_changed = (report.initial_node_count != report.final_node_count);
    for (const auto& pr : report.pass_results) {
        if (pr.changed) {
            report.any_changed = true;
            break;
        }
    }

    std::ostringstream oss;
    oss << "Optimization Summary:\n"
        << "  Initial live nodes: " << report.initial_node_count << "\n"
        << "  Final live nodes:   " << report.final_node_count << "\n"
        << "  Passes executed:\n";
    for (const auto& pr : report.pass_results) {
        oss << "    - " << pr.pass_name << ": " << (pr.changed ? "MODIFIED" : "NO-OP")
            << " (" << pr.message << ")\n";
    }
    report.summary = oss.str();

    return report;
}

} // namespace forgetune
