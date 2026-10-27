#ifndef FORGETUNE_COMPILER_HPP
#define FORGETUNE_COMPILER_HPP

#include "forgetune/ir.hpp"
#include "forgetune/graph_verifier.hpp"
#include "forgetune/interpreter.hpp"
#include "forgetune/pass_manager.hpp"
#include "forgetune/optimizer.hpp"
#include "forgetune/fusion.hpp"
#include "forgetune/kernel_config.hpp"
#include "forgetune/codegen.hpp"
#include "forgetune/json_io.hpp"

namespace forgetune {

// Full compiler compilation result
struct CompilationResult {
    bool success = false;
    std::string generated_cpp_source;
    OptimizationReport optimization_report;
    VerificationResult verification_before;
    VerificationResult verification_after;
    std::string error_message;
};

// High-level C++ compiler driver
class Compiler {
public:
    explicit Compiler(KernelConfig config = KernelConfig{})
        : config_(std::move(config)) {}

    // Compile an IR graph into standalone C++ source
    CompilationResult compile(IRGraph& graph) {
        CompilationResult res;

        // 0. Validate compiler configuration
        std::string config_err;
        if (!config_.validate(&config_err)) {
            res.success = false;
            res.error_message = "Invalid compiler configuration: " + config_err;
            return res;
        }

        // 1. Verify input graph
        try {
            res.verification_before = verify_graph(graph, true);
        } catch (const std::exception& e) {
            res.success = false;
            res.error_message = std::string("Initial verification failed: ") + e.what();
            return res;
        }

        // 2. Run optimization pipeline
        try {
            Optimizer opt;
            res.optimization_report = opt.optimize(graph, config_.enable_fusion);
        } catch (const std::exception& e) {
            res.success = false;
            res.error_message = std::string("Optimization failed: ") + e.what();
            return res;
        }

        // 3. Verify optimized graph
        try {
            res.verification_after = verify_graph(graph, true);
        } catch (const std::exception& e) {
            res.success = false;
            res.error_message = std::string("Post-optimization verification failed: ") + e.what();
            return res;
        }

        // 4. Generate C++ source
        try {
            CodeGenerator codegen(config_);
            res.generated_cpp_source = codegen.generate(graph);
            res.success = true;
        } catch (const std::exception& e) {
            res.success = false;
            res.error_message = std::string("Code generation failed: ") + e.what();
            return res;
        }

        return res;
    }

    void set_config(KernelConfig config) { config_ = std::move(config); }
    const KernelConfig& get_config() const { return config_; }

private:
    KernelConfig config_;
};

} // namespace forgetune

#endif // FORGETUNE_COMPILER_HPP
