#ifndef FORGETUNE_PASS_MANAGER_HPP
#define FORGETUNE_PASS_MANAGER_HPP

#include "forgetune/ir.hpp"
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace forgetune {

// Result of running a single optimization pass
struct PassResult {
    std::string pass_name;
    bool changed = false;     // Did this pass modify the graph?
    bool succeeded = true;    // Did this pass complete without error?
    std::string message;      // Description of what happened
    std::string ir_before;    // IR text before the pass (for debugging)
    std::string ir_after;     // IR text after the pass (for debugging)
};

// Abstract base class for optimization passes
class Pass {
public:
    virtual ~Pass() = default;
    virtual std::string name() const = 0;
    // Run the pass on the graph. Returns true if the graph was modified.
    virtual bool run(IRGraph& graph) = 0;
};

// Concrete pass implementations
class ConstantFoldingPass : public Pass {
public:
    std::string name() const override { return "ConstantFolding"; }
    bool run(IRGraph& graph) override;
};

/// AlgebraicSimplificationPass:
/// Implements standard algebraic identities:
/// 1. ADD(X, 0) -> X when 0 is all zeros of compatible shape.
///    Note on IEEE 754 signed zeros: In strict IEEE 754 round-to-nearest, (-0.0f) + (+0.0f) yields (+0.0f).
///    ForgeTune operates under standard machine learning fast-math semantics (analogous to LLVM
///    `-fno-signed-zeros`), treating +0.0f and -0.0f as numerically equivalent for inference.
/// 2. RELU(RELU(X)) -> RELU(X) idempotent property.
class AlgebraicSimplificationPass : public Pass {
public:
    std::string name() const override { return "AlgebraicSimplification"; }
    bool run(IRGraph& graph) override;
};

class DeadCodeEliminationPass : public Pass {
public:
    std::string name() const override { return "DeadCodeElimination"; }
    bool run(IRGraph& graph) override;
};

// Pass manager: runs a sequence of passes with verification
class PassManager {
public:
    void add_pass(std::unique_ptr<Pass> pass);

    // Run all passes. Verifies the graph before and after each pass.
    // Returns results for each pass.
    std::vector<PassResult> run(IRGraph& graph);

    // Enable/disable IR snapshots (for debugging)
    void set_capture_ir(bool capture) { capture_ir_ = capture; }

private:
    std::vector<std::unique_ptr<Pass>> passes_;
    bool capture_ir_ = true;
};

} // namespace forgetune

#endif // FORGETUNE_PASS_MANAGER_HPP
