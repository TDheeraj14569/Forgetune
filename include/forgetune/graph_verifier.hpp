#ifndef FORGETUNE_GRAPH_VERIFIER_HPP
#define FORGETUNE_GRAPH_VERIFIER_HPP

#include "forgetune/ir.hpp"
#include <vector>
#include <string>

namespace forgetune {

// Verification result: collects all issues found
struct VerificationResult {
    bool passed = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    void add_error(const std::string& msg) {
        passed = false;
        errors.push_back(msg);
    }
    void add_warning(const std::string& msg) {
        warnings.push_back(msg);
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "Verification " << (passed ? "PASSED" : "FAILED") << "\n";
        for (const auto& e : errors) {
            oss << "  ERROR: " << e << "\n";
        }
        for (const auto& w : warnings) {
            oss << "  WARNING: " << w << "\n";
        }
        return oss.str();
    }
};

// Verify the IR graph. Throws InvalidGraphError if check_throw is true and verification fails.
VerificationResult verify_graph(const IRGraph& graph, bool check_throw = true);

} // namespace forgetune

#endif // FORGETUNE_GRAPH_VERIFIER_HPP
