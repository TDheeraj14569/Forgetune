#ifndef FORGETUNE_CODEGEN_HPP
#define FORGETUNE_CODEGEN_HPP

#include "forgetune/ir.hpp"
#include "forgetune/kernel_config.hpp"
#include <string>
#include <limits>
#include <stdexcept>

namespace forgetune {

// Code generator: emits standalone, self-contained C++ source code from verified IR
class CodeGenerator {
public:
    explicit CodeGenerator(KernelConfig config = KernelConfig{})
        : config_(std::move(config)) {}

    // Checked workspace addition helper
    static size_t calculate_total_workspace(size_t arena_bytes, size_t conv2d_scratch_bytes) {
        if (conv2d_scratch_bytes > std::numeric_limits<size_t>::max() - arena_bytes) {
            throw std::overflow_error("Total workspace size arithmetic overflow");
        }
        return arena_bytes + conv2d_scratch_bytes;
    }

    // Generate standalone C++ source code
    std::string generate(const IRGraph& graph) const;

    // Set configuration
    void set_config(const KernelConfig& config) { config_ = config; }
    const KernelConfig& get_config() const { return config_; }

private:
    KernelConfig config_;

    // Code emission helpers
    void emit_headers(std::ostringstream& oss) const;
    void emit_constants(std::ostringstream& oss, const IRGraph& graph) const;
    void emit_arena_pool(std::ostringstream& oss) const;
    void emit_kernel_functions(std::ostringstream& oss, const IRGraph& graph) const;
    void emit_entry_point(std::ostringstream& oss, const IRGraph& graph) const;

    // Operation-specific kernel emitters
    void emit_matmul_naive(std::ostringstream& oss, const std::string& func_name,
                           int64_t K, int64_t N) const;
    void emit_matmul_ikj(std::ostringstream& oss, const std::string& func_name,
                         int64_t K, int64_t N) const;
    void emit_matmul_tiled(std::ostringstream& oss, const std::string& func_name,
                           int64_t K, int64_t N,
                           int tile_m, int tile_n, int tile_k) const;
    void emit_fused_matmul_add_relu(std::ostringstream& oss, const std::string& func_name,
                                    int64_t K, int64_t N,
                                    bool has_bias, MatMulVariant variant,
                                    int tile_m, int tile_n, int tile_k,
                                    bool bias_is_2d = false) const;
    void emit_conv2d_kernel(std::ostringstream& oss, const std::string& func_name,
                            const IRNode& node, const IRGraph& graph, bool fused_relu) const;
};

} // namespace forgetune

#endif // FORGETUNE_CODEGEN_HPP
