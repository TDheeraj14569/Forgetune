#ifndef FORGETUNE_CODEGEN_HPP
#define FORGETUNE_CODEGEN_HPP

#include "forgetune/ir.hpp"
#include "forgetune/kernel_config.hpp"
#include <string>

namespace forgetune {

// Code generator: emits standalone, self-contained C++ source code from verified IR
class CodeGenerator {
public:
    explicit CodeGenerator(KernelConfig config = KernelConfig{})
        : config_(std::move(config)) {}

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
    void emit_kernel_functions(std::ostringstream& oss, const IRGraph& graph) const;
    void emit_entry_point(std::ostringstream& oss, const IRGraph& graph) const;

    // Operation-specific kernel emitters
    void emit_matmul_naive(std::ostringstream& oss, const std::string& func_name,
                           int64_t M, int64_t K, int64_t N) const;
    void emit_matmul_ikj(std::ostringstream& oss, const std::string& func_name,
                         int64_t M, int64_t K, int64_t N) const;
    void emit_matmul_tiled(std::ostringstream& oss, const std::string& func_name,
                           int64_t M, int64_t K, int64_t N,
                           int tile_m, int tile_n, int tile_k) const;
    void emit_fused_matmul_add_relu(std::ostringstream& oss, const std::string& func_name,
                                    int64_t M, int64_t K, int64_t N,
                                    bool has_bias, MatMulVariant variant,
                                    int tile_m, int tile_n, int tile_k) const;
};

} // namespace forgetune

#endif // FORGETUNE_CODEGEN_HPP
