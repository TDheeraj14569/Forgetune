#ifndef FORGETUNE_KERNEL_CONFIG_HPP
#define FORGETUNE_KERNEL_CONFIG_HPP

#include <string>
#include <vector>
#include <sstream>

namespace forgetune {

// Matrix multiplication kernel variants
enum class MatMulVariant {
    NAIVE_IJK,        // Standard naive i, j, k loops
    LOOP_IKJ,         // Alternate cache-friendly loop order (i, k, j)
    TILED,            // 2D tiled matrix multiplication with boundary handling
    FUSED_IKJ,        // Fused MATMUL + ADD + RELU in i, k, j loop order
    FUSED_TILED       // Fused MATMUL + ADD + RELU with 2D tiling
};

inline const char* matmul_variant_to_string(MatMulVariant v) {
    switch (v) {
        case MatMulVariant::NAIVE_IJK:  return "naive_ijk";
        case MatMulVariant::LOOP_IKJ:   return "loop_ikj";
        case MatMulVariant::TILED:      return "tiled";
        case MatMulVariant::FUSED_IKJ:  return "fused_ikj";
        case MatMulVariant::FUSED_TILED: return "fused_tiled";
        default: return "unknown";
    }
}

inline MatMulVariant string_to_matmul_variant(const std::string& s) {
    if (s == "naive_ijk")  return MatMulVariant::NAIVE_IJK;
    if (s == "loop_ikj")   return MatMulVariant::LOOP_IKJ;
    if (s == "tiled")      return MatMulVariant::TILED;
    if (s == "fused_ikj")  return MatMulVariant::FUSED_IKJ;
    if (s == "fused_tiled") return MatMulVariant::FUSED_TILED;
    return MatMulVariant::NAIVE_IJK;
}

// Full configuration specification for code generation and autotuning
struct KernelConfig {
    std::string candidate_id = "baseline_naive";
    MatMulVariant matmul_variant = MatMulVariant::NAIVE_IJK;
    int tile_m = 16;
    int tile_n = 16;
    int tile_k = 16;
    bool enable_fusion = false;
    std::string compiler_opt_level = "-O2"; // e.g. -O2, -O3

    std::string to_string() const {
        std::ostringstream oss;
        oss << "KernelConfig[id=" << candidate_id
            << ", variant=" << matmul_variant_to_string(matmul_variant)
            << ", tiles=(" << tile_m << "," << tile_n << "," << tile_k << ")"
            << ", fusion=" << (enable_fusion ? "true" : "false")
            << ", opt=" << compiler_opt_level << "]";
        return oss.str();
    }
};

} // namespace forgetune

#endif // FORGETUNE_KERNEL_CONFIG_HPP
