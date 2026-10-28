#ifndef FORGETUNE_ABI_H
#define FORGETUNE_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define FORGETUNE_EXPORT __declspec(dllexport)
#else
#define FORGETUNE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ForgeTune Unified C ABI Specification (v2.0)
 *
 * Status Return Codes:
 *   0: Success
 *  -1: Null pointer contract violation (input or output is NULL)
 *  -2: Input size mismatch or invalid batch size (B <= 0 or B > max_batch)
 *  -3: Output size mismatch
 *  -4: Internal arena allocation failure or re-entrancy exhaustion
 */

// Primary / Explicit Entry Points:

// Unified C ABI Entry Point (dispatched as 4 arguments for static models, 5 arguments for dynamic models):
// For static models: int forgetune_run(const float* in, size_t in_c, float* out, size_t out_c);
// For dynamic models: int forgetune_run(const float* in, size_t in_c, float* out, size_t out_c, int64_t batch);
#ifndef FORGETUNE_EXCLUDE_GENERIC_RUN_DECLARATION
FORGETUNE_EXPORT int forgetune_run(
    const float* input,
    size_t input_count,
    float* output,
    size_t output_count
);
#endif

// Version 1 C ABI (4 arguments, fixed/inferred batch size)
FORGETUNE_EXPORT int forgetune_run_v1(
    const float* input,
    size_t input_count,
    float* output,
    size_t output_count
);

// Version 2 C ABI (5 arguments, explicit dynamic batch size)
FORGETUNE_EXPORT int forgetune_run_v2(
    const float* input,
    size_t input_count,
    float* output,
    size_t output_count,
    int64_t batch_size
);

// Runtime Metadata Query Entry Points:
FORGETUNE_EXPORT int forgetune_is_dynamic(void);
FORGETUNE_EXPORT int64_t forgetune_max_batch(void);

// Isolated Kernel Microbenchmark Entry Point:
FORGETUNE_EXPORT double forgetune_benchmark_kernel(
    const float* input,
    size_t input_count,
    float* output,
    size_t output_count,
    int iters
);

// Native Allocation Inspection Entry Points:
FORGETUNE_EXPORT size_t forgetune_get_arena_alloc_count(void);
FORGETUNE_EXPORT size_t forgetune_get_arena_payload_alloc_count(void);
FORGETUNE_EXPORT size_t forgetune_get_arena_block_insertion_count(void);
FORGETUNE_EXPORT void forgetune_reset_arena_alloc_count(void);
FORGETUNE_EXPORT void forgetune_test_inject_arena_alloc(void);

#ifdef __cplusplus
}
#endif

#endif // FORGETUNE_ABI_H
