#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <vector>

static bool run_case(ggml_backend_t backend, int64_t width, int64_t rows, std::vector<int32_t> ids_a,
                     std::vector<int32_t> ids_b) {
    ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
    ggml_tensor * ia = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ids_a.size());
    ggml_tensor * ib = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ids_b.size());
    ggml_tensor * ga = ggml_get_rows(ctx, a, ia);
    ggml_tensor * gb = ggml_get_rows(ctx, b, ib);
    ggml_tensor * sum = ggml_add(ctx, ga, gb);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }

    std::vector<float> av(width * rows), bv(width * rows), got(width * ids_a.size()), expected(got.size());
    for (int64_t r = 0; r < rows; ++r) for (int64_t i = 0; i < width; ++i) {
        av[r * width + i] = float((r * 37 + i * 13) % 251) * 0.125f - 11.0f;
        bv[r * width + i] = float((r * 19 + i * 7) % 239) * -0.0625f + 3.0f;
    }
    ggml_backend_tensor_set(a, av.data(), 0, av.size() * sizeof(float));
    ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * sizeof(float));
    ggml_backend_tensor_set(ia, ids_a.data(), 0, ids_a.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ib, ids_b.data(), 0, ids_b.size() * sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, sum);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed width=%lld rows=%lld\n", (long long)width, (long long)rows);
        ggml_backend_buffer_free(buffer); ggml_free(ctx); return false;
    }
    ggml_backend_tensor_get(sum, got.data(), 0, got.size() * sizeof(float));
    for (size_t r = 0; r < ids_a.size(); ++r) for (int64_t i = 0; i < width; ++i) {
        expected[r * width + i] = av[(size_t)ids_a[r] * width + i] + bv[(size_t)ids_b[r] * width + i];
    }
    if (got != expected) {
        fprintf(stderr, "exact output mismatch width=%lld rows=%lld\n", (long long)width, (long long)rows);
        ggml_backend_buffer_free(buffer); ggml_free(ctx); return false;
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "CUDA backend unavailable\n"); return 77; }
    const bool ok = run_case(backend, 5120, 3, {0}, {2}) &&
                    run_case(backend, 5120, 3, {2}, {0}) &&
                    run_case(backend, 1024, 3, {0, 2}, {2, 0});
    ggml_backend_free(backend);
    if (!ok) return 1;
    puts("Exp061 exact fused-boundary and generic multi-row fallback cases passed");
    return 0;
}
