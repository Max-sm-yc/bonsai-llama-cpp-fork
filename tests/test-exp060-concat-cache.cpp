#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <vector>

static bool run_case(ggml_backend_t backend, int64_t channels, int repeats) {
    const int64_t hist = 3;
    const int64_t full = 4;
    const size_t bytes = 64 * 1024 * 1024;
    ggml_init_params params = { bytes, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hist, channels);
    ggml_tensor * y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, channels);
    ggml_tensor * cache = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hist * channels);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, full, channels);
    ggml_tensor * joined = ggml_concat(ctx, x, y, 0);
    ggml_tensor * src_view = ggml_view_2d(ctx, joined, hist, channels, joined->nb[1], sizeof(float));
    ggml_tensor * dst_view = ggml_view_2d(ctx, cache, hist * channels, 1, ggml_row_size(GGML_TYPE_F32, hist * channels), 0);
    ggml_tensor * cpy = ggml_cpy(ctx, src_view, dst_view);
    ggml_tensor * conv = ggml_ssm_conv(ctx, joined, weights);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    std::vector<float> xv(hist * channels), yv(channels), wv(full * channels), cache_got(hist * channels);
    std::vector<float> joined_got(full * channels);
    for (int rep = 0; rep < repeats; ++rep) {
        for (size_t i = 0; i < xv.size(); ++i) xv[i] = (float)(i + rep * 17) * 0.125f - 11.0f;
        for (size_t i = 0; i < yv.size(); ++i) yv[i] = (float)((int64_t)i - rep * 13) * -0.25f + 7.0f;
        for (size_t i = 0; i < wv.size(); ++i) wv[i] = 0.01f * (float)(i % 29);
        std::vector<float> sentinel(hist * channels, -999.0f - rep);
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
        ggml_backend_tensor_set(y, yv.data(), 0, yv.size() * sizeof(float));
        ggml_backend_tensor_set(weights, wv.data(), 0, wv.size() * sizeof(float));
        ggml_backend_tensor_set(cache, sentinel.data(), 0, sentinel.size() * sizeof(float));

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, cpy);
        ggml_build_forward_expand(graph, conv);
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) { fprintf(stderr, "graph compute failed: %s\n", ggml_status_to_string(status)); return false; }
        ggml_backend_tensor_get(joined, joined_got.data(), 0, joined_got.size() * sizeof(float));
        ggml_backend_tensor_get(cache, cache_got.data(), 0, cache_got.size() * sizeof(float));

        std::vector<float> expected_joined(full * channels), expected_cache(hist * channels);
        for (int64_t c = 0; c < channels; ++c) {
            expected_joined[c * full + 0] = xv[c * hist + 0];
            expected_joined[c * full + 1] = xv[c * hist + 1];
            expected_joined[c * full + 2] = xv[c * hist + 2];
            expected_joined[c * full + 3] = yv[c];
            expected_cache[c * hist + 0] = xv[c * hist + 1];
            expected_cache[c * hist + 1] = xv[c * hist + 2];
            expected_cache[c * hist + 2] = yv[c];
        }
        if (joined_got != expected_joined || cache_got != expected_cache) {
            fprintf(stderr, "mismatch channels=%lld repeat=%d joined=%d cache=%d\n", (long long) channels, rep,
                    joined_got != expected_joined, cache_got != expected_cache);
            return false;
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "CUDA backend unavailable\n"); return 77; }
    const bool ok = run_case(backend, 10240, 3) && run_case(backend, 1024, 2);
    ggml_backend_free(backend);
    if (!ok) return 1;
    puts("Exp060 concat/cache fused and fallback cases passed exact output/cache checks");
    return 0;
}
