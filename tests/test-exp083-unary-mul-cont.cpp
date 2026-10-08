#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static bool run_case(ggml_backend_t cuda, int64_t sequence, bool strided,
                     std::vector<float> & expected, std::vector<float> & actual) {
    constexpr size_t context_bytes = 8 * 1024 * 1024;
    ggml_init_params params = { context_bytes, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    const int64_t source_width = strided ? 512 : 256;
    ggml_tensor * source = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, source_width, 24, sequence);
    ggml_tensor * view = ggml_view_3d(ctx, source, 256, 24, sequence,
            source_width * sizeof(float), source_width * 24 * sizeof(float), 0);
    ggml_tensor * cont = ggml_cont(ctx, view);
    ggml_tensor * other = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 24, sequence);
    ggml_tensor * sigmoid = ggml_sigmoid(ctx, cont);
    ggml_tensor * output = ggml_mul(ctx, sigmoid, other);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cuda);
    if (!buffer) { ggml_free(ctx); return false; }

    const size_t source_count = (size_t) source_width * 24 * sequence;
    const size_t output_count = (size_t) 6144 * sequence;
    std::vector<float> source_data(source_count), other_data(output_count);
    for (size_t i = 0; i < source_count; ++i) {
        source_data[i] = (float)((int)(i % 1009) - 504) * 0.00390625f;
    }
    for (size_t i = 0; i < output_count; ++i) {
        other_data[i] = (float)((int)(i % 257) - 128) * 0.0078125f;
    }
    ggml_backend_tensor_set(source, source_data.data(), 0, source_data.size() * sizeof(float));
    ggml_backend_tensor_set(other, other_data.data(), 0, other_data.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_backend_graph_compute(cuda, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    actual.resize(output_count);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    expected.resize(output_count);
    for (int64_t seq = 0; seq < sequence; ++seq) {
        for (int64_t head = 0; head < 24; ++head) {
            for (int64_t col = 0; col < 256; ++col) {
                const size_t output_index = (size_t) seq * 6144 + (size_t) head * 256 + col;
                const size_t source_index = (size_t) seq * 24 * source_width + (size_t) head * source_width + col;
                const float x = source_data[source_index];
                expected[output_index] = (1.0f / (1.0f + std::exp(-x))) * other_data[output_index];
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

static bool compare_case(ggml_backend_t cuda, int64_t sequence, bool strided) {
    std::vector<float> expected, actual;
    if (!run_case(cuda, sequence, strided, expected, actual)) {
        fprintf(stderr, "failed to run sequence=%lld strided=%d\n", (long long) sequence, (int) strided);
        return false;
    }

    float max_abs_error = 0.0f;
    size_t mismatch_count = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float error = std::abs(expected[i] - actual[i]);
        max_abs_error = std::max(max_abs_error, error);
        if (!(error <= 2.0e-6f)) ++mismatch_count;
    }
    fprintf(stderr, "sequence=%lld strided=%d elements=%zu max_abs_error=%.9g mismatches=%zu\n",
            (long long) sequence, (int) strided, actual.size(), max_abs_error, mismatch_count);
    return mismatch_count == 0;
}

int main() {
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (!cuda) {
        fprintf(stderr, "CUDA backend unavailable\n");
        return 77;
    }

    bool ok = true;
    for (int64_t sequence : { 1, 2, 128, 512, 4096 }) {
        ok = compare_case(cuda, sequence, true) && ok;
    }
    // A contiguous source is a deliberate matcher fallback; verify generic behavior too.
    ok = compare_case(cuda, 1, false) && ok;

    ggml_backend_free(cuda);
    return ok ? 0 : 1;
}
