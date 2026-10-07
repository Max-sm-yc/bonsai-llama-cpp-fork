#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static bool run_case(ggml_backend_t backend, int64_t channels, const char * output_path) {
    constexpr size_t context_bytes = 8 * 1024 * 1024;
    ggml_init_params params = { context_bytes, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, channels);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, channels);
    ggml_tensor * conv = ggml_ssm_conv(ctx, input, weights);
    ggml_tensor * silu = ggml_silu(ctx, conv);
    const int64_t groups = channels == 10240 ? 32 : channels / 128;
    ggml_tensor * qk_view = ggml_view_4d(ctx, silu, 128, groups, 1, 1,
            128 * sizeof(float), channels * sizeof(float), channels * sizeof(float), 0);
    ggml_tensor * normalized = ggml_l2_norm(ctx, qk_view, 1e-6f);
    ggml_tensor * q_view = ggml_view_2d(ctx, normalized, 128, groups / 2, 128 * sizeof(float), 0);
    ggml_tensor * k_view = ggml_view_2d(ctx, normalized, 128, groups / 2, 128 * sizeof(float), (groups / 2) * 128 * sizeof(float));
    ggml_tensor * value_view = ggml_view_1d(ctx, silu, 1, 0);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    std::vector<float> input_data(4 * channels), weight_data(4 * channels);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = (float)((int)(i % 37) - 18) * 0.03125f;
        weight_data[i] = (float)((int)(i % 19) - 9) * 0.015625f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(weights, weight_data.data(), 0, weight_data.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, q_view);
    ggml_build_forward_expand(graph, k_view);
    ggml_build_forward_expand(graph, value_view);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> silu_data(channels), norm_data(128 * groups);
    ggml_backend_tensor_get(silu, silu_data.data(), 0, silu_data.size() * sizeof(float));
    ggml_backend_tensor_get(normalized, norm_data.data(), 0, norm_data.size() * sizeof(float));
    std::ofstream output(output_path, std::ios::binary);
    const int64_t counts[] = { channels, 128 * groups };
    output.write((const char *)counts, sizeof(counts));
    output.write((const char *)silu_data.data(), silu_data.size() * sizeof(float));
    output.write((const char *)norm_data.data(), norm_data.size() * sizeof(float));
    const bool written = output.good();
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return written;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: test-exp062-ssm-l2 {model|fallback} OUTPUT\n");
        return 2;
    }
    const bool model_shape = std::string(argv[1]) == "model";
    if (!model_shape && std::string(argv[1]) != "fallback") return 2;
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "CUDA backend unavailable\n"); return 77; }
    const bool ok = run_case(backend, model_shape ? 10240 : 1024, argv[2]);
    ggml_backend_free(backend);
    if (!ok) return 1;
    return 0;
}
