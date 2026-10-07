// Direct numerical check for the Exp036 RMSNorm + signed FWHT + PT Q8_1 kernel.
// Runs the production CUDA kernel and compares its packed Q8 values, scales, and
// integer block sums against an independent host implementation.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// Keep this test on the ordinary C++ compiler; these declarations match the
// internal CUDA library ABI while avoiding a second CUDA-language scope in CMake.
enum ggml_cuda_q8_1_layout : int {
    GGML_CUDA_Q8_1_AOS = 0,
    GGML_CUDA_Q8_1_SOA = 1,
    GGML_CUDA_Q8_1_PT = 2,
};
bool fwht_rms_quantize_q8_1_supported(int n, int64_t ne00, ggml_cuda_q8_1_layout layout);
void fwht_rms_quantize_q8_1_cuda(
        const float * x, const float * weight, const float * signs, float eps, int n, void * vy,
        ggml_cuda_q8_1_layout layout, int64_t ne00, int64_t ne0, int64_t ncols, cudaStream_t stream);

namespace {

constexpr int k = 5120;
constexpr int fwht_width = 1024;
constexpr int qk8 = 32;
constexpr int QK_PTQ1_0 = 128;
constexpr int nblk = k / QK_PTQ1_0;
constexpr float eps = 1.0e-5f;

bool cuda_ok(cudaError_t err, const char * what) {
    if (err == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    return false;
}

float half_to_float(const uint16_t h) {
    const float sign = (h & 0x8000) ? -1.0f : 1.0f;
    const int exp = (h >> 10) & 0x1f;
    const int mant = h & 0x03ff;
    if (exp == 0) {
        return sign * std::ldexp((float) mant, -24);
    }
    if (exp == 0x1f) {
        return mant == 0 ? sign * INFINITY : NAN;
    }
    return sign * std::ldexp((float) (1024 + mant), exp - 25);
}

uint16_t read_u16(const uint8_t * p) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

int8_t read_q(const std::vector<uint8_t> & packed, const size_t row_base, const int elem) {
    const int kb = elem / QK_PTQ1_0;
    const int e  = elem % QK_PTQ1_0;
    const size_t offset = row_base + ((e / 16) * nblk + kb) * 16 + (e % 16);
    return (int8_t) packed[offset];
}

std::vector<float> reference_transform(
        const std::vector<float> & x, const std::vector<float> & weight,
        const std::vector<float> & signs, const int row) {
    double sum = 0.0;
    const size_t row_base = (size_t) row * k;
    for (int i = 0; i < k; ++i) {
        const double xi = x[row_base + i];
        sum += xi * xi;
    }
    const float rms = 1.0f / std::sqrt((float) (sum / k) + eps);
    const float inv_sqrt = 1.0f / std::sqrt((float) fwht_width);
    std::vector<float> out(k);
    for (int base = 0; base < k; base += fwht_width) {
        float * tile = out.data() + base;
        for (int i = 0; i < fwht_width; ++i) {
            const size_t idx = row_base + base + i;
            float value = x[idx] * rms;
            value *= weight[base + i];
            value *= signs[base + i];
            tile[i] = value * inv_sqrt;
        }
        for (int h = 1; h < fwht_width; h *= 2) {
            for (int block = 0; block < fwht_width; block += 2 * h) {
                for (int j = 0; j < h; ++j) {
                    const float a = tile[block + j];
                    const float b = tile[block + h + j];
                    tile[block + j] = a + b;
                    tile[block + h + j] = a - b;
                }
            }
        }
    }
    return out;
}

bool run_case(const int nrows, std::mt19937 & rng) {
    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> w_dist(0.65f, 1.35f);
    std::vector<float> x((size_t) nrows * k);
    std::vector<float> weight(k);
    std::vector<float> signs(k);
    for (float & v : x) {
        v = x_dist(rng);
    }
    for (float & v : weight) {
        v = w_dist(rng);
    }
    for (float & v : signs) {
        v = (rng() & 1) ? 1.0f : -1.0f;
    }

    constexpr size_t row_stride = (size_t) k * 9 / 8;
    const size_t output_bytes = (size_t) nrows * row_stride;
    float * dx = nullptr;
    float * dw = nullptr;
    float * ds = nullptr;
    uint8_t * dy = nullptr;
    cudaStream_t stream = nullptr;
    bool ok = cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void **>(&dx), x.size() * sizeof(float)), "cudaMalloc x") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void **>(&dw), weight.size() * sizeof(float)), "cudaMalloc weight") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void **>(&ds), signs.size() * sizeof(float)), "cudaMalloc signs") &&
        cuda_ok(cudaMalloc(reinterpret_cast<void **>(&dy), output_bytes), "cudaMalloc output");
    if (ok) {
        ok = cuda_ok(cudaMemcpyAsync(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy x") &&
             cuda_ok(cudaMemcpyAsync(dw, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy weight") &&
             cuda_ok(cudaMemcpyAsync(ds, signs.data(), signs.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy signs");
    }
    if (ok) {
        std::vector<uint8_t> zeros(output_bytes, 0);
        ok = cuda_ok(cudaMemcpyAsync(dy, zeros.data(), output_bytes, cudaMemcpyHostToDevice, stream), "clear output");
    }
    if (ok) {
        fwht_rms_quantize_q8_1_cuda(dx, dw, ds, eps, fwht_width, dy,
            GGML_CUDA_Q8_1_PT, k, k, nrows, stream);
        ok = cuda_ok(cudaGetLastError(), "launch RMS/FWHT/Q8 kernel") &&
             cuda_ok(cudaStreamSynchronize(stream), "synchronize RMS/FWHT/Q8 kernel");
    }

    std::vector<uint8_t> packed(output_bytes);
    if (ok) {
        ok = cuda_ok(cudaMemcpy(packed.data(), dy, output_bytes, cudaMemcpyDeviceToHost), "copy output");
    }

    float max_abs_error = 0.0f;
    float max_scaled_error = 0.0f;
    int sum_mismatches = 0;
    if (ok) {
        for (int row = 0; row < nrows; ++row) {
            const size_t row_base = (size_t) row * row_stride;
            const std::vector<float> ref = reference_transform(x, weight, signs, row);
            for (int kb = 0; kb < nblk; ++kb) {
                for (int sub = 0; sub < 4; ++sub) {
                    int expected_sum = 0;
                    for (int j = 0; j < qk8; ++j) {
                        const int elem = kb * QK_PTQ1_0 + sub * qk8 + j;
                        const int8_t q = read_q(packed, row_base, elem);
                        expected_sum += q;
                        const size_t ds_offset = row_base + 8 * nblk * 16 + (size_t) (kb * 4 + sub) * 4;
                        const float d = half_to_float(read_u16(packed.data() + ds_offset));
                        const float reconstructed = (float) q * d;
                        const float abs_error = std::fabs(reconstructed - ref[elem]);
                        max_abs_error = std::max(max_abs_error, abs_error);
                        max_scaled_error = std::max(max_scaled_error, abs_error / std::max(d, 1.0e-12f));
                    }
                    const size_t ds_offset = row_base + 8 * nblk * 16 + (size_t) (kb * 4 + sub) * 4;
                    const int16_t stored_sum = (int16_t) read_u16(packed.data() + ds_offset + 2);
                    sum_mismatches += stored_sum != expected_sum;
                }
            }
        }
        if (sum_mismatches != 0 || max_scaled_error > 0.75f) {
            std::fprintf(stderr, "nrows=%d: sum mismatches=%d, max error/scale=%.6f (max abs %.8g)\n",
                nrows, sum_mismatches, max_scaled_error, max_abs_error);
            ok = false;
        } else {
            std::printf("nrows=%d: pass; max abs error %.8g, max error/scale %.6f, block sums exact\n",
                nrows, max_abs_error, max_scaled_error);
        }
    }

    if (dy) cudaFree(dy);
    if (ds) cudaFree(ds);
    if (dw) cudaFree(dw);
    if (dx) cudaFree(dx);
    if (stream) cudaStreamDestroy(stream);
    return ok;
}

} // namespace

int main() {
    if (!fwht_rms_quantize_q8_1_supported(fwht_width, k, GGML_CUDA_Q8_1_PT) ||
        fwht_rms_quantize_q8_1_supported(fwht_width, k, GGML_CUDA_Q8_1_AOS) ||
        fwht_rms_quantize_q8_1_supported(fwht_width, k - 1, GGML_CUDA_Q8_1_PT)) {
        std::fprintf(stderr, "fused RMS/FWHT/Q8 shape support predicate mismatch\n");
        return 1;
    }
    int device = 0;
    if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice")) {
        return 1;
    }
    cudaDeviceProp prop{};
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties")) {
        return 1;
    }
    if (prop.major < 7) {
        std::fprintf(stderr, "test requires a CUDA GPU with compute capability 7.0 or newer\n");
        return 1;
    }
    std::mt19937 rng(0x036c0deu);
    if (!run_case(1, rng) || !run_case(3, rng)) {
        return 1;
    }
    return 0;
}
