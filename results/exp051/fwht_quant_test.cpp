#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

enum ggml_cuda_q8_1_layout : int { GGML_CUDA_Q8_1_PT = 2 };
void fwht_quantize_row_q8_1_cuda(const void *, ggml_type, const float *, int, void *,
    ggml_cuda_q8_1_layout, int64_t, int64_t, int64_t, cudaStream_t);
void fwht_rms_quantize_q8_1_cuda(const float *, const float *, const float *, float, int, void *,
    ggml_cuda_q8_1_layout, int64_t, int64_t, int64_t, cudaStream_t);
static bool ck(cudaError_t e, const char * what);

template <typename F>
static void bench_graph(const char * label, F launch) {
    constexpr int nodes = 64;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    if (!ck(cudaStreamCreate(&stream), "bench stream") ||
        !ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture")) return;
    for (int i = 0; i < nodes; ++i) launch(stream);
    if (!ck(cudaStreamEndCapture(stream, &graph), "end capture") ||
        !ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate graph") ||
        !ck(cudaEventCreate(&start), "event start") || !ck(cudaEventCreate(&stop), "event stop")) return;
    for (int i = 0; i < 4; ++i) cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
    std::vector<float> us;
    for (int i = 0; i < 9; ++i) {
        cudaEventRecord(start, stream);
        cudaGraphLaunch(exec, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        us.push_back(ms * 1000.0f / nodes);
    }
    std::sort(us.begin(), us.end());
    std::printf("BENCH %s graph-event us/node median=%0.4f range=%0.4f-%0.4f (9 samples, %d nodes)\n",
        label, us[us.size()/2], us.front(), us.back(), nodes);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
}

static bool ck(cudaError_t e, const char * what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return false;
}

static uint16_t f2h(float x) {
    const __half h = __float2half(x);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

static float h2f(uint16_t u) {
    __half h;
    std::memcpy(&h, &u, 2);
    return __half2float(h);
}

static bool run(int n, std::mt19937 & rng) {
    const int ne0 = std::max(128, n);
    const int nblk = ne0 / 128;
    const size_t bytes = size_t(ne0) * 9 / 8;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(n), signs(n);
    for (int i = 0; i < n; ++i) {
        x[i] = dist(rng);
        signs[i] = (rng() & 1) ? 1.0f : -1.0f;
    }
    float * dx = nullptr, * ds = nullptr;
    uint8_t * dy = nullptr;
    bool ok = ck(cudaMalloc(reinterpret_cast<void **>(&dx), n * sizeof(float)), "malloc x") &&
        ck(cudaMalloc(reinterpret_cast<void **>(&ds), n * sizeof(float)), "malloc signs") &&
        ck(cudaMalloc(reinterpret_cast<void **>(&dy), bytes), "malloc y") &&
        ck(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy x") &&
        ck(cudaMemcpy(ds, signs.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy signs");
    if (ok) {
        fwht_quantize_row_q8_1_cuda(dx, GGML_TYPE_F32, ds, n, dy,
            GGML_CUDA_Q8_1_PT, n, ne0, 1, nullptr);
        ok = ck(cudaGetLastError(), "launch") && ck(cudaDeviceSynchronize(), "sync");
    }
    std::vector<uint8_t> y(bytes);
    if (ok) ok = ck(cudaMemcpy(y.data(), dy, bytes, cudaMemcpyDeviceToHost), "copy y");
    if (ok) {
        std::ofstream out("results/exp051/output_n" + std::to_string(n) + ".bin", std::ios::binary);
        out.write(reinterpret_cast<const char *>(y.data()), y.size());
        ok = bool(out);
    }

    std::vector<float> ref(n);
    for (int i = 0; i < n; ++i) ref[i] = (x[i] * signs[i]) * (1.0f / std::sqrt(float(n)));
    for (int h = 1; h < n; h *= 2) {
        for (int j = 0; j < n; j += 2 * h) for (int k = 0; k < h; ++k) {
            const float a = ref[j + k], b = ref[j + k + h];
            ref[j + k] = a + b; ref[j + k + h] = a - b;
        }
    }
    float max_abs = 0.0f, max_scaled = 0.0f;
    int q_mismatch = 0, sum_mismatch = 0;
    for (int i = 0; ok && i < ne0; ++i) {
        const int kb = i / 128, e = i % 128;
        const size_t qoff = size_t(e / 16) * nblk * 16 + size_t(kb) * 16 + (e % 16);
        const int8_t q = static_cast<int8_t>(y[qoff]);
        const int grp = e / 32;
        const size_t doff = size_t(8 * nblk * 16) + size_t(kb * 4 + grp) * 4;
        uint16_t dh, sh;
        std::memcpy(&dh, &y[doff], 2);
        std::memcpy(&sh, &y[doff + 2], 2);
        const float d = h2f(dh);
        const float v = i < n ? ref[i] : 0.0f;
        const int8_t expected = d == 0.0f ? 0 : static_cast<int8_t>(std::round(v / d));
        q_mismatch += q != expected;
        const float err = std::fabs(float(q) * d - v);
        max_abs = std::max(max_abs, err);
        max_scaled = std::max(max_scaled, err / std::max(d, 1e-12f));
    }
    for (int kb = 0; ok && kb < nblk; ++kb) for (int grp = 0; grp < 4; ++grp) {
        int sum = 0;
        for (int j = 0; j < 32; ++j) {
            const int e = grp * 32 + j;
            const size_t qoff = size_t(e / 16) * nblk * 16 + size_t(kb) * 16 + (e % 16);
            sum += static_cast<int8_t>(y[qoff]);
        }
        const size_t doff = size_t(8 * nblk * 16) + size_t(kb * 4 + grp) * 4;
        uint16_t sh;
        std::memcpy(&sh, &y[doff + 2], 2);
        sum_mismatch += static_cast<int16_t>(sh) != sum;
    }
    std::printf("N=%d NT=%d: q_mismatch=%d sum_mismatch=%d max_abs=%g max_error/scale=%g\n",
        n, std::min(n, 256), q_mismatch, sum_mismatch, max_abs, max_scaled);
    ok = ok && sum_mismatch == 0 && max_scaled <= 0.75f;
    if (n == 1024 && std::getenv("EXP051_BENCH")) {
        bench_graph("fwht_quantize_N1024_NT256", [&](cudaStream_t s) {
            fwht_quantize_row_q8_1_cuda(dx, GGML_TYPE_F32, ds, n, dy,
                GGML_CUDA_Q8_1_PT, n, ne0, 1, s);
        });
        constexpr int rk = 5120;
        float * rx = nullptr, * rw = nullptr, * rs = nullptr;
        uint8_t * ry = nullptr;
        cudaMalloc(reinterpret_cast<void **>(&rx), rk * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&rw), rk * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&rs), rk * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&ry), size_t(rk) * 9 / 8);
        std::vector<float> host(rk), signs_r(rk), weight_r(rk);
        for (int i = 0; i < rk; ++i) { host[i] = dist(rng); signs_r[i] = (rng() & 1) ? 1.0f : -1.0f; weight_r[i] = 0.8f + 0.2f * dist(rng); }
        cudaMemcpy(rx, host.data(), rk * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(rs, signs_r.data(), rk * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(rw, weight_r.data(), rk * sizeof(float), cudaMemcpyHostToDevice);
        bench_graph("fwht_rms_quantize_N1024_NT1024", [&](cudaStream_t s) {
            fwht_rms_quantize_q8_1_cuda(rx, rw, rs, 1.0e-5f, 1024, ry,
                GGML_CUDA_Q8_1_PT, rk, rk, 1, s);
        });
        cudaFree(rx); cudaFree(rw); cudaFree(rs); cudaFree(ry);
    }
    cudaFree(dx); cudaFree(ds); cudaFree(dy);
    return ok;
}

int main() {
    int device = 0;
    if (!ck(cudaSetDevice(device), "set device")) return 1;
    std::mt19937 rng(51051);
    bool ok = true;
    for (int n : {64, 128, 256, 512, 1024, 2048}) ok &= run(n, rng);
    return ok ? 0 : 1;
}
