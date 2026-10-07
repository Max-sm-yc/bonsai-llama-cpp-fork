#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" void ggml_cuda_exp056_pair(const void *, const void *, const char *, float *, float *, int, int, cudaStream_t);
extern "C" void ggml_cuda_exp056_single(const void *, const char *, float *, int, int, cudaStream_t);

struct block_ptq1_0 { uint8_t qs[24]; uint8_t qh[2]; uint16_t d; };
static void ck(cudaError_t e) { if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); } }

int main() {
    constexpr int K = 5120, ROWS = 2048, NB = K / 128, PLANES = 9;
    std::mt19937 rng(94517);
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<block_ptq1_0> hk(ROWS * NB), hv(ROWS * NB);
    for (auto * a : {&hk, &hv}) for (auto & b : *a) {
        for (auto & q : b.qs) q = byte(rng);
        for (auto & q : b.qh) q = byte(rng);
        b.d = __half_as_ushort(__float2half(0.02f + (byte(rng) / 255.0f) * 0.02f));
    }
    std::vector<char> hy(PLANES * NB * 16);
    for (int t = 0; t < 8; ++t) for (int kb = 0; kb < NB; ++kb) {
        for (int q = 0; q < 16; ++q) hy[(t * NB + kb) * 16 + q] = (char)((int)(rng() % 255) - 127);
    }
    for (int kb = 0; kb < NB; ++kb) for (int sub = 0; sub < 4; ++sub) {
        int isum=0;
        for (int j=0;j<32;++j) {
            const int e=sub*32+j, plane=e/16, lane=e%16;
            isum += (int8_t)hy[(plane*NB+kb)*16+lane];
        }
        half2 ds = __halves2half2(__float2half(0.015f), __float2half((float)isum));
        reinterpret_cast<half2 *>(hy.data() + (8 * NB + kb) * 16)[sub] = ds;
    }
    block_ptq1_0 * dk, *dv; char *dy; float *pk, *pv, *sk, *sv;
    ck(cudaMalloc(&dk, hk.size()*sizeof(block_ptq1_0))); ck(cudaMalloc(&dv, hv.size()*sizeof(block_ptq1_0)));
    ck(cudaMalloc(&dy, hy.size())); ck(cudaMalloc(&pk, ROWS*sizeof(float))); ck(cudaMalloc(&pv, ROWS*sizeof(float)));
    ck(cudaMalloc(&sk, ROWS*sizeof(float))); ck(cudaMalloc(&sv, ROWS*sizeof(float)));
    ck(cudaMemcpy(dk, hk.data(), hk.size()*sizeof(block_ptq1_0), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dv, hv.data(), hv.size()*sizeof(block_ptq1_0), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dy, hy.data(), hy.size(), cudaMemcpyHostToDevice));
    cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ggml_cuda_exp056_single(dk, dy, sk, K, ROWS, stream);
    ggml_cuda_exp056_single(dv, dy, sv, K, ROWS, stream);
    ggml_cuda_exp056_pair(dk, dv, dy, pk, pv, K, ROWS, stream);
    ck(cudaDeviceSynchronize());
    std::vector<float> hpk(ROWS), hpv(ROWS), hsk(ROWS), hsv(ROWS);
    ck(cudaMemcpy(hpk.data(), pk, ROWS*sizeof(float), cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(hpv.data(), pv, ROWS*sizeof(float), cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(hsk.data(), sk, ROWS*sizeof(float), cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(hsv.data(), sv, ROWS*sizeof(float), cudaMemcpyDeviceToHost));
    float max_abs_k=0, max_abs_v=0, max_rel_k=0, max_rel_v=0;
    for (int i=0;i<ROWS;++i) {
        max_abs_k=std::max(max_abs_k,std::abs(hpk[i]-hsk[i])); max_abs_v=std::max(max_abs_v,std::abs(hpv[i]-hsv[i]));
        max_rel_k=std::max(max_rel_k,std::abs(hpk[i]-hsk[i])/std::max(1e-6f,std::abs(hsk[i])));
        max_rel_v=std::max(max_rel_v,std::abs(hpv[i]-hsv[i])/std::max(1e-6f,std::abs(hsv[i])));
    }
    printf("correctness rows=%d K=%d K_abs=%.8g K_rel=%.8g V_abs=%.8g V_rel=%.8g\n",ROWS,K,max_abs_k,max_rel_k,max_abs_v,max_rel_v);
    if (max_rel_k > 2e-6f || max_rel_v > 2e-6f) return 3;
    constexpr int REPLAY=1000, REPS=15;
    cudaGraph_t gp, gs; cudaGraphExec_t ep, es;
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    ggml_cuda_exp056_pair(dk,dv,dy,pk,pv,K,ROWS,stream);
    ck(cudaStreamEndCapture(stream,&gp)); ck(cudaGraphInstantiate(&ep,gp,nullptr,nullptr,0));
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    ggml_cuda_exp056_single(dk,dy,sk,K,ROWS,stream); ggml_cuda_exp056_single(dv,dy,sv,K,ROWS,stream);
    ck(cudaStreamEndCapture(stream,&gs)); ck(cudaGraphInstantiate(&es,gs,nullptr,nullptr,0));
    // Warm both graph executables before timing.
    for (int i=0;i<100;++i) { ck(cudaGraphLaunch(ep,stream)); ck(cudaGraphLaunch(es,stream)); }
    cudaEvent_t beg,end; ck(cudaEventCreate(&beg)); ck(cudaEventCreate(&end));
    std::vector<float> tp,ts;
    for (int r=0;r<REPS;++r) {
        const int first = r & 1 ? 1 : 0;
        for (int j=0;j<2;++j) {
            const int arm = j == 0 ? first : 1-first;
            ck(cudaEventRecord(beg, stream));
            auto ex=arm==0?ep:es;
            for(int i=0;i<REPLAY;++i) ck(cudaGraphLaunch(ex,stream));
            ck(cudaEventRecord(end, stream)); ck(cudaEventSynchronize(end)); float ms; ck(cudaEventElapsedTime(&ms,beg,end));
            (arm==0?tp:ts).push_back(ms*1000.0f/REPLAY);
        }
    }
    for(int i=0;i<REPS;++i) printf("rep=%d first=%s pair_us=%.3f two_us=%.3f ratio=%.4f\n",i,(i&1)?"two":"pair",tp[i],ts[i],tp[i]/ts[i]);
    cudaGraphExecDestroy(ep);cudaGraphExecDestroy(es);cudaGraphDestroy(gp);cudaGraphDestroy(gs);
    cudaEventDestroy(beg);cudaEventDestroy(end);
    cudaFree(dk);cudaFree(dv);cudaFree(dy);cudaFree(pk);cudaFree(pv);cudaFree(sk);cudaFree(sv); cudaStreamDestroy(stream);
}
