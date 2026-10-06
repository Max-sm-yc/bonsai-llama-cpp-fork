#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>

struct BaseBlock { uint8_t qs[24], qh[2]; half d; };
static_assert(sizeof(BaseBlock) == 28);
struct SideBlock { uint8_t q[32]; half d; };
static_assert(sizeof(SideBlock) == 34);

__device__ __forceinline__ uint8_t trit(uint8_t x, int t) {
    #pragma unroll
    for (int i = 0; i <= t; ++i) { int w = x * 3; x = w & 255; if (i == t) return w >> 8; }
    return 0;
}
__device__ __forceinline__ uint8_t base_weight(const BaseBlock &b, int i) {
    if (i < 120) return trit(b.qs[(i / 40) * 8 + i % 8], (i % 40) / 8);
    return trit(b.qh[(i - 120) % 2], (i - 120) / 2);
}
__global__ void convert(const BaseBlock *src, SideBlock *dst, int n) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n) return;
    #pragma unroll
    for (int i = 0; i < 128; ++i) dst[b].q[i] = base_weight(src[b], i);
    dst[b].d = src[b].d;
}
__global__ void verify(const BaseBlock *src, const SideBlock *dst, int *bad, int n) {
    int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=n)return;
    int v=0; for(int i=0;i<128;++i)v += base_weight(src[b],i)!=dst[b].q[i]; bad[b]=v;
}
// Activation bytes use the production PTQ1_0 warp-transposed layout: element
// i is stored at plane (i/16), byte (i%16). Each block has four exact q8 sums.
template<bool SIDE> __global__ void dot(const BaseBlock *base, const SideBlock *side, const int8_t *act, float *out, int n) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n) return;
    float x = 0.0f;
    int sums[4] = {};
    #pragma unroll
    for (int i = 0; i < 128; ++i) {
        uint8_t q = SIDE ? side[b].q[i] : base_weight(base[b], i);
        int a = act[b*128 + (i/16)*16 + (i%16)];
        sums[i/32] += ((int)q - 1) * a;
    }
    x = 0.0f;
    #pragma unroll
    for (int k=0;k<4;++k) x=__fmaf_rn(__half2float(SIDE ? side[b].d : base[b].d), (float)sums[k], x);
    out[b]=x;
}
static void ck(cudaError_t e) { if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);} }
int main(int argc,char **argv) {
    int n=argc>1?atoi(argv[1]):65536, reps=argc>2?atoi(argv[2]):100;
    std::mt19937 rng(1234); std::uniform_int_distribution<int> pd(0,242), ad(-127,127);
    std::vector<BaseBlock> hb(n); std::vector<int8_t> ha((size_t)n*128);
    std::uniform_int_distribution<int> hd(0,80);
    for(auto &b:hb){for(auto &x:b.qs)x=pd(rng);for(auto &x:b.qh)x=hd(rng);b.d=__float2half(0.125f);}
    for(auto &x:ha)x=ad(rng);
    BaseBlock *db; SideBlock *ds; int8_t *da; float *d0,*d1; int *dv;
    ck(cudaMalloc(&db,hb.size()*sizeof(BaseBlock)));ck(cudaMalloc(&ds,(size_t)n*sizeof(SideBlock)));ck(cudaMalloc(&da,ha.size()));ck(cudaMalloc(&d0,n*4));ck(cudaMalloc(&d1,n*4));ck(cudaMalloc(&dv,n*4));
    ck(cudaMemcpy(db,hb.data(),hb.size()*sizeof(BaseBlock),cudaMemcpyHostToDevice));ck(cudaMemcpy(da,ha.data(),ha.size(),cudaMemcpyHostToDevice));
    dim3 block(128),grid((n+127)/128); convert<<<grid,block>>>(db,ds,n); verify<<<grid,block>>>(db,ds,dv,n); dot<false><<<grid,block>>>(db,ds,da,d0,n);dot<true><<<grid,block>>>(db,ds,da,d1,n);ck(cudaDeviceSynchronize());
    std::vector<float>a(n),b(n);std::vector<int>v(n);ck(cudaMemcpy(a.data(),d0,n*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),d1,n*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(v.data(),dv,n*4,cudaMemcpyDeviceToHost));int bad=0, vb=0;for(int i=0;i<n;++i){if(a[i]!=b[i]&&bad<3)fprintf(stderr,"i=%d base=%f side=%f weight_mismatch=%d\n",i,a[i],b[i],v[i]);bad+=a[i]!=b[i];vb+=v[i];}
    cudaEvent_t s,e;ck(cudaEventCreate(&s));ck(cudaEventCreate(&e));
    for(int v=0;v<2;++v){float ms;for(int w=0;w<10;++w){if(v)dot<true><<<grid,block>>>(db,ds,da,d1,n);else dot<false><<<grid,block>>>(db,ds,da,d0,n);}ck(cudaDeviceSynchronize());ck(cudaEventRecord(s));for(int r=0;r<reps;++r){if(v)dot<true><<<grid,block>>>(db,ds,da,d1,n);else dot<false><<<grid,block>>>(db,ds,da,v?d1:d0,n);}ck(cudaEventRecord(e));ck(cudaEventSynchronize(e));ck(cudaEventElapsedTime(&ms,s,e));printf("%s %.6f ms/launch n=%d reps=%d\n",v?"2-bit-side":"base3",ms/reps,n,reps);}
    printf("mismatches=%d decoded_code_mismatches=%d payload base=%zu side=%zu delta=%.2f%%\n",bad,vb,sizeof(BaseBlock),sizeof(SideBlock),100.0*(sizeof(SideBlock)/(double)sizeof(BaseBlock)-1));
}
