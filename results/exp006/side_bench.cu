#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

struct BaseBlock { uint8_t qs[24], qh[2]; half d; };
struct SideBlock { uint8_t q[32]; half d; };
static_assert(sizeof(BaseBlock) == 28 && sizeof(SideBlock) == 34);
static void ck(cudaError_t e) { if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);} }
static uint8_t reference_code(const BaseBlock &b, int e) {
    static const uint8_t p3[5]={1,3,9,27,81}; uint8_t byte; int digit;
    if(e<80){byte=b.qs[e&15];digit=e>>4;}
    else if(e<120){int t=e-80;byte=b.qs[16+(t&7)];digit=t>>3;}
    else {int t=e-120;byte=b.qh[t&1];digit=t>>1;}
    return (uint8_t)((((byte*p3[digit])&255)*3)>>8);
}

// Exact PTQ1_0 trit digit (0, 1, 2) at canonical dequantized element e.
static __host__ __device__ __forceinline__ uint8_t base_code(const BaseBlock &b, int e) {
    uint8_t x; int n;
    if (e < 80) { x = b.qs[e & 15]; n = e >> 4; }
    else if (e < 120) { int t=e-80; x=b.qs[16+(t&7)]; n=t>>3; }
    else { int t=e-120; x=b.qh[t&1]; n=t>>1; }
    uint32_t v=x;
    #pragma unroll
    for(int i=0;i<4;i++) if(i<n) v=(v*3)&255;
    return (uint8_t)((v*3)>>8);
}
__global__ void pack(const BaseBlock *src, SideBlock *dst, int n) {
    int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=n)return;
    // Guarded byte index: exactly 32 packed bytes per block, four codes each.
    #pragma unroll
    for(int j=0;j<32;j++) {
        uint8_t x=0;
        #pragma unroll
        for(int k=0;k<4;k++) x |= (base_code(src[b],4*j+k)&3) << (2*k);
        dst[b].q[j]=x;
    }
    dst[b].d=src[b].d;
}
__device__ __forceinline__ int side_code(const SideBlock &b, int e) {
    return (b.q[e>>2] >> (2*(e&3))) & 3;
}
__global__ void verify(const BaseBlock *src,const SideBlock *dst,int *bad,int n) {
    int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=n)return;
    int v=0; for(int e=0;e<128;e++) v += base_code(src[b],e)!=side_code(dst[b],e); bad[b]=v;
}

#define GGML_CUDA_PTQ1_Q8_GROUP_WORDS (32*36)
// Production SOA_ISUM layout: 32 K-blocks are transposed by 32 lanes, then
// each block contributes 8 q8 words (4 bytes each) before its 4 ds words.
__device__ __forceinline__ size_t act_offset(int b, int e) {
    int kb=b>>2, sub=b&3, group=kb>>5, lane=kb&31, word=sub*8+(e>>2);
    return (size_t)group*GGML_CUDA_PTQ1_Q8_GROUP_WORDS*4 + ((size_t)word*32+lane)*4+(e&3);
}
// Activation and correction are exactly per 32 K values as in SOA_ISUM.
template<bool SIDE> __global__ void dot(const BaseBlock *base,const SideBlock *side,const int8_t *act,const int16_t *isums,const half *scales,float *out,int n) {
    int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=n)return;
    int sums[4]={};
    #pragma unroll
    for(int e=0;e<128;e++) {
        int code = SIDE ? side_code(side[b],e) : base_code(base[b],e);
        int a=act[act_offset(b,e)];
        sums[e/32] += code*a;
    }
    float acc=0.0f; const half d=SIDE?side[b].d:base[b].d;
    #pragma unroll
    for(int k=0;k<4;k++) acc += __half2float(scales[b*4+k])*(float)(sums[k]-isums[b*4+k]);
    out[b]=__half2float(d)*acc;
}

int main(int argc,char **argv) {
    int n=argc>1?atoi(argv[1]):65536, reps=argc>2?atoi(argv[2]):300;
    std::mt19937 rng(615006); std::uniform_int_distribution<int> bd(0,255),ad(-127,127);
    std::vector<BaseBlock> hb(n); std::vector<int8_t> ha((size_t)((n+127)/128)*GGML_CUDA_PTQ1_Q8_GROUP_WORDS*4); std::vector<int16_t> hs((size_t)n*4);
    for(auto &b:hb){for(auto &x:b.qs)x=bd(rng);for(auto &x:b.qh)x=bd(rng);b.d=__float2half(0.125f);}
    // Host-side canonical reference gate, independent of the device accessor.
    long host_checked=0; for(const auto &b:hb) for(int e=0;e<128;e++){ if(base_code(b,e)!=reference_code(b,e)){fprintf(stderr,"host reference mismatch\n");return 4;} host_checked++; }
    for(int b=0;b<n;b++) for(int g=0;g<4;g++){int sum=0;for(int e=g*32;e<(g+1)*32;e++){int8_t x=(int8_t)ad(rng); int kb=b>>2,sub=b&3,group=kb>>5,lane=kb&31,word=sub*8+(e>>2);size_t off=(size_t)group*GGML_CUDA_PTQ1_Q8_GROUP_WORDS*4+((size_t)word*32+lane)*4+(e&3);ha[off]=x;sum+=x;}hs[b*4+g]=sum;}
    std::vector<half> hscale((size_t)n*4); for(auto &x:hscale)x=__float2half(0.03125f);
    BaseBlock *db;SideBlock *ds;int8_t *da;int16_t *di;half *dy;float *d0,*d1;int *dv;
    ck(cudaMalloc(&db,(size_t)n*sizeof(BaseBlock)));ck(cudaMalloc(&ds,(size_t)n*sizeof(SideBlock)));ck(cudaMalloc(&da,ha.size()));ck(cudaMalloc(&di,hs.size()*2));ck(cudaMalloc(&dy,hscale.size()*2));ck(cudaMalloc(&d0,n*4));ck(cudaMalloc(&d1,n*4));ck(cudaMalloc(&dv,n*4));
    ck(cudaMemcpy(db,hb.data(),hb.size()*sizeof(BaseBlock),cudaMemcpyHostToDevice));ck(cudaMemcpy(da,ha.data(),ha.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(di,hs.data(),hs.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dy,hscale.data(),hscale.size()*2,cudaMemcpyHostToDevice));
    dim3 block(128),grid((n+127)/128);pack<<<grid,block>>>(db,ds,n);verify<<<grid,block>>>(db,ds,dv,n);dot<false><<<grid,block>>>(db,ds,da,di,dy,d0,n);dot<true><<<grid,block>>>(db,ds,da,di,dy,d1,n);ck(cudaDeviceSynchronize());
    std::vector<float>a(n),z(n);std::vector<int>v(n);ck(cudaMemcpy(a.data(),d0,n*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(z.data(),d1,n*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(v.data(),dv,n*4,cudaMemcpyDeviceToHost));int bad=0,vbad=0;float maxerr=0;for(int i=0;i<n;i++){bad+=a[i]!=z[i];vbad+=v[i];maxerr=fmaxf(maxerr,fabsf(a[i]-z[i]));}
    if(bad||vbad){fprintf(stderr,"FAIL output mismatches=%d code mismatches=%d maxerr=%g\n",bad,vbad,maxerr);return 3;}
    cudaEvent_t s,e;ck(cudaEventCreate(&s));ck(cudaEventCreate(&e));
    for(int v=0;v<2;v++){float ms;for(int w=0;w<30;w++){if(v)dot<true><<<grid,block>>>(db,ds,da,di,dy,d1,n);else dot<false><<<grid,block>>>(db,ds,da,di,dy,d0,n);}ck(cudaDeviceSynchronize());ck(cudaEventRecord(s));for(int r=0;r<reps;r++){if(v)dot<true><<<grid,block>>>(db,ds,da,di,dy,d1,n);else dot<false><<<grid,block>>>(db,ds,da,di,dy,d0,n);}ck(cudaEventRecord(e));ck(cudaEventSynchronize(e));ck(cudaEventElapsedTime(&ms,s,e));printf("%s %.7f ms/launch n=%d reps=%d\n",v?"2-bit-packed":"ptq1-base3",ms/reps,n,reps);}
    printf("verified_codes=%d host_reference_codes=%ld verified_outputs=%d max_abs_error=%g base_block=%zu side_block=%zu payload_delta=%.2f%%\n",n*128,host_checked,n,maxerr,sizeof(BaseBlock),sizeof(SideBlock),100.0*(sizeof(SideBlock)/(double)sizeof(BaseBlock)-1));
}
