#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>

struct BaseBlock { uint8_t qs[24], qh[2]; half d; };
struct SideBlock { uint8_t q[32]; half d; };
static_assert(sizeof(BaseBlock)==28 && sizeof(SideBlock)==34);
constexpr int GROUP_WORDS=32*36;
static void ck(cudaError_t e) { if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);} }

static uint8_t code(const BaseBlock &b,int e) {
    uint8_t x; int n;
    if(e<80){x=b.qs[e&15];n=e>>4;}
    else if(e<120){int t=e-80;x=b.qs[16+(t&7)];n=t>>3;}
    else {int t=e-120;x=b.qh[t&1];n=t>>1;}
    uint32_t v=x; for(int i=0;i<n;i++)v=(v*3)&255;
    return (uint8_t)((v*3)>>8);
}
static size_t word_index(int b,int word) {
    const int group=b>>5, lane=b&31;
    return (size_t)group*GROUP_WORDS+(size_t)word*32+lane;
}
static size_t byte_offset(int b,int e) { return word_index(b,e>>2)*4+(e&3); }

__device__ __forceinline__ int dp4a(int a,int b,int c){ return __dp4a(a,b,c); }
__device__ __forceinline__ uint32_t get4(const uint8_t *p) {
    return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}
__device__ __forceinline__ int base_code_dev(const BaseBlock &b,int e) {
    uint8_t x; int n;
    if(e<80){x=b.qs[e&15];n=e>>4;}
    else if(e<120){int t=e-80;x=b.qs[16+(t&7)];n=t>>3;}
    else {int t=e-120;x=b.qh[t&1];n=t>>1;}
    uint32_t v=x;
    #pragma unroll
    for(int i=0;i<n;i++)v=(v*3)&255;
    return (v*3)>>8;
}
__device__ __forceinline__ int side_code_dev(const SideBlock &b,int e){return (b.q[e>>2]>>(2*(e&3)))&3;}

__global__ void convert(const BaseBlock *src,SideBlock *dst,int n) {
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=n)return;
    #pragma unroll
    for(int j=0;j<32;j++){uint8_t v=0;
        #pragma unroll
        for(int k=0;k<4;k++)v|=(base_code_dev(src[b],4*j+k)&3)<<(2*k);
        dst[b].q[j]=v;
    }
    dst[b].d=src[b].d;
}
__global__ void verify_codes(const BaseBlock *src,const SideBlock *side,int *bad,int n){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=n)return;int v=0;
    for(int e=0;e<128;e++)v+=base_code_dev(src[b],e)!=side_code_dev(side[b],e);bad[b]=v;
}

// Transcription of vec_dot_ptq1_0_q8_1_multi's four sums and PTQ1_U address rule.
__global__ void dot_base(const BaseBlock *base,const int *yw,float *out,int n){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=n)return;
    int sums[4]={}; const int *lane_words=yw+(size_t)(b>>5)*GROUP_WORDS+(b&31);
    #pragma unroll
    for(int g=0;g<4;g++){
        uint32_t packed=get4(base[b].qs+4*g);
        uint32_t lo=(uint32_t)(packed&0xFF)|((uint32_t)((packed>>8)&0xFF)<<16);
        uint32_t hi=(uint32_t)((packed>>16)&0xFF)|((uint32_t)(packed>>24)<<16);
        #pragma unroll
        for(int t=0;t<5;t++){
            uint32_t wl=lo*3,wh=hi*3;lo=wl&0x00FF00FF;hi=wh&0x00FF00FF;
            int q=(int)(((wl>>8)&0xFF)|((wl>>16)&0xFF00)|(((wh>>8)&0xFF)<<16)|(wh&0xFF000000));
            int e=t*16+4*g;int u=lane_words[(e>>2)*32];sums[e>>5]=dp4a(q,u,sums[e>>5]);
        }
    }
    #pragma unroll
    for(int g=0;g<2;g++){
        uint32_t packed=get4(base[b].qs+16+4*g);
        uint32_t lo=(uint32_t)(packed&0xFF)|((uint32_t)((packed>>8)&0xFF)<<16);
        uint32_t hi=(uint32_t)((packed>>16)&0xFF)|((uint32_t)(packed>>24)<<16);
        #pragma unroll
        for(int t=0;t<5;t++){
            uint32_t wl=lo*3,wh=hi*3;lo=wl&0x00FF00FF;hi=wh&0x00FF00FF;
            int q=(int)(((wl>>8)&0xFF)|((wl>>16)&0xFF00)|(((wh>>8)&0xFF)<<16)|(wh&0xFF000000));
            int e=80+t*8+4*g;int u=lane_words[(e>>2)*32];sums[e>>5]=dp4a(q,u,sums[e>>5]);
        }
    }
    uint32_t v=(uint32_t)base[b].qh[0]|((uint32_t)base[b].qh[1]<<16);
    #pragma unroll
    for(int t=0;t<4;t+=2){uint32_t w0=v*3;v=w0&0x00FF00FF;uint32_t w1=v*3;v=w1&0x00FF00FF;
        int q=(int)(((w0>>8)&0xFF)|((w0>>16)&0xFF00)|(((w1>>8)&0xFF)<<16)|(w1&0xFF000000));
        int u=lane_words[(30+t/2)*32];sums[3]=dp4a(q,u,sums[3]);
    }
    float acc=0;
    #pragma unroll
    for(int k=0;k<4;k++){int ds=lane_words[(32+k)*32];float scale=__half2float(__ushort_as_half((uint16_t)ds));int isum=(int)(int16_t)(ds>>16);acc+=scale*(float)(sums[k]-isum);}
    out[b]=__half2float(base[b].d)*acc;
}
// Candidate changes only trit decoding: packed 2-bit extraction feeds identical activation words/DP4A.
__global__ void dot_side(const SideBlock *side,const int *yw,float *out,int n){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=n)return;
    int sums[4]={}; const int *lane_words=yw+(size_t)(b>>5)*GROUP_WORDS+(b&31);
    #pragma unroll
    for(int e=0;e<128;e+=4){int q=0;
        #pragma unroll
        for(int k=0;k<4;k++)q|=side_code_dev(side[b],e+k)<<(8*k);
        int u=lane_words[(e>>2)*32];sums[e>>5]=dp4a(q,u,sums[e>>5]);
    }
    float acc=0;
    #pragma unroll
    for(int k=0;k<4;k++){int ds=lane_words[(32+k)*32];float scale=__half2float(__ushort_as_half((uint16_t)ds));int isum=(int)(int16_t)(ds>>16);acc+=scale*(float)(sums[k]-isum);}
    out[b]=__half2float(side[b].d)*acc;
}

int main(int argc,char **argv){
    int n=argc>1?atoi(argv[1]):65536,reps=argc>2?atoi(argv[2]):300;
    if(n<=0||reps<=0)return 2;
    const size_t groups=(n+31)/32, nwords=groups*GROUP_WORDS;
    std::mt19937 rng(615007);std::uniform_int_distribution<int> bd(0,255),ad(-127,127);
    std::vector<BaseBlock> hb(n);for(auto &b:hb){for(auto &x:b.qs)x=bd(rng);for(auto &x:b.qh)x=bd(rng);b.d=__float2half(.125f);}
    long checked=0;for(auto &b:hb)for(int e=0;e<128;e++){int c=code(b,e);if(c>2){fprintf(stderr,"bad host code\n");return 3;}checked++;}
    // Explicit first/last K-block, first/last element and ds address checks.
    const size_t alloc_bytes=nwords*4;
    for(int b: {0,n-1})for(int e:{0,127}){size_t a=byte_offset(b,e);if(a+1>alloc_bytes){fprintf(stderr,"address OOB b=%d e=%d off=%zu alloc=%zu\n",b,e,a,alloc_bytes);return 4;}printf("address_check b=%d e=%d group=%d lane=%d word=%d byte=%d offset=%zu/%zu\n",b,e,b>>5,b&31,e>>2,e&3,a,alloc_bytes);}
    for(int b:{0,n-1})for(int k=0;k<4;k++)if((word_index(b,32+k)+1)*4>alloc_bytes){fprintf(stderr,"ds OOB\n");return 4;}
    std::vector<int> hy(nwords,0);
    for(int b=0;b<n;b++)for(int k=0;k<4;k++){
        int isum=0;for(int j=0;j<32;j++){int e=k*32+j;int8_t a=(int8_t)ad(rng);size_t ix=byte_offset(b,e);auto *raw=(uint8_t*)hy.data();raw[ix]=(uint8_t)a;isum+=a;}
        half sc=__float2half(.03125f);uint32_t ds=(uint16_t)__half_as_ushort(sc)|((uint32_t)(uint16_t)(int16_t)isum<<16);hy[word_index(b,32+k)]=(int)ds;
    }
    BaseBlock *db;SideBlock *ds;int *dy;float *do0,*do1;int *dv;
    ck(cudaMalloc(&db,(size_t)n*sizeof(BaseBlock)));ck(cudaMalloc(&ds,(size_t)n*sizeof(SideBlock)));ck(cudaMalloc(&dy,nwords*4));ck(cudaMalloc(&do0,n*4));ck(cudaMalloc(&do1,n*4));ck(cudaMalloc(&dv,n*4));
    ck(cudaMemcpy(db,hb.data(),hb.size()*sizeof(BaseBlock),cudaMemcpyHostToDevice));ck(cudaMemcpy(dy,hy.data(),nwords*4,cudaMemcpyHostToDevice));
    dim3 block(128),grid((n+127)/128);convert<<<grid,block>>>(db,ds,n);verify_codes<<<grid,block>>>(db,ds,dv,n);dot_base<<<grid,block>>>(db,dy,do0,n);dot_side<<<grid,block>>>(ds,dy,do1,n);ck(cudaDeviceSynchronize());
    std::vector<float>a(n),z(n);std::vector<int>badcode(n);ck(cudaMemcpy(a.data(),do0,n*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(z.data(),do1,n*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(badcode.data(),dv,n*4,cudaMemcpyDeviceToHost));int bc=0,bo=0,bh=0;float maxerr=0;
    for(int i=0;i<n;i++){bc+=badcode[i];bo+=a[i]!=z[i];maxerr=fmaxf(maxerr,fabsf(a[i]-z[i]));float ref=0;
        for(int k=0;k<4;k++){int isum=0,qsum=0;for(int e=32*k;e<32*(k+1);e++){int8_t av=(int8_t)((uint8_t*)hy.data())[byte_offset(i,e)];isum+=av;qsum+=code(hb[i],e)*av;}int dsword=hy[word_index(i,32+k)];float scale=__half2float(__ushort_as_half((uint16_t)dsword));int isum_expected=(int)(int16_t)(dsword>>16);if(isum!=isum_expected){fprintf(stderr,"host isum mismatch b=%d k=%d got=%d expected=%d\n",i,k,isum,isum_expected);return 5;}ref+=scale*(float)(qsum-isum_expected);}
        ref=__half2float(hb[i].d)*ref;if(a[i]!=ref)bh++;maxerr=fmaxf(maxerr,fabsf(a[i]-ref));
    }
    if(bc||bo||bh){fprintf(stderr,"FAIL code_mismatches=%d side_output_mismatches=%d host_output_mismatches=%d maxerr=%g\n",bc,bo,bh,maxerr);return 5;}
    cudaEvent_t s,e;ck(cudaEventCreate(&s));ck(cudaEventCreate(&e));
    std::vector<float> samples[2];
    for(int run=0;run<7;run++)for(int oi=0;oi<2;oi++){
        int v=(run&1)?1-oi:oi;
        for(int w=0;w<40;w++){if(v)dot_side<<<grid,block>>>(ds,dy,do1,n);else dot_base<<<grid,block>>>(db,dy,do0,n);}ck(cudaDeviceSynchronize());
        ck(cudaEventRecord(s));for(int r=0;r<reps;r++){if(v)dot_side<<<grid,block>>>(ds,dy,do1,n);else dot_base<<<grid,block>>>(db,dy,do0,n);}ck(cudaEventRecord(e));ck(cudaEventSynchronize(e));float ms;ck(cudaEventElapsedTime(&ms,s,e));samples[v].push_back(ms/reps);
    }
    for(int v=0;v<2;v++){printf("%s",v?"2-bit-side":"production-base3");for(float x:samples[v])printf(" %.7f",x);printf(" ms/launch n=%d reps=%d\n",n,reps);}
    printf("verified_codes=%ld code_mismatches=%d output_mismatches=%d host_output_mismatches=%d max_abs_error=%g blocks=%d alloc_words=%zu base_bytes=%zu side_bytes=%zu payload_delta=%.2f%%\n",checked,bc,bo,bh,maxerr,n,nwords,sizeof(BaseBlock),sizeof(SideBlock),100.0*(sizeof(SideBlock)/(double)sizeof(BaseBlock)-1));
}
