#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <algorithm>
struct B { uint8_t qs[24], qh[2]; half d; };
__device__ __forceinline__ uint32_t step(uint32_t &lo,uint32_t &hi){uint32_t a=lo*3,b=hi*3;lo=a&0x00ff00ff;hi=b&0x00ff00ff;return __byte_perm(a,b,0x7531);}
__device__ __forceinline__ uint32_t pack4(const uint8_t*p){volatile const uint8_t *v=p;return (uint32_t)v[0]|((uint32_t)v[1]<<8)|((uint32_t)v[2]<<16)|((uint32_t)v[3]<<24);}
__device__ __forceinline__ uint32_t initlo(uint32_t p){return __byte_perm(p,0,0x4140);}
__device__ __forceinline__ uint32_t inithi(uint32_t p){return __byte_perm(p,0,0x4342);}
__device__ __forceinline__ int act(const int*y,int nb,int b,int w){return y[((w>>2)*nb+b)*4+(w&3)];}
__device__ __forceinline__ int4 act4(const int*y,int nb,int b,int plane){return *((const int4*)y+plane*nb+b);}
__device__ __forceinline__ int byteat(int4 v,int k){return ((int*)&v)[k&3];}
__device__ __forceinline__ uint32_t packedstep(uint32_t p,int t){uint32_t lo=initlo(p),hi=inithi(p),q=0;for(int i=0;i<=t;i++)q=step(lo,hi);return q;}
__global__ void serial(const B*bq,const int*y,float*out,int nb){int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=nb)return;int s[4]={};
  #pragma unroll
  for(int g=0;g<4;g++){uint32_t p=pack4(bq[b].qs+4*g),lo=initlo(p),hi=inithi(p);for(int t=0;t<5;t++){uint32_t q=step(lo,hi);int w=4*t+g;s[(16*t+4*g)>>5]=__dp4a((int)q,act(y,nb,b,w),s[(16*t+4*g)>>5]);}}
  #pragma unroll
  for(int g=0;g<2;g++){uint32_t p=pack4(bq[b].qs+16+4*g),lo=initlo(p),hi=inithi(p);for(int t=0;t<5;t++){uint32_t q=step(lo,hi);int e=80+8*t+4*g,w=20+2*t+g;s[e>>5]=__dp4a((int)q,act(y,nb,b,w),s[e>>5]);}}
  uint32_t v=bq[b].qh[0]|((uint32_t)bq[b].qh[1]<<16);for(int t=0;t<2;t++){uint32_t a=v*3;v=a&0x00ff00ff;uint32_t c=v*3;v=c&0x00ff00ff;uint32_t q=__byte_perm(a,c,0x7531);s[3]=__dp4a((int)q,act(y,nb,b,30+2*t),s[3]);}
  float acc=0;for(int k=0;k<4;k++){uint32_t ds=act(y,nb,b,32+k);acc=__fmaf_rn(__half2float(__ushort_as_half(ds&65535)),float(s[k]-(int16_t)(ds>>16)),acc);}out[b]=__half2float(bq[b].d)*acc;
}
// Eight lanes cooperate per block: lanes 0..3 run qs[0..15] group recurrences,
// 4..5 run qs[16..23], lane 6 runs qh, lane 7 is idle. Partial DP4A integer
// sums reduce by sub-block before exact isum correction and ordered scale FMA.
__global__ void coop8(const B*bq,const int*y,float*out,int nb){int lane=threadIdx.x&31,sub=lane&7,grp=lane>>3,b=blockIdx.x*16+(threadIdx.x>>5)*4+grp;if(b>=nb)return;int s[4]={};
 if(sub<4){uint32_t p=pack4(bq[b].qs+4*sub),lo=initlo(p),hi=inithi(p);for(int t=0;t<5;t++){uint32_t q=step(lo,hi);int e=16*t+4*sub,w=4*t+sub;s[e>>5]=__dp4a((int)q,act(y,nb,b,w),s[e>>5]);}}
 else if(sub<6){int g=sub-4;uint32_t p=pack4(bq[b].qs+16+4*g),lo=initlo(p),hi=inithi(p);for(int t=0;t<5;t++){uint32_t q=step(lo,hi);int e=80+8*t+4*g,w=20+2*t+g;s[e>>5]=__dp4a((int)q,act(y,nb,b,w),s[e>>5]);}}
 else if(sub==6){uint32_t v=bq[b].qh[0]|((uint32_t)bq[b].qh[1]<<16);for(int t=0;t<2;t++){uint32_t a=v*3;v=a&0x00ff00ff;uint32_t c=v*3;v=c&0x00ff00ff;uint32_t q=__byte_perm(a,c,0x7531);s[3]=__dp4a((int)q,act(y,nb,b,30+2*t),s[3]);}}
 #pragma unroll
 for(int k=0;k<4;k++){int z=__shfl_down_sync(0xffffffff,s[k],4,8);s[k]+=z;z=__shfl_down_sync(0xffffffff,s[k],2,8);s[k]+=z;z=__shfl_down_sync(0xffffffff,s[k],1,8);s[k]+=z;}
 if(sub==0){float a=0;for(int k=0;k<4;k++){uint32_t ds=act(y,nb,b,32+k);a=__fmaf_rn(__half2float(__ushort_as_half(ds&65535)),float(s[k]-(int16_t)(ds>>16)),a);}out[b]=__half2float(bq[b].d)*a;}
}
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}
int main(int ac,char**av){int nb=ac>1?atoi(av[1]):16384,reps=ac>2?atoi(av[2]):200;std::mt19937 r(24);std::uniform_int_distribution<int> u(0,255),q(-127,127);std::vector<B> hb(nb);std::vector<int> hy((size_t)nb*36);for(auto&b:hb){for(auto&x:b.qs)x=u(r);for(auto&x:b.qh)x=u(r);b.d=__float2half(.125f);}for(int b=0;b<nb;b++){for(int w=0;w<32;w++){uint32_t x=0;for(int j=0;j<4;j++)x|=(uint32_t)(uint8_t)(int8_t)q(r)<<(8*j);hy[((w>>2)*nb+b)*4+(w&3)]=(int)x;}for(int k=0;k<4;k++){int sum=0;for(int j=0;j<32;j++){int e=k*32+j,w=e/4;sum+=(int8_t)((uint32_t)hy[((w>>2)*nb+b)*4+(w&3)]>>(8*(e&3)));}uint32_t ds=(uint16_t)__half_as_ushort(__float2half(.03125f))|((uint32_t)(uint16_t)(int16_t)sum<<16);hy[32*nb+b*4+k]=(int)ds;}}
 B*d;int*y;float*a,*c;ck(cudaMalloc(&d,nb*sizeof(B)));ck(cudaMalloc(&y,hy.size()*4));ck(cudaMalloc(&a,nb*4));ck(cudaMalloc(&c,nb*4));ck(cudaMemcpy(d,hb.data(),nb*sizeof(B),cudaMemcpyHostToDevice));ck(cudaMemcpy(y,hy.data(),hy.size()*4,cudaMemcpyHostToDevice));serial<<<(nb+127)/128,128>>>(d,y,a,nb);ck(cudaDeviceSynchronize());fprintf(stderr,"serial ok\n");coop8<<<(nb+15)/16,128>>>(d,y,c,nb);ck(cudaDeviceSynchronize());fprintf(stderr,"coop ok\n");std::vector<float>ha(nb),hc(nb);ck(cudaMemcpy(ha.data(),a,nb*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hc.data(),c,nb*4,cudaMemcpyDeviceToHost));int bad=0;for(int i=0;i<nb;i++)bad+=ha[i]!=hc[i];printf("bitwise mismatches %d/%d\n",bad,nb);if(bad)return 3;cudaEvent_t st,en;ck(cudaEventCreate(&st));ck(cudaEventCreate(&en));for(int v=0;v<2;v++){std::vector<float>ts;for(int z=0;z<9;z++){ck(cudaDeviceSynchronize());ck(cudaEventRecord(st));for(int j=0;j<reps;j++){if(v)coop8<<<(nb+15)/16,128>>>(d,y,c,nb);else serial<<<(nb+127)/128,128>>>(d,y,a,nb);}ck(cudaEventRecord(en));ck(cudaEventSynchronize(en));float ms;ck(cudaEventElapsedTime(&ms,st,en));ts.push_back(ms/reps);}std::sort(ts.begin(),ts.end());printf("%s median %.8f ms samples",v?"coop8":"serial",ts[4]);for(float x:ts)printf(" %.8f",x);puts("");} }
