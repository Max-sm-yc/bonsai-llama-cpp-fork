#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <algorithm>

struct B { uint8_t qs[24], qh[2]; half d; };
static void ck(cudaError_t e) { if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);} }
__device__ __forceinline__ uint8_t trit(const B &b, int e) {
    uint8_t x; int n;
    if(e<80){x=b.qs[e&15];n=e>>4;} else if(e<120){int t=e-80;x=b.qs[16+(t&7)];n=t>>3;} else {int t=e-120;x=b.qh[t&1];n=t>>1;}
    uint32_t v=x; for(int i=0;i<n;i++)v=(v*3)&255; return (uint8_t)((v*3)>>8);
}
__device__ __forceinline__ int word(const int *y,int b,int nb,int w){return y[((w>>2)*nb+b)*4+(w&3)];}
__global__ void base(const B *bq,const int *y,float *o,int nb){int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=nb)return;int s[4]={};
  #pragma unroll
  for(int e=0;e<128;e+=4){uint32_t q=trit(bq[b],e)|((uint32_t)trit(bq[b],e+1)<<8)|((uint32_t)trit(bq[b],e+2)<<16)|((uint32_t)trit(bq[b],e+3)<<24);s[e>>5]=__dp4a((int)q,word(y,b,nb,e>>2),s[e>>5]);}
  float a=0;for(int k=0;k<4;k++){uint32_t ds=word(y,b,nb,32+k);a=__fmaf_rn(__half2float(__ushort_as_half(ds&65535)),(float)(s[k]-(int16_t)(ds>>16)),a);}o[b]=__half2float(bq[b].d)*a;
}
// Four lanes cooperate on one block; each lane owns one 32-weight activation
// sub-block, forms eight DP4A words, applies its exact isum correction, then
// a four-lane reduction combines the scaled sub-blocks.
__global__ void coop4(const B *bq,const int *y,float *o,int nb){int lane=threadIdx.x&31,grp=lane>>2,sub=lane&3;int b=blockIdx.x*32+(threadIdx.x>>5)*8+grp;if(b>=nb)return;
  int s=0;
  #pragma unroll
  for(int z=0;z<8;z++){int e=sub*32+z*4;uint32_t q=trit(bq[b],e)|((uint32_t)trit(bq[b],e+1)<<8)|((uint32_t)trit(bq[b],e+2)<<16)|((uint32_t)trit(bq[b],e+3)<<24);s=__dp4a((int)q,word(y,b,nb,e>>2),s);}
  uint32_t ds=word(y,b,nb,32+sub);float v=__fmul_rn(__half2float(__ushort_as_half(ds&65535)),(float)(s-(int16_t)(ds>>16)));
  for(int d=2;d;d>>=1){float x=__shfl_xor_sync(0xffffffff,v,d,4);v+=x;}
  if(sub==0)o[b]=__half2float(bq[b].d)*v;
}
int main(int argc,char**argv){int nb=argc>1?atoi(argv[1]):16384,reps=argc>2?atoi(argv[2]):200;std::mt19937 r(23);std::uniform_int_distribution<int> u(0,255),av(-127,127);std::vector<B> h(nb);std::vector<int> hy((size_t)nb*9*4);
 for(auto &b:h){for(auto &x:b.qs)x=u(r);for(auto &x:b.qh)x=u(r);b.d=__float2half(.125f);}for(int b=0;b<nb;b++){for(int w=0;w<32;w++){uint32_t x=0;for(int k=0;k<4;k++)x|=(uint32_t)(uint8_t)(int8_t)av(r)<<(8*k);hy[((w>>2)*nb+b)*4+(w&3)]=(int)x;}for(int sub=0;sub<4;sub++){int isum=0;for(int e=sub*32;e<sub*32+32;e++){int x=hy[((e/16)*nb+b)*4+((e/4)&3)];isum+=(int8_t)((uint32_t)x>>(8*(e&3)));}uint32_t ds=(uint16_t)__half_as_ushort(__float2half(.03125f))|((uint32_t)(uint16_t)(int16_t)isum<<16);hy[(8*nb+b)*4+sub]=(int)ds;}}
 B *db;int *dy;float *a,*c;ck(cudaMalloc(&db,nb*sizeof(B)));ck(cudaMalloc(&dy,hy.size()*4));ck(cudaMalloc(&a,nb*4));ck(cudaMalloc(&c,nb*4));ck(cudaMemcpy(db,h.data(),nb*sizeof(B),cudaMemcpyHostToDevice));ck(cudaMemcpy(dy,hy.data(),hy.size()*4,cudaMemcpyHostToDevice));base<<<(nb+127)/128,128>>>(db,dy,a,nb);coop4<<<(nb+31)/32,128>>>(db,dy,c,nb);ck(cudaDeviceSynchronize());std::vector<float> ha(nb),hc(nb);ck(cudaMemcpy(ha.data(),a,nb*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hc.data(),c,nb*4,cudaMemcpyDeviceToHost));int bad=0;float maxe=0;for(int i=0;i<nb;i++){bad+=ha[i]!=hc[i];maxe=fmaxf(maxe,fabsf(ha[i]-hc[i]));}printf("coop4 block results bitwise_mismatch=%d/%d max_abs_error=%g\n",bad,nb,maxe);if(bad)return 3;
 cudaEvent_t s,e;ck(cudaEventCreate(&s));ck(cudaEventCreate(&e));for(int v=0;v<2;v++){std::vector<float> ts;for(int q=0;q<9;q++){if((q+v)&1){base<<<(nb+127)/128,128>>>(db,dy,a,nb);coop4<<<(nb+31)/32,128>>>(db,dy,c,nb);}else{coop4<<<(nb+31)/32,128>>>(db,dy,c,nb);base<<<(nb+127)/128,128>>>(db,dy,a,nb);}ck(cudaDeviceSynchronize());ck(cudaEventRecord(s));for(int j=0;j<reps;j++){if(v)coop4<<<(nb+31)/32,128>>>(db,dy,c,nb);else base<<<(nb+127)/128,128>>>(db,dy,a,nb);}ck(cudaEventRecord(e));ck(cudaEventSynchronize(e));float ms;ck(cudaEventElapsedTime(&ms,s,e));ts.push_back(ms/reps);}std::sort(ts.begin(),ts.end());printf("%s median_ms=%.8f samples",v?"coop4":"base",ts[4]);for(float x:ts)printf(" %.8f",x);puts("");}return 0;}
