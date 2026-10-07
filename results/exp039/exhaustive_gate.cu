#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
__device__ __forceinline__ uint32_t ps(uint32_t ol,uint32_t oh,uint32_t &pl,uint32_t &ph,int c){uint32_t fl=((ol*(uint32_t)c)>>8)&0x00ff00ffu,fh=((oh*(uint32_t)c)>>8)&0x00ff00ffu;uint32_t ql=fl-pl*3u,qh=fh-ph*3u;pl=fl;ph=fh;return __byte_perm(ql<<8,qh<<8,0x7531);}
__device__ __forceinline__ uint8_t rec(uint8_t x,int t){uint32_t v=x;for(int i=0;i<t;i++)v=(v*3)&255;return (v*3)>>8;}
__global__ void gate(int *bad){int x=blockIdx.x*blockDim.x+threadIdx.x;if(x<256){uint32_t p=(uint32_t)x|((uint32_t)((x+1)&255)<<8)|((uint32_t)((x+2)&255)<<16)|((uint32_t)((x+3)&255)<<24);uint32_t ol=__byte_perm(p,0,0x4140),oh=__byte_perm(p,0,0x4342),pl=0,ph=0;for(int t=0;t<5;t++){int c=t==0?3:t==1?9:t==2?27:t==3?81:243;uint32_t q=ps(ol,oh,pl,ph,c);for(int l=0;l<4;l++){uint8_t v=(q>>(8*l))&255;uint8_t a=(x+l)&255;if(v!=rec(a,t))atomicAdd(bad,1);}}}int pair=x;if(pair<65536){uint8_t a=pair&255,b=pair>>8;uint32_t v=(uint32_t)a|((uint32_t)b<<16);for(int t=0;t<4;t++){uint32_t w=v*3;v=w&0x00ff00ff;uint32_t w2=v*3;v=w2&0x00ff00ff;uint32_t q=__byte_perm(w,w2,0x7531);uint8_t got[4]={(uint8_t)q,(uint8_t)(q>>8),(uint8_t)(q>>16),(uint8_t)(q>>24)};uint8_t ex[4]={rec(a,t),rec(b,t),rec(a,t+1),rec(b,t+1)};for(int i=0;i<4;i++)if(got[i]!=ex[i])atomicAdd(bad,1);t++;}}
}
int main(){int *d,*h=(int*)calloc(1,sizeof(int));cudaMalloc(&d,sizeof(int));cudaMemset(d,0,sizeof(int));gate<<<512,128>>>(d);cudaDeviceSynchronize();cudaMemcpy(h,d,sizeof(int),cudaMemcpyDeviceToHost);printf("EXHAUSTIVE byte inputs=256 x 4 packed lanes x 5 digits; qh pairs=65536 x 8 interleaved digits; mismatches=%d\n",*h);return *h?1:0;}
