#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
__device__ __forceinline__ uint32_t step(uint32_t &lo,uint32_t &hi){uint32_t a=lo*3,b=hi*3;lo=a&0x00FF00FF;hi=b&0x00FF00FF;return __byte_perm(a,b,0x7531);}
__device__ __forceinline__ void pairstep(uint32_t &lo,uint32_t &hi,uint32_t &q0,uint32_t &q1){uint32_t a=lo*9,b=hi*9;uint32_t q=__byte_perm(a,b,0x7531);lo=a&0x00FF00FF;hi=b&0x00FF00FF;unsigned char x[4],y[4];
#pragma unroll
 for(int i=0;i<4;i++){unsigned v=(q>>(8*i))&255;unsigned d=v/3;x[i]=d;y[i]=v-d*3;}q0=x[0]|(uint32_t(x[1])<<8)|(uint32_t(x[2])<<16)|(uint32_t(x[3])<<24);q1=y[0]|(uint32_t(y[1])<<8)|(uint32_t(y[2])<<16)|(uint32_t(y[3])<<24);}
__global__ void gate(uint32_t *in,uint32_t *outs){int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=256)return;uint32_t x=in[idx];uint32_t lo=__byte_perm(x,0,0x4140),hi=__byte_perm(x,0,0x4342),a,b;pairstep(lo,hi,a,b);outs[idx*2]=a;outs[idx*2+1]=b;}
int main(){std::vector<uint32_t> in(256),out(512);for(int x=0;x<256;x++)in[x]=uint32_t(x)|((uint32_t)((x+17)&255)<<8)|((uint32_t)((x+91)&255)<<16)|((uint32_t)((x+203)&255)<<24);uint32_t *di,*doo;ck(cudaMalloc(&di,256*4));ck(cudaMalloc(&doo,512*4));ck(cudaMemcpy(di,in.data(),256*4,cudaMemcpyHostToDevice));gate<<<2,128>>>(di,doo);ck(cudaDeviceSynchronize());ck(cudaMemcpy(out.data(),doo,512*4,cudaMemcpyDeviceToHost));int bad=0;for(int x=0;x<256;x++){uint32_t q0=0,q1=0;for(int lane=0;lane<4;lane++){uint32_t z=(in[x]>>(lane*8))&255;uint32_t d0=z*3>>8;z=z*3&255;uint32_t d1=z*3>>8;q0|=d0<<(8*lane);q1|=d1<<(8*lane);}if(out[x*2]!=q0||out[x*2+1]!=q1){if(bad++<5)printf("x=%d got=%08x/%08x expected=%08x/%08x\n",x,out[x*2],out[x*2+1],q0,q1);}}printf("gate mismatches=%d\n",bad);return bad?1:0;}
