#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
__global__ void gate(uint32_t*out){uint32_t id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=65536)return;uint32_t v=(id&255)|((id>>8)<<16);for(int t=0;t<4;t+=2){uint32_t a=v*3;v=a&0x00ff00ff;uint32_t b=v*3;v=b&0x00ff00ff;out[id*2+t/2]=__byte_perm(a,b,0x7531);}}
int main(){std::vector<uint32_t>o(131072);uint32_t*d;ck(cudaMalloc(&d,o.size()*4));gate<<<512,128>>>(d);ck(cudaDeviceSynchronize());ck(cudaMemcpy(o.data(),d,o.size()*4,cudaMemcpyDeviceToHost));long bad=0;for(int id=0;id<65536;id++){uint32_t v[2]={uint32_t(id&255),uint32_t(id>>8)},exp[2]={};for(int t=0;t<4;t++)for(int s=0;s<2;s++){uint32_t digit=v[s]*3>>8;v[s]=v[s]*3&255;exp[t/2]|=digit<<(8*((t%2)*2+s));}for(int j=0;j<2;j++)if(o[id*2+j]!=exp[j]){if(bad++<4)printf("qh mismatch pair=%04x vector=%d got=%08x expected=%08x\n",id,j,o[id*2+j],exp[j]);}}printf("qh byte pairs=65536 interleaved vectors=131072 mismatches=%ld\n",bad);return bad?1:0;}
