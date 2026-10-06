#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
struct B { uint8_t qs[24]; };
static void ck(cudaError_t x){if(x!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(x));exit(2);}}
__device__ __forceinline__ uint32_t get4(const uint8_t*p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
__device__ __forceinline__ void init(uint32_t p,uint32_t&lo,uint32_t&hi){lo=__byte_perm(p,0,0x4140);hi=__byte_perm(p,0,0x4342);}
__device__ __forceinline__ uint32_t step(uint32_t&lo,uint32_t&hi){uint32_t a=lo*3,b=hi*3;lo=a&0x00ff00ff;hi=b&0x00ff00ff;return __byte_perm(a,b,0x7531);}
__device__ __forceinline__ void pairstep(uint32_t&lo,uint32_t&hi,uint32_t&q0,uint32_t&q1){uint32_t a=lo*9,b=hi*9,q=__byte_perm(a,b,0x7531);lo=a&0x00ff00ff;hi=b&0x00ff00ff;uint32_t x[4];for(int j=0;j<4;j++){uint32_t v=(q>>(8*j))&255; x[j]=v/3;}q0=x[0]|(x[1]<<8)|(x[2]<<16)|(x[3]<<24);for(int j=0;j<4;j++){uint32_t v=(q>>(8*j))&255; x[j]=v%3;}q1=x[0]|(x[1]<<8)|(x[2]<<16)|(x[3]<<24);}
// One thread handles one block, six qs byte-groups, and each valid pair position.
__global__ void run(const B* b,uint32_t*out,int n){int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=n*18)return;int block=id/18,slot=id%18,g=slot/3,pair=slot%3;const uint8_t*src=b[block].qs+(g<4?4*g:16+4*(g-4));uint32_t lo,hi;init(get4(src),lo,hi);for(int p=0;p<pair;p++){uint32_t a,c;pairstep(lo,hi,a,c);}uint32_t q0,q1;pairstep(lo,hi,q0,q1);uint32_t*o=out+id*4;o[0]=q0;o[1]=q1;o[2]=lo;o[3]=hi;}
static uint32_t getword(const uint8_t*p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
int main(){constexpr int n=257;std::mt19937 rng(812);std::uniform_int_distribution<int>d(0,255);std::vector<B>h(n);for(auto&b:h)for(auto&x:b.qs)x=d(rng);std::vector<uint32_t>o(n*18*4);B*db;uint32_t*do_;ck(cudaMalloc(&db,n*sizeof(B)));ck(cudaMalloc(&do_,o.size()*4));ck(cudaMemcpy(db,h.data(),n*sizeof(B),cudaMemcpyHostToDevice));run<<<(n*18+127)/128,128>>>(db,do_,n);ck(cudaDeviceSynchronize());ck(cudaMemcpy(o.data(),do_,o.size()*4,cudaMemcpyDeviceToHost));int mismatch=0;long checked=0;
 for(int b=0;b<n;b++)for(int g=0;g<6;g++){int off=g<4?4*g:16+4*(g-4);for(int p=0;p<3;p++){int ix=(b*18+g*3+p)*4;uint32_t ex0=0,ex1=0,exlo=0,exhi=0;for(int lane=0;lane<4;lane++){uint32_t x=h[b].qs[off+lane];for(int t=0;t<p*2;t++)x=(x*3)&255;uint32_t d0=((x*3)>>8);x=(x*3)&255;uint32_t d1=((x*3)>>8);x=(x*3)&255;ex0|=d0<<(8*lane);ex1|=d1<<(8*lane);if(lane%2==0)exlo|=x<<(16*(lane/2));else exhi|=x<<(8*(lane/2));}
 // recurrences packed lo/hi put even input bytes in byte0/2 and odd in byte0/2.
 uint32_t wantlo=0,wanthi=0;for(int lane=0;lane<4;lane++){uint32_t x=h[b].qs[off+lane];for(int t=0;t<(p+1)*2;t++)x=(x*3)&255;if(lane<2)wantlo|=x<<(16*lane);else wanthi|=x<<(16*(lane-2));}
 uint32_t gotlo=o[ix+2],gothi=o[ix+3];
 if(o[ix]!=ex0||o[ix+1]!=ex1||gotlo!=wantlo||gothi!=wanthi){if(mismatch++<12)printf("device mismatch block=%d stream=%c group=%d pair=%d q=%08x/%08x exp=%08x/%08x rem=%08x/%08x exp=%08x/%08x\n",b,g<4?'A':'B',g<4?g:g-4,p,o[ix],o[ix+1],ex0,ex1,gotlo,gothi,wantlo,wanthi);}
 // Verify each stream's production word and accumulator bucket placement for both emitted vectors.
 for(int j=0;j<2;j++){int digit=2*p+j;int word=g<4?4*digit+g:20+2*digit+(g-4);int elem=g<4?16*digit+4*g:80+8*digit+4*(g-4);int bucket=elem/32;if(word!=(elem>>2)||bucket!=(elem>>5)){if(mismatch++<12)printf("map mismatch stream=%c group=%d pair=%d digit=%d word=%d expectedword=%d bucket=%d expectedbucket=%d\n",g<4?'A':'B',g<4?g:g-4,p,digit,word,elem>>2,bucket,elem>>5);}}
 checked+=4;}}
 printf("groups=%d repeated_pair_checks=%ld emitted_digit_vectors=%ld mismatches=%d\n",6,checked,checked*2,mismatch);return mismatch?1:0;}
