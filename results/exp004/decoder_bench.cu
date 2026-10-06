#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

// Each packed byte represents five base-3 digits, matching PTQ1_0 qs layout.
__constant__ unsigned char trits[256][5];
__device__ __forceinline__ int dp4a(int a, int b, int c) { return __dp4a(a,b,c); }

__global__ void run(const uint8_t *packed, const int8_t *act, float *out, int n, int use_lut) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int acc = 0;
  // Six groups of four qs bytes encode 120 trits. qh is excluded to isolate
  // the base-3 qs decoder under test.
  #pragma unroll
  for (int g=0;g<6;g++) {
    uint32_t p=*reinterpret_cast<const uint32_t*>(packed+i*32+g*4);
    uint32_t lo=__byte_perm(p,0,0x4140), hi=__byte_perm(p,0,0x4342);
    #pragma unroll
    for(int t=0;t<5;t++) {
      uint32_t q;
      if(use_lut) {
        q=(uint32_t)trits[p&255][t] | ((uint32_t)trits[(p>>8)&255][t]<<8) |
          ((uint32_t)trits[(p>>16)&255][t]<<16) | ((uint32_t)trits[(p>>24)&255][t]<<24);
      } else {
        uint32_t wlo=lo*3u, whi=hi*3u;
        q=__byte_perm(wlo,whi,0x7531);
        lo=wlo&0x00ff00ff; hi=whi&0x00ff00ff;
      }
      acc=dp4a((int)q,*reinterpret_cast<const int*>(act+i*128+g*20+t*4),acc)-dp4a(0x01010101,*reinterpret_cast<const int*>(act+i*128+g*20+t*4),0);
    }
  }
  out[i]=(float)acc;
}
static void ck(cudaError_t e) { if(e!=cudaSuccess) { fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); exit(2); } }
int main(int argc,char**argv) {
  int n=argc>1?atoi(argv[1]):65536, reps=argc>2?atoi(argv[2]):100;
  unsigned char htab[256][5];
  for(int p=0;p<256;p++){int x=p;for(int t=0;t<5;t++){int w=x*3;htab[p][t]=w>>8;x=w&255;}}
  ck(cudaMemcpyToSymbol(trits,htab,sizeof(htab)));
  std::mt19937 rng(123); std::uniform_int_distribution<int> pd(0,242), ad(-127,127);
  std::vector<uint8_t> hp((size_t)n*32); std::vector<int8_t> ha((size_t)n*128);
  for(int i=0;i<n;i++){for(int b=0;b<24;b++)hp[i*32+b]=pd(rng);for(int b=24;b<32;b++)hp[i*32+b]=0;} for(auto &x:ha)x=ad(rng);
  uint8_t *dp; int8_t *da; float *d0,*d1;
  ck(cudaMalloc(&dp,hp.size()));ck(cudaMalloc(&da,ha.size()));ck(cudaMalloc(&d0,n*sizeof(float)));ck(cudaMalloc(&d1,n*sizeof(float)));
  ck(cudaMemcpy(dp,hp.data(),hp.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(da,ha.data(),ha.size(),cudaMemcpyHostToDevice));
  dim3 block(128), grid((n+127)/128);
  run<<<grid,block>>>(dp,da,d0,n,0);run<<<grid,block>>>(dp,da,d1,n,1);ck(cudaDeviceSynchronize());
  std::vector<float> a(n),b(n);ck(cudaMemcpy(a.data(),d0,n*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),d1,n*4,cudaMemcpyDeviceToHost));
  int bad=0; for(int i=0;i<n;i++) { if(a[i]!=b[i] && bad<3) fprintf(stderr,"mismatch i=%d base=%g lut=%g\n",i,a[i],b[i]); bad += a[i]!=b[i]; }
  cudaEvent_t s,e;ck(cudaEventCreate(&s));ck(cudaEventCreate(&e));
  for(int v=0;v<2;v++){float ms=0;run<<<grid,block>>>(dp,da,v?d1:d0,n,v);ck(cudaDeviceSynchronize());ck(cudaEventRecord(s));for(int r=0;r<reps;r++)run<<<grid,block>>>(dp,da,v?d1:d0,n,v);ck(cudaEventRecord(e));ck(cudaEventSynchronize(e));ck(cudaEventElapsedTime(&ms,s,e));printf("%s %.6f ms/launch (n=%d reps=%d)\n",v?"lut":"multiply",ms/reps,n,reps);}
  printf("dot mismatches %d\n",bad);
}
