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
static_assert(sizeof(BaseBlock)==28 && sizeof(SideBlock)==34);
static void ck(cudaError_t e) { if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);} }
static uint8_t code(const BaseBlock &b,int e) {
    uint8_t x; int n;
    if(e<80){x=b.qs[e&15];n=e>>4;} else if(e<120){int t=e-80;x=b.qs[16+(t&7)];n=t>>3;} else {int t=e-120;x=b.qh[t&1];n=t>>1;}
    uint32_t v=x; for(int i=0;i<n;i++)v=(v*3)&255; return (uint8_t)((v*3)>>8);
}
__device__ __forceinline__ uint32_t get4(const uint8_t *p) { return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24); }
__device__ __forceinline__ uint32_t step(uint32_t &lo,uint32_t &hi){uint32_t a=lo*3,b=hi*3;lo=a&0x00FF00FF;hi=b&0x00FF00FF;return __byte_perm(a,b,0x7531);}
__device__ __forceinline__ void pairstep(uint32_t &lo,uint32_t &hi,uint32_t &q0,uint32_t &q1){
    uint32_t a=lo*9, b=hi*9;
    uint32_t q=__byte_perm(a,b,0x7531);
    lo=a&0x00FF00FF; hi=b&0x00FF00FF;
    unsigned char z[4];
    #pragma unroll
    for(int i=0;i<4;i++){unsigned v=(q>>(8*i))&255; unsigned d0=v/3; z[i]=(unsigned char)d0;}
    q0=(uint32_t)z[0]|((uint32_t)z[1]<<8)|((uint32_t)z[2]<<16)|((uint32_t)z[3]<<24);
    #pragma unroll
    for(int i=0;i<4;i++){unsigned v=(q>>(8*i))&255; unsigned d0=v/3; unsigned d1=v-d0*3; z[i]=(unsigned char)d1;}
    q1=(uint32_t)z[0]|((uint32_t)z[1]<<8)|((uint32_t)z[2]<<16)|((uint32_t)z[3]<<24);
}
__device__ __forceinline__ int pword(const int *y,int b,int nb,int word) { int plane=word>>2, slot=word&3; return y[((plane*nb+b)*4)+slot]; }
__device__ __forceinline__ int sidecode(const SideBlock &b,int e) { return (b.q[e>>2]>>(2*(e&3)))&3; }
__device__ __forceinline__ uint32_t dec4(uint32_t packed, uint32_t &lo, uint32_t &hi) { lo=__byte_perm(packed,0,0x4140);hi=__byte_perm(packed,0,0x4342);return 0; }

// Same byte expansion and per-32-element dp4a sums as the active planar kernel.
__global__ void dot_base(const BaseBlock *base,const int *y,float *out,int nb){int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=nb)return;int sums[4]={};
    #pragma unroll
    for(int g=0;g<4;g++){uint32_t lo,hi;dec4(get4(base[b].qs+4*g),lo,hi);
        #pragma unroll
        for(int t=0;t<5;t++){uint32_t q=step(lo,hi);int word=(t*16+4*g)>>2;sums[(t*16+4*g)>>5]=__dp4a((int)q,pword(y,b,nb,word),sums[(t*16+4*g)>>5]);}
    }
    #pragma unroll
    for(int g=0;g<2;g++){uint32_t lo,hi;dec4(get4(base[b].qs+16+4*g),lo,hi);
        #pragma unroll
        for(int t=0;t<5;t++){uint32_t q=step(lo,hi);int e=80+t*8+4*g,word=e>>2;sums[e>>5]=__dp4a((int)q,pword(y,b,nb,word),sums[e>>5]);}
    }
    uint32_t v=(uint32_t)base[b].qh[0]|((uint32_t)base[b].qh[1]<<16);
    #pragma unroll
    for(int t=0;t<4;t+=2){uint32_t a=v*3;v=a&0x00FF00FF;uint32_t c=v*3;v=c&0x00FF00FF;uint32_t q=__byte_perm(a,c,0x7531);int word=30+t/2;sums[3]=__dp4a((int)q,pword(y,b,nb,word),sums[3]);}
    float acc=0;
    #pragma unroll
    for(int k=0;k<4;k++){uint32_t ds=(uint32_t)pword(y,b,nb,32+k);float sc=__half2float(__ushort_as_half((uint16_t)ds));int isum=(int)(int16_t)(ds>>16);acc=__fmaf_rn(sc,(float)(sums[k]-isum),acc);}out[b]=__half2float(base[b].d)*acc;
}
__global__ void dot_pair(const BaseBlock *base,const int *y,float *out,int nb){
 int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=nb)return;int sums[4]={};
 #pragma unroll
 for(int g=0;g<4;g++){uint32_t lo,hi;dec4(get4(base[b].qs+4*g),lo,hi);
   #pragma unroll
   for(int t=0;t<4;t+=2){uint32_t q0,q1;pairstep(lo,hi,q0,q1);int w0=t*4+g,w1=(t+1)*4+g;sums[w0>>3]=__dp4a((int)q0,pword(y,b,nb,w0),sums[w0>>3]);sums[w1>>3]=__dp4a((int)q1,pword(y,b,nb,w1),sums[w1>>3]);}
   uint32_t q=step(lo,hi);int w=16+g;sums[w>>3]=__dp4a((int)q,pword(y,b,nb,w),sums[w>>3]);
 }
 #pragma unroll
 for(int g=0;g<2;g++){uint32_t lo,hi;dec4(get4(base[b].qs+16+4*g),lo,hi);
   #pragma unroll
   for(int t=0;t<4;t+=2){uint32_t q0,q1;pairstep(lo,hi,q0,q1);int e0=80+8*t+4*g,e1=80+8*(t+1)+4*g;sums[e0>>5]=__dp4a((int)q0,pword(y,b,nb,e0>>2),sums[e0>>5]);sums[e1>>5]=__dp4a((int)q1,pword(y,b,nb,e1>>2),sums[e1>>5]);}
   uint32_t q=step(lo,hi);int e=112+4*g;sums[e>>5]=__dp4a((int)q,pword(y,b,nb,e>>2),sums[e>>5]);
 }
 // qh: exact production two-stream pair interleave and recurrence.
 uint32_t v=(uint32_t)base[b].qh[0]|((uint32_t)base[b].qh[1]<<16);
 #pragma unroll
 for(int t=0;t<4;t+=2){uint32_t a=v*3;v=a&0x00FF00FF;uint32_t c=v*3;v=c&0x00FF00FF;uint32_t q=__byte_perm(a,c,0x7531);sums[3]=__dp4a((int)q,pword(y,b,nb,30+t/2),sums[3]);}
 float acc=0;
 #pragma unroll
 for(int k=0;k<4;k++){uint32_t ds=(uint32_t)pword(y,b,nb,32+k);float sc=__half2float(__ushort_as_half((uint16_t)ds));int isum=(int)(int16_t)(ds>>16);acc=__fmaf_rn(sc,(float)(sums[k]-isum),acc);}out[b]=__half2float(base[b].d)*acc;
}
__global__ void dot_side(const SideBlock *side,const int *y,float *out,int nb){int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=nb)return;int sums[4]={};
    #pragma unroll
    for(int e=0;e<128;e+=4){uint32_t q=(uint32_t)sidecode(side[b],e)|((uint32_t)sidecode(side[b],e+1)<<8)|((uint32_t)sidecode(side[b],e+2)<<16)|((uint32_t)sidecode(side[b],e+3)<<24);sums[e>>5]=__dp4a((int)q,pword(y,b,nb,e>>2),sums[e>>5]);}
    float acc=0;
    #pragma unroll
    for(int k=0;k<4;k++){uint32_t ds=(uint32_t)pword(y,b,nb,32+k);float sc=__half2float(__ushort_as_half((uint16_t)ds));int isum=(int)(int16_t)(ds>>16);acc=__fmaf_rn(sc,(float)(sums[k]-isum),acc);}out[b]=__half2float(side[b].d)*acc;
}
// Same side block, but load one packed code byte and expand its four 2-bit
// values directly into the byte lanes consumed by one DP4A instruction.
__global__ void dot_side_expand(const SideBlock *side,const int *y,float *out,int nb){int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=nb)return;int sums[4]={};
    #pragma unroll
    for(int e=0;e<128;e+=4){uint32_t v=side[b].q[e>>2];uint32_t q=(v&3u)|((v&12u)<<6)|((v&48u)<<12)|((v&192u)<<18);sums[e>>5]=__dp4a((int)q,pword(y,b,nb,e>>2),sums[e>>5]);}
    float acc=0;
    #pragma unroll
    for(int k=0;k<4;k++){uint32_t ds=(uint32_t)pword(y,b,nb,32+k);float sc=__half2float(__ushort_as_half((uint16_t)ds));int isum=(int)(int16_t)(ds>>16);acc=__fmaf_rn(sc,(float)(sums[k]-isum),acc);}out[b]=__half2float(side[b].d)*acc;
}
__global__ void convert(const BaseBlock *src,SideBlock *dst,int n){int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=n)return;
    #pragma unroll
    for(int j=0;j<32;j++){uint8_t v=0;
        #pragma unroll
        for(int k=0;k<4;k++){int e=4*j+k;uint8_t x;int pow;if(e<80){x=src[b].qs[e&15];pow=e>>4;}else if(e<120){int t=e-80;x=src[b].qs[16+(t&7)];pow=t>>3;}else{int t=e-120;x=src[b].qh[t&1];pow=t>>1;}uint32_t z=x;for(int i=0;i<pow;i++)z=(z*3)&255;uint8_t q=(uint8_t)((z*3)>>8);v|=(q&3)<<(2*k);}dst[b].q[j]=v;
    }dst[b].d=src[b].d;
}
int main(int argc,char **argv){int nb=argc>1?atoi(argv[1]):65536,reps=argc>2?atoi(argv[2]):300;if(nb<=0||reps<=0)return 2;std::mt19937 rng(11011);std::uniform_int_distribution<int> bd(0,255),ad(-127,127);
    std::vector<BaseBlock> h(nb);for(auto &b:h){for(auto &x:b.qs)x=bd(rng);for(auto &x:b.qh)x=bd(rng);b.d=__float2half(.125f);}std::vector<int> hy((size_t)nb*9*4);std::vector<int> refs(nb*4);
    for(int b=0;b<nb;b++)for(int word=0;word<32;word++){uint32_t packed=0;for(int k=0;k<4;k++){int8_t a=(int8_t)ad(rng);packed|=(uint32_t)(uint8_t)a<<(8*k);}hy[((word>>2)*nb+b)*4+(word&3)]=(int)packed;}
    for(int b=0;b<nb;b++)for(int sub=0;sub<4;sub++){int isum=0;for(int e=sub*32;e<sub*32+32;e++){int w=e>>2;int z=hy[((w>>2)*nb+b)*4+(w&3)];isum+=(int8_t)((uint32_t)z>>(8*(e&3)));}uint32_t ds=(uint16_t)__half_as_ushort(__float2half(.03125f))|((uint32_t)(uint16_t)(int16_t)isum<<16);hy[(8*nb+b)*4+sub]=(int)ds;}
    // Verify activation planar address / four scale+isum words and independent code/dot output.
    BaseBlock *db;SideBlock *ds;int *dy,*bad;float *ob,*os,*oe,*op;ck(cudaMalloc(&db,nb*sizeof(BaseBlock)));ck(cudaMalloc(&ds,nb*sizeof(SideBlock)));ck(cudaMalloc(&dy,hy.size()*4));ck(cudaMalloc(&bad,nb*4));ck(cudaMalloc(&ob,nb*4));ck(cudaMalloc(&os,nb*4));ck(cudaMalloc(&oe,nb*4));ck(cudaMalloc(&op,nb*4));ck(cudaMemcpy(db,h.data(),nb*sizeof(BaseBlock),cudaMemcpyHostToDevice));ck(cudaMemcpy(dy,hy.data(),hy.size()*4,cudaMemcpyHostToDevice));dim3 block(128),grid((nb+127)/128);convert<<<grid,block>>>(db,ds,nb);dot_base<<<grid,block>>>(db,dy,ob,nb);dot_pair<<<grid,block>>>(db,dy,op,nb);dot_side<<<grid,block>>>(ds,dy,os,nb);dot_side_expand<<<grid,block>>>(ds,dy,oe,nb);ck(cudaDeviceSynchronize());
    std::vector<SideBlock> hs(nb);std::vector<float> hb(nb),hh(nb),he(nb),hp(nb);ck(cudaMemcpy(hs.data(),ds,nb*sizeof(SideBlock),cudaMemcpyDeviceToHost));ck(cudaMemcpy(hb.data(),ob,nb*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hh.data(),os,nb*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(he.data(),oe,nb*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hp.data(),op,nb*4,cudaMemcpyDeviceToHost));long codes=0;int codebad=0,outbad=0,base_side_bad=0,base_ref_bad=0,side_ref_bad=0,expand_ref_bad=0,refbad=0;float maxerr=0;for(int b=0;b<nb;b++){for(int e=0;e<128;e++){int c=code(h[b],e);codes++;codebad+=c>2||((hs[b].q[e>>2]>>(2*(e&3)))&3)!=c;}float ref=0;for(int k=0;k<4;k++){int isum=0,qsum=0;for(int e=k*32;e<(k+1)*32;e++){int word=e>>2;int z=hy[((word>>2)*nb+b)*4+(word&3)];int8_t a=(int8_t)((uint32_t)z>>(8*(e&3)));isum+=a;qsum+=code(h[b],e)*a;}uint32_t dsword=hy[(8*nb+b)*4+k];int stored=(int)(int16_t)(dsword>>16);if(isum!=stored)refbad++;float sc=__half2float(__ushort_as_half((uint16_t)dsword));ref=fmaf(sc,(float)(qsum-stored),ref);}ref*=__half2float(h[b].d);base_side_bad+=hb[b]!=hh[b];if(hp[b]!=hb[b]){fprintf(stderr,"PAIR_MISMATCH b=%d pair=%a base=%a\n",b,hp[b],hb[b]);return 6;}base_ref_bad+=hb[b]!=ref;side_ref_bad+=hh[b]!=ref;expand_ref_bad+=he[b]!=ref;outbad+=hb[b]!=hh[b]||hb[b]!=ref||he[b]!=ref;maxerr=fmaxf(maxerr,fmaxf(fabsf(hb[b]-hh[b]),fmaxf(fabsf(hb[b]-ref),fabsf(he[b]-ref))));}
    if(codebad||outbad||refbad){fprintf(stderr,"FAIL code=%d output=%d base_side=%d base_ref=%d side_ref=%d activation_sums=%d max_error=%g first=%a/%a\n",codebad,outbad,base_side_bad,base_ref_bad,side_ref_bad,refbad,maxerr,hb[0],hh[0]);for(int b=0;b<8&&b<nb;b++){float rr=0;for(int k=0;k<4;k++){int qs=0,ss=0;for(int ex=k*32;ex<(k+1)*32;ex++){int wo=ex>>2;int zz=hy[((wo>>2)*nb+b)*4+(wo&3)];int8_t av=(int8_t)((uint32_t)zz>>(8*(ex&3)));ss+=av;qs+=code(h[b],ex)*av;}uint32_t dd=hy[(8*nb+b)*4+k];rr=fmaf(__half2float(__ushort_as_half((uint16_t)dd)),(float)(qs-(int16_t)(dd>>16)),rr);}rr*=__half2float(h[b].d);printf("b%d base=%a side=%a ref=%a\n",b,hb[b],hh[b],rr);}return 5;}printf("EXACT blocks=%d verified_codes=%ld code_mismatches=%d outputs=%d output_mismatches=%d independent_ref_mismatches=%d max_abs_error=%g payload=%zu->%zu (+%.2f%%) planar_activation=%zu bytes\n",nb,codes,codebad,nb,outbad,refbad,maxerr,sizeof(BaseBlock),sizeof(SideBlock),100.0*(sizeof(SideBlock)/(double)sizeof(BaseBlock)-1),hy.size()*4);
    cudaEvent_t s,e;ck(cudaEventCreate(&s));ck(cudaEventCreate(&e));std::vector<double> times[2];for(int run=0;run<9;run++)for(int oi=0;oi<2;oi++){int v=(run+oi)%2;for(int w=0;w<40;w++){if(v==0)dot_base<<<grid,block>>>(db,dy,ob,nb);else dot_pair<<<grid,block>>>(db,dy,op,nb);}ck(cudaDeviceSynchronize());ck(cudaEventRecord(s));for(int r=0;r<reps;r++){if(v==0)dot_base<<<grid,block>>>(db,dy,ob,nb);else dot_pair<<<grid,block>>>(db,dy,op,nb);}ck(cudaEventRecord(e));ck(cudaEventSynchronize(e));float ms;ck(cudaEventElapsedTime(&ms,s,e));times[v].push_back(ms/reps);}for(int v=0;v<2;v++){printf("%s",v==0?"planar-base3":"planar-pairwise");for(double x:times[v])printf(" %.8f",x);printf(" ms/launch blocks=%d reps=%d\n",nb,reps);} }
