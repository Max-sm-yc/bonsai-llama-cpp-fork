#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
struct B { uint8_t qs[24], qh[2]; half d; }; static_assert(sizeof(B)==28);
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}
static uint8_t code(const B&b,int e){uint8_t x;int n;if(e<80){x=b.qs[e&15];n=e>>4;}else if(e<120){int t=e-80;x=b.qs[16+(t&7)];n=t>>3;}else{int t=e-120;x=b.qh[t&1];n=t>>1;}uint32_t v=x;for(int i=0;i<n;i++)v=(v*3)&255;return (uint8_t)((v*3)>>8);}
__device__ __forceinline__ uint32_t step(uint32_t&lo,uint32_t&hi){uint32_t a=lo*3,b=hi*3;lo=a&0x00FF00FF;hi=b&0x00FF00FF;return __byte_perm(a,b,0x7531);}
__device__ __forceinline__ int pw(const int*y,int b,int nb,int w){return y[((w>>2)*nb+b)*4+(w&3)];}
__device__ __forceinline__ float dotwords(const uint32_t*q,int d,const int*y,int b,int nb){int s[4]={};
#pragma unroll
for(int g=0;g<4;g++){uint32_t p=q[g],lo=__byte_perm(p,0,0x4140),hi=__byte_perm(p,0,0x4342);
#pragma unroll
for(int t=0;t<5;t++){uint32_t x=step(lo,hi);int e=t*16+4*g;s[e>>5]=__dp4a((int)x,pw(y,b,nb,e>>2),s[e>>5]);}}
#pragma unroll
for(int g=0;g<2;g++){uint32_t p=q[4+g],lo=__byte_perm(p,0,0x4140),hi=__byte_perm(p,0,0x4342);
#pragma unroll
for(int t=0;t<5;t++){uint32_t x=step(lo,hi);int e=80+t*8+4*g;s[e>>5]=__dp4a((int)x,pw(y,b,nb,e>>2),s[e>>5]);}}
uint32_t v=(q[6]&0xffu)|((q[6]&0xff00u)<<8);
#pragma unroll
for(int t=0;t<4;t+=2){uint32_t a=v*3;v=a&0x00FF00FF;uint32_t c=v*3;v=c&0x00FF00FF;uint32_t x=__byte_perm(a,c,0x7531);s[3]=__dp4a((int)x,pw(y,b,nb,30+t/2),s[3]);}
float acc=0;
#pragma unroll
for(int k=0;k<4;k++){uint32_t z=pw(y,b,nb,32+k);float sc=__half2float(__ushort_as_half((uint16_t)z));int isum=(int)(int16_t)(z>>16);acc=__fmaf_rn(sc,(float)(s[k]-isum),acc);}return __half2float(__ushort_as_half((uint16_t)d))*acc;}
template<int NT> __global__ void work(const B*x,const int*y,float*p,int nb,int rpc){int tid=threadIdx.x,total=rpc*nb,row0=blockIdx.x*rpc;for(int idx=tid;idx<total;idx+=NT){int r=idx/nb,b=idx-r*nb;uint32_t w[7];const uint32_t*src=(const uint32_t*)(x+(size_t)(row0+r)*nb+b);
#pragma unroll
for(int i=0;i<7;i++)w[i]=src[i];p[(size_t)(row0+r)*nb+b]=dotwords(w,w[6]>>16,y,b,nb);}}
__global__ void fold(const float*p,float*out,int nb,int nr){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=nr)return;const float*s=p+(size_t)r*nb;float a=0,b=0,c=0,d=0;int k=0;for(;k+4<=nb;k+=4){a+=s[k];b+=s[k+1];c+=s[k+2];d+=s[k+3];}for(;k<nb;k++)a+=s[k];out[r]=(a+b)+(c+d);}
static int rpc_for(int nb,int nt){int best=1;double bu=0;int rmax=std::min(16,4096/(nb+1));for(int r=1;r<=rmax;r++){int items=r*nb,iters=(items+nt-1)/nt;double u=(double)items/(iters*nt);if(u>bu+1e-9){best=r;bu=u;}if(u>.999)break;}return best;}
template<int NT> static void launch(const B*x,const int*y,float*p,int nb,int nr,int rpc){dim3 bl(NT),gr((nr+rpc-1)/rpc);work<NT><<<gr,bl>>>(x,y,p,nb,rpc);}
static void dispatch(int nt,const B*x,const int*y,float*p,int nb,int nr,int rpc){switch(nt){case 64:launch<64>(x,y,p,nb,nr,rpc);break;case 128:launch<128>(x,y,p,nb,nr,rpc);break;case 256:launch<256>(x,y,p,nb,nr,rpc);break;case 512:launch<512>(x,y,p,nb,nr,rpc);break;default:exit(3);}}
int main(int argc,char**argv){int nb=argc>1?atoi(argv[1]):40,nt=argc>2?atoi(argv[2]):128,nr=2048,reps=argc>3?atoi(argv[3]):100;if((nb!=40&&nb!=136)||(nt!=64&&nt!=128&&nt!=256&&nt!=512))return 2;int rpc=rpc_for(nb,nt);std::mt19937 gen(32042+nb);std::uniform_int_distribution<int>bd(0,255),ad(-127,127);std::vector<B>h((size_t)nb*nr);for(auto&b:h){for(auto&z:b.qs)z=bd(gen);for(auto&z:b.qh)z=bd(gen);b.d=__float2half(.125f);}std::vector<int>hy((size_t)nb*9*4);for(int b=0;b<nb;b++)for(int w=0;w<32;w++){uint32_t z=0;for(int j=0;j<4;j++)z|=(uint32_t)(uint8_t)(int8_t)ad(gen)<<(8*j);hy[((w>>2)*nb+b)*4+(w&3)]=(int)z;}for(int b=0;b<nb;b++)for(int k=0;k<4;k++){int sum=0;for(int e=k*32;e<(k+1)*32;e++){int z=hy[((e/16)*nb+b)*4+((e/4)&3)];sum+=(int8_t)((uint32_t)z>>(8*(e&3)));}uint32_t z=(uint16_t)__half_as_ushort(__float2half(.03125f))|((uint32_t)(uint16_t)(int16_t)sum<<16);hy[(8*nb+b)*4+k]=(int)z;}
B*dx;int*dy;float*dp,*out;ck(cudaMalloc(&dx,h.size()*sizeof(B)));ck(cudaMalloc(&dy,hy.size()*4));ck(cudaMalloc(&dp,(size_t)nb*nr*4));ck(cudaMalloc(&out,(size_t)nr*4));ck(cudaMemcpy(dx,h.data(),h.size()*sizeof(B),cudaMemcpyHostToDevice));ck(cudaMemcpy(dy,hy.data(),hy.size()*4,cudaMemcpyHostToDevice));dim3 fg((nr+127)/128,1,1),fb(128,1,1);dispatch(128,dx,dy,dp,nb,nr,rpc_for(nb,128));fold<<<fg,fb>>>(dp,out,nb,nr);ck(cudaDeviceSynchronize());std::vector<float>refout(nr),got(nr);for(int r=0;r<nr;r++){float a[4]={};for(int b=0;b<nb;b++){float ba=0;for(int k=0;k<4;k++){int qs=0;for(int e=k*32;e<(k+1)*32;e++){int w=e>>2,z=hy[((w>>2)*nb+b)*4+(w&3)];int av=(int8_t)((uint32_t)z>>(8*(e&3)));qs+=code(h[(size_t)r*nb+b],e)*av;}uint32_t z=hy[(8*nb+b)*4+k];float sc=__half2float(__ushort_as_half((uint16_t)z));ba=fmaf(sc,(float)(qs-(int16_t)(z>>16)),ba);}ba*=__half2float(h[(size_t)r*nb+b].d);a[b&3]+=ba;}refout[r]=(a[0]+a[1])+(a[2]+a[3]);}ck(cudaMemcpy(got.data(),out,nr*4,cudaMemcpyDeviceToHost));int bad=0;double maxerr=0;for(int r=0;r<nr;r++){bad+=refout[r]!=got[r];maxerr=fmax(maxerr,fabs((double)refout[r]-got[r]));}dispatch(nt,dx,dy,dp,nb,nr,rpc);fold<<<fg,fb>>>(dp,out,nb,nr);ck(cudaDeviceSynchronize());ck(cudaMemcpy(got.data(),out,nr*4,cudaMemcpyDeviceToHost));int candbad=0;for(int r=0;r<nr;r++){candbad+=refout[r]!=got[r];maxerr=fmax(maxerr,fabs((double)refout[r]-got[r]));}printf("CORRECT K_BLOCKS=%d CTA_THREADS=%d ROWS_PER_CTA=%d rows=%d control_mismatches=%d candidate_mismatches=%d max_abs_error=%g\n",nb,nt,rpc,nr,bad,candbad,maxerr);if(bad||candbad)return 4;
cudaEvent_t st,en;ck(cudaEventCreate(&st));ck(cudaEventCreate(&en));std::vector<double>tc,tt;for(int rep=0;rep<9;rep++)for(int z=0;z<2;z++){int arm=(rep+z)&1;ck(cudaEventRecord(st));for(int i=0;i<reps;i++){dispatch(arm?nt:128,dx,dy,dp,nb,nr,arm?rpc:rpc_for(nb,128));fold<<<fg,fb>>>(dp,out,nb,nr);}ck(cudaEventRecord(en));ck(cudaEventSynchronize(en));float ms;ck(cudaEventElapsedTime(&ms,st,en));(arm?tt:tc).push_back(ms/reps);}printf("CONTROL");for(double x:tc)printf(" %.8f",x);printf(" ms\nCTA_%d",nt);for(double x:tt)printf(" %.8f",x);printf(" ms\n");}
