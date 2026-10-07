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
struct B { uint8_t qs[24],qh[2]; half d; }; static_assert(sizeof(B)==28);
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}}
static uint8_t code(const B&b,int e){uint8_t x;int n;if(e<80){x=b.qs[e&15];n=e>>4;}else if(e<120){int t=e-80;x=b.qs[16+(t&7)];n=t>>3;}else{int t=e-120;x=b.qh[t&1];n=t>>1;}uint32_t v=x;for(int i=0;i<n;i++)v=(v*3)&255;return (uint8_t)((v*3)>>8);}
__device__ __forceinline__ uint32_t step(uint32_t&lo,uint32_t&hi){uint32_t a=lo*3,b=hi*3;lo=a&0x00FF00FF;hi=b&0x00FF00FF;return __byte_perm(a,b,0x7531);}
__device__ __forceinline__ int pw(const int*y,int b,int nb,int w){return y[((w>>2)*nb+b)*4+(w&3)];}
__device__ __forceinline__ uint32_t parallel_step(uint32_t origlo, uint32_t orighi, uint32_t &prevlo, uint32_t &prevhi, int c) {
    const uint32_t flo = ((origlo * (uint32_t)c) >> 8) & 0x00FF00FFu;
    const uint32_t fhi = ((orighi * (uint32_t)c) >> 8) & 0x00FF00FFu;
    const uint32_t qlo = flo - prevlo * 3u;
    const uint32_t qhi = fhi - prevhi * 3u;
    prevlo = flo; prevhi = fhi;
    return __byte_perm(qlo << 8, qhi << 8, 0x7531);
}
__device__ __forceinline__ float dotwords_parallel(const uint32_t *q,int d,const int*y,int b,int nb){int s[4]={};
#pragma unroll
for(int g=0;g<4;g++){uint32_t p=q[g],ol=__byte_perm(p,0,0x4140),oh=__byte_perm(p,0,0x4342),pl=0,ph=0;
#pragma unroll
for(int t=0;t<5;t++){uint32_t x=parallel_step(ol,oh,pl,ph,t==0?3:(t==1?9:(t==2?27:(t==3?81:243))));int e=t*16+4*g;s[e>>5]=__dp4a((int)x,pw(y,b,nb,e>>2),s[e>>5]);}}
#pragma unroll
for(int g=0;g<2;g++){uint32_t p=q[4+g],ol=__byte_perm(p,0,0x4140),oh=__byte_perm(p,0,0x4342),pl=0,ph=0;
#pragma unroll
for(int t=0;t<5;t++){uint32_t x=parallel_step(ol,oh,pl,ph,t==0?3:(t==1?9:(t==2?27:(t==3?81:243))));int e=80+t*8+4*g;s[e>>5]=__dp4a((int)x,pw(y,b,nb,e>>2),s[e>>5]);}}
uint32_t v=(q[6]&0xffu)|((q[6]&0xff00u)<<8);
#pragma unroll
for(int t=0;t<4;t+=2){uint32_t a=v*3;v=a&0x00FF00FF;uint32_t c=v*3;v=c&0x00FF00FF;uint32_t x=__byte_perm(a,c,0x7531);s[3]=__dp4a((int)x,pw(y,b,nb,30+t/2),s[3]);}
float acc=0;
#pragma unroll
for(int k=0;k<4;k++){uint32_t z=pw(y,b,nb,32+k);float sc=__half2float(__ushort_as_half((uint16_t)z));int isum=(int)(int16_t)(z>>16);acc=__fmaf_rn(sc,(float)(s[k]-isum),acc);}return __half2float(__ushort_as_half((uint16_t)d))*acc;}
__device__ __forceinline__ float dotwords(const uint32_t *q,int d,const int*y,int b,int nb){int s[4]={};
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
__global__ void work_aos(const B*x,const int*y,float*p,int nb,int nr){int item=blockIdx.x*blockDim.x+threadIdx.x,total=nb*nr;if(item>=total)return;int r=item/nb,b=item-r*nb;uint32_t w[7];const uint32_t *src=(const uint32_t*)(x+(size_t)r*nb+b);
#pragma unroll
for(int i=0;i<7;i++)w[i]=src[i];p[item]=dotwords(w,w[6]>>16,y,b,nb);}
__global__ void work_parallel(const B*x,const int*y,float*p,int nb,int nr){int item=blockIdx.x*blockDim.x+threadIdx.x,total=nb*nr;if(item>=total)return;int r=item/nb,b=item-r*nb;uint32_t w[7];const uint32_t *src=(const uint32_t*)(x+(size_t)r*nb+b);
#pragma unroll
for(int i=0;i<7;i++)w[i]=src[i];p[item]=dotwords_parallel(w,w[6]>>16,y,b,nb);}
__global__ void work_soa(const uint32_t*x,const int*y,float*p,int nb,int nr){int item=blockIdx.x*blockDim.x+threadIdx.x,total=nb*nr;if(item>=total)return;int r=item/nb,b=item-r*nb;uint32_t w[7];const uint32_t*row=x+(size_t)r*7*nb;
#pragma unroll
for(int i=0;i<7;i++)w[i]=row[(size_t)i*nb+b];p[item]=dotwords(w,w[6]>>16,y,b,nb);}
__global__ void work_staged(const B*x,const int*y,float*p,int nb,int nr){__shared__ uint32_t tile[128*7];int tid=threadIdx.x,base=blockIdx.x*128,total=nb*nr;const uint32_t*src=(const uint32_t*)x;for(int i=tid;i<128*7;i+=128){int g=base*7+i;tile[i]=g<total*7?src[g]:0;}__syncthreads();int item=base+tid;if(item<total){int b=item%nb;const uint32_t*w=tile+tid*7;p[item]=dotwords(w,w[6]>>16,y,b,nb);}}
// One warp loads 28 contiguous words (four PTQ blocks).  Four lanes then
// gather seven words each from registers with shuffles and execute the packed dot.
__global__ void work_warp_register(const B*x,const int*y,float*p,int nb,int nr){
 int lane=threadIdx.x&31, warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
 int base=warp*4, total=nb*nr; const uint32_t*src=(const uint32_t*)x;
 uint32_t v=0; if(lane<28 && base*7+lane<total*7) v=src[base*7+lane];
 if(base>=total)return;
 uint32_t w[7];
 #pragma unroll
 for(int j=0;j<7;j++) w[j]=__shfl_sync(0xffffffffu,v,lane<4?lane*7+j:0);
 if(lane<4){int item=base+lane; if(item<total){int b=item%nb;p[item]=dotwords(w,w[6]>>16,y,b,nb);}}
}
__global__ void emit_codes(const B*x,uint8_t*out,int nb,int nr){int item=blockIdx.x*blockDim.x+threadIdx.x,total=nb*nr;if(item>=total)return;const B*b=x+item;int base=item*128;
#pragma unroll
for(int g=0;g<4;g++){uint32_t p;memcpy(&p,b->qs+4*g,4);uint32_t lo=__byte_perm(p,0,0x4140),hi=__byte_perm(p,0,0x4342);
#pragma unroll
for(int t=0;t<5;t++){uint32_t q=step(lo,hi);int e=16*t+4*g;
#pragma unroll
for(int l=0;l<4;l++)out[base+e+l]=(q>>(8*l))&255;}}
#pragma unroll
for(int g=0;g<2;g++){uint32_t p;memcpy(&p,b->qs+16+4*g,4);uint32_t lo=__byte_perm(p,0,0x4140),hi=__byte_perm(p,0,0x4342);
#pragma unroll
for(int t=0;t<5;t++){uint32_t q=step(lo,hi);int e=80+8*t+4*g;
#pragma unroll
for(int l=0;l<4;l++)out[base+e+l]=(q>>(8*l))&255;}}
uint32_t v=(uint32_t)b->qh[0]|((uint32_t)b->qh[1]<<16);
#pragma unroll
for(int t=0;t<4;t+=2){uint32_t a=v*3;v=a&0x00FF00FF;uint32_t c=v*3;v=c&0x00FF00FF;uint32_t q=__byte_perm(a,c,0x7531);int e=120+(t/2)*4;
#pragma unroll
for(int l=0;l<4;l++)out[base+e+l]=(q>>(8*l))&255;}}
__global__ void fold(const float*p,float*out,int nb,int nr){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=nr)return;const float*s=p+(size_t)r*nb;float a=0,b=0,c=0,d=0;int k=0;for(;k+4<=nb;k+=4){a+=s[k];b+=s[k+1];c+=s[k+2];d+=s[k+3];}for(;k<nb;k++)a+=s[k];out[r]=(a+b)+(c+d);}
int main(int argc,char**argv){int nb=argc>1?atoi(argv[1]):40,nr=argc>2?atoi(argv[2]):4096,reps=argc>3?atoi(argv[3]):1000;if(nb!=40&&nb!=136)return 2;std::mt19937 gen(32032+nb);std::uniform_int_distribution<int>bd(0,255),ad(-127,127);std::vector<B> h((size_t)nb*nr);for(auto&b:h){for(auto&z:b.qs)z=bd(gen);for(auto&z:b.qh)z=bd(gen);b.d=__float2half(.125f);}std::vector<uint32_t> hs((size_t)nr*nb*7);for(int r=0;r<nr;r++)for(int b=0;b<nb;b++){const uint32_t*src=(const uint32_t*)&h[(size_t)r*nb+b];for(int i=0;i<7;i++)hs[((size_t)r*7+i)*nb+b]=src[i];}
std::vector<int> hy((size_t)nb*9*4);for(int b=0;b<nb;b++)for(int w=0;w<32;w++){uint32_t z=0;for(int j=0;j<4;j++)z|=(uint32_t)(uint8_t)(int8_t)ad(gen)<<(8*j);hy[((w>>2)*nb+b)*4+(w&3)]=(int)z;}for(int b=0;b<nb;b++)for(int k=0;k<4;k++){int sum=0;for(int e=k*32;e<(k+1)*32;e++){int z=hy[((e/16)*nb+b)*4+((e/4)&3)];sum+=(int8_t)((uint32_t)z>>(8*(e&3)));}uint32_t z=(uint16_t)__half_as_ushort(__float2half(.03125f))|((uint32_t)(uint16_t)(int16_t)sum<<16);hy[(8*nb+b)*4+k]=(int)z;}
B *dx;uint32_t*ds;int*dy;float*dp,*doa,*dos,*dst,*dwr;uint8_t*dc;ck(cudaMalloc(&dx,h.size()*sizeof(B)));ck(cudaMalloc(&ds,hs.size()*4));ck(cudaMalloc(&dy,hy.size()*4));ck(cudaMalloc(&dp,(size_t)nb*nr*4));ck(cudaMalloc(&dc,(size_t)nr*nb*128));ck(cudaMalloc(&doa,(size_t)nr*4));ck(cudaMalloc(&dos,(size_t)nr*4));ck(cudaMalloc(&dst,(size_t)nr*4));ck(cudaMalloc(&dwr,(size_t)nr*4));ck(cudaMemcpy(dx,h.data(),h.size()*sizeof(B),cudaMemcpyHostToDevice));ck(cudaMemcpy(ds,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(dy,hy.data(),hy.size()*4,cudaMemcpyHostToDevice));dim3 bl(128),gr((nb*nr+127)/128),fg((nr+127)/128);emit_codes<<<gr,bl>>>(dx,dc,nb,nr);work_aos<<<gr,bl>>>(dx,dy,dp,nb,nr);fold<<<fg,bl>>>(dp,doa,nb,nr);work_parallel<<<gr,bl>>>(dx,dy,dp,nb,nr);fold<<<fg,bl>>>(dp,dst,nb,nr);work_soa<<<gr,bl>>>(ds,dy,dp,nb,nr);fold<<<fg,bl>>>(dp,dos,nb,nr);work_staged<<<gr,bl>>>(dx,dy,dp,nb,nr);fold<<<fg,bl>>>(dp,dst,nb,nr);work_parallel<<<gr,bl>>>(dx,dy,dp,nb,nr);fold<<<fg,bl>>>(dp,dwr,nb,nr);ck(cudaDeviceSynchronize());std::vector<float>a(nr),s(nr),staged(nr);ck(cudaMemcpy(a.data(),doa,nr*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(s.data(),dos,nr*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(staged.data(),dst,nr*4,cudaMemcpyDeviceToHost));std::vector<float> wr(nr);ck(cudaMemcpy(wr.data(),dwr,nr*4,cudaMemcpyDeviceToHost));int bad=0,stagebad=0,wrbad=0,refbad=0;double maxerr=0;std::vector<uint8_t>devcodes((size_t)nr*nb*128);ck(cudaMemcpy(devcodes.data(),dc,devcodes.size(),cudaMemcpyDeviceToHost));for(int r=0;r<nr;r++){bad+=a[r]!=s[r];stagebad+=a[r]!=staged[r];wrbad+=a[r]!=wr[r];float accs[4]={};for(int b=0;b<nb;b++){float ba=0;for(int k=0;k<4;k++){int isum=0,qsum=0;for(int e=k*32;e<(k+1)*32;e++){int w=e>>2;int z=hy[((w>>2)*nb+b)*4+(w&3)];int av=(int8_t)((uint32_t)z>>(8*(e&3)));isum+=av;qsum+=code(h[(size_t)r*nb+b],e)*av;}uint32_t z=hy[(8*nb+b)*4+k];float sc=__half2float(__ushort_as_half((uint16_t)z));ba=fmaf(sc,(float)(qsum-(int16_t)(z>>16)),ba);}ba*=__half2float(h[(size_t)r*nb+b].d);switch(b&3){case 0:accs[0]+=ba;break;case 1:accs[1]+=ba;break;case 2:accs[2]+=ba;break;case 3:accs[3]+=ba;break;}}float ref=(accs[0]+accs[1])+(accs[2]+accs[3]);maxerr=fmax(maxerr,fabs((double)ref-a[r]));maxerr=fmax(maxerr,fabs((double)ref-s[r]));maxerr=fmax(maxerr,fabs((double)ref-staged[r]));maxerr=fmax(maxerr,fabs((double)ref-wr[r]));refbad+=(fabs(ref-a[r])>1e-5*fmax(1.0,fabs(ref)))||(fabs(ref-s[r])>1e-5*fmax(1.0,fabs(ref)))||(fabs(ref-staged[r])>1e-5*fmax(1.0,fabs(ref)))||(fabs(ref-wr[r])>1e-5*fmax(1.0,fabs(ref)));}long nc=(long)nr*nb*128;long invalid=0;for(int r=0;r<nr;r++)for(int b=0;b<nb;b++)for(int e=0;e<128;e++)invalid+=(code(h[(size_t)r*nb+b],e)>2)||(devcodes[((size_t)r*nb+b)*128+e]!=code(h[(size_t)r*nb+b],e));printf("CORRECT rows=%d blocks_per_row=%d verified_device_codes=%ld code_mismatches=%ld aos_soa_mismatches=%d aos_staged_mismatches=%d warp_register_mismatches=%d independent_host_output_mismatches=%d max_error=%g\n",nr,nb,nc,invalid,bad,stagebad,wrbad,refbad,maxerr);if(bad||stagebad||wrbad||refbad||invalid)return 5;
cudaEvent_t st,en;ck(cudaEventCreate(&st));ck(cudaEventCreate(&en));std::vector<double>ta,tp;for(int rep=0;rep<9;rep++){for(int z=0;z<2;z++){int v=(rep+z)&1;ck(cudaEventRecord(st));for(int i=0;i<reps;i++){if(v==0){work_aos<<<gr,bl>>>(dx,dy,dp,nb,nr);fold<<<fg,bl>>>(dp,doa,nb,nr);}else{work_parallel<<<gr,bl>>>(dx,dy,dp,nb,nr);fold<<<fg,bl>>>(dp,dwr,nb,nr);}}ck(cudaEventRecord(en));ck(cudaEventSynchronize(en));float ms;ck(cudaEventElapsedTime(&ms,st,en));(v==0?ta:tp).push_back(ms/reps);}}for(double x:ta)printf("RECURRENCE %.8f ms_pair\n",x);for(double x:tp)printf("FIXED_POINT %.8f ms_pair\n",x);}
