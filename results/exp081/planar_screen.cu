#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Active PTQ1_0 planar GEMV screen: <ncols=1, ROWS=1>, 128-thread CTAs,
// flattened (row-group,K-block) work, per-block shared partials, and the
// production modulo-four FP32 epilogue. Activation planes match mmvq-ptq1_0.cuh.
struct Base { uint8_t qs[24], qh[2]; __half d; };
struct Side { uint8_t q[32]; __half d; };
struct block_ptq1_0 { uint8_t qs[24], qh[2]; uint16_t d; };
extern "C" void dequantize_row_ptq1_0(const block_ptq1_0 *, float *, int64_t);
static_assert(sizeof(Base)==28 && sizeof(Side)==34, "block sizes");

__device__ __forceinline__ uint32_t trit_step(uint32_t &lo, uint32_t &hi) {
    uint32_t a=lo*3, b=hi*3; lo=a&0x00FF00FF; hi=b&0x00FF00FF;
    return __byte_perm(a,b,0x7531);
}
__device__ __forceinline__ uint32_t four2(const Side *w, int e) {
    const uint32_t p=w->q[e>>2];
    return (p&3u)|(((p>>2)&3u)<<8)|(((p>>4)&3u)<<16)|(((p>>6)&3u)<<24);
}
__device__ __forceinline__ uint32_t word_at(const int4 &v,int i) { return (uint32_t)((int*)&v)[i]; }

template<bool DIRECT>
__device__ __forceinline__ float block_dot(const void *wp, const int8_t *act, int nblk, int kb) {
    int sumi=0; float acc[4]={0,0,0,0};
    const int4 dsraw=*reinterpret_cast<const int4*>(act+8*nblk*16+kb*16);
    const Base *bw=(const Base*)wp; const Side *sw=(const Side*)wp;
    uint32_t lo[4],hi[4];
    if constexpr(!DIRECT) for(int g=0;g<4;g++){uint32_t p;memcpy(&p,bw->qs+4*g,4);lo[g]=__byte_perm(p,0,0x4140);hi[g]=__byte_perm(p,0,0x4342);}
    for(int t=0;t<5;t++) {
        const int4 u=*reinterpret_cast<const int4*>(act+(t*nblk+kb)*16);
        for(int g=0;g<4;g++) {
            uint32_t q;
            if constexpr(DIRECT) q=four2(sw,16*t+4*g);
            else q=trit_step(lo[g],hi[g]);
            sumi=__dp4a((int)q,(int)word_at(u,g),sumi);
        }
        if(t==1||t==3){int k=t/2;__half2 hs=((const __half2*)&dsraw)[k];acc[k]=__fmaf_rn(__half2float(__low2half(hs)),(float)(sumi-__half_as_short(__high2half(hs))),acc[k]);sumi=0;}
    }
    uint32_t lo2[2],hi2[2];
    if constexpr(!DIRECT) for(int g=0;g<2;g++){uint32_t p;memcpy(&p,bw->qs+16+4*g,4);lo2[g]=__byte_perm(p,0,0x4140);hi2[g]=__byte_perm(p,0,0x4342);}
    int4 u2{};
    for(int t=0;t<5;t++) {
        if((t&1)==0) u2=*reinterpret_cast<const int4*>(act+((5+t/2)*nblk+kb)*16);
        for(int g=0;g<2;g++) {
            uint32_t q;
            if constexpr(DIRECT) q=four2(sw,80+8*t+4*g);
            else q=trit_step(lo2[g],hi2[g]);
            int w=20+2*t+g; sumi=__dp4a((int)q,(int)word_at(u2,w&3),sumi);
        }
        if(t==1){__half2 hs=((const __half2*)&dsraw)[2];acc[2]=__fmaf_rn(__half2float(__low2half(hs)),(float)(sumi-__half_as_short(__high2half(hs))),acc[2]);sumi=0;}
    }
    // qh bytes interleave by trit plane: element 120+2*t+h. The packed
    // recurrence below is the production qh path; direct codes follow e order.
    if constexpr(DIRECT) {
        for(int h=0;h<2;h++) { uint32_t q=four2(sw,120+4*h); sumi=__dp4a((int)q,(int)word_at(u2,2+h),sumi); }
    } else {
        uint32_t v=(uint32_t)bw->qh[0]|((uint32_t)bw->qh[1]<<16);
        for(int t=0;t<4;t+=2) {
            uint32_t w0=v*3; v=w0&0x00FF00FF;
            uint32_t w1=v*3; v=w1&0x00FF00FF;
            uint32_t q=__byte_perm(w0,w1,0x7531);
            sumi=__dp4a((int)q,(int)word_at(u2,2+t/2),sumi);
        }
    }
    __half2 hs=((const __half2*)&dsraw)[3];acc[3]=__fmaf_rn(__half2float(__low2half(hs)),(float)(sumi-__half_as_short(__high2half(hs))),acc[3]);
    float sum=(acc[0]+acc[1])+(acc[2]+acc[3]);
    float wd=__half2float(((const Base*)wp)->d);
    if constexpr(DIRECT) wd=__half2float(((const Side*)wp)->d);
    return __fmul_rn(wd,sum);
}

template<bool DIRECT>
__global__ void active_like(const void *weights,const int8_t *act,float *out,int rows,int bpr,int rpc) {
    extern __shared__ float part[];
    int tid=threadIdx.x, row0=blockIdx.x*rpc, nitems=rpc*bpr;
    int nblk=bpr; // screen uses no padded K rows
    for(int idx=tid;idx<nitems;idx+=128) {
        int rg=idx/bpr,kb=idx-rg*bpr;
        int real_rows=min(rpc,rows-row0), row=rg<real_rows?rg:real_rows-1;
        const char *w=(const char*)weights+(size_t)(row0+row)*bpr*(DIRECT?34:28)+(size_t)kb*(DIRECT?34:28);
        part[rg*(bpr+1)+kb]=block_dot<DIRECT>(w,act,nblk,kb);
    }
    __syncthreads();
    for(int r=tid;r<rpc;r+=128) {
        int row=row0+r; float s0=0,s1=0,s2=0,s3=0;
        const float *src=part+r*(bpr+1); int k=0;
        for(;k+4<=bpr;k+=4){s0+=src[k];s1+=src[k+1];s2+=src[k+2];s3+=src[k+3];}
        for(;k<bpr;k++)s0+=src[k];
        if(row<rows)out[row]=(s0+s1)+(s2+s3);
    }
}

__global__ void side_pack(const uint8_t *codes, Side *out, int n) {
    int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=n)return;
    for(int i=0;i<32;i++) { uint8_t v=0; for(int j=0;j<4;j++) v|=(codes[b*128+i*4+j]&3)<<(2*j); out[b].q[i]=v; }
    out[b].d=__float2half(1.0f);
}
__global__ void side_unpack_check(const Side *in,const uint8_t *ref,int *bad,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n*128)return;
    int got=(in[i/128].q[(i%128)/4]>>(((i%128)%4)*2))&3;
    if(got!=ref[i] || got==3) atomicAdd(bad,1);
}

static void encode_codes(const uint8_t *c, Base &b) {
    size_t j=0;
    const int stages[3]={32,16,8};
    for(int s=0;s<3;s++){int n=stages[s];for(;j+n<=24;j+=n)for(int m=0;m<n;m++){
        int q=0; for(int t=0;t<5;t++)q=q*3+c[(j<16 ? 16*t+m : 80+8*t+m)];
        b.qs[j+m]=(uint8_t)((q*256+242)/243);
    }}
    for(int h=0;h<2;h++){int q=0;for(int t=0;t<4;t++)q=q*3+c[120+2*t+h];q*=3;b.qh[h]=(uint8_t)((q*256+242)/243);}
    b.d=__float2half(1.0f);
}
static uint8_t decode_canonical(const Base &b,int e) {
    static const uint8_t p3[6]={1,3,9,27,81,243};
    uint8_t q; int n;
    if(e<80){int t=e/16,m=e%16; q=b.qs[m];n=t;}
    else if(e<120){int t=(e-80)/8,m=(e-80)%8;q=b.qs[16+m];n=t;}
    else {int t=(e-120)/2,h=(e-120)%2;q=b.qh[h];n=t;}
    const uint8_t rem=(uint8_t)(q*p3[n]);
    return (uint8_t)(((uint16_t)rem*3)>>8); // raw 0/1/2 digit
}
static std::vector<float> cpu_reference(const std::vector<Base>&w,const int8_t*act,int rows,int bpr) {
    std::vector<float> out(rows); float qw[128];
    for(int r=0;r<rows;r++) {
        float s[4]={0,0,0,0};
        for(int kb=0;kb<bpr;kb++) {
            const Base&b=w[(size_t)r*bpr+kb];
            dequantize_row_ptq1_0(reinterpret_cast<const block_ptq1_0*>(&b),qw,128);
            float acc[4]={0,0,0,0};
            for(int g=0;g<4;g++) {
                int dot=0;
                for(int x=0;x<32;x++){int e=g*32+x;int plane=e/16,ix=e%16;dot+=(int)qw[e]*(int)act[((size_t)plane*bpr+kb)*16+ix];}
                acc[g]=std::fma(0.015625f,(float)dot,0.0f);
            }
            float partial=(acc[0]+acc[1])+(acc[2]+acc[3]);
            int k=kb&3;s[k]=s[k]+partial;
        }
        out[r]=(s[0]+s[1])+(s[2]+s[3]);
    }
    return out;
}

int main(int argc,char**argv) {
    int iters=200; if(argc>1)iters=atoi(argv[1]);
    bool check_only=argc>2 && strcmp(argv[2],"--check-only")==0;
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop,0); printf("gpu=%s cc=%d.%d iters=%d\n",prop.name,prop.major,prop.minor,iters);
    std::mt19937 rng(81); std::uniform_int_distribution<int> trit(0,2), byte(-127,127);
    // Full 128-code canonical host packing and round trip (all digit positions exercised).
    uint8_t codes[128]; for(int e=0;e<128;e++)codes[e]=e%3;
    Base test{}; encode_codes(codes,test); int bad=0; float deq[128];
    dequantize_row_ptq1_0(reinterpret_cast<const block_ptq1_0*>(&test),deq,128);
    for(int e=0;e<128;e++) if(decode_canonical(test,e)!=codes[e] || deq[e]+1.0f!=codes[e])bad++;
    if(bad){printf("host canonical mismatch=%d\n",bad);return 2;}

    for(int bpr: {40,136}) for(int rows: {257,1025,4099}) {
        int n=rows*bpr; std::vector<Base> base(n); std::vector<Side> side(n); std::vector<uint8_t> ref(n*128);
        for(int i=0;i<n;i++){for(int e=0;e<128;e++)ref[i*128+e]=trit(rng);encode_codes(&ref[i*128],base[i]);}
        // construct planar Q8_1 PT: 8 x (nblk*16) activation bytes, then 4 half2(d,isum) records.
        int8_t *ha=(int8_t*)calloc((size_t)9*bpr*16,1);
        for(int t=0;t<8;t++)for(int kb=0;kb<bpr;kb++)for(int x=0;x<16;x++)ha[(size_t)(t*bpr+kb)*16+x]=(int8_t)byte(rng);
        for(int kb=0;kb<bpr;kb++)for(int g=0;g<4;g++){
            int sum=0;for(int x=0;x<32;x++){int e=g*32+x;int t=e/16, ix=e%16;sum+=ha[(size_t)(t*bpr+kb)*16+ix];}
            __half2 *ds=(__half2*)(ha+8*bpr*16+kb*16); ds[g]=__halves2half2(__float2half(0.015625f),__ushort_as_half((unsigned short)(short)sum));
        }
        Base *db;Side *ds;uint8_t *dr;int *d_bad;int8_t *da;float *do0,*do1;
        cudaMalloc(&db,n*sizeof(Base));cudaMalloc(&ds,n*sizeof(Side));cudaMalloc(&dr,n*128);cudaMalloc(&d_bad,sizeof(int));cudaMalloc(&da,(size_t)9*bpr*16);cudaMalloc(&do0,rows*sizeof(float));cudaMalloc(&do1,rows*sizeof(float));
        cudaMemcpy(db,base.data(),n*sizeof(Base),cudaMemcpyHostToDevice);cudaMemcpy(dr,ref.data(),n*128,cudaMemcpyHostToDevice);cudaMemcpy(da,ha,(size_t)9*bpr*16,cudaMemcpyHostToDevice);
        side_pack<<<(n+127)/128,128>>>(dr,ds,n);cudaMemset(d_bad,0,sizeof(int));side_unpack_check<<<(n*128+255)/256,256>>>(ds,dr,d_bad,n);
        int unpack=0;cudaMemcpy(&unpack,d_bad,sizeof(int),cudaMemcpyDeviceToHost);if(unpack){printf("device unpack mismatch=%d\n",unpack);return 3;}
        int rpc=1; // ROWS=1, active host picker below
        int rmax=4096/(bpr+1);if(rmax>16)rmax=16;int best=1;double bu=0;
        for(int r=1;r<=rmax;r++){int items=r*bpr,it=(items+127)/128;double u=(double)items/(it*128);if(u>bu+1e-9){bu=u;best=r;}if(u>.999)break;}rpc=best;
        size_t smem=(size_t)rpc*(bpr+1)*sizeof(float);dim3 grid((rows+rpc-1)/rpc);
        active_like<false><<<grid,128,smem>>>(db,da,do0,rows,bpr,rpc);
        active_like<true><<<grid,128,smem>>>(ds,da,do1,rows,bpr,rpc);
        cudaDeviceSynchronize();std::vector<float> a(rows),c(rows);cudaMemcpy(a.data(),do0,rows*4,cudaMemcpyDeviceToHost);cudaMemcpy(c.data(),do1,rows*4,cudaMemcpyDeviceToHost);
        std::vector<float> cpu=cpu_reference(base,ha,rows,bpr);
        int mism=0,cpu_mism=0;float maxd=0;for(int i=0;i<rows;i++){if(memcmp(&a[i],&c[i],4))mism++;if(memcmp(&a[i],&cpu[i],4))cpu_mism++;maxd=fmaxf(maxd,fabsf(a[i]-c[i]));}
        if(mism||cpu_mism){printf("K=%d rows=%d baseline_direct_mismatch=%d cpu_mismatch=%d max=%g first %g %g %g\n",bpr,rows,mism,cpu_mism,maxd,a[0],c[0],cpu[0]);return 4;}
        if(check_only){printf("checked K=%d rows=%d device_unpack=exact baseline_direct=exact\n",bpr,rows);cudaFree(db);cudaFree(ds);cudaFree(dr);cudaFree(d_bad);cudaFree(da);cudaFree(do0);cudaFree(do1);free(ha);continue;}
        cudaStream_t stream;cudaStreamCreate(&stream);cudaGraph_t graph[2];cudaGraphExec_t exec[2];
        for(int variant=0;variant<2;variant++) {
            cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal);
            if(variant==0) active_like<false><<<grid,128,smem,stream>>>(db,da,do0,rows,bpr,rpc);
            else           active_like<true><<<grid,128,smem,stream>>>(ds,da,do1,rows,bpr,rpc);
            cudaStreamEndCapture(stream,&graph[variant]);cudaGraphInstantiate(&exec[variant],graph[variant],nullptr,nullptr,0);
        }
        cudaEvent_t st,en;cudaEventCreate(&st);cudaEventCreate(&en);std::vector<float> tb,tc;
        for(int r=0;r<20;r++){bool rev=r&1;for(int v=0;v<2;v++){bool cand=(v==0)^rev;cudaEventRecord(st,stream);for(int z=0;z<iters;z++)cudaGraphLaunch(exec[cand?1:0],stream);cudaEventRecord(en,stream);cudaEventSynchronize(en);float ms;cudaEventElapsedTime(&ms,st,en);(cand?tc:tb).push_back(ms/iters);}}
        auto stats=[](std::vector<float>x){std::sort(x.begin(),x.end());return std::array<float,3>{x[x.size()/2],x.front(),x.back()};};auto B=stats(tb),C=stats(tc);
        printf("K=%d rows=%d rpc=%d nblk=%d base_bytes=%zu direct_bytes=%zu base_us=%.6f range=[%.6f,%.6f] direct_us=%.6f range=[%.6f,%.6f] delta_pct=%.3f mismatches=%d\n",bpr,rows,rpc,bpr,n*sizeof(Base),n*sizeof(Side),B[0]*1000,B[1]*1000,B[2]*1000,C[0]*1000,C[1]*1000,C[2]*1000,100*(C[0]/B[0]-1),mism);
        printf("  samples_base_us=");for(float x:tb)printf("%.6f,",x*1000);printf("\n  samples_direct_us=");for(float x:tc)printf("%.6f,",x*1000);printf("\n");
        cudaEventDestroy(st);cudaEventDestroy(en);for(int v=0;v<2;v++){cudaGraphExecDestroy(exec[v]);cudaGraphDestroy(graph[v]);}cudaStreamDestroy(stream);
        cudaFree(db);cudaFree(ds);cudaFree(dr);cudaFree(d_bad);cudaFree(da);cudaFree(do0);cudaFree(do1);free(ha);
    }
}
