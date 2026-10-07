#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
extern "C" void ggml_cuda_exp079_ptq1_gemv(const void*,const void*,float*,cudaStream_t);
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2;} } while(0)
struct W { uint8_t qs[24], qh[2]; __half d; };
__global__ void evict(uint8_t *p,size_t n){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]=(uint8_t)(i*13+7);}
static std::vector<uint8_t> readfile(const std::string&p){std::ifstream f(p,std::ios::binary);return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),{});}
static int trit(const W&w,int e){uint8_t b;int n;if(e<80){b=w.qs[e&15];n=e>>4;}else if(e<120){int t=e-80;b=w.qs[16+(t&7)];n=t>>3;}else{int t=e-120;b=w.qh[t&1];n=t>>1;}uint32_t v=b;for(int i=0;i<4;i++)if(i<n)v=(v*3)&255;return int((v*3)>>8)-1;}
int main(int argc,char**argv){if(argc!=3)return 1;std::string dir=argv[1],actpath=argv[2];std::vector<std::vector<uint8_t>> hW(32);for(int i=0;i<32;i++)hW[i]=readfile(dir+"/"+std::to_string(i)+".bin");auto hY=readfile(actpath);if(hY.size()!=1152||hW[0].size()!=1146880)return 3;
 std::vector<int8_t> aq(1024);std::vector<float> yd(32);std::vector<int16_t> isum(32);for(int kb=0;kb<8;kb++){for(int t=0;t<8;t++)for(int b=0;b<16;b++)aq[kb*128+t*16+b]=(int8_t)hY[t*128+kb*16+b];for(int s=0;s<4;s++){uint8_t* p=hY.data()+1024+kb*16+s*4;uint16_t half;memcpy(&half,p,2);yd[kb*4+s]=__half2float(__ushort_as_half(half));memcpy(&isum[kb*4+s],p+2,2);}}
 std::vector<float> ref(5120);for(int row=0;row<5120;row++){float part[8]={};for(int kb=0;kb<8;kb++){const W&w=((const W*)hW[0].data())[row*8+kb];float a=0;for(int s=0;s<4;s++){int sum=0;for(int i=0;i<32;i++)sum+=trit(w,kb*128+s*32+i)*aq[kb*128+s*32+i];a=std::fma(yd[kb*4+s],float(sum-isum[kb*4+s]),a);}part[kb]=__half2float(w.d)*a;}ref[row]=(part[0]+part[1])+(part[2]+part[3]);ref[row]+=(part[4]+part[5])+(part[6]+part[7]);}
 cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,0));int lim=0;CK(cudaDeviceGetAttribute(&lim,cudaDevAttrMaxPersistingL2CacheSize,0));fprintf(stderr,"L2=%d persisting_max=%d window_max=%d\n",prop.l2CacheSize,lim,prop.accessPolicyMaxWindowSize);
 std::vector<void*> dW(32);for(int i=0;i<32;i++){CK(cudaMalloc(&dW[i],hW[i].size()));CK(cudaMemcpy(dW[i],hW[i].data(),hW[i].size(),cudaMemcpyHostToDevice));}void*dY,*dOut,*dEv;CK(cudaMalloc(&dY,hY.size()));CK(cudaMalloc(&dOut,5120*sizeof(float)));size_t evbytes=32ull<<20;CK(cudaMalloc(&dEv,evbytes));CK(cudaMemcpy(dY,hY.data(),hY.size(),cudaMemcpyHostToDevice));cudaStream_t st;CK(cudaStreamCreate(&st));
 ggml_cuda_exp079_ptq1_gemv(dW[0],dY,(float*)dOut,st);CK(cudaStreamSynchronize(st));std::vector<float>got(5120);CK(cudaMemcpy(got.data(),dOut,got.size()*4,cudaMemcpyDeviceToHost));int mism=0;float maxabs=0;for(int i=0;i<5120;i++){if(got[i]!=ref[i])mism++;maxabs=fmaxf(maxabs,fabsf(got[i]-ref[i]));}fprintf(stderr,"CPU exact equality mismatches=%d maxabs=%g out0=%g ref0=%g\n",mism,maxabs,got[0],ref[0]);
 for(int count: {1,4,32})for(int policy=0;policy<=1;policy++){cudaGraph_t g;cudaGraphExec_t ge;CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));for(int i=0;i<count;i++)ggml_cuda_exp079_ptq1_gemv(dW[i],dY,(float*)dOut,st);CK(cudaStreamEndCapture(st,&g));size_t nn=0;CK(cudaGraphGetNodes(g,nullptr,&nn));std::vector<cudaGraphNode_t>nodes(nn);CK(cudaGraphGetNodes(g,nodes.data(),&nn));int kn=0;if(policy){cudaKernelNodeAttrValue v{};v.accessPolicyWindow={dW[0],size_t(count)*1146880,1.0f,cudaAccessPropertyPersisting,cudaAccessPropertyStreaming};for(auto n:nodes){cudaGraphNodeType ty;CK(cudaGraphNodeGetType(n,&ty));if(ty==cudaGraphNodeTypeKernel){cudaError_t e=cudaGraphKernelNodeSetAttribute(n,cudaKernelNodeAttributeAccessPolicyWindow,&v);if(e!=cudaSuccess){fprintf(stderr,"policy count=%d rejected: %s\n",count,cudaGetErrorString(e));policy=-1;break;}kn++;}}}CK(cudaGraphInstantiate(&ge,g,0,0,0));cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));for(int z=0;z<10;z++)CK(cudaGraphLaunch(ge,st));CK(cudaStreamSynchronize(st));
 for(int cold=0;cold<=1;cold++){for(int rep=0;rep<25;rep++){if(cold){evict<<<(evbytes+255)/256,256,0,st>>>((uint8_t*)dEv,evbytes);CK(cudaGetLastError());}CK(cudaEventRecord(a,st));CK(cudaGraphLaunch(ge,st));CK(cudaEventRecord(b,st));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));printf("%d,%d,%d,%d,%.6f\n",count,policy,cold,rep,ms);} }CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));CK(cudaGraphExecDestroy(ge));CK(cudaGraphDestroy(g));}
 cudaDeviceSynchronize();for(void*p:dW)cudaFree(p);cudaFree(dY);cudaFree(dOut);cudaFree(dEv);cudaStreamDestroy(st);return mism?4:0;}
