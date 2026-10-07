#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void alias_kernel(const float *x, const float *w, float *out, float eps) {
  int tid=threadIdx.x, group=blockIdx.y;
  float v=0.f;
  #pragma unroll
  for(int j=0;j<4;++j) v += x[group*128*4 + tid*4+j]*w[group*128*4 + tid*4+j];
  v = v / (1.f + expf(-v));
  out[group*128+tid]=v;
  cg::this_grid().sync();
  if(group<32) {
    extern __shared__ float sm[];
    if(tid<32) { float sum=0.f; for(int c=tid;c<128;c+=32){float a=out[group*128+c];sum += a*a;} sm[tid]=sum; }
    __syncthreads();
    if(tid<32) { float sum=sm[tid]+sm[(tid+1)%32]+sm[(tid+2)%32]+sm[(tid+3)%32]+sm[(tid+4)%32]+sm[(tid+5)%32]+sm[(tid+6)%32]+sm[(tid+7)%32]; sm[tid]=sum; }
    __syncthreads();
    float norm=rsqrtf(fmaxf(sm[0],eps*eps));
    out[group*128+tid] *= norm;
  }
}
#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 2;} } while(0)
int main(){
 int dev=0, coop=0, sms=0; CK(cudaSetDevice(dev)); CK(cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,dev)); CK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev));
 int active=0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,alias_kernel,128,0)); cudaFuncAttributes a{}; CK(cudaFuncGetAttributes(&a,alias_kernel));
 printf("coop=%d sms=%d maxBlocksPerSM=%d totalGridCap=%d regs/thread=%d static_smem=%zu local=%zu\n",coop,sms,active,sms*active,a.numRegs,a.sharedSizeBytes,a.localSizeBytes);
 float *x,*w,*o; CK(cudaMalloc(&x,10240*4)); CK(cudaMalloc(&w,10240*4)); CK(cudaMalloc(&o,10240*4)); CK(cudaMemset(x,0,10240*4)); CK(cudaMemset(w,0,10240*4));
 void *args[]={&x,&w,&o,(void*)nullptr}; float eps=1e-6f; args[3]=&eps;
 CK(cudaLaunchCooperativeKernel((void*)alias_kernel,dim3(1,80,1),dim3(128),args,32*4,0)); CK(cudaDeviceSynchronize());
 cudaStream_t s; CK(cudaStreamCreate(&s)); CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal)); CK(cudaLaunchCooperativeKernel((void*)alias_kernel,dim3(1,80,1),dim3(128),args,32*4,s)); cudaGraph_t g; CK(cudaStreamEndCapture(s,&g)); cudaGraphExec_t ex; CK(cudaGraphInstantiate(&ex,g,0)); CK(cudaGraphLaunch(ex,s)); CK(cudaStreamSynchronize(s)); size_t n=0; CK(cudaGraphGetNodes(g,nullptr,&n)); printf("capture=success nodes=%zu replay=success\n",n);
 CK(cudaGraphExecDestroy(ex)); CK(cudaGraphDestroy(g)); CK(cudaStreamDestroy(s)); CK(cudaFree(x)); CK(cudaFree(w)); CK(cudaFree(o));
}
