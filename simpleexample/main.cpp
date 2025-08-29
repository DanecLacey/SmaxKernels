#include "SmaxKernels/interface.hpp"

__global__ void my_kernel()
{
    printf("the tid.x %d and blkidx.x %d blkdim.x %d\n", threadIdx.x, blockIdx.x, blockDim.x);
}

int main(int argc, char const *argv[])
{
    SMAX::gpu_stream str;
    my_kernel<<<2,4,0,str.get()>>>();
    cudaDeviceSynchronize();
    return 0;
}
