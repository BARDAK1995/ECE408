#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return 1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);

        std::cout << "Device " << i << " properties:" << std::endl;
        std::cout << "  Name: " << devProp.name << std::endl;
        std::cout << "  Total global memory: " << devProp.totalGlobalMem << std::endl;
        std::cout << "  Shared memory per block: " << devProp.sharedMemPerBlock << std::endl;
        std::cout << "  Registers per block: " << devProp.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << devProp.warpSize << std::endl;
        std::cout << "  Maximum threads per block: " << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Maximum thread dimensions: (" 
                  << devProp.maxThreadsDim[0] << ", " 
                  << devProp.maxThreadsDim[1] << ", " 
                  << devProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Maximum grid dimensions: (" 
                  << devProp.maxGridSize[0] << ", " 
                  << devProp.maxGridSize[1] << ", " 
                  << devProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Compute capability: " << devProp.major << "." << devProp.minor << std::endl;
        std::cout << "  Max threads per multi-processor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Max blocks per multi-processor: " << devProp.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "  Multi-processor count: " << devProp.multiProcessorCount << std::endl;
    }

    return 0;
}
