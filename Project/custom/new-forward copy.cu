#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>
// #include <chrono>
cudaStream_t stream1;
__constant__ half KERNEL_DEVICE_CST[3136];

// __global__ void conv_forward_kernel_basic(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
// {
//     /*
//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     B - batch_size (number of images in x)
//     M - number of output feature maps
//     C - number of input feature maps
//     H - input height dimension
//     W - input width dimension
//     K - kernel height and width (K x K)
//     S - stride step length
//     */
//     const int H_out = (H - K)/S + 1;
//     const int W_out = (W - K)/S + 1;
//     #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]    // out_4d(b, m, h, w)
//     #define in_4d_global(i3, i2, i1, i0) (input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0])     // in_4d(b, c, cell_height, cell_width)
//     #define mask_4d(i3, i2, i1, i0) (KERNEL_DEVICE_CST[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0])                         // mask_4d(m, c, mask_heightindex, mask_widthindex)
//     // Insert your GPU convolution kernel code here
//     const int tile_width = blockDim.x;
//     const int tile_height = blockDim.y;
//     const int W_grid_blocks = (W_out - 1) / tile_width + 1;  //tiles in outputWidth
//     const int m_feature = blockIdx.x;
//     const int b = blockIdx.z;
//     const int output_h = (blockIdx.y / W_grid_blocks) * tile_height + threadIdx.y;
//     const int output_w = (blockIdx.y % W_grid_blocks) * tile_width + threadIdx.x;
//     // starting index for current Block
//     const int input_h_start = output_h * S; 
//     const int input_w_start = output_w * S;
//     int input_x;// input-x index
//     int input_y;// input-y index
//     float acc = 0.0f;
//     if((output_h < H_out) && (output_w < W_out)){
//         for(int c = 0; c < C; ++c){   // sum over all input channels
//             for(int j = 0; j < K; ++j){   // KxK filter (height)
//                 input_y = input_h_start + j;
//                 for(int i = 0; i < K; ++i){   // KxK filter (width)
//                     input_x = input_w_start + i;
//                     acc += in_4d_global(b, c, input_y, input_x) * mask_4d(m_feature, c, j, i); 
//                 }
//             }
//         }
//         out_4d(b, m_feature, output_h, output_w) = acc;
//     }
//     #undef out_4d
//     #undef in_4d_global
//     #undef mask_4d
// }

__global__ void conv_forward_kernel_basic_16FP(float* __restrict__ output, const half* __restrict__ input, const half* __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]    // out_4d(b, m, h, w)
    #define in_4d_global(i3, i2, i1, i0) (input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0])     // in_4d(b, c, cell_height, cell_width)
    #define mask_4d(i3, i2, i1, i0) (KERNEL_DEVICE_CST[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0])                         // mask_4d(m, c, mask_heightindex, mask_widthindex)
    // Insert your GPU convolution kernel code here
    const int tile_width = blockDim.x;
    const int tile_height = blockDim.y;
    const int W_grid_blocks = (W_out - 1) / tile_width + 1;  //tiles in outputWidth
    const int m_feature = blockIdx.x;
    const int b = blockIdx.z;
    const int output_h = (blockIdx.y / W_grid_blocks) * tile_height + threadIdx.y;
    const int output_w = (blockIdx.y % W_grid_blocks) * tile_width + threadIdx.x;
    // starting index for current Block
    const int input_h_start = output_h * S; 
    const int input_w_start = output_w * S;
    int input_x;// input-x index
    int input_y;// input-y index
    half acc = __float2half(0.0f);
    // float acc = 0.0f;
    if((output_h < H_out) && (output_w < W_out)){
        for(int c = 0; c < C; ++c){   // sum over all input channels
            for(int j = 0; j < K; ++j){   // KxK filter (height)
                input_y = input_h_start + j;
                for(int i = 0; i < K; ++i){   // KxK filter (width)
                    input_x = input_w_start + i;
                    acc = __hadd(acc, __hmul(in_4d_global(b, c, input_y, input_x), mask_4d(m_feature, c, j, i)));
                    // acc += in_4d_global(b, c, input_y, input_x) * mask_4d(m_feature, c, j, i);
                    // acc = __hfma(in_4d_global(b, c, input_y, input_x), mask_4d(m_feature, c, j, i), acc);
                }
            }
        }
        out_4d(b, m_feature, output_h, output_w) = __half2float(acc);
    }
    #undef out_4d
    #undef in_4d_global
    #undef mask_4d
}

__global__ void conv_forward_kernel_basic_16FP_convLayerK7_CnstMask(float* __restrict__ output, const half* __restrict__ input, const half* __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]    // out_4d(b, m, h, w)
    #define in_4d_global(i3, i2, i1, i0) (input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0])     // in_4d(b, c, cell_height, cell_width)
    #define mask_4d(i3, i2, i1, i0) (KERNEL_DEVICE_CST[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0])                         // mask_4d(m, c, mask_heightindex, mask_widthindex)
    // Insert your GPU convolution kernel code here
    const int tile_width = blockDim.x;
    const int tile_height = blockDim.y;
    const int W_grid_blocks = (W_out - 1) / tile_width + 1;  //tiles in outputWidth
    const int m_feature = blockIdx.x;
    const int b = blockIdx.z;
    const int output_h = (blockIdx.y / W_grid_blocks) * tile_height + threadIdx.y;
    const int output_w = (blockIdx.y % W_grid_blocks) * tile_width + threadIdx.x;
    // starting index for current Block
    const int input_h_start = output_h * S; 
    const int input_w_start = output_w * S;
    int input_x;// input-x index
    int input_y;// input-y index
    half acc = __float2half(0.0f);
    if((output_h < H_out) && (output_w < W_out)){
        for(int c = 0; c < C; ++c){   // sum over all input channels
            #pragma unroll 7
            for(int j = 0; j < K; ++j){   // KxK filter (height)
                input_y = input_h_start + j;
                #pragma unroll 7
                for(int i = 0; i < K; ++i){   // KxK filter (width)
                    input_x = input_w_start + i;
                    acc = __hadd(acc, __hmul(in_4d_global(b, c, input_y, input_x), mask_4d(m_feature, c, j, i)));
                    // acc += in_4d_global(b, c, input_y, input_x) * mask_4d(m_feature, c, j, i);
                }
            }
        }
        out_4d(b, m_feature, output_h, output_w) = __half2float(acc);
    }
    #undef out_4d
    #undef in_4d_global
    #undef mask_4d
}
// converts arrays to half in gpu
__global__ void convertFloatToHalf(half *output, const float *input, const int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        output[idx] = __float2half(input[idx]);
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    cudaStreamCreate(&stream1);
    // Allocate memory and copy over the relevant data structures to the GPU
    const int nInputElements = (B * C * H * W);
    const int memSizeInput = nInputElements * sizeof(float);
    const int mMaskElements = (M * C * K * K);
    const int memSizeMask = mMaskElements * sizeof(float);
    // std::cout << mMaskElements << "   n mask elements " << std::endl;
    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    const int nOutputElements = (B * M * outputHeight * outputWidth);
    const int memSizeOutput = nOutputElements * sizeof(float);
    
    // cudaStreamCreate(&stream2);
    cudaMalloc((void **)device_input_ptr, memSizeInput);
    cudaMalloc((void **)device_mask_ptr, memSizeMask);
    // cudaMemcpyToSymbol(KERNEL_DEVICE_CST, host_mask, memSizeMask);
    cudaMemcpyAsync(*device_input_ptr, host_input, memSizeInput, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(*device_mask_ptr, host_mask, memSizeMask, cudaMemcpyHostToDevice, stream1);
    cudaHostRegister(const_cast<float*>(host_output), memSizeOutput, cudaHostRegisterDefault);
    cudaMalloc((void **)device_output_ptr, memSizeOutput);
    // std::cout<<"mMaskElements: "<<mMaskElements<<std::endl;
    // auto start6 = std::chrono::high_resolution_clock::now();
    // cudaHostRegister(const_cast<float*>(host_output), memSizeOutput, cudaHostRegisterDefault);
    // auto stop6 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration6 = stop6 - start6;
    // std::cout << "Output memory Pinning took " << duration6.count()*1000 << " ms" << std::endl;
    // get_device_properties();
    // // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int mMaskElements = (M * C * K * K);
    const int memSizeMask = mMaskElements * sizeof(float);
    const int memSizeMaskHalf = mMaskElements * sizeof(half);

    half *device_input_half;
    half *device_mask_half;
    const int nInputElements = (B * C * H * W);
    const int memSizeInput_half = nInputElements * sizeof(half);

    cudaMalloc((void **)&device_input_half, memSizeInput_half);
    cudaMalloc((void **)&device_mask_half, memSizeMaskHalf);
    const int blockSizeFP16Converter = 128;
    const int blockSizeFP16mask = 32;
    const int gridSizeFP16ConverterInput = (nInputElements + blockSizeFP16Converter - 1) / blockSizeFP16Converter;
    const int gridSizeFP16ConverterMask = (mMaskElements + blockSizeFP16Converter - 1) / blockSizeFP16mask;
    convertFloatToHalf<<<gridSizeFP16ConverterMask, blockSizeFP16mask, 0, stream1>>>(device_mask_half, device_mask, mMaskElements);
    convertFloatToHalf<<<gridSizeFP16ConverterInput, blockSizeFP16Converter, 0, stream1>>>(device_input_half, device_input, nInputElements);
    cudaMemcpyToSymbol(KERNEL_DEVICE_CST, device_mask_half, memSizeMaskHalf);

    
    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    // std::cout << outputWidth << " x " << outputHeight << " x " << C << " and K is " << K << " and S is " << S << std::endl;
    int TILE_WIDTH = 6;
    int TILE_HEIGHT = 48;
    if(outputWidth==80){
        TILE_WIDTH = 16;
        TILE_HEIGHT = 16;
    }
    else if(outputWidth==34){
        TILE_WIDTH = 8;
        TILE_HEIGHT = 48;
    }
    int H_grid_blocks = (outputHeight - 1) / TILE_HEIGHT + 1; //tiles in outputHeight
    int W_grid_blocks = (outputWidth - 1) / TILE_WIDTH + 1;  //tiles in outputWidth
    int nTiles = H_grid_blocks * W_grid_blocks; // total tiles
    // int sharedMemConvSize = (TILE_WIDTH * TILE_HEIGHT * S * S * C) * sizeof(float);
    // while (sharedMemConvSize > 49152){
    //     TILE_HEIGHT /= 2;
    //     // W_grid_blocks = (outputWidth - 1) / TILE_WIDTH + 1;
    //     H_grid_blocks = (outputHeight - 1) / TILE_HEIGHT + 1; //tiles in outputHeight
    //     nTiles = H_grid_blocks * W_grid_blocks; // total tiles
    //     sharedMemConvSize = (TILE_WIDTH * TILE_HEIGHT * S * S * C) * sizeof(float);
    //     // std::cout<<"REsizing "<<std::endl;
    // }
    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT, 1);
    dim3 dimGrid(M, nTiles, B); // Ensuring all elements are covered
    // cudaStreamSynchronize(stream1);
    if(K==7){
        conv_forward_kernel_basic_16FP_convLayerK7_CnstMask<<<dimGrid, dimBlock, 0, stream1>>>(device_output, device_input_half, device_mask_half, B, M, C, H, W, K, S);
    }
    else{
        conv_forward_kernel_basic_16FP<<<dimGrid, dimBlock, 0 , stream1>>>(device_output, device_input_half, device_mask_half, B, M, C, H, W, K, S);
    }
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Calculate output size and memory size for half-precision
    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    const int nOutputElements = (B * M * outputHeight * outputWidth);
    const int memSizeOutput = nOutputElements * sizeof(float);
    // cudaHostRegister(host_output, memSizeOutput, cudaHostRegisterDefault);
    // cudaStreamSynchronize(stream1);
    cudaMemcpyAsync(host_output, device_output, memSizeOutput, cudaMemcpyDeviceToHost,stream1);
    cudaHostUnregister(host_output);
    
    // auto start4 = std::chrono::high_resolution_clock::now();
    // cudaMemcpy(host_output, device_output, memSizeOutput, cudaMemcpyDeviceToHost);
    // cudaHostUnregister(host_output);
    // auto stop4 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration4 = stop4 - start4;
    // std::cout << "FinalMemOps took " << duration4.count()*1000 << " ms" << std::endl;
    
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
    cudaStreamDestroy(stream1);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
