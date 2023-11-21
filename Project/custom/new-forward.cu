#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

__constant__ half KERNEL_DEVICE_CST[3136];

__global__ void conv_forward_kernel_basic(half *output, const half *input, const half *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
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
    #define in_4d_global(i3, i2, i1, i0) __half2float(input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0])     // in_4d(b, c, cell_height, cell_width)
    #define mask_4d(i3, i2, i1, i0) __half2float(mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0])                         // mask_4d(m, c, mask_heightindex, mask_widthindex)
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
    float acc = 0.0f;
    if((output_h < H_out) && (output_w < W_out)){
        for(int c = 0; c < C; ++c){   // sum over all input channels
            for(int j = 0; j < K; ++j){   // KxK filter (height)
                input_y = input_h_start + j;
                for(int i = 0; i < K; ++i){   // KxK filter (width)
                    input_x = input_w_start + i;
                    acc += in_4d_global(b, c, input_y, input_x) * mask_4d(m_feature, c, j, i); 
                }
            }
        }
        out_4d(b, m_feature, output_h, output_w) = __float2half(acc);
    }
    #undef out_4d
    #undef in_4d_global
    #undef mask_4d
}
__global__ void conv_forward_kernel_basic_16FP(half* __restrict__ output, const half* __restrict__ input, const half* __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
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
    #define in_4d_global(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]     // in_4d(b, c, cell_height, cell_width)
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]                         // mask_4d(m, c, mask_heightindex, mask_widthindex)
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
            for(int j = 0; j < K; ++j){   // KxK filter (height)
                input_y = input_h_start + j;
                for(int i = 0; i < K; ++i){   // KxK filter (width)
                    input_x = input_w_start + i;
                    acc = __hadd(acc, __hmul(in_4d_global(b, c, input_y, input_x), mask_4d(m_feature, c, j, i)));
                    // acc = __hfma(in_4d_global(b, c, input_y, input_x), mask_4d(m_feature, c, j, i), acc);
                }
            }
        }
        out_4d(b, m_feature, output_h, output_w) = acc;
    }
    #undef out_4d
    #undef in_4d_global
    #undef mask_4d
}

__global__ void conv_forward_kernel_basic_16FP_convLayerK7(half* __restrict__ output, const half* __restrict__ input, const half* __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
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
    #define in_4d_global(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]     // in_4d(b, c, cell_height, cell_width)
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]                         // mask_4d(m, c, mask_heightindex, mask_widthindex)
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
                    // acc = __hfma(in_4d_global(b, c, input_y, input_x), mask_4d(m_feature, c, j, i), acc);
                }
            }
        }
        out_4d(b, m_feature, output_h, output_w) = acc;
    }
    #undef out_4d
    #undef in_4d_global
    #undef mask_4d
}
// __global__ void conv_forward_kernel_basic_16FP_half2vectorized(half2 *output, const half2 *input, const half2 *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
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
//     #define in_4d_global(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]     // in_4d(b, c, cell_height, cell_width)
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]                         // mask_4d(m, c, mask_heightindex, mask_widthindex)
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
//     half acc = __float2half2_rn(0.0f);
//     if((output_h < H_out) && (output_w < W_out)){
//         for(int c = 0; c < C; ++c){   // sum over all input channels
//             for(int j = 0; j < K; ++j){   // KxK filter (height)
//                 input_y = input_h_start + j;
//                 for(int i = 0; i < K; ++i){   // KxK filter (width)
//                     input_x = input_w_start + i;
//                     acc = __hadd2(acc, __hmul(in_4d_global(b, c, input_y, input_x), mask_4d(m_feature, c, j, i)));
//                     // acc = __hfma(in_4d_global(b, c, input_y, input_x), mask_4d(m_feature, c, j, i), acc);
//                 }
//             }
//         }
//         out_4d(b, m_feature, output_h, output_w) = acc;
//     }
//     #undef out_4d
//     #undef in_4d_global
//     #undef mask_4d
// }
// __global__ void conv_forward_kernel_ConstantMem(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
// {
//     /*
//     Function paramter definitions:
//     output - output
//     input - input
//     KERNEL_DEVICE_CST - convolution kernel mask in constant MEM
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
//     #define in_4d_global(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]                          // in_4d(b, c, cell_height, cell_width)
//     #define mask_4d(i3, i2, i1, i0) KERNEL_DEVICE_CST[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]                         // mask_4d(m, c, mask_heightindex, mask_widthindex)
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

// __global__ void conv_forward_kernel_ConstantMem_bigstride(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
// {
//     /*
//     Function paramter definitions:
//     output - output
//     input - input
//     KERNEL_DEVICE_CST - convolution kernel mask in constant MEM
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
//     #define in_4d_global(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]                          // in_4d(b, c, cell_height, cell_width)
//     #define mask_4d(i3, i2, i1, i0) KERNEL_DEVICE_CST[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]                         // mask_4d(m, c, mask_heightindex, mask_widthindex)
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
// __global__ void conv_forward_kernel_ConstantMem_SharedMem(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
// {
//     /*
//     Function paramter definitions:
//     output - output
//     input - input
//     KERNEL_DEVICE_CST - convolution kernel mask in constant MEM
//     B - batch_size (number of images in x)
//     M - number of output feature maps
//     C - number of input feature maps
//     H - input height dimension
//     W - input width dimension
//     K - kernel height and width (K x K)
//     S - stride step length
//     */
//     extern __shared__ float N_ds[]; //size determined dynamicly at runtime, we will rely on cache to catch others
//     const int H_out = (H - K)/S + 1;
//     const int W_out = (W - K)/S + 1;
//     const int tile_width = blockDim.x;
//     const int tile_height = blockDim.y;
//     const int SharedMatrix_width = tile_width * S;
//     const int SharedMatrix_height = tile_height * S;
//     #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]    // out_4d(b, m, h, w)
//     #define in_4d_global(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]                          // in_4d_global(b, c, cell_height, cell_width)
//     #define in_4d_shared(i2, i1, i0) N_ds[(i2) * (SharedMatrix_height * SharedMatrix_width) + (i1) * (SharedMatrix_width) + i0]                          // in_4d_shared(c, cell_height, cell_width)
//     #define mask_4d(i3, i2, i1, i0) KERNEL_DEVICE_CST[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]                         // mask_4d(m, c, mask_heightindex, mask_widthindex)
//     // Insert your GPU convolution kernel code here
//     const int W_grid_blocks = (W_out - 1) / tile_width + 1;  //tiles in outputWidth
//     const int m_feature = blockIdx.x;
//     const int b = blockIdx.z;
//     // y and x indexes for the output matrix
//     const int output_h = (blockIdx.y / W_grid_blocks) * tile_height + threadIdx.y;
//     const int output_w = (blockIdx.y % W_grid_blocks) * tile_width + threadIdx.x;
//     // corresponding y and x starting indexes for input matrix for the current block 
//     const int input_y_Block_start = ((blockIdx.y / W_grid_blocks) * tile_height) * S; 
//     const int input_x_Block_start = ((blockIdx.y % W_grid_blocks) * tile_width) * S;
//     int input_x;// input-x index
//     int input_y;// input-y index
//     int shared_x;// shared-x index
//     int shared_y;// shared-y index
//     // starting index for current Block
//     //load Shared Memory
//     for(int c = 0; c < C; ++c){
//         for (int scountery = 0; scountery < S; ++scountery){
//             shared_y = threadIdx.y + scountery * tile_height;
//             input_y = input_y_Block_start + shared_y;
//             bool is_Y_outbound = input_y > H;
//             for (int scounterx = 0; scounterx < S; ++scounterx){
//                 shared_x = threadIdx.x + scounterx * tile_width;
//                 input_x = input_x_Block_start + shared_x;
//                 bool is_X_outbound = input_x > W;
//                 //INDEXING OVER C in the outermost layer, to not mess up the coalescedd memory acces
//                 if(is_Y_outbound && is_X_outbound){
//                     in_4d_shared(c, shared_y, shared_x) = 0.0f;
//                 }
//                 else {
//                     in_4d_shared(c, shared_y, shared_x) = in_4d_global(b, c, input_y, input_x);
//                 }  
//             }
//         }
//     }
//     __syncthreads();
//     const int input_h_start = output_h * S; 
//     const int input_w_start = output_w * S;
//     float acc = 0.0f;
//     if((output_h < H_out) && (output_w < W_out)){
//         for(int c = 0; c < C; ++c){   // sum over all input channels
//             for(int j = 0; j < K; ++j){   // KxK filter (height)
//                 input_y = input_h_start + j;
//                 shared_y = input_y - input_y_Block_start; //where it is in corresponding input tile, we use this to determine if its in shared mem or not.
//                 bool is_y_in_bounds = shared_y < SharedMatrix_height;
//                 for(int i = 0; i < K; ++i){   // KxK filter (width)
//                     input_x = input_w_start + i;
//                     shared_x = input_x - input_x_Block_start; //where it is in corresponding input tile, we use this to determine if its in shared mem or not.
//                     bool is_x_in_bounds = shared_x < SharedMatrix_width;
//                     if(is_y_in_bounds && is_x_in_bounds){
//                         acc += in_4d_shared(c, shared_y, shared_x) * mask_4d(m_feature, c, j, i); 
//                     }
//                     else{
//                         acc += in_4d_global(b, c, input_y, input_x) * mask_4d(m_feature, c, j, i); 
//                     }
//                 }
//             }
//         }
//         out_4d(b, m_feature, output_h, output_w) = acc;
//     }
//     #undef out_4d
//     #undef in_4d_global
//     #undef mask_4d
// }

// __host__ void convertFp32ToFp16(half* output, const float* input, const int size) {
//     for (int i = 0; i < size; i++) {
//         output[i] = __float2half(input[i]);
//     }
// }
__host__ void convertFp32ToFp16(half* __restrict__ output, const float* __restrict__ input, const int size) {
    int i;
    // Process 8 elements per iteration
    for (i = 0; i <= size - 16; i += 16) {
        output[i] = __float2half(input[i]);
        output[i + 1] = __float2half(input[i + 1]);
        output[i + 2] = __float2half(input[i + 2]);
        output[i + 3] = __float2half(input[i + 3]);
        output[i + 4] = __float2half(input[i + 4]);
        output[i + 5] = __float2half(input[i + 5]);
        output[i + 6] = __float2half(input[i + 6]);
        output[i + 7] = __float2half(input[i + 7]);
        output[i + 8] = __float2half(input[i + 8]);
        output[i + 9] = __float2half(input[i + 9]);
        output[i + 10] = __float2half(input[i + 10]);
        output[i + 11] = __float2half(input[i + 11]);
        output[i + 12] = __float2half(input[i + 12]);
        output[i + 13] = __float2half(input[i + 13]);
        output[i + 14] = __float2half(input[i + 14]);
        output[i + 15] = __float2half(input[i + 15]);
    }
    // Process remaining elements
    for (; i < size; i++) {
        output[i] = __float2half(input[i]);
    }
}

__host__ void convertFp16ToFp32Inplace(float* output, half* input, const int size) {
    int i;
    // Process 8 elements per iteration
    for (i = size - 1; i >= 15; i -= 16) {
        output[i] = __half2float(input[i]);
        output[i - 1] = __half2float(input[i - 1]);
        output[i - 2] = __half2float(input[i - 2]);
        output[i - 3] = __half2float(input[i - 3]);
        output[i - 4] = __half2float(input[i - 4]);
        output[i - 5] = __half2float(input[i - 5]);
        output[i - 6] = __half2float(input[i - 6]);
        output[i - 7] = __half2float(input[i - 7]);
        output[i - 8] = __half2float(input[i - 8]);
        output[i - 9] = __half2float(input[i - 9]);
        output[i - 10] = __half2float(input[i - 10]);
        output[i - 11] = __half2float(input[i - 11]);
        output[i - 12] = __half2float(input[i - 12]);
        output[i - 13] = __half2float(input[i - 13]);
        output[i - 14] = __half2float(input[i - 14]);
        output[i - 15] = __half2float(input[i - 15]);
    }
    // Process remaining elements
    for (; i >= 0; i--) {
        output[i] = __half2float(input[i]);
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int nInputElements = (B * C * H * W);
    const int memSizeInputHalf = nInputElements * sizeof(half);
    const int mMaskElements = (M * C * K * K);
    const int memSizeMaskHalf = mMaskElements * sizeof(half);
    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    const int memSizeOutputhalf = (B * M * outputHeight * outputWidth) * sizeof(half);
    half* hostInput_half;
    half* hostMask_half;
    cudaHostAlloc((void**)&hostInput_half, memSizeInputHalf, cudaHostAllocDefault);
    cudaHostAlloc((void**)&hostMask_half, memSizeMaskHalf, cudaHostAllocDefault);
    cudaMalloc((void **)device_input_ptr, memSizeInputHalf);
    cudaMalloc((void **)device_mask_ptr, memSizeMaskHalf);
    cudaMalloc((void **)device_output_ptr, memSizeOutputhalf);

    convertFp32ToFp16(hostInput_half, host_input, nInputElements);
    convertFp32ToFp16(hostMask_half, host_mask, mMaskElements);
    cudaMemcpy(*device_input_ptr, hostInput_half, memSizeInputHalf, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, hostMask_half, memSizeMaskHalf, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(KERNEL_DEVICE_CST, hostMask_half, memSizeMaskHalf);
    cudaFreeHost(hostInput_half);
    cudaFreeHost(hostMask_half);
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
    // const int memSizeMask = (M * C * K * K) * sizeof(float);
    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    std::cout << outputWidth << " x " << outputHeight << " x " << C << " and K is " << K << std::endl;
    int TILE_WIDTH = 6;
    int TILE_HEIGHT = 48;
    if(outputWidth==80){
        TILE_WIDTH = 16;
        TILE_HEIGHT = 16;
    }
    if(outputWidth==34){
        TILE_WIDTH = 8;
        TILE_HEIGHT = 48;
    }
    int H_grid_blocks = (outputHeight - 1) / TILE_HEIGHT + 1; //tiles in outputHeight
    int W_grid_blocks = (outputWidth - 1) / TILE_WIDTH + 1;  //tiles in outputWidth
    int nTiles = H_grid_blocks * W_grid_blocks; // total tiles
    int sharedMemConvSize = (TILE_WIDTH * TILE_HEIGHT * S * S * C) * sizeof(float);
    while (sharedMemConvSize > 49152){
        TILE_HEIGHT /= 2;
        // W_grid_blocks = (outputWidth - 1) / TILE_WIDTH + 1;
        H_grid_blocks = (outputHeight - 1) / TILE_HEIGHT + 1; //tiles in outputHeight
        nTiles = H_grid_blocks * W_grid_blocks; // total tiles
        sharedMemConvSize = (TILE_WIDTH * TILE_HEIGHT * S * S * C) * sizeof(float);
        std::cout<<"REsizing "<<std::endl;
    }
    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT, 1);
    dim3 dimGrid(M, nTiles, B); // Ensuring all elements are covered
    // std::cout << "The memsize of sharedELementMatrix is: " << sharedMemConvSize << std::endl;     // max size is 49152
    const half* deviceInputHalf = reinterpret_cast<const half*>(device_input);
    const half* device_maskHalf = reinterpret_cast<const half*>(device_mask);
    half* device_outputHalf = reinterpret_cast<half*>(device_output);
    // conv_forward_kernel_basic<<<dimGrid, dimBlock>>>(device_outputHalf, deviceInputHalf, device_maskHalf, B, M, C, H, W, K, S);
    if(K==7){
        conv_forward_kernel_basic_16FP_convLayerK7<<<dimGrid, dimBlock>>>(device_outputHalf, deviceInputHalf, device_maskHalf, B, M, C, H, W, K, S);
    }
    else{
        conv_forward_kernel_basic_16FP<<<dimGrid, dimBlock>>>(device_outputHalf, deviceInputHalf, device_maskHalf, B, M, C, H, W, K, S);
    }
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Calculate output size and memory size for half-precision
    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    const int nOutputElements = (B * M * outputHeight * outputWidth);
    const int memSizeOutputHalf = nOutputElements * sizeof(half);
    half* device_outputHalf = reinterpret_cast<half*>(device_output);
    // Copy the half-precision output directly to the host_output array
    cudaMemcpy(host_output, device_outputHalf, memSizeOutputHalf, cudaMemcpyDeviceToHost);
    // In-place conversion from FP16 to FP32 within the host_output array
    convertFp16ToFp32Inplace(reinterpret_cast<float*>(host_output), reinterpret_cast<half*>(host_output), nOutputElements);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
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
