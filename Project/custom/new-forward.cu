#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
__constant__ float KERNEL_DEVICE_CST[3136];

__global__ void conv_forward_kernel_basic(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
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
    #define in_4d_global(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]                          // in_4d(b, c, cell_height, cell_width)
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
        out_4d(b, m_feature, output_h, output_w) = acc;
    }
    #undef out_4d
    #undef in_4d_global
    #undef mask_4d
}

__global__ void conv_forward_kernel_ConstantMemRestrict_Unroll(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Function paramter definitions:
    output - output
    input - input
    KERNEL_DEVICE_CST - convolution kernel mask in constant MEM
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
    #define in_4d_global(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]                          // in_4d(b, c, cell_height, cell_width)
    #define mask_4d(i3, i2, i1, i0) KERNEL_DEVICE_CST[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]                         // mask_4d(m, c, mask_heightindex, mask_widthindex)
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
    float acc = 0.0f;
    
    if((output_h < H_out) && (output_w < W_out)){
        for(int c = 0; c < C; ++c){   // sum over all input channels
            acc +=  in_4d_global(b, c, input_h_start + 0, input_w_start) * mask_4d(m_feature, c, 0, 0)
                +   in_4d_global(b, c, input_h_start + 1, input_w_start) * mask_4d(m_feature, c, 1, 0)
                +   in_4d_global(b, c, input_h_start + 2, input_w_start) * mask_4d(m_feature, c, 2, 0) 
                +   in_4d_global(b, c, input_h_start + 3, input_w_start) * mask_4d(m_feature, c, 3, 0) 
                +   in_4d_global(b, c, input_h_start + 4, input_w_start) * mask_4d(m_feature, c, 4, 0) 
                +   in_4d_global(b, c, input_h_start + 5, input_w_start) * mask_4d(m_feature, c, 5, 0) 
                +   in_4d_global(b, c, input_h_start + 6, input_w_start) * mask_4d(m_feature, c, 6, 0); 
            //x+1
            acc +=  in_4d_global(b, c, input_h_start + 0, input_w_start + 1) * mask_4d(m_feature, c, 0, 1)
                +   in_4d_global(b, c, input_h_start + 1, input_w_start + 1) * mask_4d(m_feature, c, 1, 1)
                +   in_4d_global(b, c, input_h_start + 2, input_w_start + 1) * mask_4d(m_feature, c, 2, 1) 
                +   in_4d_global(b, c, input_h_start + 3, input_w_start + 1) * mask_4d(m_feature, c, 3, 1) 
                +   in_4d_global(b, c, input_h_start + 4, input_w_start + 1) * mask_4d(m_feature, c, 4, 1) 
                +   in_4d_global(b, c, input_h_start + 5, input_w_start + 1) * mask_4d(m_feature, c, 5, 1) 
                +   in_4d_global(b, c, input_h_start + 6, input_w_start + 1) * mask_4d(m_feature, c, 6, 1); 
            //x+2
            acc +=  in_4d_global(b, c, input_h_start + 0, input_w_start + 2) * mask_4d(m_feature, c, 0, 2)
                +   in_4d_global(b, c, input_h_start + 1, input_w_start + 2) * mask_4d(m_feature, c, 1, 2)
                +   in_4d_global(b, c, input_h_start + 2, input_w_start + 2) * mask_4d(m_feature, c, 2, 2) 
                +   in_4d_global(b, c, input_h_start + 3, input_w_start + 2) * mask_4d(m_feature, c, 3, 2) 
                +   in_4d_global(b, c, input_h_start + 4, input_w_start + 2) * mask_4d(m_feature, c, 4, 2) 
                +   in_4d_global(b, c, input_h_start + 5, input_w_start + 2) * mask_4d(m_feature, c, 5, 2) 
                +   in_4d_global(b, c, input_h_start + 6, input_w_start + 2) * mask_4d(m_feature, c, 6, 2); 
            //x+3
            acc +=  in_4d_global(b, c, input_h_start + 0, input_w_start + 3) * mask_4d(m_feature, c, 0, 3)
                +   in_4d_global(b, c, input_h_start + 1, input_w_start + 3) * mask_4d(m_feature, c, 1, 3)
                +   in_4d_global(b, c, input_h_start + 2, input_w_start + 3) * mask_4d(m_feature, c, 2, 3) 
                +   in_4d_global(b, c, input_h_start + 3, input_w_start + 3) * mask_4d(m_feature, c, 3, 3) 
                +   in_4d_global(b, c, input_h_start + 4, input_w_start + 3) * mask_4d(m_feature, c, 4, 3) 
                +   in_4d_global(b, c, input_h_start + 5, input_w_start + 3) * mask_4d(m_feature, c, 5, 3) 
                +   in_4d_global(b, c, input_h_start + 6, input_w_start + 3) * mask_4d(m_feature, c, 6, 3); 
            //x+4
            acc +=  in_4d_global(b, c, input_h_start + 0, input_w_start + 4) * mask_4d(m_feature, c, 0, 4)
                +   in_4d_global(b, c, input_h_start + 1, input_w_start + 4) * mask_4d(m_feature, c, 1, 4)
                +   in_4d_global(b, c, input_h_start + 2, input_w_start + 4) * mask_4d(m_feature, c, 2, 4) 
                +   in_4d_global(b, c, input_h_start + 3, input_w_start + 4) * mask_4d(m_feature, c, 3, 4) 
                +   in_4d_global(b, c, input_h_start + 4, input_w_start + 4) * mask_4d(m_feature, c, 4, 4) 
                +   in_4d_global(b, c, input_h_start + 5, input_w_start + 4) * mask_4d(m_feature, c, 5, 4) 
                +   in_4d_global(b, c, input_h_start + 6, input_w_start + 4) * mask_4d(m_feature, c, 6, 4); 
            //x+5
            acc +=  in_4d_global(b, c, input_h_start + 0, input_w_start + 5) * mask_4d(m_feature, c, 0, 5)
                +   in_4d_global(b, c, input_h_start + 1, input_w_start + 5) * mask_4d(m_feature, c, 1, 5)
                +   in_4d_global(b, c, input_h_start + 2, input_w_start + 5) * mask_4d(m_feature, c, 2, 5) 
                +   in_4d_global(b, c, input_h_start + 3, input_w_start + 5) * mask_4d(m_feature, c, 3, 5) 
                +   in_4d_global(b, c, input_h_start + 4, input_w_start + 5) * mask_4d(m_feature, c, 4, 5) 
                +   in_4d_global(b, c, input_h_start + 5, input_w_start + 5) * mask_4d(m_feature, c, 5, 5) 
                +   in_4d_global(b, c, input_h_start + 6, input_w_start + 5) * mask_4d(m_feature, c, 6, 5);
            //x+6
            acc +=  in_4d_global(b, c, input_h_start + 0,  input_w_start + 6) * mask_4d(m_feature, c, 0, 6)
                +   in_4d_global(b, c, input_h_start + 1,  input_w_start + 6) * mask_4d(m_feature, c, 1, 6)
                +   in_4d_global(b, c, input_h_start + 2,  input_w_start + 6) * mask_4d(m_feature, c, 2, 6) 
                +   in_4d_global(b, c, input_h_start + 3,  input_w_start + 6) * mask_4d(m_feature, c, 3, 6) 
                +   in_4d_global(b, c, input_h_start + 4,  input_w_start + 6) * mask_4d(m_feature, c, 4, 6) 
                +   in_4d_global(b, c, input_h_start + 5,  input_w_start + 6) * mask_4d(m_feature, c, 5, 6) 
                +   in_4d_global(b, c, input_h_start + 6,  input_w_start + 6) * mask_4d(m_feature, c, 6, 6);
        }
        out_4d(b, m_feature, output_h, output_w) = acc;
    }
    #undef out_4d
    #undef in_4d_global
    #undef mask_4d
}

__global__ void conv_forward_kernel_ConstantMem_SharedMem(float* output, const float* input, const float* mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Function paramter definitions:
    output - output
    input - input
    KERNEL_DEVICE_CST - convolution kernel mask in constant MEM
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */
    extern __shared__ float N_ds[]; //size determined dynamicly at runtime, we will rely on cache to catch others
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int tile_width = blockDim.x;
    const int tile_height = blockDim.y;
    const int SharedMatrix_width = tile_width * S;
    const int SharedMatrix_height = tile_height * S;
    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]    // out_4d(b, m, h, w)
    #define in_4d_global(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]                          // in_4d_global(b, c, cell_height, cell_width)
    #define in_4d_shared(i2, i1, i0) N_ds[(i2) * (SharedMatrix_height * SharedMatrix_width) + (i1) * (SharedMatrix_width) + i0]                          // in_4d_shared(c, cell_height, cell_width)
    #define mask_4d(i3, i2, i1, i0) KERNEL_DEVICE_CST[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]                         // mask_4d(m, c, mask_heightindex, mask_widthindex)
    // Insert your GPU convolution kernel code here
    const int W_grid_blocks = (W_out - 1) / tile_width + 1;  //tiles in outputWidth
    const int m_feature = blockIdx.x;
    const int b = blockIdx.z;
    // y and x indexes for the output matrix
    const int output_h = (blockIdx.y / W_grid_blocks) * tile_height + threadIdx.y;
    const int output_w = (blockIdx.y % W_grid_blocks) * tile_width + threadIdx.x;
    // corresponding y and x starting indexes for input matrix for the current block 
    const int input_y_Block_start = ((blockIdx.y / W_grid_blocks) * tile_height) * S; 
    const int input_x_Block_start = ((blockIdx.y % W_grid_blocks) * tile_width) * S;
    int input_x;// input-x index
    int input_y;// input-y index
    int shared_x;// shared-x index
    int shared_y;// shared-y index
    // starting index for current Block
    //load Shared Memory
    for(int c = 0; c < C; ++c){
        for (int scountery = 0; scountery < S; ++scountery){
            shared_y = threadIdx.y + scountery * tile_height;
            input_y = input_y_Block_start + shared_y;
            bool is_Y_outbound = input_y > H;
            for (int scounterx = 0; scounterx < S; ++scounterx){
                shared_x = threadIdx.x + scounterx * tile_width;
                input_x = input_x_Block_start + shared_x;
                bool is_X_outbound = input_x > W;
                //INDEXING OVER C in the outermost layer, to not mess up the coalescedd memory acces
                if(is_Y_outbound && is_X_outbound){
                    in_4d_shared(c, shared_y, shared_x) = 0.0f;
                }
                else {
                    in_4d_shared(c, shared_y, shared_x) = in_4d_global(b, c, input_y, input_x);
                }  
            }
        }
    }
    __syncthreads();
    const int input_h_start = output_h * S; 
    const int input_w_start = output_w * S;
    bool is_y_in_bounds;
    bool is_x_in_bounds;
    float acc = 0.0f;
    if((output_h < H_out) && (output_w < W_out)){
        for(int c = 0; c < C; ++c){   // sum over all input channels
            for(int j = 0; j < K; ++j){   // KxK filter (height)
                input_y = input_h_start + j;
                shared_y = input_y - input_y_Block_start; //where it is in corresponding input tile, we use this to determine if its in shared mem or not.
                is_y_in_bounds = shared_y < SharedMatrix_height;
                for(int i = 0; i < K; ++i){   // KxK filter (width)
                    input_x = input_w_start + i;
                    shared_x = input_x - input_x_Block_start; //where it is in corresponding input tile, we use this to determine if its in shared mem or not.
                    is_x_in_bounds = shared_x < SharedMatrix_width;
                    if(is_y_in_bounds && is_x_in_bounds){
                        acc += in_4d_shared(c, shared_y, shared_x) * mask_4d(m_feature, c, j, i); 
                    }
                    else{
                        acc += in_4d_global(b, c, input_y, input_x) * mask_4d(m_feature, c, j, i); 
                    }
                }
            }
        }
        out_4d(b, m_feature, output_h, output_w) = acc;
    }
    #undef out_4d
    #undef in_4d_global
    #undef mask_4d
}

__global__ void conv_forward_kernel_ConstantMem_SharedMem2(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Function paramter definitions:
    output - output
    input - input
    KERNEL_DEVICE_CST - convolution kernel mask in constant MEM
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */
    __shared__ float SharedMask[196];
    extern __shared__ float N_ds[]; //size determined dynamicly at runtime, we will rely on cache to catch others
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int tile_width = blockDim.x;
    const int tile_height = blockDim.y;
    const int SharedMatrix_width = (tile_width+K-1) * S;
    const int SharedMatrix_height = (tile_height+K-1) * S;
    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]    // out_4d(b, m, h, w)
    #define in_4d_global(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]                          // in_4d_global(b, c, cell_height, cell_width)
    #define in_4d_shared(i2, i1, i0) N_ds[(i2) * (SharedMatrix_height * SharedMatrix_width) + (i1) * (SharedMatrix_width) + i0]                          // in_4d_shared(c, cell_height, cell_width)
    #define mask_4d(i3, i2, i1, i0) KERNEL_DEVICE_CST[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]                         // mask_4d(m, c, mask_heightindex, mask_widthindex)
    // Insert your GPU convolution kernel code here
    const int W_grid_blocks = (W_out - 1) / tile_width + 1;  //tiles in outputWidth
    const int m_feature = blockIdx.x;
    const int b = blockIdx.z;
    // y and x indexes for the output matrix
    const int output_h = (blockIdx.y / W_grid_blocks) * tile_height + threadIdx.y;
    const int output_w = (blockIdx.y % W_grid_blocks) * tile_width + threadIdx.x;
    // corresponding y and x starting indexes for input matrix for the current block 
    const int input_y_Block_start = ((blockIdx.y / W_grid_blocks) * tile_height) * S; 
    const int input_x_Block_start = ((blockIdx.y % W_grid_blocks) * tile_width) * S;
    int input_x;
    int input_y;
    int shared_x;
    int shared_y;
    // starting index for current Block
    //load Shared Memory
    bool is_X_inbound;
    #pragma unroll 4
    for(int c = 0; c < C; ++c){
        shared_y = threadIdx.y;
        shared_x = threadIdx.x;
        input_y = input_y_Block_start + shared_y;
        input_x = input_x_Block_start + shared_x;
        // is_Y_inbound = shared_y < SharedMatrix_height;
        is_X_inbound = shared_x < SharedMatrix_width;
        if(is_X_inbound){in_4d_shared(c, shared_y, shared_x) = in_4d_global(b, c, input_y, input_x);}
        //rightupper
        shared_x = threadIdx.x + tile_width;
        input_x = input_x_Block_start + shared_x;
        is_X_inbound = shared_x < SharedMatrix_width;
        if(is_X_inbound){in_4d_shared(c, shared_y, shared_x) = in_4d_global(b, c, input_y, input_x);}
    }
    __syncthreads();
    const int input_h_start = output_h * S; 
    const int input_w_start = output_w * S;
    float acc = 0.0f;
    if((output_h < H_out) && (output_w < W_out)){
        for(int c = 0; c < C; ++c){   // sum over all input channels
            #pragma unroll 7
            for(int j = 0; j < K; ++j){   // KxK filter (height)
                #pragma unroll 7
                for(int i = 0; i < K; ++i){   // KxK filter (width)
                    acc += in_4d_shared(c, input_h_start - input_y_Block_start + j, input_w_start - input_x_Block_start + i) * mask_4d(m_feature, c, j, i); 
                }
            }
        }
        out_4d(b, m_feature, output_h, output_w) = acc;
    }
    #undef out_4d
    #undef in_4d_global
    #undef mask_4d
}

__global__ void conv_forward_kernel_ConstantMem_SharedMemUnroll(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Function paramter definitions:
    output - output
    input - input
    KERNEL_DEVICE_CST - convolution kernel mask in constant MEM
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */
    extern __shared__ float N_ds[]; //size determined dynamicly at runtime, we will rely on cache to catch others
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int tile_width = blockDim.x;
    const int tile_height = blockDim.y;
    const int SharedMatrix_width = tile_width * S;
    const int SharedMatrix_height = tile_height * S;
    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]    // out_4d(b, m, h, w)
    #define in_4d_global(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]                          // in_4d_global(b, c, cell_height, cell_width)
    #define in_4d_shared(i2, i1, i0) N_ds[(i2) * (SharedMatrix_height * SharedMatrix_width) + (i1) * (SharedMatrix_width) + i0]                          // in_4d_shared(c, cell_height, cell_width)
    #define mask_4d(i3, i2, i1, i0) KERNEL_DEVICE_CST[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]                         // mask_4d(m, c, mask_heightindex, mask_widthindex)
    // Insert your GPU convolution kernel code here
    const int W_grid_blocks = (W_out - 1) / tile_width + 1;  //tiles in outputWidth
    const int m_feature = blockIdx.x;
    const int b = blockIdx.z;
    // y and x indexes for the output matrix
    const int output_h = (blockIdx.y / W_grid_blocks) * tile_height + threadIdx.y;
    const int output_w = (blockIdx.y % W_grid_blocks) * tile_width + threadIdx.x;
    // corresponding y and x starting indexes for input matrix for the current block 
    const int input_y_Block_start = ((blockIdx.y / W_grid_blocks) * tile_height) * S; 
    const int input_x_Block_start = ((blockIdx.y % W_grid_blocks) * tile_width) * S;
    int input_x;// input-x index
    int input_y;// input-y index
    int shared_x;// shared-x index
    int shared_y;// shared-y index
    // starting index for current Block
    //load Shared Memory
    bool is_X_outbound;
    bool is_Y_outbound;
    if(S==1){
        for(int c = 0; c < C; ++c){
            shared_y = threadIdx.y;
            input_y = input_y_Block_start + shared_y;
            is_Y_outbound = input_y > H;
            shared_x = threadIdx.x;
            input_x = input_x_Block_start + shared_x;
            is_X_outbound = input_x > W;
            if(is_Y_outbound && is_X_outbound){in_4d_shared(c, shared_y, shared_x) = 0.0f;}
            else {in_4d_shared(c, shared_y, shared_x) = in_4d_global(b, c, input_y, input_x);}  
        }
    }
    else{
        for(int c = 0; c < C; ++c){
            for (int scountery = 0; scountery < S; ++scountery){
                shared_y = threadIdx.y + scountery * tile_height;
                input_y = input_y_Block_start + shared_y;
                bool is_Y_outbound = input_y > H;
                for (int scounterx = 0; scounterx < S; ++scounterx){
                    shared_x = threadIdx.x + scounterx * tile_width;
                    input_x = input_x_Block_start + shared_x;
                    bool is_X_outbound = input_x > W;
                    //INDEXING OVER C in the outermost layer, to not mess up the coalescedd memory acces
                    if(is_Y_outbound && is_X_outbound){
                        in_4d_shared(c, shared_y, shared_x) = 0.0f;
                    }
                    else {
                        in_4d_shared(c, shared_y, shared_x) = in_4d_global(b, c, input_y, input_x);
                    }  
                }
            }
        }
    }
    
    __syncthreads();
    const int input_h_start = output_h * S; 
    const int input_w_start = output_w * S;
    float acc = 0.0f;
    bool is_y_in_bounds;
    bool is_x_in_bounds;
    int shared_x1;
    int shared_x2;
    int shared_x3;
    int shared_x4;
    int shared_x5;
    int shared_x6;
    if((output_h < H_out) && (output_w < W_out)){
        for(int c = 0; c < C; ++c){   // sum over all input channels
            for(int j = 0; j < K; ++j){   // KxK filter (height)
                input_y = input_h_start + j;
                shared_y = input_y - input_y_Block_start; //where it is in corresponding input tile, we use this to determine if its in shared mem or not.
                is_y_in_bounds = shared_y < SharedMatrix_height;
                //0
                shared_x = input_w_start - input_x_Block_start + 0;
                shared_x1 = input_w_start - input_x_Block_start + 1;
                shared_x2 = input_w_start - input_x_Block_start + 2;
                shared_x3 = input_w_start - input_x_Block_start + 3;
                shared_x4 = input_w_start - input_x_Block_start + 4;
                shared_x5 = input_w_start - input_x_Block_start + 5;
                shared_x6 = input_w_start - input_x_Block_start + 6;
                if(is_y_in_bounds && (shared_x < SharedMatrix_width)){acc += in_4d_shared(c, shared_y, shared_x) * mask_4d(m_feature, c, j, 0); }
                else                                {acc += in_4d_global(b, c, input_y, input_w_start) * mask_4d(m_feature, c, j, 0);}
                //1
                if(is_y_in_bounds && (shared_x1 < SharedMatrix_width)){acc += in_4d_shared(c, shared_y, shared_x1) * mask_4d(m_feature, c, j, 1); }
                else                                {acc += in_4d_global(b, c, input_y, input_w_start + 1) * mask_4d(m_feature, c, j, 1);}
                //2
                if(is_y_in_bounds && (shared_x2 < SharedMatrix_width)){acc += in_4d_shared(c, shared_y, shared_x2) * mask_4d(m_feature, c, j, 2); }
                else                                {acc += in_4d_global(b, c, input_y, input_w_start + 2) * mask_4d(m_feature, c, j, 2);}
                //3
                if(is_y_in_bounds && (shared_x3 < SharedMatrix_width)){acc += in_4d_shared(c, shared_y, shared_x3) * mask_4d(m_feature, c, j, 3); }
                else                                {acc += in_4d_global(b, c, input_y, input_w_start + 3) * mask_4d(m_feature, c, j, 3);}
                //4
                
                if(is_y_in_bounds && (shared_x4 < SharedMatrix_width)){acc += in_4d_shared(c, shared_y, shared_x4) * mask_4d(m_feature, c, j, 4); }
                else                                {acc += in_4d_global(b, c, input_y, input_w_start + 4) * mask_4d(m_feature, c, j, 4);}
                //5
                
                if(is_y_in_bounds && (shared_x5 < SharedMatrix_width)){acc += in_4d_shared(c, shared_y, shared_x5) * mask_4d(m_feature, c, j, 5); }
                else                                {acc += in_4d_global(b, c, input_y, input_w_start + 5) * mask_4d(m_feature, c, j, 5);}
                //6
                
                if(is_y_in_bounds && shared_x6 < SharedMatrix_width){acc += in_4d_shared(c, shared_y, shared_x6) * mask_4d(m_feature, c, j, 6); }
                else                                {acc += in_4d_global(b, c, input_y, input_w_start + 6) * mask_4d(m_feature, c, j, 6);}
            }
        }
        out_4d(b, m_feature, output_h, output_w) = acc;
    }
    #undef out_4d
    #undef in_4d_global
    #undef mask_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int memSizeInput = (B * C * H * W) * sizeof(float);
    const int memSizeMask = (M * C * K * K) * sizeof(float);
    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    const int memSizeOutput = (B * M * outputHeight * outputWidth) * sizeof(float);
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    cudaMalloc((void **)device_input_ptr, memSizeInput);
    cudaMalloc((void **)device_mask_ptr, memSizeMask);
    cudaMalloc((void **)device_output_ptr, memSizeOutput);
    cudaMemcpy(*device_input_ptr, host_input, memSizeInput, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, memSizeMask, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(KERNEL_DEVICE_CST, host_mask, memSizeMask);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    // const int memSizeMask = (M * C * K * K) * sizeof(float);
    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    int TILE_WIDTH = 8;
    int TILE_HEIGHT = 48;
    bool kernelFlag = false;
    //first layer
    if(outputWidth==80){
        // TILE_WIDTH = 16;
        // TILE_HEIGHT = 16;
        kernelFlag = true;
    }
    //second layer
    else if(outputWidth==34){
        // TILE_WIDTH = 8;
        // TILE_HEIGHT = 48;
        kernelFlag = true;
    }
    int H_grid_blocks;
    int W_grid_blocks;
    int nTiles;
    int sharedMemConvSize;


    if(kernelFlag){
        if(outputWidth==34){
            TILE_WIDTH = 8;
            TILE_HEIGHT = 40;
            H_grid_blocks = (outputHeight - 1) / TILE_HEIGHT + 1; //tiles in outputHeight
            W_grid_blocks = (outputWidth - 1) / TILE_WIDTH + 1;  //tiles in outputWidth
            nTiles = H_grid_blocks * W_grid_blocks; // total tiles
            sharedMemConvSize = ((TILE_WIDTH+K-1) * (TILE_HEIGHT+K-1) * S * S * C) * sizeof(float);
            // W_grid_blocks = (outputWidth - 1) / TILE_WIDTH + 1;
            dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT, 1);
            dim3 dimGrid(M, nTiles, B); // Ensuring all elements are covered
            conv_forward_kernel_ConstantMem_SharedMem2<<<dimGrid, dimBlock, sharedMemConvSize>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
            std::cout << "C" << C << std::endl;
        }
        else{
            //layer1
            TILE_WIDTH = 16;
            TILE_HEIGHT = 16;
            H_grid_blocks = (outputHeight - 1) / TILE_HEIGHT + 1; //tiles in outputHeight
            W_grid_blocks = (outputWidth - 1) / TILE_WIDTH + 1;  //tiles in outputWidth
            nTiles = H_grid_blocks * W_grid_blocks; // total tiles
            sharedMemConvSize = (TILE_WIDTH * TILE_HEIGHT * S * S * C) * sizeof(float);
            dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT, 1);
            dim3 dimGrid(M, nTiles, B); // Ensuring all elements are covered
            conv_forward_kernel_ConstantMem_SharedMemUnroll<<<dimGrid, dimBlock, sharedMemConvSize>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
        }
    }
    else{
        H_grid_blocks = (outputHeight - 1) / TILE_HEIGHT + 1; //tiles in outputHeight
        W_grid_blocks = (outputWidth - 1) / TILE_WIDTH + 1;  //tiles in outputWidth
        nTiles = H_grid_blocks * W_grid_blocks; // total tiles
        sharedMemConvSize = (TILE_WIDTH * TILE_HEIGHT * S * S * C) * sizeof(float);
        while (sharedMemConvSize > 49152){
            TILE_HEIGHT /= 2;
            // W_grid_blocks = (outputWidth - 1) / TILE_WIDTH + 1;
            H_grid_blocks = (outputHeight - 1) / TILE_HEIGHT + 1; //tiles in outputHeight
            nTiles = H_grid_blocks * W_grid_blocks; // total tiles
            sharedMemConvSize = (TILE_WIDTH * TILE_HEIGHT * S * S * C) * sizeof(float);
        }
        dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT, 1);
        dim3 dimGrid(M, nTiles, B); // Ensuring all elements are covered
        conv_forward_kernel_ConstantMem_SharedMem<<<dimGrid, dimBlock, sharedMemConvSize>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    const int memSizeOutput = (B * M * outputHeight * outputWidth) * sizeof(float);
    cudaMemcpy(host_output, device_output, memSizeOutput,cudaMemcpyDeviceToHost);
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
    // Free device memory
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
