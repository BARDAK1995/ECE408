#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

__constant__ half KERNEL_DEVICE_CST[3136];
#define TILE_WIDTH_MATMUL 32

__global__ void matrixMultiplyShared(float *OUTPUT_C, half *B_Matrix, const int B, const int M, const int C, const int H, const int W, const int K, const int S) {
    extern __shared__ half tileAB[];  // Declaration of the shared memory array
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // matrix sizes
    const int B_width = H_out*W_out; //numBColumns;
    const int B_height = C * K * K; // C*K*K; //numBrows;
    const int TILE_depth = C * K * K; //TILE_depth;
    const int Aheight = M; // numARows;
    const int A_width = C * K * K;
    const int A_Tilewidth = TILE_WIDTH_MATMUL;

    const int sharedMatmulA_Nelements = TILE_WIDTH_MATMUL*M;
    #define A_2d(i1, i0) (KERNEL_DEVICE_CST[(i1) * (C*K*K) + i0]) // mask_4d(m, c, mask_heightindex, mask_widthindex) = mask_4d(m=y, x)
    #define B_3d_UNROLED(i2, i1, i0) B_Matrix[(i2) * (B_height * B_width) + (i1) * (B_width) + i0]     // outputUnrolles(b, cell_height, cell_width)
    #define B_2d_shared(i1, i0) tileAB[ (i1) * (TILE_WIDTH_MATMUL) + i0 + sharedMatmulA_Nelements]     // outputUnrolles(b, cell_height, cell_width)
    #define A_2d_shared(i1, i0) tileAB[ (i0) * (M) + i1]     // A_2d_shared(cell_height, cell_width) in column major format
    #define C_4d(i3, i2, i1) OUTPUT_C[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1)]    // out_4d(b, m, h, w)

    //   index values for the global output matrix
    const int column_X_outputElement = blockDim.x * blockIdx.x + threadIdx.x;
    const int row_Y_m_feature = blockDim.y * blockIdx.y + threadIdx.y;
    const int batchN = blockIdx.z;
    half Cvalue =  __float2half(0.0f);
    const int nTilesA = ceil(A_width / (float)(TILE_WIDTH_MATMUL)); // load tile to shared memory For A
    const int nTilesB = ceil(B_height / (float)(M)); //load tile to shared memory For A
    const int B_tile_multiplier = ceil(TILE_WIDTH_MATMUL / (float)M);
    int colxx;
    int rowyy;
    int sharedRow_y;
    for (int tileNoX = 0; tileNoX < nTilesA; tileNoX++){
        colxx = (tileNoX * blockDim.x) + threadIdx.x;
        //first A shared memory
        if ((colxx < A_width) && (row_Y_m_feature < Aheight)){
            A_2d_shared(threadIdx.y, threadIdx.x) = A_2d(row_Y_m_feature, colxx);}
        else{
            A_2d_shared(threadIdx.y, threadIdx.x) = __float2half(0.0f);}
        //Load B to shared Memory
        for(int counterB = 0; counterB<B_tile_multiplier; counterB++){
            rowyy = ((tileNoX*B_tile_multiplier + counterB) * blockDim.y) + threadIdx.y;
            sharedRow_y = counterB*blockDim.y + threadIdx.y;
            if ((column_X_outputElement < B_width) && (rowyy < B_height) && (sharedRow_y < TILE_WIDTH_MATMUL)){
                B_2d_shared(sharedRow_y, threadIdx.x) = B_3d_UNROLED(batchN, rowyy, column_X_outputElement);
            }
            else{
                B_2d_shared(sharedRow_y, threadIdx.x) = __float2half(0.0f);
            }
        }
        __syncthreads();
        #pragma unroll 32
        for (int kk = 0; kk < TILE_WIDTH_MATMUL; kk++){
            // const half a = A_2d_shared(threadIdx.y, kk);
            // const half b = B_2d_shared(kk, threadIdx.x);
            // Cvalue = __hfma(A_2d_shared(threadIdx.y, kk), B_2d_shared(kk, threadIdx.x), Cvalue);
            Cvalue = __hadd(Cvalue, __hmul(A_2d_shared(threadIdx.y, kk), B_2d_shared(kk, threadIdx.x)));
        }
    }
    __syncthreads();
    //   put the correct summed up multiplication result
    if ((row_Y_m_feature < Aheight) && (column_X_outputElement < B_width))
        C_4d(batchN, row_Y_m_feature, column_X_outputElement) = __half2float(Cvalue);
    #undef A_2d
    #undef B_3d_UNROLED
    #undef B_2d_shared
    #undef A_2d_shared
    #undef C_4d
}

__global__ void unroll_Kernel(half *outputUnrolled, const half *inputX, const int B, const int C, const int H, const int W, const int K,const int S) {
    //1d block need C*Hout*Wout threads in x and batchn B blocks in y 
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    const int WIDTH_unroll = H_out * W_out;
    const int HIGHT_unroll = C * K * K;
    #define in_4d_global(i3, i2, i1, i0) inputX[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]     // in_4d(b, c, cell_height, cell_width)
    #define out_3d_UNROLL(i2, i1, i0) outputUnrolled[(i2) * (HIGHT_unroll * WIDTH_unroll) + (i1) * (WIDTH_unroll) + i0]     // outputUnrolles(b, cell_height, cell_width)
    const int batchN = blockIdx.y;
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    // Width of the unrolled input feature matrix
    if (thread < C * WIDTH_unroll) {
        const int c = thread / WIDTH_unroll; 
        const int x_unroll = thread % WIDTH_unroll;
        const int h_out = S * (x_unroll / W_out);
        const int w_out = S * (x_unroll % W_out);
        const int w_base_unrolled = (c * K * K);
        #pragma unroll 7
        for(int p = 0; p < K; p++) {
            #pragma unroll 7
            const int input_y = h_out + p;
            for(int q = 0; q < K; q++) {
                const int y_unroll = w_base_unrolled + (p * K) + q;
                const int input_x = w_out + q;
                out_3d_UNROLL(batchN, y_unroll, x_unroll) = in_4d_global(batchN, c, input_y, input_x);
            }
        }
    }
    #undef in_4d_global
    #undef out_3d_UNROLL
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
    // Allocate memory and copy over the relevant data structures to the GPU
    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    //input sizes
    const int nInputElements = (B * C * H * W);
    const int memSizeInput = nInputElements * sizeof(float);
    const int mMaskElements = (M * C * K * K);
    const int memSizeMask = mMaskElements * sizeof(float);
    //OutputSizes
    const int nOutputElements = (B * M * outputHeight * outputWidth);
    const int memSizeOutput = nOutputElements * sizeof(float);
    
    cudaMalloc((void **)device_input_ptr, memSizeInput);
    cudaMalloc((void **)device_mask_ptr, memSizeMask);

    cudaMemcpyAsync(*device_input_ptr, host_input, memSizeInput, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(*device_mask_ptr, host_mask, memSizeMask, cudaMemcpyHostToDevice);
    cudaMalloc((void **)device_output_ptr, memSizeOutput);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int mMaskElements = (M * C * K * K);
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
    convertFloatToHalf<<<gridSizeFP16ConverterMask, blockSizeFP16mask>>>(device_mask_half, device_mask, mMaskElements);
    cudaMemcpyToSymbol(KERNEL_DEVICE_CST, device_mask_half, memSizeMaskHalf);
    convertFloatToHalf<<<gridSizeFP16ConverterInput, blockSizeFP16Converter>>>(device_input_half, device_input, nInputElements);

    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    //Unrolled X matrix 
    const int unrolledMatrixSize = B * C * K * K * outputHeight * outputWidth;
    const int memSizeunrolledX = unrolledMatrixSize * sizeof(half);
    // std::cout <<"MATRIX SIZE is =" << outputHeight <<  std::endl;
    half *device_UnrolledX;
    cudaMalloc((void **)&device_UnrolledX, memSizeunrolledX);
    int blocksizeUnroll = 32;
    int blocksPerInput = (outputHeight*outputWidth*C - 1) / blocksizeUnroll + 1; //tiles in outputHeight
    dim3 dimGridUnroll(blocksPerInput, B, 1); // Ensuring all elements are covered
    dim3 dimBlockUnroll(blocksizeUnroll, 1, 1);
    unroll_Kernel<<<dimGridUnroll, dimBlockUnroll>>>(device_UnrolledX, device_input_half, B, C, H, W, K, S);
    const int Matmul_Output_height = M;
    const int Matmul_Output_width = outputHeight * outputWidth;
    //for shared tiled matrix multiplyu
    int grid_blocks_X = (Matmul_Output_width - 1) / TILE_WIDTH_MATMUL + 1; // TILE_WIDTH_MATMUL = 32
    int grid_blocks_Y = (Matmul_Output_height - 1) / M + 1; //tiles in outputHeight should be 1
    dim3 DimBlock_sharedMatmul(TILE_WIDTH_MATMUL, M, 1); // TILE_WIDTH_MATMUL=32
    dim3 DimGrid_sharedMatmul(grid_blocks_X, grid_blocks_Y, B);
    const int BtileHeight = ceil(TILE_WIDTH_MATMUL/(float)M) * M;
    const int sharedMatmulBsize = BtileHeight*TILE_WIDTH_MATMUL * sizeof(half);
    const int sharedMatmulAsize = M * TILE_WIDTH_MATMUL * sizeof(half);
    const int sharedMem = sharedMatmulAsize + sharedMatmulBsize;
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<DimGrid_sharedMatmul,DimBlock_sharedMatmul,sharedMem>>>(device_output, device_UnrolledX, B, M, C, H, W, K, S);
    cudaFree(device_mask_half);
    cudaFree(device_input_half);
    cudaFree(device_UnrolledX);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Calculate output size and memory size for half-precision
    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    const int nOutputElements = (B * M * outputHeight * outputWidth);
    const int memSizeOutput = nOutputElements * sizeof(float);
    cudaHostRegister(host_output, memSizeOutput, cudaHostRegisterDefault);
    cudaMemcpyAsync(host_output, device_output, memSizeOutput, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
    cudaHostUnregister(host_output);
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