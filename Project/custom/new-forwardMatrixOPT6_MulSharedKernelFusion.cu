#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>
// #include <chrono>
cudaStream_t stream1;
__constant__ half KERNEL_DEVICE_CST[3136];
#define TILE_WIDTH_MATMUL 64
__global__ void matrixMultiplySharedFusion_unroll(float* __restrict__ OUTPUT_C, half* __restrict__ inputX, const int B, const int M, const int C, const int H, const int W, const int K, const int S) {
    extern __shared__ half tileAB[];  // Declaration of the shared memory array
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // matrix sizes
    const int B_width = H_out*W_out; //numBColumns;
    const int B_height = C * K * K; // C*K*K; //numBrows;
    const int TILE_depth = C * K * K; //TILE_depth;
    const int Aheight = M; // numARows;
    const int A_width = C * K * K;
    const int sharedMatmulA_Nelements = A_width*M;

    const int WIDTH_unroll_tile = TILE_WIDTH_MATMUL;
    const int HIGHT_unroll = C * K * K;
    #define in_4d_global(i3, i2, i1, i0) inputX[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]     // in_4d(b, c, cell_height, cell_width)
    #define A_2d(i1, i0) (KERNEL_DEVICE_CST[(i1) * (C*K*K) + i0]) // mask_4d(m, c, mask_heightindex, mask_widthindex) = mask_4d(m=y, x)
    #define B_2d_shared(i1, i0) tileAB[ (i1) * (TILE_WIDTH_MATMUL) + i0 + sharedMatmulA_Nelements]     // outputUnrolles(b, cell_height, cell_width)
    #define A_2d_shared(i1, i0) tileAB[ (i1) * (A_width) + i0]     // outputUnrolles(b, cell_height, cell_width)
    #define C_4d(i3, i2, i1) OUTPUT_C[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1)]    // out_4d(b, m, h, w)

    //   index values for the global output matrix
    const int column_X_outputElement = blockDim.x * blockIdx.x + threadIdx.x;
    const int row_Y_m_feature = blockDim.y * blockIdx.y + threadIdx.y;
    const int batchN = blockIdx.z;
    half Cvalue =  __float2half(0.0f);
    
    const int nTilesA = ceil(A_width / (float)(blockDim.x)); // load tile to shared memory For A
    for (int tileNoX = 0; tileNoX < nTilesA; tileNoX++){
        const int colxx = (tileNoX * blockDim.x) + threadIdx.x;
        if ((colxx < A_width) && (row_Y_m_feature < Aheight))
            A_2d_shared(threadIdx.y, colxx) = A_2d(row_Y_m_feature, colxx);
        // __syncthreads();
    }
    __syncthreads();
    //Unroll B
    const int Unroll_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int Unroll_channel = threadIdx.y;
    
    if ((Unroll_thread < B_width) && (Unroll_channel < C)) {
        // Channel of the input feature map being collected by the thread
        const int cc = Unroll_channel; 
        // const int x_unroll = thread % WIDTH_unroll;
        const int x_unroll = Unroll_thread;
        // Horizontal and vertical indices of the output element
        const int h_out = S * (x_unroll / W_out);
        const int w_out = S * (x_unroll % W_out);
        // Starting row index for the unrolled matrix section for channel c
        const int y_base_unrolled = cc * K * K;
        #pragma unroll
        for(int q = 0; q < K; q++) {
            #pragma unroll
            for(int p = 0; p < K; p++) {
                const int input_y = h_out + p;
                const int y_unroll = y_base_unrolled + (p * K) + q;
                // Compute linearized indices for X and X_unroll
                const int input_x = w_out + q;
                B_2d_shared(y_unroll, threadIdx.x) = in_4d_global(batchN, cc, input_y, input_x);
            }
        }
    }
    __syncthreads();
    // calculate partial multiplication result for this tile 
    for (int kk = 0; kk < TILE_depth; kk++){
        const half a = A_2d_shared(threadIdx.y, kk);
        const half b = B_2d_shared(kk, threadIdx.x);
        Cvalue = __hfma(a, b, Cvalue);
    }
    __syncthreads();
    //   put the correct summed up multiplication result
    if ((row_Y_m_feature < Aheight) && (column_X_outputElement < B_width))
        C_4d(batchN, row_Y_m_feature, column_X_outputElement) = __half2float(Cvalue);
    #undef in_4d_global
    #undef A_2d
    #undef B_2d_shared
    #undef A_2d_shared
    #undef C_4d
}

__global__ void matrixMultiplyShared(float* __restrict__ OUTPUT_C, const half* __restrict__ B_Matrix, const int B, const int M, const int C, const int H, const int W, const int K, const int S) {
    extern __shared__ half tileAB[];  // Declaration of the shared memory array
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // matrix sizes
    
    const int B_width = H_out*W_out; //numBColumns;
    const int B_height = C * K * K; // C*K*K; //numBrows;
    const int TILE_depth = C * K * K; //TILE_depth;
    const int Aheight = M; // numARows;
    const int A_width = C * K * K;
    const int sharedMatmulA_Nelements = A_width*M;
    #define A_2d(i1, i0) (KERNEL_DEVICE_CST[(i1) * (C*K*K) + i0]) // mask_4d(m, c, mask_heightindex, mask_widthindex) = mask_4d(m=y, x)
    #define B_3d_UNROLED(i2, i1, i0) B_Matrix[(i2) * (B_height * B_width) + (i1) * (B_width) + i0]     // outputUnrolles(b, cell_height, cell_width)
    #define B_2d_shared(i1, i0) tileAB[ (i1) * (TILE_WIDTH_MATMUL) + i0 + sharedMatmulA_Nelements]     // outputUnrolles(b, cell_height, cell_width)
    #define A_2d_shared(i1, i0) tileAB[ (i1) * (A_width) + i0]     // outputUnrolles(b, cell_height, cell_width)
    #define C_4d(i3, i2, i1) OUTPUT_C[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1)]    // out_4d(b, m, h, w)

    //   index values for the global output matrix
    const int column_X_outputElement = blockDim.x * blockIdx.x + threadIdx.x;
    const int row_Y_m_feature = blockDim.y * blockIdx.y + threadIdx.y;
    const int batchN = blockIdx.z;
    half Cvalue =  __float2half(0.0f);
    
    const int nTilesA = ceil(A_width / (float)(blockDim.x)); // load tile to shared memory For A
    for (int tileNoX = 0; tileNoX < nTilesA; tileNoX++){
        const int colxx = (tileNoX * blockDim.x) + threadIdx.x;
        if ((colxx < A_width) && (row_Y_m_feature < Aheight))
            A_2d_shared(threadIdx.y, colxx) = A_2d(row_Y_m_feature, colxx);
        // __syncthreads();
    }
    __syncthreads();
    const int nTilesB = ceil(B_height / (float)(M)); //load tile to shared memory For A
    for (int tileNoB = 0; tileNoB < nTilesB; tileNoB++){
        const int rowyy = (tileNoB * blockDim.y) + threadIdx.y;
        if ((column_X_outputElement < B_width) && (rowyy < B_height))
            B_2d_shared(rowyy, threadIdx.x) = B_3d_UNROLED(batchN, rowyy, column_X_outputElement);
        // __syncthreads();
    }
    __syncthreads();
    // calculate partial multiplication result for this tile 
    for (int kk = 0; kk < TILE_depth; kk++){
        const half a = A_2d_shared(threadIdx.y, kk);
        const half b = B_2d_shared(kk, threadIdx.x);
        Cvalue = __hfma(a, b, Cvalue);
    }
    __syncthreads();
    //   put the correct summed up multiplication result
    if ((row_Y_m_feature < Aheight) && (column_X_outputElement < B_width))
        C_4d(batchN, row_Y_m_feature, column_X_outputElement) = __half2float(Cvalue);
    #undef A_2d
    #undef B_3d_UNROLED
    #undef B_2d_shared
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
        // Channel of the input feature map being collected by the thread
        const int c = thread / WIDTH_unroll; 
        // Column index of the unrolled matrix to write a strip of
        // input elements into (also, the linearized index of the output
        // element for which the thread is collecting input elements)
        const int x_unroll = thread % WIDTH_unroll;
        // Horizontal and vertical indices of the output element
        const int h_out = S * (x_unroll / W_out);
        const int w_out = S * (x_unroll % W_out);
        // Starting row index for the unrolled matrix section for channel c
        const int w_base_unrolled = (c * K * K);
        for(int p = 0; p < K; p++) {
            for(int q = 0; q < K; q++) {
                // Row index of the unrolled matrix for the thread to write
                // the input element into for the current iteration
                const int y_unroll = w_base_unrolled + (p * K) + q;
                // Compute linearized indices for X and X_unroll
                const int input_x = w_out + q;
                const int input_y = h_out + p;
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
    cudaStreamCreate(&stream1);
    // Allocate memory and copy over the relevant data structures to the GPU
    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    //input sizes
    const int nInputElements = (B * C * H * W);
    const int memSizeInput = nInputElements * sizeof(float);
    const int mMaskElements = (M * C * K * K);
    const int memSizeMask = mMaskElements * sizeof(float);
    // std::cout <<"Hin =" << H << "Win = " << W << "batch =" << B << "Chanelinput =" << C << std::endl;
    // std::cout <<"H out =" << outputHeight << "Wout = " << outputWidth << "output features =" << M << "Kernelsize =" << K << std::endl;
    //OutputSizes
    const int nOutputElements = (B * M * outputHeight * outputWidth);
    const int memSizeOutput = nOutputElements * sizeof(float);

    // std::cout << mMaskElements << "   n mask elements " << std::endl;
    
    cudaMalloc((void **)device_input_ptr, memSizeInput);
    cudaMalloc((void **)device_mask_ptr, memSizeMask);


    

    // cudaMemcpyToSymbol(KERNEL_DEVICE_CST, host_mask, memSizeMask);
    cudaMemcpyAsync(*device_input_ptr, host_input, memSizeInput, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(*device_mask_ptr, host_mask, memSizeMask, cudaMemcpyHostToDevice, stream1);
    cudaMalloc((void **)device_output_ptr, memSizeOutput);
    std::cout<<"B is : "<<B<<" M is : "<<M<<std::endl;
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
    // const int memSizeMask = mMaskElements * sizeof(float);
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
    cudaMemcpyToSymbol(KERNEL_DEVICE_CST, device_mask_half, memSizeMaskHalf);
    convertFloatToHalf<<<gridSizeFP16ConverterInput, blockSizeFP16Converter, 0, stream1>>>(device_input_half, device_input, nInputElements);
    //Pointers to acces the fp16 portion of the arrays
    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    //Unrolled X matrix 
    const int unrolledMatrixSize = B * C * K * K * outputHeight * outputWidth;
    const int memSizeunrolledX = unrolledMatrixSize * sizeof(half);
    // std::cout <<"MATRIX SIZE is =" << outputHeight <<  std::endl;
    // half *device_UnrolledX;
    
    // cudaMalloc((void **)&device_UnrolledX, memSizeunrolledX);
    int blocksizeUnroll = 32;
    int blocksPerInput = (outputHeight*outputWidth*C - 1) / blocksizeUnroll + 1; //tiles in outputHeight
    // dim3 dimGridUnroll(blocksPerInput, B, 1); // Ensuring all elements are covered
    // dim3 dimBlockUnroll(blocksizeUnroll, 1, 1);
    // cudaStreamSynchronize(stream1);
    // unroll_Kernel<<<dimGridUnroll, dimBlockUnroll,0,stream1>>>(device_UnrolledX, device_input_half, B, C, H, W, K, S);
    // std::cout <<"Output MATRIX SIZE is =" << M << "x" << outputWidth*outputHeight << std::endl;
    std::cout <<"CxKxK =" << C*K*K << std::endl;


   
    const int Matmul_Output_height = M;
    const int Matmul_Output_width = outputHeight * outputWidth;

    //for shared tiled matrix multiplyu
    int grid_blocks_X = (Matmul_Output_width - 1) / TILE_WIDTH_MATMUL + 1; // TILE_WIDTH_MATMUL = 32
    int grid_blocks_Y = (Matmul_Output_height - 1) / M + 1; //tiles in outputHeight should be 1
    dim3 DimBlock_sharedMatmul(TILE_WIDTH_MATMUL, M, 1); // TILE_WIDTH_MATMUL=32
    // std::cout <<"DimBlock_sharedMatmul  is =" << TILE_WIDTH_MATMUL << "x" << M << std::endl;
    dim3 DimGrid_sharedMatmul(grid_blocks_X, grid_blocks_Y, B);
    // std::cout <<"DimGrid_sharedMatmul  is =" << grid_blocks_X << "x" << grid_blocks_Y << "x" << B << std::endl;
    const int sharedMatmulBsize = C*K*K*TILE_WIDTH_MATMUL * sizeof(half);
    const int sharedMatmulAsize = C*K*K*M * sizeof(half);
    const int sharedMem = sharedMatmulAsize + sharedMatmulBsize;
    //@@ Launch the GPU Kernel here
    cudaStreamSynchronize(stream1);
    // matrixMultiplyShared<<<DimGrid_sharedMatmul,DimBlock_sharedMatmul,sharedMem,stream1>>>(device_output, device_UnrolledX, B, M, C, H, W, K, S);

    matrixMultiplySharedFusion_unroll<<<DimGrid_sharedMatmul,DimBlock_sharedMatmul,sharedMem,stream1>>>(device_output, device_input_half, B, M, C, H, W, K, S);

    cudaStreamSynchronize(stream1);
    // cudaFree(device_UnrolledX);
    cudaFree(device_input_half);
    cudaFree(device_mask_half);
    // std::cout << outputWidth << " x " << outputHeight << " x " << C << " and K is " << K << " and S is " << S << std::endl;
    
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Calculate output size and memory size for half-precision
    const int outputHeight = (H - K)/S + 1;
    const int outputWidth = (W - K)/S + 1;
    const int nOutputElements = (B * M * outputHeight * outputWidth);
    const int memSizeOutput = nOutputElements * sizeof(float);
    cudaHostRegister(host_output, memSizeOutput, cudaHostRegisterDefault);
    cudaStreamSynchronize(stream1);
    cudaMemcpyAsync(host_output, device_output, memSizeOutput, cudaMemcpyDeviceToHost);
    cudaHostUnregister(host_output);
    
    // auto start4 = std::chrono::high_resolution_clock::now();
    // cudaMemcpy(host_output, device_output, memSizeOutput, cudaMemcpyDeviceToHost);
    // cudaHostUnregister(host_output);
    cudaStreamDestroy(stream1);
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