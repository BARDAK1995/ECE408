// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
}
__global__ void Brent_Kung_scan_kernel(float *input, float *output, unsigned int len) {
  const int sectionsize = blockDim.x*2;
  __shared__ float XY[BLOCK_SIZE*2];
  unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  // Load data into shared memory
  if(i < len) 
    XY[threadIdx.x] = input[i];
  else 
    XY[threadIdx.x] = 0.0f;
  if(i + blockDim.x < len) 
    XY[threadIdx.x + blockDim.x] = input[i + blockDim.x];
  else 
    XY[threadIdx.x + blockDim.x] = 0.0f;
  // Reduction phase
  for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
      __syncthreads();
      unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
      if(index < sectionsize && (index-stride) >= 0) { 
          XY[index] += XY[index - stride];
      }
  }
  // Down-sweep phase
  for (int stride = sectionsize / 4; stride > 0; stride /= 2) {
      __syncthreads();
      unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
      if(index + stride < sectionsize) {
          XY[index + stride] += XY[index];
      }
  }
  __syncthreads();
  // Write results back to global memory
  if (i < len) output[i] = XY[threadIdx.x];
  if ((i + blockDim.x) < len) output[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((numElements - 1) / (BLOCK_SIZE * 2) + 1, 1, 1); // Ensuring all elements are covered

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  Brent_Kung_scan_kernel<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
