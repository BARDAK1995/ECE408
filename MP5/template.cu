// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
  
  __global__ void kernelAdd1(float *input, float *output, int len) {
    int i = 2 * threadIdx.x + blockIdx.x * blockDim.x * 2; // Adjusted indexing with block offset
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0 && (i + stride) < len) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    // Every block writes its result to deviceOutput.
    if (threadIdx.x == 0) {
        output[blockIdx.x] = input[i];
    }
  }
  
  __global__ void ConvergentSumReductionKernel(float* input, float* output, int len) {
    // Continuous thread assignment
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x * 2;

    // Reverse stride logic for reduced control divergence
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        __syncthreads(); // Ensure all previous updates in this block are visible
        if (threadIdx.x < stride && (i + stride) < len) {
            input[i] += input[i + stride];
        }
    }

    // First thread of each block writes the block's result to output
    if (threadIdx.x == 0) {
        output[blockIdx.x] = input[i];
    }
}

__global__ void total(float* input, float* output, int len) {
  __shared__ float input_s[BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int i = blockIdx.x * (2 * BLOCK_SIZE) + threadIdx.x;

  // Ensure that we don't read out-of-bounds data
  if (i + BLOCK_SIZE < len) {
      input_s[t] = input[i] + input[i + BLOCK_SIZE];
  } else if (i < len) {
      input_s[t] = input[i];
  } else {
      input_s[t] = 0;
  }
  
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      __syncthreads();
      if (t < stride) {
          input_s[t] += input_s[t + stride];
      }
  }

  // Write the result for this block to the output
  if (t == 0) {
      output[blockIdx.x] = input_s[0];
  }
}

  //@@ Load a segment of the input vector into shared memory
  //@@ Traverse the reduction tree
  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index


int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ",
        numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceInput, numInputElements * sizeof(float));
  cudaMalloc((void **)&deviceOutput, numOutputElements * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((numInputElements - 1) / (BLOCK_SIZE * 2) + 1, 1, 1); // Ensuring all elements are covered


  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  // kernelAdd1<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numInputElements);
  // ConvergentSumReductionKernel<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numInputElements);
  total<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numInputElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. For simplicity, we do not
   * require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  free(hostInput);
  free(hostOutput);

  return 0;
}
