#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_DIM 3
#define FilterRadius 1
#define input_TILE_WIDTH 6
#define O_TILE_WIDTH ((input_TILE_WIDTH) - 2 * (FilterRadius))
//@@ Define constant memory for device kernel here
__constant__ float filter3d_device[FilterRadius][FilterRadius][FilterRadius];

__global__ void conv3d(float *input, float *output, const int z_size, const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
  
  int col_o = blockIdx.x * O_TILE_WIDTH + tx;
  int row_o = blockIdx.y * O_TILE_WIDTH + ty;
  int depth_o = blockIdx.z * O_TILE_WIDTH + tz;

  int col_i = col_o - FilterRadius;
  int row_i = row_o - FilterRadius;
  int depth_i = depth_o - FilterRadius;

  int global_index = depth_i * x_size * y_size + row_i * x_size + col_i;
  // load shared mem
  __shared__ float N_ds[input_TILE_WIDTH][input_TILE_WIDTH][input_TILE_WIDTH];
  //load nonghost elements, put zero if ghost
  if ((row_i >= 0) && (row_i < y_size) && (col_i >= 0) && (col_i < x_size) && (depth_i >= 0) && (depth_i < z_size)) {
    N_ds[tx][ty][tz] = input[global_index];
  }
  else{
    N_ds[tx][ty][tz] = 0.0f;
  }
  __syncthreads();
  int output_column = threadIdx.x - FilterRadius;
  int output_row = threadIdx.y - FilterRadius;
  int output_depth = threadIdx.z - FilterRadius;
  //  check if actual output cell is inside
  if ((row_i >= 0) && (row_i < y_size) && (col_i >= 0) && (col_i < x_size) && (depth_i >= 0) && (depth_i < z_size)){
    // then check if it really is a output point, and not a input memory loading thread
    if ((output_column >= 0) && (output_column < O_TILE_WIDTH) && (output_row >= 0) && (output_row < O_TILE_WIDTH) && (output_depth >= 0) && (output_depth < O_TILE_WIDTH)){
      float Pvalue = 0.0f;
      for(int i=0; i < MASK_DIM; i++){
        for(int j=0; j<MASK_DIM; j++){
          for(int k=0; k<MASK_DIM; k++){
            // Pvalue += filter3d_device[i][j][k] * N_ds[output_column + i][output_row + j][output_depth + k];
            Pvalue += filter3d_device[i][j][k] * N_ds[threadIdx.x  + i][threadIdx.y + j][threadIdx.z + k];
          }
        }
      }
      output[global_index] = Pvalue;
    }
  }
  __syncthreads();
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  
  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  // wbLog(TRACE, hostKernel[1]);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  const int datalength = inputLength - 3;                                           // Recall that inputLength is 3 elements longer than the input data because the first  three elements were the dimensions
  cudaMalloc((void**)&deviceInput, datalength*sizeof(float));
  cudaMalloc((void**)&deviceOutput, datalength*sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  cudaMemcpyToSymbol(filter3d_device, hostKernel, kernelLength*sizeof(float));
  cudaMemcpy(deviceInput, hostInput+3, datalength*sizeof(float), cudaMemcpyHostToDevice);   // Recall that the first three elements of hostInput are dimensions and
  cudaMemcpy(deviceOutput, hostOutput+3, datalength*sizeof(float), cudaMemcpyHostToDevice); // do not need to be copied to the gpu
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  const int kerneldim = 3;
  const int gridXdim = ceil(x_size / (float)O_TILE_WIDTH);
  const int gridYdim = ceil(y_size / (float)O_TILE_WIDTH);
  const int gridZdim = ceil(z_size / (float)O_TILE_WIDTH);

  dim3 dimGrid(gridXdim, gridYdim, gridZdim);
  dim3 dimBlock(input_TILE_WIDTH, input_TILE_WIDTH, input_TILE_WIDTH);
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid,  dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // wbLog(TRACE, "Host Input 5 BEfore is ", hostInput[0], " Host Input 6 BEfore is ", hostInput[datalength+2]);
  cudaMemcpy(hostInput+3, deviceInput, datalength*sizeof(float), cudaMemcpyDeviceToHost);            // Recall that the first three elements of the output are the dimensions
  cudaMemcpy(hostOutput+3, deviceOutput, datalength*sizeof(float), cudaMemcpyDeviceToHost);          // and should not be set here (they are set below)
  // wbLog(TRACE, "Host Input 5 after is ", hostInput[0], " Host Input 6 after is ", hostInput[datalength+2]);
  
  
  wbTime_stop(Copy, "Copying data from the GPU");
  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
