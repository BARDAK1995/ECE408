
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


#define TILE_WIDTH 16
// Compute C = A * B

__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
    int numAColumns, int numBRows,
    int numBColumns, int numCRows,
    int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
  const int width = numAColumns;
  const int Adepth = numARows;
  const int Bdepth = numBColumns;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int blockx = blockDim.x;
  const int blocky = blockDim.y;
  //   index values for the global output matrix
  const int column = blockx * blockIdx.x + threadIdx.x;
  const int row = blocky * blockIdx.y + threadIdx.y;

  float Cvalue = 0.0f;
  const int n_tiles = ceil(width/(float)TILE_WIDTH);
  for (int tileNo = 0; tileNo < n_tiles; tileNo++){
    // load tile to shared memory
    // forA
    if ((row < Adepth) && ((tx + tileNo * TILE_WIDTH) < width))
      tileA[ty][tx] = A[(row * width) + tx + (tileNo * TILE_WIDTH)];
    else tileA[ty][tx] = 0.0f;
    // For B
    if ((column < Bdepth) && ((ty + tileNo * TILE_WIDTH) < width))
      tileB[ty][tx] = B[column + (ty * Bdepth) + (tileNo * Bdepth * TILE_WIDTH)];
    else tileB[ty][tx] = 0.0f;
    __syncthreads();
    // calculate partial multiplication result for this tile 
    for (int k = 0; k < TILE_WIDTH; k++)
      Cvalue += tileA[ty][k] * tileB[k][tx];
    __syncthreads();
  }
  //   put the correct summed up multiplication result
  if ((row < numARows) && (column < numBColumns))
    C[row*numCColumns + column] = Cvalue;
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  const int sizeA = numARows * numAColumns;
  const int sizeB = numBRows * numBColumns;
  const int sizeC = numCRows * numCColumns;
  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(sizeC * sizeof(float));
  
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceA, sizeA * sizeof(float));
  cudaMalloc((void**)&deviceB, sizeB * sizeof(float));
  cudaMalloc((void**)&deviceC, sizeC * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeB * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, sizeC * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  const int gridXdim = ceil(numCColumns/(float)TILE_WIDTH);
  const int gridYdim = ceil(numCRows/(float)TILE_WIDTH);
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 DimGrid(gridXdim, gridYdim, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaError_t errA = cudaMemcpy(hostA, deviceA, sizeA * sizeof(float), cudaMemcpyDeviceToHost);
  cudaError_t errB = cudaMemcpy(hostB, deviceB, sizeB * sizeof(float), cudaMemcpyDeviceToHost);
  cudaError_t errC = cudaMemcpy(hostC, deviceC, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA); 
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
