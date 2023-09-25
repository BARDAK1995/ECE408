#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

void matrixMultiplyCPU(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns){
    for (int i = 0; i < numAColumns; i++)
    {
        for (int j = 0; j < numBRows; j++)
        {   
            float sum = 0;
            for (int k = 0; k < numCColumns; k++)
            { 
                float a = A[(i * numAColumns) + k];
                float b = B[j + (k * numBColumns)];
                sum += a*b;
            }
            C[(i * numCColumns) + j] = sum;
        }
    }
}


__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //   index values for the global matrix
  int column = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  if (column < numCColumns && row < numCRows)
  {
    float cvalue = 0;
    for (int i = 0; i < numAColumns; i++)
    {
        const float a = A[(row * numAColumns) + i];
        const float b = B[column + (numBColumns * i)];
        cvalue += a * b;
    }
    C[column + (row * numCColumns)] = cvalue;
  }
}

__global__ void matrixMultiplyGPUTiled(float *A, float *B, float *C, int numARows,
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

void writeMatrixToFile(const float* matrix, int numRows, int numColumns, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open the file" << std::endl;
        return;
    }
    outfile << numRows << " " << numColumns << "\n";
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numColumns; ++j) {
            outfile << matrix[i * numColumns + j];
            if (j < numColumns - 1) {
                outfile << "  ";
            }
        }
        outfile << '\n';
    }
    outfile.close();
}

float* readMatrixFromFile(const std::string& filePath, int* rows, int* cols) {
    std::ifstream file(filePath);
    if(!file) {
        std::cerr << "Could not open the file!" << std::endl;
        return nullptr;
    }
    file >> *rows >> *cols;

    if(!file || *rows <= 0 || *cols <= 0) {
        std::cerr << "Error reading matrix dimensions!" << std::endl;
        return nullptr;
    }
    float* matrix = (float*) malloc( (*rows) * (*cols) * sizeof(float));
    for(int i = 0; i < *rows; ++i) {
        for(int j = 0; j < *cols; ++j) {
            file >> matrix[i * *cols + j];
        }
    }
    if(file.fail()) {
        std::cerr << "Error reading file!" << std::endl;
        delete[] matrix;
        return nullptr;
    }
    file.close();
    return matrix;
}


int main() {
    const std::string filePathA = "./data/9/input0.raw";
    const std::string filePathB = "./data/9/input1.raw";

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

    hostA = readMatrixFromFile(filePathA, &numARows, &numAColumns);
    hostB = readMatrixFromFile(filePathB, &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;

    int sizeA = numARows * numAColumns;
    int sizeB = numBRows * numBColumns;
    int sizeC = numCRows * numCColumns;

    //@@ Allocate the hostC matrix
    hostC = (float*)malloc(sizeC * sizeof(float));
    // solutionCPU
    // matrixMultiplyCPU(hostA, hostB, hostC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    // writeMatrixToFile(hostC, numCRows, numCColumns, "output.raw");
    
    //@@ Allocate GPU memory here
    cudaMalloc((void**)&deviceA, sizeA * sizeof(float));
    cudaMalloc((void**)&deviceB, sizeB * sizeof(float));
    cudaMalloc((void**)&deviceC, sizeC * sizeof(float));
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceC, hostC, sizeC * sizeof(float), cudaMemcpyHostToDevice);

    //@@ Initialize the grid and block dimensions here
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    const int gridXdim = ceil(numCColumns/(float)TILE_WIDTH);
    const int gridYdim = ceil(numCRows/(float)TILE_WIDTH);
    std::cout << gridXdim << gridYdim <<std::endl;
    dim3 DimGrid(gridXdim, gridYdim, 1);

    //@@ Launch the GPU Kernel here
    // matrixMultiplyGPU<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    matrixMultiplyGPUTiled<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    //@@ Copy the GPU memory back to the CPU here
    cudaError_t errA = cudaMemcpy(hostA, deviceA, sizeA * sizeof(float), cudaMemcpyDeviceToHost);
    cudaError_t errB = cudaMemcpy(hostB, deviceB, sizeB * sizeof(float), cudaMemcpyDeviceToHost);
    cudaError_t errC = cudaMemcpy(hostC, deviceC, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
    // if (errA != cudaSuccess || errB != cudaSuccess || errC != cudaSuccess) {
    //     fprintf("Failed to Allocate deviceA memory:");
    // }
    //@@ Free the GPU memory here
    writeMatrixToFile(hostC, numCRows, numCColumns, "output2.raw");

    free(hostA); free(hostB); free(hostC);
    cudaFree(deviceA); cudaFree(deviceB); cudaFree(deviceC);

    return 0;
}
