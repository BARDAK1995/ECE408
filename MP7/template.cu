// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCKWIDTH 32
#define HISTOGRAM_BLOCK_SIZE 128
//@@ insert code here
__global__ void castToUchar(unsigned char *output, float *input, int width, int height, int channels) {
  // Calculate the two-dimensional thread index
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  // Check if the thread is within image bounds
  if (x < width && y < height) {
    // Calculate the index within the array for the pixel's first channel
    int start_idx = (y * width + x) * channels;
    // Convert and store each channel's value
    for (int channel = 0; channel < channels; ++channel) {
      output[start_idx + channel] = (unsigned char)(255 * input[start_idx + channel]);
    }
  }
}

__global__ void rgbToGrayscaleKernel(unsigned char *grayImage, unsigned char *rgbImage, int width, int height) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col < width && row < height) {
    // Calculate the index within the array
    int rgbIdx = (row * width + col) * 3; // 3 channels for RGB
    int grayIdx = row * width + col;      // 1 channel for grayscale
    unsigned char r = rgbImage[rgbIdx];     // Red value
    unsigned char g = rgbImage[rgbIdx + 1]; // Green value
    unsigned char b = rgbImage[rgbIdx + 2]; // Blue value
    // Compute the grayscale value using luminosity method
    grayImage[grayIdx] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
  }
}

__global__ void histogram_plusProbility(unsigned int *histogram, double *histProbability, unsigned char *grayImage, int imagesize){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  while (idx < imagesize) {
    atomicAdd(&(histogram[grayImage[idx]]), 1);
    histProbability[grayImage[idx]] = (histogram[grayImage[idx]]) / (double)imagesize;
    idx += stride;
  }
}

__global__ void paralelScanPFD(double *output, double *histogramProb,  double *AuxilarySum, int len) {
  const int sectionsize = blockDim.x*2;
  __shared__ double XY[HISTOGRAM_BLOCK_SIZE*2];
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  // Load data into shared memory
  XY[threadIdx.x]              = (i < len)              ? histogramProb[i]              : 0.0f;
  XY[threadIdx.x + blockDim.x] = (i + blockDim.x < len) ? histogramProb[i + blockDim.x] : 0.0f;
  // Reduction forward
  for(int stride = 1; stride <= blockDim.x; stride *= 2) {
      __syncthreads();
      int index = (threadIdx.x + 1) * 2 * stride - 1;
      if(index < sectionsize && (index-stride) >= 0) { 
          XY[index] += XY[index - stride];
      }
  }
  // Traverse back up
  for (int stride = sectionsize / 4; stride > 0; stride /= 2) {
      __syncthreads();
      int index = (threadIdx.x + 1) * stride * 2 - 1;
      if(index + stride < sectionsize) {
          XY[index + stride] += XY[index];
      }
  }
  __syncthreads();
  // Write results back to global memory
  //if its the last thread of the block, make it do this work, we do this first to make the kernel usable for SUM scanning as well
  if(threadIdx.x == (blockDim.x-1)) AuxilarySum[blockIdx.x] =  XY[sectionsize-1];
  if (i < len) output[i] = XY[threadIdx.x];
  if ((i + blockDim.x) < len) output[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}
__global__ void accumulateSums(double *DataArray, double *Sums, int len) {
  int index = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  int index2 = 2 * blockIdx.x * blockDim.x + threadIdx.x + blockDim.x;
  if((blockIdx.x > 0) && (index < len)) DataArray[index] += Sums[blockIdx.x-1];
  if((blockIdx.x > 0) && (index2 < len)) DataArray[index2] += Sums[blockIdx.x-1];
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  float *deviceInputImageData;
  unsigned char *deviceCharImageData;
  unsigned char *deviceGreyScaleImageData;

  
  unsigned int *devicehistogram;
  
  double *deviceProbabilitiesHist;
  //@@ Insert more code here
  
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  int imageSize = imageWidth * imageHeight;
  wbLog(TRACE, imageWidth, " x ", imageHeight);
  
  cudaMalloc((void **)&deviceInputImageData, (imageSize*imageChannels) * sizeof(float));
  cudaMalloc((void **)&deviceCharImageData, (imageSize*imageChannels) * sizeof(unsigned char));
  cudaMalloc((void **)&deviceGreyScaleImageData, imageSize * sizeof(unsigned char));
  cudaMalloc((void **)&devicehistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&deviceProbabilitiesHist, HISTOGRAM_LENGTH * sizeof(double));
  cudaMemset(devicehistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset(deviceProbabilitiesHist, 0.0 , HISTOGRAM_LENGTH * sizeof(double));

  // Copy host memory to device memory (for the float input image)
  cudaMemcpy(deviceInputImageData, hostInputImageData, (imageSize*imageChannels) * sizeof(float), cudaMemcpyHostToDevice);
  
  
  // Set up the execution configuration
  dim3 dimGrid2d(ceil(imageWidth / (float)BLOCKWIDTH), ceil(imageHeight / (float)BLOCKWIDTH), 1);
  dim3 dimBlock2d(BLOCKWIDTH, BLOCKWIDTH, 1);
  castToUchar<<<dimGrid2d, dimBlock2d>>>(deviceCharImageData, deviceInputImageData, imageWidth, imageHeight, imageChannels);// we cast it to char first.
  rgbToGrayscaleKernel<<<dimGrid2d, dimBlock2d>>>(deviceGreyScaleImageData, deviceCharImageData, imageWidth, imageHeight);//to grayscale

  //Histogram
  int stride = 4;
  int blockdim = HISTOGRAM_LENGTH;
  dim3 dimGrid1d(ceil(imageWidth*imageHeight/(float)blockdim/(float)stride), 1, 1);
  dim3 dimBlock1d(HISTOGRAM_LENGTH, 1, 1);
  histogram_plusProbility<<<dimGrid1d, dimBlock1d>>>(devicehistogram, deviceProbabilitiesHist, deviceGreyScaleImageData, imageSize);
  
  unsigned int *hosthistogram; //sil
  double *hostProbHist; //sil
  hosthistogram = (unsigned int *)malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));//sil
  hostProbHist = (double *)malloc(HISTOGRAM_LENGTH * sizeof(double));//sil
  cudaMemcpy(hosthistogram, devicehistogram, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);//sil
  cudaMemcpy(hostProbHist, deviceProbabilitiesHist, HISTOGRAM_LENGTH * sizeof(double), cudaMemcpyDeviceToHost);//sil
  wbLog(TRACE, hosthistogram[0],", " ,hosthistogram[1],", " , hosthistogram[2],", " , hosthistogram[3],", " , hosthistogram[4]);//sil
  // wbLog(TRACE, hosthistogram[251],", " ,hosthistogram[252],", " , hosthistogram[253],", " , hosthistogram[254],", " , hosthistogram[255]);//sil
  // wbLog(TRACE, hostProbHist[0],", " ,hostProbHist[1],", " , hostProbHist[2],", " , hostProbHist[3],", " , hostProbHist[4]);//sil
  // wbLog(TRACE, hostProbHist[251],", " ,hostProbHist[252],", " , hostProbHist[253],", " , hostProbHist[254],", " , hostProbHist[255]);//sil


  //paralel scan
  double *deviceHistAuxSum;
  double *deviceCDF;
  int numElements = HISTOGRAM_LENGTH;
  int numBlocks = (numElements - 1) / (HISTOGRAM_BLOCK_SIZE * 2) + 1;
  cudaMalloc((void **)&deviceHistAuxSum, numBlocks * sizeof(double));  
  cudaMemset(deviceHistAuxSum, 0.0, numBlocks * sizeof(double));
  cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(double));  
  cudaMemset(deviceCDF, 0.0, HISTOGRAM_LENGTH * sizeof(double));

  dim3 dimBlockScan(HISTOGRAM_BLOCK_SIZE, 1, 1);
  dim3 dimGridScan(numBlocks, 1, 1); // Ensuring all elements are covered
  paralelScanPFD<<<dimGridScan, dimBlockScan>>>(deviceCDF, deviceProbabilitiesHist, deviceHistAuxSum, numElements);
  paralelScanPFD<<<dim3(1, 1, 1), dimBlockScan>>>(deviceHistAuxSum, deviceHistAuxSum, deviceHistAuxSum, numBlocks);
  accumulateSums<<<dimGridScan, dimBlockScan>>>(deviceCDF, deviceHistAuxSum, numElements);
  double *hostCDF; //sil
  hostCDF = (double *)malloc(HISTOGRAM_LENGTH * sizeof(double));//sil
  cudaMemcpy(hostCDF, deviceCDF, HISTOGRAM_LENGTH * sizeof(double), cudaMemcpyDeviceToHost);//sil
  wbLog(TRACE, hostCDF[0],", " ,hostCDF[1],", " , hostCDF[2],", " , hostCDF[3],", " , hostCDF[4]);//sil
  wbLog(TRACE, hostCDF[251],", " ,hostCDF[252],", " , hostCDF[253],", " , hostCDF[254],", " , hostCDF[255]);//sil


  cudaDeviceSynchronize();  // Wait for GPU to finish before accessing on host

  // Copy device memory to host (for the unsigned char output image)
  

  // Now you can continue with the rest of the image processing steps...

  // Remember to free device memory
  cudaFree(deviceInputImageData);
  cudaFree(deviceCharImageData);
  cudaFree(deviceGreyScaleImageData);
  cudaFree(devicehistogram);
  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}
