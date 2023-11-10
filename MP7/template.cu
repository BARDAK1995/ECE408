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
    int start_idx = (y * width + x) * channels;
    for (int channel = 0; channel < channels; ++channel) {
      output[start_idx + channel] = (unsigned char)(255 * input[start_idx + channel]);
    }
  }
}

__global__ void rgbToGrayscaleKernel(unsigned char *grayImage, unsigned char *rgbImage, int width, int height) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x < width && y < height) {
    int rgbIdx = (y * width + x) * 3; // 3 channels for RGB
    int grayIdx = y * width + x;      // 1 channel for grayscale
    unsigned char r = rgbImage[rgbIdx];     // Red value
    unsigned char g = rgbImage[rgbIdx + 1]; // Green value
    unsigned char b = rgbImage[rgbIdx + 2]; // Blue value
    grayImage[grayIdx] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);// Compute the grayscale value using luminosity method
  }
}

__global__ void histogram_privatized(unsigned int *histogram, float *histProbability, unsigned char *grayImage, int imagesize){
  __shared__ unsigned int privateHistogram[HISTOGRAM_LENGTH];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride = blockDim.x * gridDim.x;
  const int localIdx = threadIdx.x;
  if (localIdx < HISTOGRAM_LENGTH) privateHistogram[localIdx] = 0; // Initialize private histogram with zeros
  __syncthreads();
  // Populate private histogram
  while (idx < imagesize) {
      atomicAdd(&(privateHistogram[grayImage[idx]]), 1);
      idx += stride;
  }
  __syncthreads();
  // Merge private histogram into global histogram
  if (localIdx < HISTOGRAM_LENGTH) {
      atomicAdd(&(histogram[localIdx]), privateHistogram[localIdx]);
  }
  //we can do the hist probability calculation in this kernel, no need for additional launches
  while (idx > 0) {
    histProbability[grayImage[idx]] = histogram[grayImage[idx]] / (float)imagesize; //hist probability also calculated here, not the perfect implementation, but works
    idx -= stride;
  }
}
__global__ void paralelScanPFD(float *output, float *histogramProb,  float *AuxilarySum, int len) {
  const int sectionsize = blockDim.x*2;
  __shared__ float XY[HISTOGRAM_BLOCK_SIZE*2];
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  // Load data into shared memory
  XY[threadIdx.x]              = (i < len)              ? histogramProb[i]              : 0.0f;
  XY[threadIdx.x + blockDim.x] = (i + blockDim.x < len) ? histogramProb[i + blockDim.x] : 0.0f;
  // Reduction forward
  for(int stride = 1; stride <= blockDim.x; stride *= 2) {
      __syncthreads();
      int index = (threadIdx.x + 1) * 2 * stride - 1;
      if(index < sectionsize && (index-stride) >= 0)
        XY[index] += XY[index - stride];
  }
  // Traverse back up
  for (int stride = sectionsize / 4; stride > 0; stride /= 2) {
      __syncthreads();
      int index = (threadIdx.x + 1) * stride * 2 - 1;
      if(index + stride < sectionsize)
        XY[index + stride] += XY[index];
  }
  __syncthreads();
  // Write results back to global memory
  //if its the last thread of the block, make it do this work, we do this first to make the kernel usable for SUM scanning as well
  if(threadIdx.x == (blockDim.x-1)) AuxilarySum[blockIdx.x] =  XY[sectionsize-1];
  if (i < len) output[i] = XY[threadIdx.x];
  if ((i + blockDim.x) < len) output[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}
__global__ void accumulateSums(float *DataArray, float *Sums, int len) {
  int index = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  int index2 = 2 * blockIdx.x * blockDim.x + threadIdx.x + blockDim.x;
  if((blockIdx.x > 0) && (index < len)) DataArray[index] += Sums[blockIdx.x-1];
  if((blockIdx.x > 0) && (index2 < len)) DataArray[index2] += Sums[blockIdx.x-1];
}

__global__ void applyHistogramEqualizationRGB(float *outputImage, unsigned char *rgbImage_unchar, float *cdf, int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  const float cdfmin = (float)cdf[0];
  __syncthreads();
  //we need the cdfmin value before starting
  if (x < width && y < height) {
    int idx = y * width + x;
    int rgbIdx = idx * channels; // Index for RGB image
    // Apply the histogram equalization function to each channel
    for (int channel = 0; channel < channels; ++channel) {
      unsigned char pixel = rgbImage_unchar[rgbIdx + channel]; // Pixel value from each channel
      float correctedValue = 255.0f * (cdf[pixel] - cdfmin) / (1.0f - cdfmin); 
      correctedValue = fmin(fmax(correctedValue, 0.0f), 255.0f); // Clamp the value
      outputImage[rgbIdx + channel] = correctedValue / 255.0f;
    }
  }
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

  //Part 1: casting to unsigned int and making it greyscale
  int imageSize = imageWidth * imageHeight;
  float *deviceInputImageData;
  float* deviceOutputImageData;
  unsigned char *deviceCharImageData;
  unsigned char *deviceGreyScaleImageData;
  cudaMalloc((void **)&deviceInputImageData, (imageSize*imageChannels) * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData, (imageSize*imageChannels) * sizeof(float));
  cudaMalloc((void **)&deviceCharImageData, (imageSize*imageChannels) * sizeof(unsigned char));
  cudaMalloc((void **)&deviceGreyScaleImageData, imageSize * sizeof(unsigned char));
  cudaMemcpy(deviceInputImageData, hostInputImageData, (imageSize*imageChannels) * sizeof(float), cudaMemcpyHostToDevice); // Copy host memory to device memory (for the float input image)
  dim3 dimGrid2d(ceil(imageWidth / (float)BLOCKWIDTH), ceil(imageHeight / (float)BLOCKWIDTH), 1);
  dim3 dimBlock2d(BLOCKWIDTH, BLOCKWIDTH, 1);
  castToUchar<<<dimGrid2d, dimBlock2d>>>(deviceCharImageData, deviceInputImageData, imageWidth, imageHeight, imageChannels);// we cast it to char first.
  rgbToGrayscaleKernel<<<dimGrid2d, dimBlock2d>>>(deviceGreyScaleImageData, deviceCharImageData, imageWidth, imageHeight);//to grayscale

  //Part 2: Histogram
  unsigned int *devicehistogram;
  float *deviceProbabilitiesHist;
  cudaMalloc((void **)&devicehistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&deviceProbabilitiesHist, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemset(devicehistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset(deviceProbabilitiesHist, 0.0 , HISTOGRAM_LENGTH * sizeof(float));
  int partition = 4;
  int blockdim = HISTOGRAM_LENGTH;
  dim3 dimGrid1d(ceil((imageWidth*imageHeight) / ((float)blockdim*partition)), 1, 1);
  dim3 dimBlock1d(HISTOGRAM_LENGTH, 1, 1);
  histogram_privatized<<<dimGrid1d, dimBlock1d>>>(devicehistogram, deviceProbabilitiesHist, deviceGreyScaleImageData, imageSize);

  //Part 3: paralel scan to get CDF which will be used in histogram equalization function
  float *deviceHistAuxSum;
  float *deviceCDF;

  int numElements = HISTOGRAM_LENGTH;
  int numBlocks = (numElements - 1) / (HISTOGRAM_BLOCK_SIZE * 2) + 1;
  cudaMalloc((void **)&deviceHistAuxSum, numBlocks * sizeof(float));  
  cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));  
  cudaMemset(deviceHistAuxSum, 0.0f, numBlocks * sizeof(float));
  cudaMemset(deviceCDF, 0.0f, HISTOGRAM_LENGTH * sizeof(float));
  dim3 dimBlockScan(HISTOGRAM_BLOCK_SIZE, 1, 1);
  dim3 dimGridScan(numBlocks, 1, 1); 
  paralelScanPFD<<<dimGridScan, dimBlockScan>>>(deviceCDF, deviceProbabilitiesHist, deviceHistAuxSum, numElements);
  paralelScanPFD<<<dim3(1, 1, 1), dimBlockScan>>>(deviceHistAuxSum, deviceHistAuxSum, deviceHistAuxSum, numBlocks);
  accumulateSums<<<dimGridScan, dimBlockScan>>>(deviceCDF, deviceHistAuxSum, numElements);
  applyHistogramEqualizationRGB<<<dimGrid2d, dimBlock2d>>>(deviceOutputImageData, deviceCharImageData, deviceCDF, imageWidth, imageHeight, imageChannels);  //Part 4: histogram equalization function applied! to get the corrected image
  
  cudaDeviceSynchronize(); 

  cudaMemcpy(hostOutputImageData, deviceOutputImageData, (imageSize*imageChannels) * sizeof(float), cudaMemcpyDeviceToHost);  // Copy device memory to host (for the unsigned char output image)

  // Remember to free device memory
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceCDF);
  cudaFree(deviceHistAuxSum);
  cudaFree(deviceProbabilitiesHist);
  cudaFree(deviceCharImageData);
  cudaFree(deviceGreyScaleImageData);
  cudaFree(devicehistogram);
  wbSolution(args, outputImage);
  return 0;
}
