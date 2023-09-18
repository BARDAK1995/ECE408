#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

void vecAddCPU(const double* in1, const double* in2, double* out, int len) {
    for(int i = 0; i < len; ++i) {
        out[i] = in1[i] + in2[i];
    }
}
__global__ void vecAddKernel(const double* a, const double* b, double* c, int n, int tastPerThread) {
    for(int threadTask = 1; threadTask <=tastPerThread; ++threadTask) {
        int i = (blockIdx.x * blockDim.x) * tastPerThread + threadIdx.x + blockDim.x * (tastPerThread - 1);
        if (i < n) {c[i] = a[i] + b[i]; }
    }
}

void vecAdd(const double* in1, const double* in2, double* out, int n) {
    double *d_in1, *d_in2, *d_out;
    size_t size = n * sizeof(double);

    cudaMalloc((void**)&d_in1, size);
    cudaMalloc((void**)&d_in2, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in1, in1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, in2, size, cudaMemcpyHostToDevice);
    int threadxx = 2;
    int blockSize = 256;
    int gridSize = ceil(n/blockSize) / blockSize / 2; //

    vecAddKernel<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, n, threadxx);

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);
}


bool readDataFromFile(const std::string& fileName, double*& data, int& n) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << fileName << std::endl;
        return false;
    }
    
    file >> n;
    data = new double[n];
    for(int i = 0; i < n; ++i) {
        file >> data[i];
    }

    return true;
}

bool writeDataToFile(const std::string& fileName, const double* data, int n) {
    std::ofstream outfile(fileName);
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file" << std::endl;
        return false;
    }

    outfile << n << '\n';
    for(int i = 0; i < n; ++i) {
        outfile << data[i] << '\n';
    }

    return true;
}

int main() {
    int n1, n2;
    double *nums1 = nullptr, *nums2 = nullptr, *results = nullptr;
    std::cout << "READING FILE" << std::endl;
    if (!readDataFromFile("input0.raw", nums1, n1) || 
        !readDataFromFile("input1.raw", nums2, n2) ||
        n1 != n2) 
    {
        std::cerr << "The number of elements in the files do not match or file reading error" << std::endl;
        delete[] nums1;
        delete[] nums2;
        return 1;
    }
    std::cout << "ADD START" << std::endl;
    results = new double[n1];
    // vecAddCPU(nums1, nums2, results, n1);
    vecAdd(nums1, nums2, results, n1);
    std::cout << "ADD FINISH Start writing" << std::endl;

    if (!writeDataToFile("output.raw", results, n1)) {
        delete[] nums1;
        delete[] nums2;
        delete[] results;
        return 1;
    }

    std::cout << "Operation successful, results written to output file" << std::endl;

    delete[] nums1;
    delete[] nums2;
    delete[] results;
    return 0;
}
