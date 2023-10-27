#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "matplotlibcpp.h"

#define BLOCK_SIZE 16

__global__ void matrixMul(float *A, float *B, float *C, int m, int n, int p) {
    // 计算当前线程处理的元素位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保当前线程处理的位置不超过矩阵维度
    if (row < m && col < p) {
        float sum = 0.0f;

        // 计算矩阵乘法
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * p + col];
        }

        // 保存结果到矩阵C
        C[row * p + col] = sum;
    }
}

void matrixMul_CPU(float *A, float *B, float *C, int m, int n, int p) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < p; col++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += A[row * n + i] * B[i * p + col];
            }
            C[row * p + col] = sum;
        }
    }
}

void printMatrix(float *matrix, int rows, int columns) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
            std::cout << std::setw(4) << matrix[row * columns + col] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::vector<int> matrixSizes = {100, 200, 300, 400, 500};

    std::vector<float> executionTimesCPU(matrixSizes.size());
    std::vector<float> executionTimesGPU(matrixSizes.size());

    for (int i = 0; i < matrixSizes.size(); i++) {
        int size = matrixSizes[i];

        // Allocate CPU memory for matrices A, B, and C
        float *h_A = new float[size * size];
        float *h_B = new float[size * size];
        float *h_C = new float[size * size];

        // Initialize matrices A and B with random values
        for (int row = 0; row < size; row++) {
            for (int col = 0; col < size; col++) {
                h_A[row * size + col] = static_cast<float>(rand()) / RAND_MAX;
                h_B[row * size + col] = static_cast<float>(rand()) / RAND_MAX;
            }
        }

        // Allocate GPU memory for matrices A, B, and C
        float *d_A, *d_B, *d_C;
        cudaMalloc((void **)&d_A, sizeof(float) * size * size);
        cudaMalloc((void **)&d_B, sizeof(float) * size * size);
        cudaMalloc((void **)&d_C, sizeof(float) * size * size);

        // Copy matrices A and B from CPU memory to GPU memory
        cudaMemcpy(d_A, h_A, sizeof(float) * size * size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, sizeof(float) * size * size, cudaMemcpyHostToDevice);

        // Define grid and block dimensions for GPU matrix multiplication
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y);

        // Record start time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Call the kernel function for GPU matrix multiplication
        matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, size, size, size);

        // Record end time
        cudaEventRecord(stop);

        // Wait for the GPU to finish
        cudaDeviceSynchronize();

        // Copy matrix C from GPU memory to CPU memory
        cudaMemcpy(h_C, d_C, sizeof(float) * size * size, cudaMemcpyDeviceToHost);

        // Print a part of the result for validation
        std::cout << "Partial result for size " << size << "x" << size << ":" << std::endl;
        printMatrix(h_C, std::min(size, 4), std::min(size, 4));
        std::cout << std::endl;

        // Calculate the execution time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        executionTimesGPU[i] = milliseconds;

        // Call the function for CPU matrix multiplication
        auto startTimeCPU = std::chrono::high_resolution_clock::now();
        matrixMul_CPU(h_A, h_B, h_C, size, size, size);
        auto endTimeCPU = std::chrono::high_resolution_clock::now();

        // Calculate the execution time
        std::chrono::duration<float, std::chrono::milliseconds::period> durationCPU = endTimeCPU - startTimeCPU;
        executionTimesCPU[i] = durationCPU.count();

        // Print a part of the result for validation
        std::cout << "Partial result for size " << size << "x" << size << ":" << std::endl;
        printMatrix(h_C, std::min(size, 4), std::min(size, 4));
        std::cout << std::endl;

        // Free CPU and GPU memory
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    // Save data to a file
    std::ofstream outputFile("execution_times.csv");
    outputFile << "matrixSize,executionTimeCPU,executionTimeGPU" << std::endl;
    for (int i = 0; i < matrixSizes.size(); i++) {
        outputFile << matrixSizes[i] << ","
                    << executionTimesCPU[i] << ","
                    << executionTimesGPU[i] << std::endl;
    }
    outputFile.close();

    // Read the saved data from the file
    std::ifstream inputFile("execution_times.csv");
    std::string line;
    std::vector<int> sizes;
    std::vector<float> timesCPU;
    std::vector<float> timesGPU;
    std::getline(inputFile, line); // Read and discard the header line
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        std::string sizeStr, timeCPU, timeGPU;
        std::getline(iss, sizeStr, ',');
        std::getline(iss, timeCPU, ',');
        std::getline(iss, timeGPU, ',');
        sizes.push_back(std::stoi(sizeStr));
        timesCPU.push_back(std::stof(timeCPU));
        timesGPU.push_back(std::stof(timeGPU));
    }
    inputFile.close();

    // Plot the comparison chart
    namespace plt = matplotlibcpp;
    plt::plot(sizes, timesCPU, "-o", "CPU");
    plt::plot(sizes, timesGPU, "-o", "GPU");
    plt::xlabel("Matrix Size");
    plt::ylabel("Execution Time (ms)");
    plt::title("CPU vs GPU Matrix Multiplication Execution Time");
    plt::legend();
    plt::show();

    return 0;
}
