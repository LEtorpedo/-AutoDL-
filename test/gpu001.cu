#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// CUDA 核函数
__global__ void matrix_multiply(float* a, float* b, float* c, int m, int n, int p)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * p + col];
        }
        c[row * p + col] = sum;
    }
}

// 生成随机矩阵
void generate_matrix(float* matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

// CPU 串行版本的矩阵乘法
void cpu_matrix_multiply(float* a, float* b, float* c, int m, int n, int p)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * p + j];
            }
            c[i * p + j] = sum;
        }
    }
}

int main()
{
    int m = 100;
    int n = 100;
    int p = 100;

    float* h_a = new float[m * n];
    float* h_b = new float[n * p];
    float* h_c_cpu = new float[m * p];
    float* h_c_gpu = new float[m * p];

    // 生成随机矩阵
    generate_matrix(h_a, m, n);
    generate_matrix(h_b, n, p);

    // 在 GPU 上进行矩阵乘法
    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, m * n * sizeof(float));
    cudaMalloc((void**)&d_b, n * p * sizeof(float));
    cudaMalloc((void**)&d_c, m * p * sizeof(float));

    cudaMemcpy(d_a, h_a, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * p * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((p + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 清空GPU事件流
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录 GPU 开始时间
    cudaEventRecord(start, 0);

    matrix_multiply<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, m, n, p);

    // 记录 GPU 结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算 GPU 执行时间
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(h_c_gpu, d_c, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // 在 CPU 上进行矩阵乘法
    cpu_matrix_multiply(h_a, h_b, h_c_cpu, m, n, p);

    // 比较 GPU 和 CPU 的结果
    float max_diff = 0.0f;
    for (int i = 0; i < m * p; i++) {
        float diff = abs(h_c_gpu[i] - h_c_cpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    std::cout << "Max difference between GPU and CPU results: " << max_diff << std::endl;

    // 计算加速比
    float cpu_time = 0.0f;  // CPU 执行时间
    float speedup = 0.0f;  // 加速比

    cudaEvent_t cpu_start, cpu_stop;
    cudaEventCreate(&cpu_start);
    cudaEventCreate(&cpu_stop);

    // 记录 CPU 开始时间
    cudaEventRecord(cpu_start, 0);

    cpu_matrix_multiply(h_a, h_b, h_c_cpu, m, n, p);

    // 记录 CPU 结束时间
    cudaEventRecord(cpu_stop, 0);
    cudaEventSynchronize(cpu_stop);

    // 计算 CPU 执行时间
    cudaEventElapsedTime(&cpu_time, cpu_start, cpu_stop);

    // 计算加速比
    if (gpu_time > 0 && cpu_time > 0) {
        speedup = cpu_time / gpu_time;
    }

    std::cout << "GPU execution time: " << gpu_time << " ms" << std::endl;
    std::cout << "CPU execution time: " << cpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << speedup << std::endl;

    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;

    return 0;
}