#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cublas_v2.h>

__global__ void conv_gpu_kernel(const float* input, const float* kernel, float* output, int input_height, int input_width, int kernel_size, int input_channels, int kernel_channels) {
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.z;

    float sum = 0.0;
    for (int ki = 0; ki < kernel_size; ki++) {
        for (int kj = 0; kj < kernel_size; kj++) {
            int input_row = output_row + ki;
            int input_col = output_col + kj;

            int input_idx = (input_row * input_width + input_col) * input_channels + channel;
            int kernel_idx = (ki * kernel_size + kj) * kernel_channels + channel;
            
            sum += input[input_idx] * kernel[kernel_idx];
        }
    }

    output[(output_row * (input_width - kernel_size + 1) + output_col) * input_channels + channel] = sum;
}

void conv_gpu(const float* input, const float* kernel, float* output, int input_channels, int input_height, int input_width, int kernel_channels, int kernel_size) {
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;

    float* input_dev;
    float* kernel_dev;
    float* output_dev;

    cudaMalloc((void**)&input_dev, sizeof(float) * input_channels * input_height * input_width);
    cudaMalloc((void**)&kernel_dev, sizeof(float) * kernel_channels * kernel_size * kernel_size);
    cudaMalloc((void**)&output_dev, sizeof(float) * input_channels * output_height * output_width);

    cudaMemcpy(input_dev, input, sizeof(float) * input_channels * input_height * input_width, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_dev, kernel, sizeof(float) * kernel_channels * kernel_size * kernel_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y, input_channels);

    conv_gpu_kernel<<<numBlocks, threadsPerBlock>>>(input_dev, kernel_dev, output_dev, input_height, input_width, kernel_size, input_channels, kernel_channels);

    cudaMemcpy(output, output_dev, sizeof(float) * input_channels * output_height * output_width, cudaMemcpyDeviceToHost);

    cudaFree(input_dev);
    cudaFree(kernel_dev);
    cudaFree(output_dev);
}

void randomize(float* arr, int size) {
    srand(42);
    for (int i = 0; i < size; i++) {
        arr[i] = (float)rand() / RAND_MAX * 10;
    }
}

int main() {
    // Convolution parameters
    int input_channels = 16;
    int kernel_channels = 8;
    int input_height = 64;
    int input_width = 64;
    int kernel_size = 3;

    // Allocate memory for input, kernel, and output arrays
    int input_size = input_channels * input_height * input_width;
    int kernel_size_3d = kernel_channels * kernel_size * kernel_size;
    float* input = new float[input_size];
    float* kernel = new float[kernel_size_3d];
    float* output_gpu = new float[input_size - (kernel_size - 1) * (input_height - kernel_size + 1) * (input_width - kernel_size + 1)];

    // Initialize input and kernel arrays with random values
    randomize(input, input_size);
    randomize(kernel, kernel_size_3d);

    // GPU Convolution
    conv_gpu(input, kernel, output_gpu, input_channels, input_height, input_width, kernel_channels, kernel_size);

    // CPU Convolution (Baseline)
    float* output_cpu = new float[input_size - (kernel_size - 1) * (input_height - kernel_size + 1) * (input_width - kernel_size + 1)];
    for (int ci = 0; ci < input_channels; ci++) {
        for (int hi = 0; hi < input_height - kernel_size + 1; hi++) {
            for (int wi = 0; wi < input_width - kernel_size + 1; wi++) {
                float sum = 0.0;
                for (int ck = 0; ck < kernel_channels; ck++) {
                    for (int ki = 0; ki < kernel_size; ki++) {
                        for (int kj = 0; kj < kernel_size; kj++) {
                            int input_idx = ((hi + ki) * input_width + (wi + kj)) * input_channels + ci;
                            int kernel_idx = ((ki * kernel_size + kj) * kernel_channels + ck);
                            sum += input[input_idx] * kernel[kernel_idx];
                        }
                    }
                }
                output_cpu[(hi * (input_width - kernel_size + 1) + wi) * input_channels + ci] = sum;
            }
        }
    }

    // Compare GPU and CPU results
    bool success = true;
    for (int i = 0; i < input_size - (kernel_size - 1) * (input_height - kernel_size + 1) * (input_width - kernel_size + 1); i++) {
        if (fabs(output_gpu[i] - output_cpu[i]) > 1e-5) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "GPU Convolution matches CPU Convolution!" << std::endl;
    } else {
        std::cout << "GPU Convolution does not match CPU Convolution." << std::endl;
    }

    // Clean up memory
    delete[] input;
    delete[] kernel;
    delete[] output_gpu;
    delete[] output_cpu;

    return 0;
}
