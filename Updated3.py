import torch
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

gemm_execution_times = []
conv_execution_times = []

def matrix_multiplication_gpu(matrix_a, matrix_b):
    return torch.mm(matrix_a, matrix_b)

def convolution_gpu(input_data, kernel):
    return torch.nn.functional.conv2d(input_data, kernel, padding=1)

def generate_random_matrix(rows, columns):
    return torch.randn(rows, columns, dtype=torch.float)

def generate_random_image(height, width):
    return torch.randn(1, height, width)

def generate_random_kernel(out_channels, in_channels, kernel_height, kernel_width):
    return torch.randn(out_channels, in_channels, kernel_height, kernel_width, dtype=torch.float)

if __name__ == "__main__":
    sizes = list(range(500, 3001, 500))  # 不同尺寸范围
    num_experiments = 10  # 重复实验次数

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gemm_execution_times = np.empty((len(sizes), num_experiments))
    conv_execution_times = np.empty((len(sizes), num_experiments))

    for i, size in enumerate(sizes):
        total_gemm_execution_time = 0
        total_conv_execution_time = 0

        avg_gemm_execution_times = []
        avg_conv_execution_times = []

        for j in range(num_experiments):
            matrix_a = generate_random_matrix(size, size).to(device)
            matrix_b = generate_random_matrix(size, size).to(device)
            image = generate_random_image(size, size).to(device)
            kernel = generate_random_kernel(3, 1, 3, 3).to(device)

            start_time = time.time()
            result_gemm = matrix_multiplication_gpu(matrix_a, matrix_b)
            torch.cuda.synchronize()
            end_time = time.time()
            execution_time = end_time - start_time
            total_gemm_execution_time += execution_time

            start_time = time.time()
            result_conv = convolution_gpu(image, kernel)
            torch.cuda.synchronize()
            end_time = time.time()
            execution_time = end_time - start_time
            total_conv_execution_time += execution_time

            avg_gemm_execution_times.append(total_gemm_execution_time / (j + 1))
            avg_conv_execution_times.append(total_conv_execution_time / (j + 1))

        gemm_execution_times[i] = avg_gemm_execution_times
        conv_execution_times[i] = avg_conv_execution_times

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = torch.tensor(sizes)
    y = torch.tensor(range(1, num_experiments + 1))

    # 创建网格
    x, y = torch.meshgrid(x, y)
    x = x.numpy()
    y = y.numpy()

    # 绘制通用矩阵乘（GEMM）执行时间曲面
    ax.plot_surface(x, y, gemm_execution_times, cmap='viridis', label='GEMM')

    # 绘制卷积计算执行时间曲面
    ax.plot_surface(x, y, conv_execution_times, cmap='magma', label='Convolution')

    # 添加标签和标题
    ax.set_xlabel('Matrix/Image Size')
    ax.set_ylabel('Experiment')
    ax.set_zlabel('Execution Time (s)')
    ax.set_title('Performance Comparison')

    # 添加图例
    ax.legend()

    # 显示图像
    plt.show()