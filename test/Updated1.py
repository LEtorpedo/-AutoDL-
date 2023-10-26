import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def matrix_mult_cpu(M, N, P):
    np.random.seed(42)  # 设置随机数种子以获得可重复的结果

    A = np.random.rand(M, N) * 10
    B = np.random.rand(N, P) * 10

    start = time.time()

    # 执行矩阵乘法
    result = np.matmul(A, B)

    end = time.time()
    avg_time = end - start

    return avg_time


def matrix_mult_gpu(M, N, P):
    torch.manual_seed(42)  # 设置随机数种子以获得可重复的结果

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)  # 在GPU上设置随机数种子

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.rand(M, N, device=device) * 10
    B = torch.rand(N, P, device=device) * 10

    start = time.time()

    # 执行矩阵乘法
    result = torch.matmul(A, B)

    end = time.time()
    avg_time = end - start

    return avg_time


# 定义矩阵复杂度范围
M_range = np.arange(500, 3001, 500)
N_range = np.arange(500, 3001, 500)

# 存储 CPU 和 GPU 上的执行时间
cpu_times = np.zeros((len(M_range), len(N_range)))
gpu_times = np.zeros((len(M_range), len(N_range)))
performance_ratio = np.zeros((len(M_range), len(N_range)))

# 计算执行时间并存储到数组中
for i, M in enumerate(M_range):
    for j, N in enumerate(N_range):
        cpu_time = matrix_mult_cpu(M, N, 1)
        gpu_time = matrix_mult_gpu(M, N, 1)

        # 如果执行时间大于一个阈值，将其设为 0
        if cpu_time > 10 or gpu_time > 10:
            cpu_time = 0
            gpu_time = 0

        cpu_times[i, j] = cpu_time
        gpu_times[i, j] = gpu_time

        # 计算 CPU 和 GPU 的性能比
        if gpu_time != 0:
            performance_ratio[i, j] = cpu_time / gpu_time

# 创建网格数据
M_grid, N_grid = np.meshgrid(M_range, N_range)

# 绘制三维图表
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Execution Time vs Matrix Complexity")
ax.set_xlabel("Matrix Size (log10(M))")
ax.set_ylabel("Matrix Size (log10(N))")
ax.set_zlabel("Execution Time (seconds)")

# 将坐标单位转换为对数形式
M_log = np.log10(M_grid)
N_log = np.log10(N_grid)

# 绘制 CPU 执行时间
ax.plot_surface(M_log, N_log, cpu_times, cmap='viridis', label='CPU')

# 绘制 GPU 执行时间
ax.plot_surface(M_log, N_log, gpu_times, cmap='plasma', linestyle='dashed', linewidth=0.2, label='GPU')

ax.legend()
plt.show()

# 绘制 CPU 和 GPU 的计算时间比
fig2 = plt.figure(figsize=(12, 8))
ax2 = fig2.add_subplot(111)
ax2.set_title("CPU vs GPU Execution Time Ratio")
ax2.set_xlabel("Matrix Size (log10(M))")
ax2.set_ylabel("Matrix Size (log10(N))")

# 绘制 CPU 和 GPU 的计算时间比
ax2.plot_surface(M_log, N_log, performance_ratio, cmap='coolwarm', linewidth=0.2)

plt.show()