import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# ...

# 在主循环中计算执行时间并存储到数组中
for i, M in enumerate(M_range):
    for j, N in enumerate(N_range):
        cpu_time = matrix_mult_cpu(M, N, 1)
        gpu_time = matrix_mult_gpu(M, N, 1)

        # ...

# ...