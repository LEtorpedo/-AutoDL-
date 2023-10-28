import numpy as np
import time

def matrix_mult_cpu(M, N, P, num_experiments=10):
    np.random.seed(42)  # 设置随机数种子以获得可重复的结果

    A = np.random.rand(M, N)
    B = np.random.rand(N, P)

    avg_time = 0

    for _ in range(num_experiments):
        start = time.time()

        # 执行矩阵乘法
        result = np.matmul(A, B)

        end = time.time()
        avg_time += end - start

    avg_time /= num_experiments

    print(f"Average time on CPU: {avg_time} seconds.")
    return result

# 输入矩阵的尺寸
M = int(input("请输入矩阵的行数 M："))
N = int(input("请输入矩阵 A 的列数（或矩阵 B 的行数） N："))
P = int(input("请输入矩阵的列数 P："))

# 输入实验次数
num_experiments = int(input("请输入实验次数："))

# 调用函数
out_cpu = matrix_mult_cpu(M, N, P, num_experiments)