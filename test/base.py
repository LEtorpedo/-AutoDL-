import numpy as np
import time

def matrix_mult_cpu(M, N, P):
    np.random.seed(42)  # 设置随机数种子以获得可重复的结果

    A = np.random.rand(M, N)
    B = np.random.rand(N, P)

    start = time.time()

    # 执行矩阵乘法
    result = np.matmul(A, B)

    end = time.time()
    avg_time = end - start

    print(f"Average time on CPU: {avg_time} seconds.")
    return result

# 输入矩阵的尺寸
M = int(input("请输入矩阵的行数 M："))
N = int(input("请输入矩阵 A 的列数（或矩阵 B 的行数） N："))
P = int(input("请输入矩阵的列数 P："))

# 调用函数
out_cpu = matrix_mult_cpu(M, N, P)