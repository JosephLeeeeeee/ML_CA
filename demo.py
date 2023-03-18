import numpy as np

# 创建一个 120 行 1 列的数组
arr = np.random.rand(120, 1)

# 将数组转换为矩阵
mat = np.matrix(arr)

# 将结果赋值给目标矩阵的第一列
target_matrix = np.zeros((120, 3))  # 创建一个 120 行 3 列的零矩阵
target_matrix[0] = mat             # 将 mat 的所有元素赋值给 target_matrix 的第一列

print(target_matrix)
