import numpy as np

# 输入: 一维数据:np 数组, 列表, 元组
# 输出: 一维数据:np 数组
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 输入: 一维数据:np 数组, 列表, 元组
# 输出: 一维数据:np 数组
def relu(x):
    return np.maximum(0,x)

# 输入: np 数组
# 输出: np 数组
def softmax(x):
    if x.ndim == 2:
        x = x.T     # 转置后使得 np 的广播功能能够方便实现
        x = x - np.max(x, axis=0)   # 转置后减去第 0 维度上的最大值
        y = np.exp(x) / np.sum(np.exp(x), axis=0)   # 除以第 0 维度上的和
        return y.T  # 转置以恢复之前的 shape

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


# a = np.array([[1,2,3],[4,5,6]])
# c = np.max(a, axis=1)
# print(type(c))
# print(c)
# b = softmax(a)
#
# print(type(b))
# print(b)