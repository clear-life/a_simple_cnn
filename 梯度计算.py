import numpy as np

# 计算单个数据的梯度
def single_gradient(f, x):
    # 初始化
    h = 1e-4

    # 转化为浮点型数组
    float_x = x.astype("float64")
    grad = np.zeros_like(float_x)

    # 每个分量单独计算偏导数
    for i in range(x.size):

        # 记录下当前分量的值
        tmp = float_x[i]

        # f(x_i + h)
        float_x[i] = tmp + h
        f1 = f(float_x)

        # f(x_i - h)
        float_x[i] = tmp - h
        f2 = f(float_x)

        # 计算该分量的偏导数
        grad[i] = (f1 - f2) / (2 * h)

    return grad


# 计算批量数据的梯度
# f 是函数, X 是批量数据
def batch_gradient(f, X):
    # 单个数据
    if X.ndim == 1:
        return single_gradient(f, X)
    # 批量数据
    else:
        grad = np.zeros_like(X)
        # 挨个计算, grad 是二维 np 数组
        for i, x in enumerate(X):
            grad[i] = single_gradient(f, x)

        return grad


# 梯度计算的函数
# 输入: x 是 np 数组, 因为需要调用 ndim 成员
# 功能: 计算 x 的平方和
# 输出: np 数组
def function(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)

if __name__ == '__main__':
    X = [1,2,3,4]
    Y = [4,3,2,1]

    grad = batch_gradient(function, np.array(Y))
    print(grad)