import numpy as np

# 批数据的平均交叉熵误差
# y 为预测数据, t 为正确数据(也就是所说的标签)
def cross_entropy_error(y, t):
    # 维度为一时转化为二维数组
    if y.ndim == 1:
        t = t.reshape(1, t.size)    # 1 表明二维数组的第 0 维度大小为 1 , 即二维数组第 0 维只有一个元素
        y = y.reshape(1, y.size)    # [1,2,3] 转化为 [[1,2,3]], (3,) 转化为 (1,3)

    # 数据格式样例
    # y : [[1,2,3],
    #      [4,5,6]]
    # t : [[0,1,0],
    #      [0,0,1]]
    if t.size == y.size:
        t = t.argmax(axis=1)    # t 表示最大值的索引, t 是一维数组

    batch_num = y.shape[0]      # 批处理的数据个数就是 y 的第 0 维大小, 也可以用 t.size 表示

    # y[np.arange(batch_num), t] 表示所有数据中 t = 1 对应的分量, 是一个一维数组
    # 加上 1e-7 是防止对数 log 的对象为 0
    # 除以 batch_num 得到单个数据的平均损失函数值
    return -np.sum(np.log(y[np.arange(batch_num), t] + 1e-7)) / batch_num

# y = np.array([[1,2,3],
#      [4,5,6]])
# t = np.array([[0,1,0],
#      [0,0,1]])
# a = cross_entropy_error(np.array(y), np.array(t))
# print(a)
