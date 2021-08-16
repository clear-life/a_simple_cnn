import numpy as np

# 随机梯度下降法
# 随机二字体现在批处理是每次随机挑选一批数据进行计算梯度来更新权重
class SGD:
    # 学习率默认为 0.01
    def __init__(self, lr=0.01):
        self.lr = lr

    # 更新方法就是 参数 减去 学习率与偏导数的乘积
    # params 为要更新的参数, grads 为当前的梯度
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# Adam 优化方法
# 融合 Momentum 和 AdaGrad 方法
class Adam:
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
