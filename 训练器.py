from 优化器 import *

### 进行神经网络训练的类

class Trainer:
    """
    network: 神经网络类的对象
    x_train, t_train, x_test, t_test: 训练数据和测试数据
    epochs: 一共遍历数据集多少次, 每次遍历一遍数据集就是一个 epoch, 类似于吃一盒药, epochs 就是一共要吃几盒药
    mini_batch_size: 批数据的大小, 即每次从数据集中随机取出用于正向和反向传播更新参数的数据数量
    相当于一盒药每次吃几粒, 几粒中的几就是 mini_batch_size
    还有一个概念: iterations 迭代次数, 等于 一盒药的药粒数量 除以 每次吃的药粒数量, 就是一盒药能吃几次. 原本含义是一个 epoch 能迭代几批数据
    optimizer: 优化方法
    optimizer_param: 优化方法中的参数, 其实就是超参数
    evaluate_sample_num_per_epoch: 统计精度时样本的数量,即有 10000 个数据, 但统计预测准确度时只计算 100 个数据的准确度, 默认为全部
    verbose: 是否打印提示信息
    """
    def __init__(self, network,
                 x_train, t_train, x_test, t_test,
                 epochs = 20, mini_batch_size = 100,
                 optimizer = 'SGD', optimizer_param = {'lr':0.01},
                 evaluate_sample_num_per_epoch = None, verbose = True):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.verbose = verbose

        # 优化器
        optimizer_class_dict = {'sgd': SGD, 'adam': Adam}   # 字典的值为类对象
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]                  # 训练数据的个数
        self.iter_per_epoch = max(self.train_size / mini_batch_size , 1)      # 每盒药能吃几次, 每个 epoch 能迭代几次
        self.max_iter = int(epochs * self.iter_per_epoch)   # 一共要吃几次药, 一共要迭代几次
        self.current_iter = 0       # 当前吃药数, iterator 数
        self.current_epoch = 0      # 当前药盒数, epoch 数

        self.train_loss_list = []   # 损失函数列表, 记录每一次迭代计算出的损失函数值
        self.train_acc_list = []    # 训练数据的精度, 记录每一 epoch 的训练数据精度
        self.test_acc_list = []     # 测试数据的精度, 记录每一 epoch 的测试数据精度

    # 训练的每一步要做的事情, 也就是训练的每一次迭代要做的事情
    def train_step(self):
        # 从总数据中随机选取一批数据, 返回这批数据的下标数组
        batch_index = np.random.choice(self.train_size,self.batch_size)

        # 这批数据的输入数据与标签数据
        x_batch = self.x_train[batch_index]
        t_batch = self.t_train[batch_index]

        # 反向传播计算这批数据的梯度
        grads = self.network.gradient(x_batch, t_batch)

        # 根据优化方法更新参数
        self.optimizer.update(self.network.params, grads)

        # 获取该批数据的损失函数值, 添加到列表中
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose:
            print("train loss:" + str(loss))

        # 判断当前迭代是不是该 epoch 的最后一次迭代
        # 如果不是, 无事发生
        # 如果是, 就更新一些数据
        if self.current_iter % self.iter_per_epoch == 0:
            # epoch 数加一
            self.current_epoch += 1

            # 如果没给样本数
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test

            # 如果给了样本数
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            # 计算并保存在在训练集和测试集上的精度
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            # epoch 结束时打印一些信息
            if self.verbose:
                print("=== epoch:" + str(self.current_epoch)
                      + ", train acc:" + str(train_acc)
                      + ", test acc:" + str(test_acc)
                      + " ===")
        # 迭代次数加一
        self.current_iter += 1

    # 总的训练要做的事情
    def train(self):
        # 迭代所有步
        for i in range(self.max_iter):
            self.train_step()

        # 训练完成后在测试集计算总的准确度
        test_acc = self.network.accuracy(self.x_test, self.t_test)

        # 打印训练结束后的信息
        if self.verbose:
            print("训练已完成!")
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))