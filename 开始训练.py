from 加载数据集 import *
from 神经网络 import *
from 训练器 import *
import matplotlib.pylab as plt

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 设置遍历数据集的次数
max_epochs = 20

# 生成神经网络的对象
# hidden_size 就是第一个仿射层的神经元数量
# output_size 就是第二个仿射层的神经元数据, 也是输出层的大小
network = SimpleConvNet(input_dim = (1,28,28),
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'filter_stride':1, 'filter_pad':0},
                        hidden_size = 100, output_size = 10, weight_init_std = 0.01)

# 生成训练器的对象
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=10000)

# 开始训练
trainer.train()

# 保存训练出来的参数结果
network.save_params("params.pkl")
print("Saved Network Parameters!")

# 图形化展示训练结果
markers = {'train': 'o', 'test': 's'}
# 展示每一 epoch 的结果
x = np.arange(max_epochs)
# 传递训练精度数据
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
# 传递测试精度数据
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
# 设置 x 和 y 轴的标签
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()