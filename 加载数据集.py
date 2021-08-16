try:
    import urllib.request   # 抓取网络资源的库
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path              # 操作系统路径相关的库
import gzip                 # 压缩解压缩文件的库
import pickle               # 读取和保存文件的库
import os
import numpy as np

# mnist 数据集的官网
url_base = 'http://yann.lecun.com/exdb/mnist/'
# 记录 mnist 数据集的四个文件名称
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',      # 训练数据的输入数据
    'train_label': 'train-labels-idx1-ubyte.gz',    # 训练数据的标签数据
    'test_img': 't10k-images-idx3-ubyte.gz',        # 测试数据的输入数据
    'test_label': 't10k-labels-idx1-ubyte.gz'       # 测试数据的标签数据
}


save_file = "./mnist.pkl"

train_num = 60000       # 训练数据个数
test_num = 10000        # 测试数据个数
img_dim = (1, 28, 28)   # 输入数据的维度
img_size = 784          # 数据数据的 size

# 下载单个 mnist 数据集的文件
def _download(file_name):
    file_path = "./" + file_name

    # 如果要下载的文件已存在, 就直接返回
    print("*******")
    print(file_path)
    print(os.path.exists(file_path))
    if os.path.exists(file_path):
        print("afdaf")
        return

    # 如果要下载的文件不存在, 就根据 url 和 文件名下载文件
    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path) # 下载资源的 url 和 保存到本地的路径
    print("Done")

# 下载 mnist 数据集的四个文件
def download_mnist():
    # 遍历四个文件的名字, 挨个调用函数下载相应文件, 如果文件已存在, 就不用下载
    for v in key_file.values():
        _download(v)

# 将标签数据从 mnist 数据集文件中以 np 数组格式读取出来
def _load_label(file_name):
    file_path =  "./" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels

# 将输入数据从 mnist 数据集文件中以 np 数组格式读取出来
def _load_img(file_name):
    file_path =  "./" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        """
        f.read() 读数据的缓冲区
        np.uint8 数据类型
        offset=16 读数据的起始位置
        """
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)   # 重塑数据 shape, img_size 是 784
    print("Done")

    return data

# 将下载的四个 mnist 文件以 np 数组的格式读取出来, 并返回
def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset

# 下载并保存 minst 数据集
# 保存的文件名为 save_file = "mnist.pkl"
def init_mnist():
    # 下载 mnist 数据集
    download_mnist()
    # 以 np 数组格式从下载的 mnist 数据集文件中读数据
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        # 将数据 dataset 序列化后保存到文件 f 中, -1 表示用最高的序列化协议
        pickle.dump(dataset, f, -1)
    print("Done!")

# 将 X 转换成 one-hot 格式
# X 是一系列正确结果的下标,
# [5,8,3] 指第一个数据的正确预测结果是下标为 5 的物品
# 第二个数据的正确预测结果是下标为 8 的物品
# 第三个数据的正确预测结果是下标为 3 的物品
# 假设一共有 10 个物品, 则 one-hot 格式的表示为
# [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]]
def _change_one_hot_label(X):
    # 一共有 10 个结果, 所以对于 X 的每个标签, 都要生成一个 10 大小的数组来用 one-hot 格式表示标签数据
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

# 加载并按照给定格式返回 训练数据 和 测试数据
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """
    返回
    (训练图像, 训练标签), (测试图像, 测试标签)
    参数
    normalize: 是否将输入数据正规化, 即将输入数据转换为 0.0 ~ 1.0 之间的数
    flatten: 是否将数据一维化, 即将多维数据转换为一维数据
    one_hot_label: 是否返回 one-hot 格式的标签数据
    """

    # 如果文件未下载保存好, 就下载 mnist 数据集并保存为 pkl 格式
    if not os.path.exists(save_file):
        # 由于需要翻墙下载 MNIST 数据集
        # 同时直接在 MNIST 官网都不能下载第二个数据集文件(可能是因为用的 vpn ip 地址被禁止访问的原因)
        # 所以最好手动在网上找到 MNIST 的文件放到当前目录下
        init_mnist()

    # 从 mnist.pkl 文件中加载数据
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    # 输入数据的正规化处理
    if normalize:
        # 仅遍历处理输入数据
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    # 标签数据的 one-hot 处理
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    # 如果不返回一维数据, 就要将一维序列数据转化为多维数据
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)  # 数据个数 N 自动计算得出, 通道数 C 为 1 , 高 H 为 28, 宽 W 为 28

    # 返回训练数据和测试数据
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()
