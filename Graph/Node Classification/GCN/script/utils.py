"""辅助函数
"""


import os
import yaml

from collections import namedtuple
import torch


def create_dir(dir_path):
    # 生成文件夹
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return


# 定义数据结构
Data = namedtuple(
    typename='Data',
    field_names=[
        'X',           # 节点特征
        'y',           # 节点类别标签
        'adjacency',   # 邻接矩阵
        'test_mask',   # 测试集样本mask
        'train_mask',  # 训练集样本mask
        'valid_mask'   # 验证集样本mask
    ]
)


# 加载全局配置
def load_config(config_file):
    """加载全局配置

        加载模型参数和训练超参数, 用于不同的数据集训练模型

    """

    with open(config_file, 'r', encoding='utf-8') as f:
        # 读取yaml文件内容
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def save_checkpoint(dir, epoch, name="checkpoint", **kwargs):
    state = {"epoch": epoch}
    state.update(kwargs)
    filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
    torch.save(state, filepath)

def cyclic_linear_learning_rate(epoch, start, max, min, cycle):
    n = (epoch - start) // cycle
    k = (epoch - start - n * cycle) / cycle
    lr = max - (max - min) * k
    return lr
