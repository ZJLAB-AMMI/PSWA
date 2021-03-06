"""GCN模型训练与预测
"""


import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from .model import GCN

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

class Pipeline(object):
    """GCN模型训练与预测
    """

    def __init__(self, **params):
        """GCN模型训练与预测

            加载GCN模型, 生成训练必要组件实例

            Input:
            ------
            params: dict, 模型参数和超参数, 格式为:
                    {
                        'random_state': 42,
                        'model': {
                            'input_dim': 1433,
                            'output_dim': 7,
                            'hidden_dim': 16,
                            'use_bias': True,
                            'dropout': 0.5
                        },
                        'hyper': {
                            'lr': 1e-2,
                            'epochs': 100,
                            'weight_decay': 5e-4
                        }
                    }

        """

        self.__init_environment(params['random_state'])
        self.__build_model(**params['model'])
        self.__build_components(**params['hyper'])

        return

    def __init_environment(self, random_state):
        """初始化环境

            Input:
            ------
            random_state: int, 随机种子

        """

        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        return

    def __build_model(self, **model_params):
        """加载模型

            Input:
            ------
            model_params: dict, 模型相关参数

        """

        self.model = GCN(**model_params)
        self.swa_model = deepcopy(self.model)
        if torch.cuda.is_available():
            self.model.cuda()
            self.swa_model.cuda()

        return

    def __build_components(self, **hyper_params):
        """加载组件

            Input:
            ------
            hyper_params: dict, 超参数

        """

        self.epochs = hyper_params['epochs']

        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 定义优化器
        self.optimizer = optim.SGD(
            momentum=0,
            params=self.model.parameters(),
            lr=hyper_params['lr'],
            weight_decay=hyper_params['weight_decay']
        )

        return

    def train(self, dataset):
        """训练模型

            Input:
            ------
            dataset: Data, 包含X, y, adjacency, test_mask,
                     train_mask和valid_mask

        """

        # 训练集标签
        train_y = dataset.y[dataset.train_mask]

        # 记录验证集效果最佳模型
        best_model = None

        # 记录验证集最佳准确率
        best_valid_acc = 0

        swa_n = 0

        def schedule(epoch):
            t = (epoch) / 10000
            lr_ratio = 0.1
            if t <= 0.5:
                factor = 1.0
            elif t <= 0.9:
                factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
            else:
                factor = lr_ratio
            return 0.1 * factor

        for epoch in range(self.epochs):
            # 模型训练模式
            self.model.train()
            if epoch < 9990:
                lr = schedule(epoch)
                adjust_learning_rate(self.optimizer, lr)


            if epoch >= 9990:
                lr = 0.1
                adjust_learning_rate(self.optimizer, lr)

            # 模型输出
            logits = self.model(dataset.adjacency, dataset.X)
            train_logits = logits[dataset.train_mask]

            # 计算损失函数
            loss = self.criterion(train_logits, train_y)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 计算训练集准确率
            train_acc = self.predict(dataset, 'train')
            # 计算验证集准确率
            valid_acc = self.predict(dataset, 'valid')
            # 计算测试集准确率
            test_acc = self.predict(dataset, 'test')

            print('lr:{:.4f}, Epoch:{:03d}, Loss:{:.4f},TrainAcc:{:.4f},ValidAcc:{:.4f}, TestAcc:{:.4f}'.format(lr, epoch, loss, train_acc, valid_acc, test_acc))
            if epoch >= 9990:
                moving_average(self.swa_model, self.model, 1 / (swa_n + 1))
                swa_n += 1


            if valid_acc >= best_valid_acc:
                # 获得最佳验证集准确率
                best_model = copy.deepcopy(self.model)
                best_valid_acc = valid_acc

        # 最终模型为验证集效果最佳的模型
        self.model = best_model

        return

    def predict(self, dataset, split='train'):
        """模型预测

            Inputs:
            -------
            dataset: Data, Data, 包含X, y, adjacency, test_mask,
                     train_mask和valid_mask
            split: string, 待预测的节点

            Output:
            -------
            accuracy: float, 节点分类准确率

        """

        # 模型推断模式
        self.model.eval()


        # 节点mask
        if split == 'train':
            mask = dataset.train_mask
        elif split == 'valid':
            mask = dataset.valid_mask
        else:  # split == 'test'
            mask = dataset.test_mask

        # 获得待预测节点的输出
        logits = self.model(dataset.adjacency, dataset.X)
        predict_y = logits[mask].max(1)[1]


        # 计算预测准确率
        y = dataset.y[mask]
        accuracy = torch.eq(predict_y, y).float().mean()

        return accuracy
    def predicts(self, dataset, split='train'):
        """模型预测

            Inputs:
            -------
            dataset: Data, Data, 包含X, y, adjacency, test_mask,
                     train_mask和valid_mask
            split: string, 待预测的节点

            Output:
            -------
            accuracy: float, 节点分类准确率

        """

        # 模型推断模式
        self.model.eval()
        self.swa_model.eval()

        # 节点mask
        if split == 'train':
            mask = dataset.train_mask
        elif split == 'valid':
            mask = dataset.valid_mask
        else:  # split == 'test'
            mask = dataset.test_mask

        # 获得待预测节点的输出
        logits = self.model(dataset.adjacency, dataset.X)
        logits1 = self.swa_model(dataset.adjacency, dataset.X)
        predict_y = logits[mask].max(1)[1]
        predict_y1 = logits1[mask].max(1)[1]

        # 计算预测准确率
        y = dataset.y[mask]
        accuracy = torch.eq(predict_y, y).float().mean()
        swa_accuracy = torch.eq(predict_y1, y).float().mean()

        return accuracy, swa_accuracy
