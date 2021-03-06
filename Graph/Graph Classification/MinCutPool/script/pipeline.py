"""MinCutPool训练与预测
"""


import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from .model import MinCutPoolModel
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader
from copy import deepcopy

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

class Pipeline(object):
    """MinCutPool训练与预测
    """

    def __init__(self, **params):
        """SAGPool训练与预测

            加载MinCutPool模型, 生成训练必要组件实例

            Input:
            ------
            params: dict, 模型参数和超参数, 格式为:
                    {
                        'random_state': 42,
                        'split': {
                            'test_prop': 0.2,
                            'valid_prop': 0.1
                        },
                        'model': {
                            'input_dim': 89,
                            'output_dim': 2,
                            'hidden_dim': 16,
                            'use_bias': True,
                            'dropout': 0.2,
                            'aggregate': 'sum'
                        },
                        'hyper': {
                            'lr': 1e-3,
                            'epochs': 50,
                            'patience': 50,
                            'batch_size': 8,
                            'weight_decay': 5e-4
                        }
                    }

        """

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

        # 加载模型
        self.model = MinCutPoolModel(**model_params)
        self.swa_model = deepcopy(self.model)
        self.model.to(self.device)
        self.swa_model.to(self.device)

        return

    def __build_components(self, **hyper_params):
        """加载组件

            Input:
            ------
            hyper_params: dict, 超参数

        """

        self.epochs = hyper_params['epochs']
        self.patience = hyper_params['patience']
        self.batch_size = hyper_params['batch_size']

        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 定义优化器
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=hyper_params['lr'],
            weight_decay=hyper_params['weight_decay']
        )

        return

    def train(self, dataset):
        """训练模型

            Input:
            ------
            dataset: Dataset, 包含test、valid和train

        """

        # 训练集batch加载器
        train_loader = DataLoader(
            dataset=dataset.train,
            batch_size=self.batch_size,
            shuffle=True
        )

        # 记录验证集效果最佳模型
        best_model = None

        # 记录验证集最佳准确率
        best_valid_acc = 0

        # 获得最佳的验证集后计数轮次
        epochs_after_best = 0

        swa_n = 0

        for epoch in range(self.epochs):
            # 模型训练模式
            self.model.train()
            if epoch >= 450:
                lr = 0.001
                adjust_learning_rate(self.optimizer, lr)

            # 用于记录每个epoch中所有batch的loss
            epoch_losses = []


            for i, data in enumerate(train_loader):
                # 模型输出
                data = data.to(self.device)
                logits, pool_loss = self.model(data)

                # 计算损失函数
                loss = self.criterion(logits, data.y)
                loss += pool_loss
                epoch_losses.append(loss.item())

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 计算epoch中所有batch的loss的均值
            epoch_loss = np.mean(epoch_losses)

            # 计算验证集loss和Accuracy
            valid_loss, valid_acc = self.predict(dataset, 'valid')

            print('[Epoch:{:03d}]-[TrainLoss:{:.4f}]-[ValidLoss:{:.4f}]-[ValidAcc:{:.4f}]'.format(
                epoch, epoch_loss, valid_loss, valid_acc))
            if epoch >= 450:
                moving_average(self.swa_model, self.model, 1 / (swa_n + 1))
                swa_n += 1

            if valid_acc >= best_valid_acc:
                best_model = copy.deepcopy(self.model)
                # 获得最佳验证集准确率
                best_valid_acc = valid_acc
                # 从新计数轮次
                epochs_after_best = 0
            else:
                # 未获得最佳验证集准确率
                # 增加计数轮次
                epochs_after_best += 1

            if epochs_after_best == self.patience:
                # 符合早停条件
                self.model = best_model
                break

        return

    def predict(self, dataset, split='train'):
        """模型预测

            Inputs:
            -------
            dataset: Dataset, 包含test、valid和train
            split: string, 使用的数据集划分

            Output:
            -------
            loss: float, 数据集样本平均损失
            accuracy: float, 节点分类准确率

        """

        # 模型推断模式
        self.model.eval()

        # 使用的数据集划分
        if split == 'train':
            eval_dataset = dataset.train
        elif split == 'valid':
            eval_dataset = dataset.valid
        else:  # split == 'test'
            eval_dataset = dataset.test

        # 数据集样本加载器, 每个batch加载1个样本
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=1, shuffle=False
        )

        # 用于记录每个样本的损失、标签和预测类别
        losses, y_true, y_pred = [], [], []
        for i, data in enumerate(eval_loader):
            # 模型输出
            data = data.to(self.device)
            logits, pool_loss = self.model(data)
            predict_y = logits.max(1)[1]
            loss = self.criterion(logits, data.y)
            loss += pool_loss

            y_true.append(predict_y.cpu().numpy()[0])
            y_pred.append(data.y.cpu().numpy()[0])
            losses.append(loss.item())

        # 计算所有样本的平均损失
        loss = np.mean(losses)
        # 计算准确率
        acc = accuracy_score(y_true, y_pred)
        return loss, acc

    def predicts(self, dataset, split='train'):
        """模型预测

            Inputs:
            -------
            dataset: Dataset, 包含test、valid和train
            split: string, 使用的数据集划分

            Output:
            -------
            loss: float, 数据集样本平均损失
            accuracy: float, 节点分类准确率

        """

        # 模型推断模式
        self.model.eval()
        self.swa_model.eval()

        # 使用的数据集划分
        if split == 'train':
            eval_dataset = dataset.train
        elif split == 'valid':
            eval_dataset = dataset.valid
        else:  # split == 'test'
            eval_dataset = dataset.test

        # 数据集样本加载器, 每个batch加载1个样本
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=1, shuffle=False
        )

        # 用于记录每个样本的损失、标签和预测类别
        losses, y_true, y_pred = [], [], []
        losses1, y_true, y_pred1 = [], [], []
        for i, data in enumerate(eval_loader):
            # 模型输出
            data = data.to(self.device)
            logits, pool_loss = self.model(data)
            logits1, pool_loss1 = self.swa_model(data)
            predict_y = logits.max(1)[1]
            predict_y1 = logits1.max(1)[1]
            loss = self.criterion(logits, data.y)
            loss1 = self.criterion(logits1, data.y)
            loss += pool_loss
            loss1 += pool_loss1

            y_pred.append(predict_y.cpu().numpy()[0])
            y_pred1.append(predict_y1.cpu().numpy()[0])
            y_true.append(data.y.cpu().numpy()[0])
            losses.append(loss.item())
            losses1.append(loss1.item())

        # 计算所有样本的平均损失
        loss = np.mean(losses)
        loss1 = np.mean(losses1)
        # 计算准确率
        acc = accuracy_score(y_true, y_pred)
        acc1 = accuracy_score(y_true, y_pred1)
        return loss, loss1, acc, acc1
