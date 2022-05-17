import argparse
import os, sys

import time
import tabulate

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import torch.nn as nn
from script.model import GraphSAGE
from script.dataset import Dataset
from script.prepare import prepare
from script.utils import load_config
from copy import deepcopy
from script import utils


def train_one_epoch(model, optimizer, dataset, criterion):
    # 训练集标签
    train_y = dataset.y[dataset.train_mask]
    model.train()
    # 模型输出
    logits = model(dataset.adjacency, dataset.X)
    train_logits = logits[dataset.train_mask]

    # 计算损失函数
    loss = criterion(train_logits, train_y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 计算训练集准确率
    # 获得待预测节点的输出
    logits = model(dataset.adjacency, dataset.X)
    predict_y = logits[dataset.train_mask].max(1)[1]
    y = dataset.y[dataset.train_mask]
    accuracy = torch.eq(predict_y, y).float().mean()

    return {"loss": loss, "accuracy": accuracy}

def eval(model, dataset, split):
    model.eval()

    # 节点mask
    if split == 'train':
        mask = dataset.train_mask
    elif split == 'valid':
        mask = dataset.valid_mask
    else:  # split == 'test'
        mask = dataset.test_mask
    # 获得待预测节点的输出
    logits = model(dataset.adjacency, dataset.X)
    predict_y = logits[mask].max(1)[1]

    # 计算预测准确率
    y = dataset.y[mask]
    accuracy = torch.eq(predict_y, y).float().mean()

    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SWA training")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="training directory (default: None)",
    )

    parser.add_argument(
        "--dataset", type=str, default="cora", help="dataset name"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../Dataset",
        metavar="PATH",
        help="path to datasets location (default: None)",
    )
    parser.add_argument(
        "--use_test",
        default=True,
        dest="use_test",
        action="store_true",
        help="use test dataset instead of validation (default: False)",
    )
    parser.add_argument("--split_classes", type=int, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
        metavar="MODEL",
        help="model name (default: None)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="checkpoint to resume training from (default: None)",
    )

    parser.add_argument(

        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=150,
        metavar="N",
        help="save frequency (default: 25)",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=5,
        metavar="N",
        help="evaluation frequency (default: 5)",
    )
    parser.add_argument(
        "--lr_init",
        type=float,
        default=0.1,
        metavar="LR",
        help="initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--wd", type=float, default=5e-4, help="weight decay (default: 1e-4)"
    )

    parser.add_argument("--swa", action="store_true", default=True, help="swa usage flag (default: off)")
    parser.add_argument(
        "--swa_start",
        type=float,
        default=160,
        metavar="N",
        help="SWA start epoch number (default: 161)",
    )
    parser.add_argument(
        "--swa_lr", type=float, default=0.05, metavar="LR", help="SWA LR (default: 0.02)"
    )
    parser.add_argument(
        "--swa_c_epochs",
        type=int,
        default=1,
        metavar="N",

        help="SWA model collection frequency/cycle length in epochs (default: 1)",
    )

    parser.add_argument(
        "--seed", type=int, default=40, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument("--no_schedule", action="store_true", help="store schedule")

    args = parser.parse_args()

    args.device = None

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    print("Preparing directory %s" % args.dir)
    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, "command.sh"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    print("Using model %s" % args.model)
    print("Loading dataset %s from %s" % (args.dataset, args.data_path))
    # 加载数据集
    dataset = Dataset(args.dataset, args.data_path)
    prep_dataset = prepare(dataset)

    # 加载模型
    config = load_config(config_file='config.yaml')
    model_params = config[args.dataset]['model']
    model = GraphSAGE(**model_params)
    swa_model = deepcopy(model)
    model.to(args.device)
    swa_model.to(args.device)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(params=model.parameters(), weight_decay=args.wd, lr=args.lr_init)

    columns = ["ep", "lr", "tr_loss", "tr_acc", "te_acc", "time"]
    if args.swa:
        columns = columns[:-1] + ["swa_te_acc"] + columns[-1:]
        swa_acc = None



    start_epoch = 0
    swa_n = 0
    train_loss = []
    tes_acc = []
    swa_sample = []

    for epoch in range(start_epoch, args.epochs):
        time_ep = time.time()

        if (epoch + 1) <= args.swa_start:
            lr = args.lr_init
            utils.adjust_learning_rate(optimizer, lr)
        else:
            lr = args.swa_lr
            utils.adjust_learning_rate(optimizer, lr)


        if (epoch + 1) <= args.swa_start:
            train_res = train_one_epoch(model, optimizer, prep_dataset, criterion)
            test_acc = eval(model, prep_dataset, 'test')
            train_loss.append(train_res["loss"].cpu().detach().numpy())
            tes_acc.append(test_acc.cpu().detach().numpy())
            if (epoch + 1) == args.swa_start:
                utils.save_checkpoint(args.dir, epoch + 1, state_dict=model.state_dict(), optimizer=optimizer.state_dict())

        if (
             args.swa
             and (epoch + 1) > args.swa_start
        ):
            train_res = train_one_epoch(model, optimizer, prep_dataset, criterion)
            test_acc = eval(model, prep_dataset, 'test')
            train_loss.append(train_res["loss"].cpu().detach().numpy())
            tes_acc.append(test_acc.cpu().detach().numpy())
            utils.moving_average(swa_model, model, 1 / (swa_n + 1))
            swa_n += 1
            swa_acc = eval(swa_model, prep_dataset, 'test')
            swa_sample.append(test_acc.cpu().detach().numpy())



        time_ep = time.time() - time_ep
        values = [
            epoch + 1,
            lr,
            train_res["loss"],
            train_res["accuracy"],
            test_acc,
            time_ep,
        ]
        if args.swa:
            values = values[:-1] + [swa_acc] + values[-1:]
        table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
        if epoch % 40 == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        print(table)
    swa_acc = swa_acc.cpu().numpy()
    np.savez(os.path.join(args.dir, "summary.npz"), train_loss=train_loss, test_acc=tes_acc, swa_acc=swa_acc, swa_sample=swa_sample)













