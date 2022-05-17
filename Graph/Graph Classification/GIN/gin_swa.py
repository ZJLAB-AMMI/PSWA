import os.path as osp
import os, sys
import argparse
import torch

import torch.nn as nn
import time
import torch_geometric.transforms as T


from copy import deepcopy
import tabulate
import numpy as np
from torch_geometric.nn import GINConv, global_add_pool

import os.path as osp

import torch
import torch.nn.functional as F

from torch.nn import BatchNorm1d, Linear, ReLU, Sequential

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import utils




class GIN(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels):
        super().__init__()

        self.conv1 = GINConv(
            Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv5 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)





def train(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return {"loss": total_loss / len(loader.dataset)}


def eval(model, loader, device):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        total_correct += int((out.argmax(-1) == data.y).sum())
    return {"accuracy": total_correct / len(loader.dataset)}



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SWA training")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="training directory (default: None)",
    )

    parser.add_argument(
        "--dataset", type=str, default="MUTAG", help="dataset name"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
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
        default="GIN",
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
        default=300,
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
        default=0.01,
        metavar="LR",
        help="initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--wd", type=float, default=0e-4, help="weight decay (default: 1e-4)"
    )

    parser.add_argument("--swa", action="store_true", default=True, help="swa usage flag (default: off)")
    parser.add_argument(
        "--swa_start",
        type=float,
        default=270,
        metavar="N",
        help="SWA start epoch number (default: 161)",
    )
    parser.add_argument(
        "--swa_lr", type=float, default=0.01, metavar="LR", help="SWA LR (default: 0.02)"
    )
    parser.add_argument(
        "--swa_c_epochs",
        type=int,
        default=1,
        metavar="N",

        help="SWA model collection frequency/cycle length in epochs (default: 1)",
    )
    parser.add_argument('--use_gdc', action='store_true',
                        help='Use GDC preprocessing.')

    parser.add_argument(
        "--seed", type=int, default=20, metavar="S", help="random seed (default: 1)"
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
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MUTAG')
    dataset = TUDataset(path, name='MUTAG').shuffle()

    train_dataset = dataset[len(dataset) // 10:]
    test_dataset = dataset[:len(dataset) // 10]
    test_loader = DataLoader(test_dataset, batch_size=64)
    train_loader = DataLoader(train_dataset, batch_size=64)



    model = GIN(dataset.num_features, 32, dataset.num_classes)
    swa_model = deepcopy(model)
    model.to(args.device)
    swa_model.to(args.device)


    criterion = F.nll_loss

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init)

    columns = ["ep", "lr", "tr_loss", "tr_acc", "te_acc", "time"]
    if args.swa:
        columns = columns[:-1] + ["swa_te_acc"] + columns[-1:]
        swa_acc = None


    if args.resume is not None:
        print("Resume training from %s" % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = 0
    swa_n = 0
    train_loss = []
    tes_acc = []
    swa_sample = []

    for epoch in range(start_epoch, args.epochs):
        time_ep = time.time()

        if (epoch + 1) <= args.swa_start:
            lr = args.lr_init
        else:
            lr = 0.02
            utils.adjust_learning_rate(optimizer, lr)


        if (epoch + 1) <= args.swa_start:
            train_res = train(model, train_loader, criterion, optimizer, args.device)
            train_acc = eval(model, train_loader, args.device)["accuracy"]
            test_acc = eval(model, test_loader, args.device)["accuracy"]
            train_loss.append(train_res["loss"])
            tes_acc.append(test_acc)
            if (epoch + 1) == args.swa_start:
                utils.save_checkpoint(args.dir, epoch + 1, state_dict=model.state_dict(), optimizer=optimizer.state_dict())

        if (
             args.swa
             and (epoch + 1) > args.swa_start
        ):
            train_res = train(model, train_loader, criterion, optimizer, args.device)
            train_acc = eval(model, train_loader, args.device)["accuracy"]
            test_acc = eval(model, test_loader, args.device)["accuracy"]
            train_loss.append(train_res["loss"])
            tes_acc.append(test_acc)
            utils.moving_average(swa_model, model, 1 / (swa_n + 1))
            swa_n += 1
            utils.update_bn(train_loader, swa_model, device=args.device)
            swa_acc = eval(swa_model, test_loader, args.device)["accuracy"]
            swa_sample.append(test_acc)


        time_ep = time.time() - time_ep
        values = [
            epoch + 1,
            lr,
            train_res["loss"],
            train_acc,
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
    np.savez(os.path.join(args.dir, "result.npz"), train_loss=train_loss, test_acc=tes_acc, swa_acc=swa_acc, swa_sample=swa_sample)