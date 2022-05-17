import argparse
import os, sys
import time
import tabulate

import torch
import torch.nn.functional as F
import torchvision
import numpy as np

import data, models, utils, losses
from copy import deepcopy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DSWA training")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        required=True,
        help="training directory (default: None)",
    )

    parser.add_argument(
        "--dataset", type=str, default="CIFAR10", help="dataset name (default: CIFAR10)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        required=True,
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
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 128)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        metavar="N",
        help="number of workers (default: 4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
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
        default=90,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=160,
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
    parser.add_argument("--dswa", action="store_true", default=True, help="dswa usage flag (default: off)")
    parser.add_argument("--tswa", action="store_true", default=True, help="tswa usage flag (default: off)")
    parser.add_argument(
        "--swa_start",
        type=float,
        default=30,
        metavar="N",
        help="SWA start epoch number (default: 30)",
    )
    parser.add_argument(
        "--dswa_start",
        type=float,
        default=50,
        metavar="N",
        help="DSWA start epoch number (default: 50)",
    )
    parser.add_argument(
        "--tswa_start",
        type=float,
        default=70,
        metavar="N",
        help="TSWA start epoch number (default: 70)",
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

    parser.add_argument(
        "--swa_resume",

        type=str,
        default=None,
        metavar="CKPT",
        help="checkpoint to restor SWA from (default: None)",
    )
    parser.add_argument(

        "--loss",
        type=str,
        default="CE",
        help="loss to use for training model (default: Cross-entropy)",
    )

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
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
    model_cfg = getattr(models, args.model)

    print("Loading dataset %s from %s" % (args.dataset, args.data_path))
    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        model_cfg.transform_train,
        model_cfg.transform_test,
        use_validation=not args.use_test,
        split_classes=args.split_classes,
    )

    print("Preparing model")
    print(*model_cfg.args)
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    model.to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd)
    swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    swa_model.to(args.device)
    swa_n = 0


    def schedule(epoch):
        t = (epoch) / (args.swa_start if args.swa else args.epochs)
        lr_ratio = args.swa_lr / args.lr_init
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return args.lr_init * factor


    criterion = losses.cross_entropy

    start_epoch = 0
    if args.resume is not None:
        print("Resume training from %s" % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    if args.swa and args.swa_resume is not None:
        checkpoint = torch.load(args.swa_resume)
        start_epoch = checkpoint["epoch"]
        swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        swa_model.to(args.device)
        swa_model.load_state_dict(checkpoint["state_dict"])

    columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]
    if args.swa:
        columns = columns[:-1] + ["swa_te_loss", "swa_te_acc"] + columns[-1:]
        swa_res = {"loss": None, "accuracy": None}

    utils.save_checkpoint(
        args.dir,
        start_epoch,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )

    test_acc = []
    train_loss = []

    for epoch in range(start_epoch, args.epochs):
        time_ep = time.time()

        if not args.no_schedule:
            lr = schedule(epoch)
        else:
            lr = args.lr_init
        if (
                epoch == 0
                or epoch % args.eval_freq == args.eval_freq - 1
                or epoch == args.epochs - 1
        ):
            test_res = utils.eval(loaders["test"], model, criterion, cuda=use_cuda)
        else:
            test_res = {"loss": None, "accuracy": None}

        if (epoch + 1) <= args.swa_start:
            train_res = utils.train_epoch(loaders["train"], model, criterion, optimizer, lr_schedule=lr, cuda=use_cuda)
            train_loss.append(train_res["loss"])
            test_res = utils.eval(loaders["test"], model, criterion, cuda=use_cuda)

        if (
                (epoch + 1) > args.swa_start
                and args.swa
                and (epoch + 1) <= args.dswa_start
        ):
            train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, lr_schedule=lr, cuda=use_cuda)
            train_loss.append(train_res["loss"])
            test_res = utils.eval(loaders['test'], model, criterion, cuda=use_cuda)
            utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
            swa_n += 1
            utils.bn_update(loaders["train"], swa_model)
            swa_res = utils.eval(loaders["test"], swa_model, criterion)
            test_acc.append(swa_res["accuracy"])
            if (epoch + 1 - args.dswa_start) == 0:
                model = deepcopy(swa_model)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.wd)
                swa_n = 0
        if (
                (epoch + 1) > args.dswa_start
                and args.dswa
                and (epoch + 1) <= args.tswa_start
        ):
            train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, lr_schedule=lr, cuda=use_cuda)
            train_loss.append(train_res["loss"])
            test_res = utils.eval(loaders["test"], model, criterion, cuda=use_cuda)
            utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
            swa_n += 1
            utils.bn_update(loaders["train"], swa_model)
            swa_res = utils.eval(loaders["test"], swa_model, criterion)
            test_acc.append(swa_res["accuracy"])
            if (epoch + 1 - args.tswa_start) == 0:
                model = deepcopy(swa_model)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.wd)
                swa_n = 0
        if (
                (epoch + 1) > args.tswa_start
                and args.tswa
        ):
            train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, lr_schedule=lr, cuda=use_cuda)
            train_loss.append(train_res["loss"])
            test_res = utils.eval(loaders["test"], model, criterion, cuda=use_cuda)
            utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
            swa_n += 1
            utils.bn_update(loaders["train"], swa_model)
            swa_res = utils.eval(loaders["test"], swa_model, criterion)
            test_acc.append(swa_res["accuracy"])



        time_ep = time.time() - time_ep

        values = [
            epoch + 1,
            lr,
            train_res["loss"],
            train_res["accuracy"],
            test_res["loss"],
            test_res["accuracy"],
            time_ep
        ]
        if args.swa:
            values = values[:-1] + [swa_res["loss"], swa_res["accuracy"]] + values[-1:]
        table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
        if epoch % 40 == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        print(table)

    if args.epochs % args.save_freq != 0:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
    np.savez(os.path.join(args.dir, "summary.npz"), train_loss=train_loss, test_acc=test_acc)