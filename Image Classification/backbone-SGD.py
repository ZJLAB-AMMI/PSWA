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
    parser = argparse.ArgumentParser(description="SGD training")
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
        default=200,
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



    def schedule(epoch):
        t = (epoch) / args.epochs
        lr_ratio = 0.2
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


    columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]


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

        train_res = utils.train_epoch(loaders["train"], model, criterion, optimizer, lr_schedule=lr, cuda=use_cuda)
        train_loss.append(train_res["loss"])
        test_res = utils.eval(loaders["test"], model, criterion, cuda=use_cuda)


        if (epoch + 1) % args.save_freq == 0:
            utils.save_checkpoint(
                args.dir,
                epoch + 1,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )



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