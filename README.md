# PSWA
This repository contains a PyTorch implementation of the periodic Stochastic Weight Averaging(PSWA) procedures described in the paper "Stochastic Weight Averaging Revisited" by Hao Guo, Jiyong Jin and Bin Liu (https://arxiv.org/abs/2201.00519). 

# Usage
The code in this repository implements the periodic Stochastic Weight Averaging (PSWA) algorithm, with examples on the CIFAR10 and CIFAR100 datasets.

## PSWA Training
You can train model using the following command

```
python3 pswa.py \
        --dir=<DIR> \
        --dataset=<DATASET> \
        --data_path=<PATH> \
        --model=<MODEL> \
        --epochs=<EPOCHS> \
        --lr_init=<LR> \
        --wd=<WD> \
        --pswa \
        --pswa_start=<PSWA_START> \
        --P=<P> 
 ```
 
 Parameters: \
 ```DIR``` — path to training directory where checkpoints will be stored \
```DATASET``` —  dataset name (default: CIFAR10) \
```PATH``` — path to the data directory \
```MODEL``` — DNN model name: VGG16, PreResNet164 and WideResNet28x10 \
```EPOCHS``` — number of training epochs \
```LR``` — initial learning rate \
```WD``` — weight decay \
```PSWA_START``` — the number of epoch after which PSWA will start to average models (default: 40) \
```P``` — model recording period(default:20)
 
 ### Example
 VGG16: 
 
 ```
CIFAR10
 python3 pswa.py --dir=<DIR> --data_path=<PATH> --dataset=CIFAR10 --model=VGG16 --epochs=160 --lr_init=0.05 \
                 --wd=5e-4 --pswa --pswa_start=40 --P=20
CIFAR100
 python3 pswa.py --dir=<DIR> --data_path=<PATH> --dataset=CIFAR100 --model=VGG16 --epochs=160 --lr_init=0.05 \
                 --wd=5e-4 --pswa --pswa_start=40 --P=20
 ```
        
        
