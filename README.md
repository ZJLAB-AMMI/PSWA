# PSWA

This repository contains a PyTorch implementation of methods described in the paper "Stochastic Weight Averaging Revisited" by Hao Guo, Jiyong Jin, and Bin Liu (https://arxiv.org/abs/2201.00519). 

## Dependencies

PyTorch 1.9.0 

## Usage

### Image Classification

The code in this sub-folder implements periodic Stochastic Weight Averaging (PSWA), conventional SGD, Double Stochastic Weight Averaging (DSWA), and Triple Stochastic Weight Averaging (TSWA) on datasets CIFAR-10 and CIFAR-100.

#### PSWA Training

You can train a DNN with PSWA using the following command

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

##### Example

 VGG16: 

 ```python
CIFAR10
 python3 pswa.py --dir=<DIR> --data_path=<PATH> --dataset=CIFAR10 --model=VGG16 --epochs=160 --lr_init=0.05 \
                 --wd=5e-4 --pswa --pswa_start=40 --P=20
CIFAR100
 python3 pswa.py --dir=<DIR> --data_path=<PATH> --dataset=CIFAR100 --model=VGG16 --epochs=160 --lr_init=0.05 \
                 --wd=5e-4 --pswa --pswa_start=40 --P=20
 ```

#### Backbone SGD Training

You can run conventional SGD training use the following command:

```python
python3 backbone-SGD.py \
        --dir=<DIR> \
        --dataset=<DATASET> \
        --data_path=<PATH> \
        --model=<MODEL> \
        --epochs=<EPOCHS> \
        --momentum=<MOMENTUM> \
        --lr_init=<LR> \
        --wd=<WD> \
```

SGD with momentum and weight decay

```python
VGG16 CIFAR10
python3 backbone-SGD.py --dir=<DIR> --data_path=<PATH> --dataset=CIFAR10 --model=VGG16 --epochs=160 --lr_init=0.05 \
                        --momentum=0.9 --wd=5e-4
```

SGD without momentum and weight decay

```python
VGG16 CIFAR10
python3 backbone-SGD.py --dir=<DIR> --data_path=<PATH> --dataset=CIFAR10 --model=VGG16 --epochs=160 --lr_init=0.05 \
                        --momentum=0 --wd=0
```

#### DSWA Training

```
VGG16 CIFAR10
python3 dswa.py --dir=<DIR> --data_path=<PATH> --resume=<MODEL PATH> --dataset=CIFAR10 --model=VGG16 --epochs=40 \
                --wd=5e-4 --swa --dswa --swa_start=0 --dswa_start=20
```

#### TSWA Training

```python
VGG16 CIFAR10
python3 tswa.py --dir=<DIR> --data_path=<PATH> --resume=<MODEL PATH> --dataset=CIFAR10 --model=VGG16 --epochs=60 
                --wd=5e-4 --swa --dswa --tswa --swa_start=0 --dswa_start=20 --tswa_start=40
```

### Graph Classification   

```python
python3 gin_swa.py --dir=<DIR> --data_path=<PATH> --dataset=MUTAG --epochs=300 --lr_init=0.01 --swa_start=270
```

### Text Classification

```python
python3 text-classification.py --dir=<DIR> --task_name=mrpc --model_name_or_path=<PATH> --epochs=50 \
                               --learning_rate=1e-4 --weight_decay=1e-2 --swa_start=45
```



## Reference

Provided model implementations were adapted from

- VGG: (https://github.com/pytorch/vision/)
- PreResNet: (https://github.com/bearpaw/pytorch-classification)
- WideResNet: (https://github.com/meliketoy/wide-resnet.pytorch)
- GIN:(https://github.com/pyg-team/pytorch_geometric)
- Roberta: (https://github.com/huggingface/transformers)
