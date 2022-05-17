import torch
import torch.nn.functional as F


def cross_entropy(model, input, target):
    # standard cross-entropy loss function

    output = model(input)

    loss = F.cross_entropy(output, target)

    return loss, output


