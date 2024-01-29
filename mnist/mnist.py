import torch
from torch.utils import data as data_
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
import os
import pathlib

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False  # 下載過就不用再下載了
plt.ion()


dataSet = torchvision.datasets.MNIST("./mnist/data", train = True, transform = torchvision.transforms.ToTensor(), download = DOWNLOAD_MNIST)


# train_loader = data_.DataLoader(dataset = dataSet, batch_size = BATCH_SIZE, shuffle = True)




