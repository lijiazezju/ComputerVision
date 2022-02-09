
# coding ：utf-8

import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

def main():
    print("666")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #CIFA10  50000张训练图片
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36, shuffle=True, num_workers=0)

    #CIFA10  10000张验证图片
    val_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000, shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()
    print(len(val_label))

if __name__=='__main__':
    main()