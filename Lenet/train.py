
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

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(10):
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 500 == 499:
                with torch.no_grad():
                    outputs = net(val_image)
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item()/val_label.size(0)
                    print('[%d, %5d] train_loss %.3f test_accuracy: %.3f' %(epoch+1, step+1, running_loss/500, accuracy))
                    running_loss = 0
    print("Finished Training")

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)

if __name__=='__main__':
    main()
