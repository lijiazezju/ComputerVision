import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
from matplotlib import pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    }
    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    image_path = os.path.join(data_root, 'flower_data')
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform['train'])
    train_num = len(train_dataset)
    #print(train_dataset)
    flower_list = train_dataset.class_to_idx
    print(flower_list)
    cla_dict = dict((val, key) for key, val in flower_list.items())
    print(cla_dict)
    json_str = json.dumps(cla_dict, indent=4)
    with open("class.json", 'w') as json_file:
        json_file.write(json_str)
    batch_size = 32
    nw = min([os.cpu_count(), batch_size, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'), transform=data_transform['val'])
    val_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=4, shuffle=True, num_workers=nw)
    val_num = len(validate_dataset)
    print("using {} images for trainging, and using {} images for val.".format(train_num, val_num))

    val_data_iter = iter(val_loader)
    test_image, test_label = val_data_iter.next()

    # def imshow(img):
    #     img = img/2 + 0.5
    #     nping = img.numpy()
    #     plt.imshow(np.transpose(nping, (1, 2, 0)))
    #     plt.show()
    #
    # print(''.join('%10s'% cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))
    net = AlexNet(num_classes=5, init_weights=True)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    #pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)


    save_path = './Alexnet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    epochs = 10
    for epoch in range(10):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar, start=0):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, epochs, loss)
        net.eval()
        acc = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_images, val_labels in val_bar:
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = acc/val_num
        print('[epoch %d] train_loss: %.3f val_accuracy: %.3f' %(epoch+1, running_loss/train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print("Training Finished")

if __name__=='__main__':
    main()

