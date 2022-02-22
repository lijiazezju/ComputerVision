import os
import json

import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

from model import AlexNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_path = './1.jpg'
    assert os.path.exists(img_path), "file:{} does not exists".format(img_path)
    img = Image.open(img_path)
    print("666")
    #print(img)
    #type(img)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = "./class.json"
    assert os.path.exists(json_path), "file:'{}' does not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    model = AlexNet(num_classes=5).to(device)

    weights_path = './Alexnet.pth'
    assert os.path.exists(weights_path), "file:'{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        output = model(img.to(device))
        output = torch.squeeze(output, dim=0).cpu()
        output = torch.softmax(output, dim=0)
        predict_d = torch.argmax(output)
        predict_d = predict_d.numpy()
    print_res = "class:{} prob:{:.3}".format(class_indict[str(predict_d)], output[int(predict_d)].numpy())

    plt.title(print_res)
    for i in range(len(output)):
        print("class:{:10} prob:{:.3}".format(class_indict[str(i)], output[i].numpy()))
    plt.show()
if __name__=='__main__':
    main()