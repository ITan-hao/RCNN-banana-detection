import os
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import utils
import download
import bb_regression
import matplotlib
matplotlib.use('TkAgg') #Qt5Agg  TkAgg


if not os.path.exists('./data'):
    raise FileNotFoundError('数据不存在, 请先运行selective_search.py')


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((227, 227)),
                                transforms.Normalize(mean=[0.564, 0.529, 0.385], std=[1, 1, 1])])#标准化
train_data = ImageFolder('./data/ss/train', transform=transform)
val_data = ImageFolder('./data/ss/val', transform=transform)#使用ImageFolder创建train_data和val_data数据集
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=128)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = utils.get_Alexnet().to(device)
epochs = 15
lr = 0.001
criterion = [nn.CrossEntropyLoss(), nn.MSELoss()]

fun1 = lambda x: x.startswith('classify') #用于确定是否已经完成了分类模型的训练
if not sum([fun1(n) for n in os.listdir('./model')]):
    utils.train(train_dataloader, val_dataloader, net, epochs, lr, criterion[0], device, len(train_data), len(val_data))

net.load_state_dict(torch.load('./model/classify_5th_model.pth', map_location=device))
net.eval()

fun2 = lambda x: x.startswith('regression')
if not sum([fun2(n) for n in os.listdir('./model')]):
    bb_regression.train(net, epochs, lr, criterion[1], device, transform)


utils.show_predict(val_data, net, device, './model/classify_5th_model.pth', transform, val_data.classes)
