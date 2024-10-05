# 利用最后一个池化层pool输出的特征去训练一个线性回归模型，以预测新的检测框
import utils
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch import nn

"""
定义了一个用于训练神经网络回归任务的数据集类Reg_dataset。
它主要用于香蕉检测项目中，利用给定的CSV文件（ss_train_loc.csv 和 gt_train_loc.csv）作为输入图像路径和标签信息，结合transform转换图像，net模型以及设备设置来创建训练和验证数据加载器
"""
def train(net, epochs, lr, criterion, device, transform):
    class Reg_dataset(Dataset):
        # 读取两个CSV文件（ss_train_loc.csv和gt_train_loc.csv），分别存储输入图像位置（ss_loc）和真实标签（gt_loc）
        def __init__(self, ss_csv_path, gt_csv_path, transform, net, device):
            self.ss_csv = pd.read_csv(ss_csv_path)
            self.gt_csv = pd.read_csv(gt_csv_path)
            self.transform = transform
            self.net = net
            self.device = device
        
        def __getitem__(self, index):#返回单个样本数据
            img_path, *ss_loc = self.ss_csv.iloc[index, :]#从CSV文件中获取当前图像的路径、输入位置（ss_loc）
            index = img_path.split('/')[-1].split('_')[0]+'.png'
            gt_loc = self.gt_csv[self.gt_csv.img_name==index].iloc[0, 2:].tolist()#和对应的GT标签（gt_loc）
            label = torch.tensor(gt_loc, dtype=torch.float32) - torch.tensor(ss_loc, dtype=torch.float32)#计算标签，将GT减去输入位置，转换为PyTorch张量
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                img = self.transform(img).to(device).unsqueeze(0)
                return net.features(img).squeeze(0), label
                #使用网络模型提取特征（net.features(img)）并去掉第一个维度（unsqueeze和squeeze）
        
        def __len__(self):
            return len(self.ss_csv)#返回数据集中样本的数量，等于ss_csv中的行数
    
    ss_train_loc = './data/ss_train_loc.csv'
    gt_train_loc = './data/banana-detection/bananas_train/label.csv'
    ss_val_loc = './data/ss_val_loc.csv'
    gt_val_loc = './data/banana-detection/bananas_val/label.csv'

    train_data = Reg_dataset(ss_train_loc, gt_train_loc, transform, net=net, device=device)
    val_data = Reg_dataset(ss_val_loc, gt_val_loc, transform, net=net, device=device)
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=128)

    linear_net = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(), nn.Linear(256*6*6, 4))
    nn.init.xavier_normal_(linear_net[-1].weight)#使用Xavier初始化权重。
    utils.train(train_dataloader, val_dataloader, linear_net, epochs, lr, criterion, device, len(train_data), len(val_data), mode='regression')

