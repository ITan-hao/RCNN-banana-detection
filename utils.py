from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import models, transforms
import torchvision.transforms.functional as tf
from torchvision.datasets import ImageFolder
import torch
from torch import nn, optim
import random
import os

def cal_IoU(box: np.array, gt_box):
    '''
    Args:
        box: nX4维的数组, 列为xmin, ymin, xmax, ymax
        gt_box: 真实框坐标[xmin, ymin, xmax, ymax]
    '''
    box_area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    inter_w = np.minimum(box[:, 2], gt_box[2]) - np.maximum(box[:, 0], gt_box[0])
    inter_h = np.minimum(box[:, 3], gt_box[3]) - np.maximum(box[:, 1], gt_box[1])
    
    inter = np.maximum(inter_w, 0) * np.maximum(inter_h, 0) #计算交集的面积
    union = box_area + gt_area - inter #并集的面积等于两个矩形框的面积之和减去它们的交集面积。


    return inter/union #返回交并比，衡量两个矩形框的重合度


def draw_box(img, boxes=None):
    '''在图像上显示检测框
    Args:
        img: 图像RGB
        boxes: 锚框, num*(xmin, ymin, xmax, ymax)
    '''
    plt.imshow(img)
    currentAxis = plt.gca()#使用plt.gca()获取当前轴
    if boxes is not None:
        for i in boxes:#左上角坐标（x, y） 宽度 高
            rect = patches.Rectangle((int(i[0]), int(i[1])),int(i[2])-int(i[0]),int(i[3])-int(i[1]),linewidth=1,edgecolor='r',facecolor='none')#facecolor='none' 让内部区域不显示颜色
            currentAxis.add_patch(rect)
    plt.show()


def get_Alexnet(pretrained=True):
    cnn = models.alexnet(pretrained=pretrained)
    cnn.classifier[-1] = nn.Linear(4096, 2)
    return cnn


def cal_RGB(): #计算给定图像数据集中RGB通道的像素平均值
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((227, 227)),
                                    ])
    temp_data = ImageFolder('./data/ss', transform=transform)
    R, G, B = 0, 0, 0
    print('正在计算数据集RGB均值...')
    for i, d in enumerate(temp_data):
        mean = d[0].mean(dim=(1, 2))
        R += mean[0].item()
        G += mean[1].item()
        B += mean[2].item()
        if (i+1)%1000 == 0:
            print(f'已完成{i+1}张图片的计算, 共{len(temp_data)}张图片')
        print('已完成计算！')
    return R/(i+1), G/(i+1), B/(i+1)

def train(train_dataloader, val_dataloader, net, epochs, lr, criterion, device, train_num, val_num, mode='classify'):
    os.makedirs('./model', exist_ok=True)
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    #net.parameters()指定了要优化的网络的所有可学习参数
    #momentum动量项，用来加速梯度下降过程并减少震荡，数值通常在0到1之间
    #weight_decay权重衰减，也叫L2正则化，防止过拟合的一种策略，通过惩罚较大的权重值来约束模型。

    print(f'=====mode={mode}, 开始训练...======')
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for i in range(epochs):
        # 训练
        net.train()
        temp_loss, temp_correct = 0, 0
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 如果是分类任务，计算当前批次的正确预测数。
            label_hat = torch.argmax(y_hat, dim=1)
            if mode=='classify':
                temp_correct += (label_hat == y).sum()
            temp_loss += loss#更新总损失

        print(f'epoch:{i+1}  train loss:{temp_loss/len(train_dataloader):.3f}, train Aacc:{temp_correct/train_num*100:.2f}%', end='\t')
        train_loss.append((temp_loss/len(train_dataloader)).item())
        if mode=='classify':
            train_acc.append((temp_correct/train_num).item())
        if (i+1)%5 == 0:
            torch.save(net.state_dict(), './model/'+mode+'_'+str(i+1)+'th_model.pth')


        temp_loss, temp_correct = 0, 0
        net.eval()
        with torch.no_grad():
            for X, y in val_dataloader:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                #自动计算预测值与真实标签之间的差异，返回的是一个标量值，表示当前批次数据的整体损失。
                loss = criterion(y_hat, y) #损失函数（模型对输入数据 x 的预测概率分布，对应的真实标签）

                label_hat = torch.argmax(y_hat, dim=1)
                if mode=='classify':
                    temp_correct += (label_hat == y).sum()
                temp_loss += loss

            print(f'test loss:{temp_loss/len(val_dataloader):.3f}, test acc:{temp_correct/val_num*100:.2f}%')

            val_loss.append((temp_loss/len(val_dataloader)).item())
            if mode=='classify':
                val_acc.append((temp_correct/val_num).item())
            

def show_predict(val_data, net, device, state_dict, transform, classes):
    net.load_state_dict(torch.load(state_dict, map_location=device))
    net.eval()
    
    plt.figure(figsize=(20, 20))
    for i in range(12):
        img_data, label_id = random.choice(val_data.imgs)
        img = Image.open(img_data)
        predict_id = torch.argmax(net(transform(img).unsqueeze(0).to(device)))

        predict = classes[predict_id]
        label = classes[label_id]
        plt.subplot(2, 6, i+1)#定义子图的位置，这里是一个2行6列网格中的第(i+1)个位置。
        plt.imshow(img)
        plt.title(f'truth:{label}\npredict:{predict}')
    plt.show()