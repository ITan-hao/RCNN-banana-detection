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
    box_area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]) #计算所有输入框的面积 box[:, 2]和box[:, 0]分别获取所有框的右下角和左上角的x坐标
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])#计算gt_box的真实框的面积

    inter_w = np.minimum(box[:, 2], gt_box[2]) - np.maximum(box[:, 0], gt_box[0])
    #计算交集的宽度。使用np.minimum函数获取两个矩形框左右边界的较小值，使用np.maximum获取两个矩形框左右边界的较大值，两者相减得到交集的宽度。
    inter_h = np.minimum(box[:, 3], gt_box[3]) - np.maximum(box[:, 1], gt_box[1]) #计算交集的高度
    
    inter = np.maximum(inter_w, 0) * np.maximum(inter_h, 0) #计算交集的面积
    union = box_area + gt_area - inter #计算并集的面积。并集的面积等于两个矩形框的面积之和减去它们的交集面积。


    return inter/union #返回交并比，即交集面积除以并集面积。这个值可以用来衡量两个矩形框的重合度，值越高表示重合度越大。


def draw_box(img, boxes=None):
    '''在图像上显示检测框
    Args:
        img: 图像RGB
        boxes: 锚框, num*(xmin, ymin, xmax, ymax)
    '''
    plt.imshow(img)
    currentAxis = plt.gca()
    if boxes is not None:
        for i in boxes:
            #参数分别是：左下角顶点坐标，width即向右，height即向上，facecolor='none'会只有框本身没颜色
            rect = patches.Rectangle((int(i[0]), int(i[1])),int(i[2])-int(i[0]),int(i[3])-int(i[1]),linewidth=1,edgecolor='r',facecolor='none')
            currentAxis.add_patch(rect)
    plt.show()


def get_Alexnet(pretrained=True):
    cnn = models.alexnet(pretrained=pretrained)
    cnn.classifier[-1] = nn.Linear(4096, 2)
    #classifier是该模型的一个属性，通常指代模型中的全连接层
    #cnn.classifier[-1]指的是classifier这个列表（或序列）中的最后一个元素，也就是最接近输出层的那个全连接层。
    # 这条语句的作用是将这个最顶层的全连接层替换为一个新的全连接层，其输入特征的数量是4096，输出的类别数是2。
    # 这意味着模型将被调整为适用于具有两个类别的分类任务。
    return cnn


def cal_RGB(): #计算给定图像数据集中RGB通道的像素平均值
    transform = transforms.Compose([transforms.ToTensor(), #用Compose把多个步骤整合到一起
                                    transforms.Resize((227, 227)),
                                    ])
    temp_data = ImageFolder('./data/ss', transform=transform) #ImageFolder 加载图像数据集
    R, G, B = 0, 0, 0
    print('正在计算数据集RGB均值...')
    #enumerate(sequence, [start = 0]) sequence：一个序列、迭代器或其他支持迭代对象。start：下标起始位置。
    for i, d in enumerate(temp_data): #遍历数据集中的每一项
        mean = d[0].mean(dim=(1, 2))#沿着标量dim指定的维数上的元素的平均值 d[0] 通常代表图像本身 mean(dim=(1, 2)) 计算每个样本在宽度和高度维度上的像素均值，得到一个一维向量，对应于三个颜色通道。
        R += mean[0].item() #.item() 以列表返回可遍历的(键, 值) 元组数组。
        G += mean[1].item()# .item() 方法将数值从张量转换为浮点数便于累加
        B += mean[2].item()
        if (i+1)%1000 == 0:
            print(f'已完成{i+1}张图片的计算, 共{len(temp_data)}张图片')
        print('已完成计算！')
    return R/(i+1), G/(i+1), B/(i+1) #计算每个图像的RGB通道像素均值，并返回这三个均值的平均值 标准化训练数据，使其具有相同的尺度。

def train(train_dataloader, val_dataloader, net, epochs, lr, criterion, device, train_num, val_num, mode='classify'):
    #训练数据集（train_dataloader）、验证数据集（val_dataloader）、损失函数（criterion） 训练样本数量（train_num）、验证样本数量（val_num），以及训练模式（mode，这里默认为分类，classify）
    os.makedirs('./model', exist_ok=True) #创建名为 ‘model’ 的目录
    net = net.to(device) #将网络（net）从CPU设备移动到指定的计算设备（device）
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005) #随机梯度下降（SGD）
    #net.parameters()指定了要优化的网络的所有可学习参数
    #momentum动量项，用来加速梯度下降过程并减少震荡，数值通常在0到1之间
    #weight_decay权重衰减，也叫L2正则化，防止过拟合的一种策略，通过惩罚较大的权重值来约束模型。

    print(f'=====mode={mode}, 开始训练...======')
    train_loss, val_loss, train_acc, val_acc = [], [], [], [] # 定义空列表来存储训练和验证期间的损失和准确率
    for i in range(epochs):
        # 训练
        net.train()#设置网络处于训练模式（net.train()）
        temp_loss, temp_correct = 0, 0 #初始化临时变量来计算损失和正确预测的总数
        for X, y in train_dataloader: #DataLoader便于分批输入到网络中
            X, y = X.to(device), y.to(device)
            y_hat = net(X)#前向传播得到预测结果y_hat
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()#计算损失并更新梯度

            # 如果是分类任务，计算当前批次的正确预测数。
            label_hat = torch.argmax(y_hat, dim=1)
            if mode=='classify':
                temp_correct += (label_hat == y).sum()
            temp_loss += loss#更新总损失

        print(f'epoch:{i+1}  train loss:{temp_loss/len(train_dataloader):.3f}, train Aacc:{temp_correct/train_num*100:.2f}%', end='\t')
        train_loss.append((temp_loss/len(train_dataloader)).item())
        if mode=='classify':
            train_acc.append((temp_correct/train_num).item())#记录训练损失和精度（仅对分类任务）
        if (i+1)%5 == 0:
            torch.save(net.state_dict(), './model/'+mode+'_'+str(i+1)+'th_model.pth')#每5个epoch保存一次模型权重

        # 验证集精度
        temp_loss, temp_correct = 0, 0
        net.eval()
        with torch.no_grad():
            for X, y in val_dataloader:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                #在交叉熵损失函数 (nn.CrossEntropyLoss) 中，它会自动计算预测值与真实标签之间的差异，返回的是一个标量值，
                # 表示当前批次数据的整体损失。
                loss = criterion(y_hat, y) #损失函数（模型对输入数据 x 的预测概率分布，对应的真实标签）

                label_hat = torch.argmax(y_hat, dim=1) #argmax 函数返回的是概率最高的类别编号 dim=1 表示沿着第二个维度（通常对应于分类标签的数量）查找最大值
                if mode=='classify':
                    temp_correct += (label_hat == y).sum()
                temp_loss += loss

            print(f'test loss:{temp_loss/len(val_dataloader):.3f}, test acc:{temp_correct/val_num*100:.2f}%')
            # 在验证集上计算损失（loss）并将其添加到验证集损失列表 val_loss 中
            val_loss.append((temp_loss/len(val_dataloader)).item()) #temp_loss 是当前迭代验证集时的总损失
            if mode=='classify':
                val_acc.append((temp_correct/val_num).item())
            

def show_predict(val_data, net, device, state_dict, transform, classes): #展示模型在验证集上的预测结果
    net.load_state_dict(torch.load(state_dict, map_location=device))
    net.eval()
    
    plt.figure(figsize=(20, 20)) # 创建一个新的图像窗口，设置其大小为20 x 20英寸
    for i in range(12):
        img_data, label_id = random.choice(val_data.imgs)
        #从验证数据集中随机选择一个图像（img_data）及其对应的标签ID（label_id）。
        img = Image.open(img_data)#使用PIL库打开选中的图像
        predict_id = torch.argmax(net(transform(img).unsqueeze(0).to(device)))#使用选定的模型（net）对处理后的图像做预测，通过argmax找出概率最高的类别索引（即最大概率）
        #对图像应用预处理（transform），增加一维以便输入网络（unsqueeze(0)），并将其移动到设备上（如GPU）
        predict = classes[predict_id]#根据预测的索引从classes列表中获取预测的类别名称
        label = classes[label_id]#获取真实标签的类别名称
        plt.subplot(2, 6, i+1)#定义子图的位置，这里是一个2行6列网格中的第(i+1)个位置。
        plt.imshow(img)#显示图像
        plt.title(f'truth:{label}\npredict:{predict}')
    plt.show()#最后显示整个图像