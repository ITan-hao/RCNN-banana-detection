from torchvision import transforms
import torch
from selective_search import Selective_search
import utils
from torch import nn
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg') #Qt5Agg  TkAgg

def predict(img_path, alexnet, linear_net):
    proposal_region = Selective_search.cal_pro_region(img_path)#使用Selective Search算法获取图像的候选区域。
    img = Image.open(img_path).convert('RGB')#打开图像文件，并将其转换为RGB格式以便于处理。
    img_tensor = transforms.functional.to_tensor(img)#加载图像并转换成PyTorch张量，便于后续处理

    transform = transforms.Compose([transforms.Resize((227, 227)), #创建预处理变换，包括调整大小、归一化等操作
                                    transforms.Normalize(mean=[0.564, 0.529, 0.385], std=[1, 1, 1])])
                                    #mean=[0.564, 0.529, 0.385]和std=[1, 1, 1]是均值和标准差，用来规范化像素值。
    locs, offset = [], []
    for loc in proposal_region:
        with torch.no_grad():#确保计算不计入反向传播
            crop_region = img_tensor[:, loc[1]:loc[3], loc[0]:loc[2]]#剪裁图像的一部分对应于当前的候选区域
            crop = transform(crop_region).unsqueeze(0)#应用预处理，添加额外的批次维度 以便于Alexnet网络的输入形状匹配。
            if torch.argmax(alexnet(crop)).item():
                features = alexnet.features(crop) #使用预训练的Alexnet进行分类决策，如果预测结果为True，则提取特征并传递给linear_net。
                offset.append(linear_net(features).squeeze(0)) #将Alexnet的特征映射通过线性网络，得到位置偏移信息。
                #通过linear_net(features)将这些特征映射到线性空间，得到位置偏移量，并用squeeze(0)消除多余的维度
                locs.append(torch.tensor(loc, dtype=torch.float32))#记录候选区域的位置
    if locs is not None: #当有候选区域时，计算偏移量和位置，并找到具有最小偏移总和的区域作为预测结果
        offset, locs = torch.vstack(offset), torch.vstack(locs) #垂直堆叠（vertically stack）张量列表
        #vstack 被用来将这些列表中的张量按行方向合并成一个新的二维张量，这样就可以方便地处理所有位置的结果。
        index = offset.abs().sum(dim=1).argmin().item()
        result = locs[index] + offset[index]
        utils.draw_box(img, np.array(result.unsqueeze(0))) #squeeze()用于压缩维度，unsqueeze()则用于提升维度
    else:
        utils.draw_box(img)#如果没有候选区域，直接绘制原始图像


device = torch.device('cpu')
# =====================加载Alexnet训练参数====================
alexnet_state_dict = './model/classify_5th_model.pth' #要加载的第一个模型（Alexnet）的权重文件路径
alexnet = utils.get_Alexnet()#创建一个Alexnet模型实例alexnet，通过utils.get_Alexnet()函数得到
alexnet.load_state_dict(torch.load(alexnet_state_dict, map_location=device))#map_location=device在从磁盘读取权重文件后，将它们映射到CPU设备上
#使用load_state_dict()函数加载预训练的Alexnet模型状态
alexnet.to(device)
alexnet.eval()

# =====================加载linear_net训练参数==================
linear_net = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(), nn.Linear(256*6*6, 4)) #(256*6*6输入 4输出
#是一个自适应平均池化层，用于从输入张量中提取特征表示。它会根据输入的尺寸动态调整池化的窗口大小，通常用于减小维度并保持不变的特征图大小。
#将前面池化后的二维张量展平成一维向量，便于后续的一维操作（如全连接层）
linear_state_dict = './model/regression_15th_model.pth'
linear_net.load_state_dict(torch.load(linear_state_dict, map_location=device))
linear_net.to(device)
linear_net.eval()

if __name__=='__main__':
    img_path = './test_imgs/73.png'
    predict(img_path, alexnet, linear_net)

