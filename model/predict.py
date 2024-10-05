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
    proposal_region = Selective_search.cal_pro_region(img_path)
    img = Image.open(img_path).convert('RGB')
    img_tensor = transforms.functional.to_tensor(img)

    transform = transforms.Compose([transforms.Resize((227, 227)),
                                    transforms.Normalize(mean=[0.564, 0.529, 0.385], std=[1, 1, 1])])
                                    #mean和std均值和标准差，用来规范化像素值。
    locs, offset = [], []
    for loc in proposal_region:
        with torch.no_grad():#确保计算不计入反向传播
            crop_region = img_tensor[:, loc[1]:loc[3], loc[0]:loc[2]]
            crop = transform(crop_region).unsqueeze(0)
            if torch.argmax(alexnet(crop)).item():
                features = alexnet.features(crop)
                offset.append(linear_net(features).squeeze(0))

                locs.append(torch.tensor(loc, dtype=torch.float32))
    if locs is not None:
        offset, locs = torch.vstack(offset), torch.vstack(locs) #垂直堆叠（vertically stack）张量列表
        #vstack 被用来将这些列表中的张量按行方向合并成一个新的二维张量，这样就可以方便地处理所有位置的结果。
        index = offset.abs().sum(dim=1).argmin().item()
        result = locs[index] + offset[index]
        utils.draw_box(img, np.array(result.unsqueeze(0)))
    else:
        utils.draw_box(img)


device = torch.device('cpu')
# =====================加载Alexnet训练参数====================
alexnet_state_dict = './model/classify_5th_model.pth'
alexnet = utils.get_Alexnet()
alexnet.load_state_dict(torch.load(alexnet_state_dict, map_location=device))#map_location=device在从磁盘读取权重文件后，将它们映射到CPU设备上

alexnet.to(device)
alexnet.eval()

# =====================加载linear_net训练参数==================
linear_net = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(), nn.Linear(256*6*6, 4))
#是一个自适应平均池化层，用于从输入张量中提取特征表示。它会根据输入的尺寸动态调整池化的窗口大小，通常用于减小维度并保持不变的特征图大小。
#将前面池化后的二维张量展平成一维向量，便于后续的一维操作（如全连接层）
linear_state_dict = './model/regression_15th_model.pth'
linear_net.load_state_dict(torch.load(linear_state_dict, map_location=device))
linear_net.to(device)
linear_net.eval()

if __name__=='__main__':
    img_path = './test_imgs/001.png'
    predict(img_path, alexnet, linear_net)

