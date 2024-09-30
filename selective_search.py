# slective search生成训练集，IoU>=0.5为positive
import cv2
import pandas as pd
from utils import cal_IoU
import os
import numpy as np
import multiprocessing
import download


class Selective_search():
    def __init__(self, dir_path, ):
        '''
        Args:
            dir_path: 训练或验证数据所在文件夹
            接收训练或验证数据的目录路径作为输入，设置csv文件路径（标签文件）、图片文件夹路径以及标志（flag）
        '''
        self.csv_path = os.path.join(dir_path, 'label.csv')
        self.imgs_path = os.path.join(dir_path, 'images')
        self.flag = dir_path.split('_')[-1]
        self.num_per_image = 8 #self.num_per_image 是一个类变量，用于设置每张图片最多保存多少个候选区域
        
    @staticmethod #标记类的方法为静态方法
    def cal_pro_region(img_path):
        '''计算每张图片的proposal region
        Args:
            img_path: 图片所在路径
        Returns:
            np.array: proposal region的坐标, 大小为num*4, 4列分别[xmin, ymin, xmax, ymax]
            OpenCV的ximgproc模块计算每个图像的候选区域（rects）
        '''
        try:
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            #创建一个Selective Search Segmenter实例。
        except AttributeError:
            raise Exception('需要安装opencv-contrib-python, 安装前请先删除原有的opencv-python')
        ss.setBaseImage(cv2.imread(img_path))
        #setBaseImage 函数用于初始化 selective search 分割算法，以便后续对图像进行区域提议（region proposals）的计算。
        # 这是 selective search 算法的第一步，即确定图像中可能包含对象的候选区域。
        ss.switchToSelectiveSearchFast()
        rects = ss.process()#对图像应用选择性搜索算法，返回一个包含边界框信息的矩形数组
        #xmin, ymin, xmax, ymax
        rects[:, 2] += rects[:, 0]
        rects[:, 3] += rects[:, 1]
        #扩展了每个候选区域以包括其整个矩形。这样做的目的是为了得到完整的边界信息，以便后续可能的分析或操作。
        return rects

    def save(self, num_workers=1):
        '''可选多进程计算proposal regions
        Args:
            num_workers: 进程个数
            加载CSV文件中的数据，创建正样本和负样本的存储路径
            并通过多线程并行处理图像，根据索引分批调用save_pr方法
        '''
        self.csv = pd.read_csv(self.csv_path, header=0, index_col=None)#用于读取csv格式的文件并将其转换为DataFrame格式
        self.positive_path = './data/ss/' + self.flag + '/banana/'
        self.negative_path = './data/ss/' + self.flag + '/background/'
        os.makedirs(self.positive_path, exist_ok=True)
        #如果指定的 path 目录已经存在，exist_ok=True 参数会让这个函数不抛出异常，而是直接继续执行，不会因为目标目录已存在而中断。
        os.makedirs(self.negative_path, exist_ok=True)
        index = self.csv.index.to_list()
        span = len(index)//num_workers#平均每个进程应该处理的索引范围
        print(f'=======开始计算proposal regions of {self.flag} imgs...=======')
        # (i * span)和(i + 1) * span表示当前进程负责处理的索引范围
        # multiprocessing.Process 创建多个子进程，每个进程处理一部分图像，从而加快计算提案区域的速度。
        for i in range(num_workers):
            if i != num_workers-1:
                multiprocessing.Process(target=self.save_pr, 
                            kwargs={'index': index[i*span:(i+1)*span]}).start()
                #start() 方法会启动一个新的线程来执行 save_pr 函数，针对这些特定的索引进行操作
            else:
                multiprocessing.Process(target=self.save_pr, 
                            kwargs={'index': index[i*span:]}).start()
              
    
    def save_pr(self, index):
        '''根据索引保存该图片proposal regions坐标-xmin, ymin, xmax, ymax
        Args:
            index(list): 索引
        根据给定的索引列表处理单个批次的图像
        对于每个图像，读取图像、GT（Ground Truth）边界框、计算候选区域与GT的IoU（Intersection over Union）
        根据IoU阈值筛选出正样本（IoU >= 0.5）和负样本（0.1 < IoU < 0.5），并将它们裁剪出来保存为PNG图像。
        同时记录每个positive proposal的位置到CSV文件中
        '''
        for row in index:
            img_name = self.csv.iloc[row, 0]#csv.iloc允许访问DataFrame（一种二维表格）中的特定行和列
            gt_box = self.csv.iloc[row, 2:].values
            # 表示从CSV文件读取指定行的第二列之后的所有列（即GT box，ground truth boxes）
            # 并转换为NumPy数组形式，这样可以方便地进行后续的计算操作，
            # 如IoU（Intersection over Union）匹配
            img_path = os.path.join(self.imgs_path, img_name)#os.path.abspath 和 os.path.join 来构建绝对路径
            region_pro = self.cal_pro_region(img_path) # self.cal_pro_region是一个静态方法，用于计算给定图像 img_path 的候选区域 proposal region坐标--num*4大小的np.array
            IoU = cal_IoU(region_pro, gt_box)
            # cal_IoU用于计算两个区域（region_pro是候选区域，gt_box是GroundTruthBox，即标注的边界框）
            # 之间的IntersectionoverUnion(IoU，交并比) 的函数
            # IoU是衡量两个形状重叠程度的一个指标，其值范围在0到1之间，如果
            # IoU大于某个阈值（如0.5），通常认为候选区域与真实框有较高的匹配度。
            locs_p = region_pro[np.where(IoU>=0.5)]  # IoU超过0.5，positive
            locs_n = region_pro[np.where((IoU<0.5) & (0.1<IoU))] # IoU<0.5，negative
            
            img = cv2.imread(img_path)
            for (j, loc) in enumerate(locs_p):
                crop = img[loc[1]:loc[3], loc[0]:loc[2], :]
                crop_img = self.positive_path + img_name.split('.')[0]+'_'+str(j)+'.png'
                with open('./data/ss_'+self.flag+'_loc.csv', 'a') as f:
                    f.writelines([crop_img, ',', str(loc[0]), ',', str(loc[1]), ',', str(loc[2]), ',', str(loc[3]), '\n'])
                cv2.imwrite(crop_img, crop)#调用cv2.imwrite()保存裁剪后的子图到相应的文件路径
                if j==self.num_per_image-1:
                    break
            print(f'{img_name}: {j+1}个positive', end='\t')

            for (j, loc) in enumerate(locs_n):
                crop = img[loc[1]:loc[3], loc[0]:loc[2], :]
                crop_img = self.negative_path + img_name.split('.')[0]+'_'+str(j)+'.png'
                cv2.imwrite(crop_img, crop)
                if j==self.num_per_image-1:
                    break
            print(f'{j+1}个negative')


if __name__ == '__main__':
    download.download_extract()
    Selective_search('./data/banana-detection/bananas_val').save(os.cpu_count())
    Selective_search('./data/banana-detection/bananas_train').save(os.cpu_count())
