RCNN-for-Banana-Dataset
该项目提供了一个针对香蕉数据集(d2l)的RCNN对象检测算法的简化版本实现。本项目采用PyTorch作为主要开发工具，并且包括模型训练和预测的脚本。

快速开始:
预测步骤

直接执行predict.py来使用已有的模型对测试图片进行预测。

注意微调过的AlexNet网络参数文件（classify_5th_model.pth）体积较大，请从此链接下载：
https://pan.baidu.com/s/1rjL266XTO2uBjRHXzclcVQ 
提取码为：lufh。

下载完成后将该文件放置于项目的model目录下即可。

训练步骤:
运行selective_search.py以生成所需的数据集；

接着运行train.py来训练模型；

最后再次运行predict.py查看预测结果。

  
项目参考：[https://blog.csdn.net/Myshrry/article/details/124181381?spm=1001.2014.3001.5502](https://github.com/Myshrry/RCNN-for-Banana-Dataset)

大概讲一下每个文件中模块的作用
selective_search.py:
  __init__ 设置csv文件路径，图片文件夹路径以及标志（flag）
  cal_pro_region 用到OpenCV的ximgproc模块，用于执行选择性搜索，得到完整的边界信息
  save 通过多线程并行处理图像
  save_pr 根据给定的索引列表处理图像，调用utils.py文件中的cal_IoU计算出交并比，大于0.5则有效，同时保存裁剪后的图片到相应文件夹中

  
train.py
 先对图像进行预处理，组织和加载训练和评估的图像数据集，读取并处理，设置相关超参数等，读取模型权重变差，设置评估模式，调用bb_regression中train模块，利用给定的CSV文件作为输入图像路径和标签信息，结合transform转换图像，net模型以及设备设置来创建训练和验证数据加载器，查看训练结果
  
