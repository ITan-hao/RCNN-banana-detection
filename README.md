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
