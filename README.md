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

 predict.py
 调用Selective_search文件中的cal_pro_region函数，再将图片转化转为rgb格式，调用transforms中functional功能转化给张量，定义一些预处理操作，两个列表用来存储候选区位置和位置偏移信息，for循环裁剪图像，经过预处理使用alexnet进行分类决策，再将其结果经过线性网络得到位置偏移量存储到offset中，同时将图像位置张量存储到locs列表中，如果候选区域位置列表不为空垂直堆叠偏移量和位置的张量列表，再将偏移量取绝对值，按照第一维度（每个位置的偏移量）求和，在用argmin（）求最小偏移量的索引，用item（）将其转化为整数，最终的结果为图像位置和偏移量的总和，将位置提升一个维度然后转化给np.array格式，调用utils文件中draw_box显示出检测区域，没有就直接显示原图。
 将设备转化为cpu运行，因为我没给电脑配置好gpu，一直用不了gpu。定义加载的模型参数，接着调用utils中的get_alexnet函数输出两个类别，因为改了最后一层的全连接，本项目也只需要两个结果，positive或negitive，加载模型的权重，将其移动到cpu设备，并将模型设置为评估模式
 构建linear_net网络模型，其中包含平均池化组件，再用flatten将其转化为一维向量，最后全连接输出4个单元，全连接网络权重来自已经训练好的模型，加载转移设备设置为评估模式
