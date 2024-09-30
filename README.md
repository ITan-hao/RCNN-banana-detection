# RCNN-for-Banana-Dataset
#### RCNN物体检测的简化实现

--直接运行predict.py，则会利用提供的model与test_imgs进行测试

注: 微调的Alexnet网络参数文件（classify_5th_model.pth）较大，放在网盘了，下载后放入model文件夹就行

链接：https://pan.baidu.com/s/1rjL266XTO2uBjRHXzclcVQ 
提取码：lufh

--从头训练，先运行selective_search.py生成相应数据集后，再运行train.py，然后运行predict.py就能得到结果
  
项目参考：[https://blog.csdn.net/Myshrry/article/details/124181381?spm=1001.2014.3001.5502](https://github.com/Myshrry/RCNN-for-Banana-Dataset)
