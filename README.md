# Pytorch-lightning-template
一个自用的pytorchlightning训练框架
## 安装
1.安装依赖：
```shell
pip install -r requirements.txt
```
2.具体步骤：

2.1 准备数据集：

数据集文件夹目录结构如下：
```kotlin
data
|---DRIVE
|   |--images  
|     |--01.tif
|     |--......
|   |--masks
|     |--01.gif
|     |--......
|---......  
```

制作文件索引(执行命令的位置在DRIVE的images)：
```shell
(linux):find  -name "*.tif" > ../drive_train.txt
(windows):dir /s /b *.tif > ..\drive_train.txt
```
执行后会生成一个drive_train.txt

2.2训练
```shell
python main.py -c drive_unet
```
c代表当前的任务配置文件，该文件处于根目录下的congifs文件夹下，以yaml文件存储

# 代码来源
* WZMIAOMIAO / deep-learning-for-image-processing 
* thuanz123 / enhancing-transformers 
* JunMa11 / SegLoss 

# TODO
-[ ] 当前的学习率衰减策略会导致在第16个epoch时为0，因此需要调整
-[x] 实现VQGAN+Seg
-[x] 实现灰度图在tensorboard中的彩色显示



