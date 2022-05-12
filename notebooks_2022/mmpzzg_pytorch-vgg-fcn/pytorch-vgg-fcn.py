#!/usr/bin/env python
# coding: utf-8

# **导入必要的包**

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import PIL.Image as pil_image

import matplotlib.pyplot as plt


# **加载自定义的数据**
# 
# *参考链接： https://zhuanlan.zhihu.com/p/32506912

# In[ ]:


# 分割的类别
classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']

# 每个类别对应的RGB值
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

cm2lbl = np.zeros(256**3) # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i # 建立索引
    
# 将图片映射成索引数组
def image_to_label(image):
    """将图片映射成类别索引的数组"""
    data=np.array(image,dtype='int32')
    # 按照上面一样的计算规则，得到对应的值
    index = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[index],dtype='int64')

def random_crop(image,width,height):
    """随机裁剪"""
    pass


# 测试
image=pil_image.open('../input/voc2012/data/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png').convert('RGB')
# image = transforms.RandomCrop((224, 224))(image)
# print(image)
# plt.imshow(image)
# label = transforms.FixedCrop(*rect)(label)
# image_array=image_to_label(image)


# In[ ]:


class VOCDataset(Dataset):
    """自定义数据类加载规则"""
    def __init__(self,file_path=None,transform=None):
        """初始化函数"""
        
        images_labels=[]
        file=open(file_path)
        for name in file.readlines():
            
            # 移除空格和换行符
            name=name.strip()
            
            image="../input/voc2012/data/VOCdevkit/VOC2012/JPEGImages/"+name+".jpg"
            label="../input/voc2012/data/VOCdevkit/VOC2012/SegmentationClass/"+name+".png"
            images_labels.append((image,label))
            
        self.images_labels=images_labels
        self.transform=transform
    
    def __getitem__(self,index):
        """在DataLoader中会调用这个方法读取一个batch的数据"""
        
        image_path,label_path=self.images_labels[index]

        # 使用image.open加载目标图和特征图
        image=pil_image.open(image_path)
        label=pil_image.open(label_path).convert('RGB')
        
#         # 裁剪图片，使其所有的图片输入一致   
#         x,y,width,height=transforms.RandomCrop.get_params(img=image,output_size=(224,224))
#         image=function.crop(image,x,y,width,height)
#         label=function.crop(label,x,y,width,height)

        image=transforms.Resize((512,512))(image)
        label=transforms.Resize((512,512))(label)
        
        # 转化特征图
        if self.transform is not None:
            image=self.transform(image)
            
        # 映射目标图
        label=image_to_label(label)
        # 从numpy数组转化成张量
        label=torch.from_numpy(label)
        
        # 返回
        return image,label
    
    def __len__(self):
        """获取整个dataset的数据大小"""
        return len(self.images_labels)
    
    
# 数据预处理，增强，归一化
transform_train=transforms.Compose([
    # 将数据转化成张量，并且归一化到[0,1]
    transforms.ToTensor(),
    # 将数据标准化到[-1,1]，image=(image-mean)/std
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

transform_test=transforms.Compose([
    # 将数据转化成张量，并且归一化到[0,1]
    transforms.ToTensor(),
    # 将数据标准化到[-1,1]，image=(image-mean)/std
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


train_datasets=VOCDataset(file_path="../input/voc2012/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt",transform=transform_train)
test_datasets=VOCDataset(file_path="../input/voc2012/data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",transform=transform_test)

train_loader=DataLoader(dataset=train_datasets,batch_size=8,shuffle=False,sampler=None)
test_loader=DataLoader(dataset=test_datasets,batch_size=8,shuffle=False,sampler=None)

print(len(train_loader))
# print(next(iter(train_loader)))


# **FCN网络结构如下**
# 
# 前面由VGG16组成，分5个卷积层，后面有3个（1，1）卷积层，输出32s，16s，8s，然后在进行相加融合，上采样得出结果
# 

# ![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fwww.itdaan.com%2Fi%2Fd91a35b4436eba676dc2ee6cc6e45276.jpg&refer=http%3A%2F%2Fwww.itdaan.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1653238248&t=2c48611d3c312d0356b18fff8f203e7f)

# **构建FCN网络模型**
# * 参考链接：https://github.com/wkentaro/pytorch-fcn/blob/main/torchfcn/models/fcn8s.py

# In[ ]:


class FCN8s(nn.Module):
    def __init__(self):
        super(FCN8s,self).__init__()
        
        # 本项目有20个类别，一个背景，一共21类
        n_class=21
        
        # conv1 
        # 输入图像为3通道，输出64个特征图，卷积核大小为（3，3），步长为1，padding为100（避免图片不兼容，其实也可以为1的）
        # 卷积输出公式：output=(input+2*padding-kernel_size)/stride+1
        #  512=(512+2*1-3)/1+1
        self.conv1_1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn1_1=nn.BatchNorm2d(num_features=64)
        self.relu1_1=nn.ReLU(inplace=True)
        
        self.conv1_2=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn1_2=nn.BatchNorm2d(num_features=64)
        self.relu1_2=nn.ReLU(inplace=True)
        
        # 最大池化层进行下采样
        # 采样输出公式：output=(input+2*padding-kernel_size)/stride+1
        # 256=(512+2*0-2)/2+1
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=2)
        
        # conv2
        self.conv2_1=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn2_1=nn.BatchNorm2d(num_features=128)
        self.relu2_1=nn.ReLU(inplace=True)
        
        self.conv2_2=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn2_2=nn.BatchNorm2d(num_features=128)
        self.relu2_2=nn.ReLU(inplace=True)
        
        # 最大池化层进行下采样
        self.maxpool2=nn.MaxPool2d(kernel_size=2,stride=2)
        
        
        # conv3
        self.conv3_1=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn3_1=nn.BatchNorm2d(num_features=256)
        self.relu3_1=nn.ReLU(inplace=True)
        
        self.conv3_2=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn3_2=nn.BatchNorm2d(num_features=256)
        self.relu3_2=nn.ReLU(inplace=True)
        
        self.conv3_3=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn3_3=nn.BatchNorm2d(num_features=256)
        self.relu3_3=nn.ReLU(inplace=True)
        
        # 最大池化层进行下采样
        self.maxpool3=nn.MaxPool2d(kernel_size=2,stride=2)
        
        
        # conv4
        self.conv4_1=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn4_1=nn.BatchNorm2d(num_features=512)
        self.relu4_1=nn.ReLU(inplace=True)
        
        self.conv4_2=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn4_2=nn.BatchNorm2d(num_features=512)
        self.relu4_2=nn.ReLU(inplace=True)
        
        self.conv4_3=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn4_3=nn.BatchNorm2d(num_features=512)
        self.relu4_3=nn.ReLU(inplace=True)
        
        # 最大池化层进行下采样
        self.maxpool4=nn.MaxPool2d(kernel_size=2,stride=2)
        
        
        # conv5
        # 输入图像为3通道，输出64个特征图，卷积核大小为（3，3），步长为1，padding为100（避免图片不兼容）
        self.conv5_1=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn5_1=nn.BatchNorm2d(num_features=512)
        self.relu5_1=nn.ReLU(inplace=True)
        
        self.conv5_2=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn5_2=nn.BatchNorm2d(num_features=512)
        self.relu5_2=nn.ReLU(inplace=True)
        
        self.conv5_3=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn5_3=nn.BatchNorm2d(num_features=512)
        self.relu5_3=nn.ReLU(inplace=True)
        
        # 最大池化层进行下采样
        self.maxpool5=nn.MaxPool2d(kernel_size=2,stride=2)
        
        
        # cnov6
        self.conv6=nn.Conv2d(in_channels=512,out_channels=4096,kernel_size=7,stride=1,padding=1)
        self.bn6=nn.BatchNorm2d(num_features=4096)
        self.relu6=nn.ReLU(inplace=True)
        self.drop6=nn.Dropout2d(p=0.5)
        
        # cnov7
        self.conv7=nn.Conv2d(in_channels=4096,out_channels=4096,kernel_size=1,stride=1,padding=1)
        self.bn7=nn.BatchNorm2d(num_features=4096)
        self.relu7=nn.ReLU(inplace=True)
        self.drop7=nn.Dropout2d(p=0.5)

        # cnov8，本项目有20个类别，一个背景，一共21类
        self.conv8=nn.Conv2d(in_channels=4096,out_channels=n_class,kernel_size=1,stride=1,padding=1)
        
        # 上采样2倍（16，16，21）————>（32，32，21）
        self.up_conv8_2=nn.ConvTranspose2d(in_channels=n_class,out_channels=n_class,kernel_size=2,stride=2,bias=False)
        
        # 反卷积ConvTranspose2d操作输出宽高公式
        # output=((input-1)*stride)+outputpadding-(2*padding)+kernelsize
        # 34=(16-1)*2+0-(2*0)+4
        
        # 第4层maxpool值做卷积运算
        self.pool4_conv=nn.Conv2d(in_channels=512,out_channels=n_class,kernel_size=1,stride=1)

        # 利用反卷积上采样2倍
        self.up_pool4_2=nn.ConvTranspose2d(in_channels=n_class,out_channels=n_class,kernel_size=2,stride=2,bias=False)
    
        # 第3层maxpool值做卷积运算
        self.pool3_conv=nn.Conv2d(in_channels=256,out_channels=n_class,kernel_size=1,stride=1)

        # 利用反卷积上采样8倍
        self.up_pool3_8=nn.ConvTranspose2d(in_channels=n_class,out_channels=n_class,kernel_size=8,stride=8,bias=False)


    def forward(self,x):
        """正向传播"""
        
        # 记录初始图片的大小（32，21，512，512）
        h=x

        # conv1
        x=self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x=self.relu1_2(self.bn1_2(self.conv1_2(x)))
        x=self.maxpool1(x)
        
        # conv2
        x=self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x=self.relu2_2(self.bn2_2(self.conv2_2(x)))
        x=self.maxpool2(x)

        # conv3
        x=self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x=self.relu3_2(self.bn3_2(self.conv3_2(x)))
        x=self.relu3_3(self.bn3_3(self.conv3_3(x)))
        x=self.maxpool3(x)
        pool3=x
        
        # conv4
        x=self.relu4_1(self.bn4_1(self.conv4_1(x)))
        x=self.relu4_2(self.bn4_2(self.conv4_2(x)))
        x=self.relu4_3(self.bn4_3(self.conv4_3(x)))
        x=self.maxpool4(x)
        pool4=x

        # conv5
        x=self.relu5_1(self.bn5_1(self.conv5_1(x)))
        x=self.relu5_2(self.bn5_2(self.conv5_2(x)))
        x=self.relu5_3(self.bn5_3(self.conv5_3(x)))
        x=self.maxpool5(x)
        
        # conv6
#         print(self.conv6(x).shape)
#         print(self.bn6(self.conv6(x)).shape)
#         print(self.relu6(self.bn6(self.conv6(x))).shape)
#         print(self.drop6(self.relu6(self.bn6(self.conv6(x)))).shape)
        x=self.drop6(self.relu6(self.bn6(self.conv6(x))))
        
        # conv7
        x=self.drop7(self.relu7(self.bn7(self.conv7(x))))
        
        # conv8
        x=self.up_conv8_2(self.conv8(x))
        up_conv8=x 
        
        # 计算第4层的值
        x2=self.pool4_conv(pool4)
        # 相加融合
        x2=up_conv8+x2
        # 反卷积上采样8倍
        x2=self.up_pool4_2(x2)
        up_pool4=x2
        
        # 计算第3层的值
        x3=self.pool3_conv(pool3)
        x3=up_pool4+x3
        
        # 反卷积上采样8倍
        x3=self.up_pool3_8(x3)
        return x3
    
model=FCN8s()
# print(model)


# **确定语义分割的评判标准**
# 1. 准确率Acc 对应像素准确率PA
# 1. 精准率Precision 对应类别像素准确率CPA
# 1. 交并比IOU
# 
# 参考链接：https://blog.csdn.net/weixin_38353277/article/details/121029978?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.pc_relevant_default&utm_relevant_index=1

# In[ ]:


"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""
class SegmentationMetric():
    """语义分割评判标准"""
    
    def __init__(self,numClass):
        """初始化"""
        # 分类个数
        self.numClass=numClass
        # 混淆矩阵
        self.confusionMatrix=np.zeros((self.numClass,self.numClass))
        
    def addBatch(self,imgPredict,imgLabel):
        """添加一个batch_size数据"""
        # 判断预测值和真实值大小是否一致，不一致直接抛出异常
#         print("imgPredict",imgPredict.shape,"imgLabel",imgLabel.shape)
        assert imgPredict.shape==imgLabel.shape
        self.confusionMatrix+=self.getConfusionMatrix(imgPredict,imgLabel)
        return self.confusionMatrix
    
    def getConfusionMatrix(self,imgPredict,imgLabel):
        """获取混淆矩阵"""
        # 筛选>=0,<类别数的标签
        mask=(imgLabel>=0)&(imgLabel<self.numClass)
        label=self.numClass*imgLabel[mask]+imgPredict[mask]
        count=np.bincount(label,minlength=self.numClass**2)
        # 调整形状
        confusionMatrix=count.reshape(self.numClass,self.numClass)
        return confusionMatrix
    
    def pixelAccuracy(self):
        """像素准确率，对应分类混淆矩阵中的准确率Accuracy"""
        # PA=(TP+TN)/(TP+TN+FP+FN)
        # 对角线相加之和/像素点之和
        pa=np.diag(self.confusionMatrix).sum()/self.confusionMatrix.sum()
        return pa
    
    def classPixelAccuracy(self):
        """类别像素准确率，对应分类混淆矩阵精准率Precision"""
        # CPA=TP/(TP+FP)
        # 计算横向的比值
        cpa=np.diag(self.confusionMatrix)/self.confusionMatrix.sum(axis=1)
        return cpa
    
    def meanPixelAccuracy(self):
        """平均类别像素准确率"""
        cpa=self.classPixelAccuracy()
        # 求平均值，遇到nan的填充0
        mpa=np.nanmean(cpa)
        return mpa
    
    def intersectionOverUnion(self):
        """计算交并比IOU"""
        # IOU=TP/((TP+FP)+(TP+FN)-TP)
        # 对角线的值是预测正确的值，作为交集
        intersection=np.diag(self.confusionMatrix)
        # 预测值+真实值-预测正确的值，作为并集
        union=np.sum(self.confusionMatrix,axis=1)+np.sum(self.confusionMatrix,axis=0)-intersection
        iou=intersection/union
        return iou
    
    def meanIntersectionOverUnion(self):
        """计算交并比的平均值"""
        iou=self.intersectionOverUnion()
        miou=np.nanmean(iou)
        return miou
        


# **模型训练**

# In[ ]:


# 配置训练参数

# 选择设置，优先GPU
device=torch.device("cuda" if torch.cuda.is_available else "cpu")
# 训练次数
epochs=100
# 损失函数，交叉熵
lossfunciton=torch.nn.CrossEntropyLoss()
# 优化方法
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

# 将模型赋值到GPU中
model=model.to(device)

for epoch in range(epochs):
    loss_add=0
    pa_add=0
    mpa_add=0
    miou_add=0
    for i,(image,label) in enumerate(train_loader):
        
        # 将数据复制到GPU中
        image=image.to(device)
        label=label.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 正向传播
        output=model(image)
        # 计算损失
        loss=lossfunciton(output,label)
        # 反向传播
        loss.backward()
        # 更新梯度
        optimizer.step()
        
        # 获取当前损失
        loss_add+=loss.data.item()
        
        # 获取评判标准
        label_pred = output.max(dim=1)[1].data.cpu().numpy()
        label_true = label.data.cpu().numpy()
        metric=SegmentationMetric(21)
        metric.addBatch(label_pred,label_true)
        pa_add+=metric.pixelAccuracy()
        mpa_add+=metric.meanPixelAccuracy()
        miou_add+=metric.meanIntersectionOverUnion()
#         print(loss_add,pa_add,cpa_add,iou_add)
        
    # 计算整体损失和评判标准
    epoch_loss=loss_add/len(train_loader)
    epoch_pa=pa_add/len(train_loader)
    epoch_mpa=mpa_add/len(train_loader)
    epoch_miou=miou_add/len(train_loader)
    
    print("epochs",epoch,"loss",epoch_loss,"pa",epoch_pa,"mpa",epoch_mpa,"miou",epoch_miou)
    
# 保存模型
torch.save(model.state_dict(),"./fcn8s.pt")


# **模型预测**

# In[ ]:


# 加载模型
# model=torch.load("./fcn8s.pt")
# 进入评估模式
model.eval()

cm = np.array(colormap).astype('uint8')

# 测试集梯度不更新
with torch.no_grad():
    for image,label in test_loader:
    
        # 将数据复制到GPU中
        image=image.to(device)
        label=label.to(device)
        
        print("image",image.shape,"label",label.shape)

        # 正向传播
        output=model(image)
        
        # 把数据从GPU复制到CPU中，plt才能调用
        output=output.max(1)[1].squeeze().cpu().data.numpy()
        pred = cm[output]
        plt.imshow(pred[0])
#         output=output.cpu().numpy()
#         print(output.shape)
        
#         for i,eval_image in enumerate(output):
#             plt.imshow(eval_image)
            
        break
    

