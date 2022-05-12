#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import torch.utils.data as Data
import torch


def dataloader(batch_size, shuffle=True):
    # from sklearn.datasets import load_digits
    # digits = load_digits()
    # X = digits.data
    # y = digits.target
    # Y = []
    # for i in y:
    #     Y.append([i])

    train_data = pd.read_csv('/kaggle/input/ml2021-2022-2-cnn/train.csv').values
    test_data = pd.read_csv('/kaggle/input/ml2021-2022-2-cnn/test.csv').values
    train_X = train_data[:, 1:]
    train_y = train_data[:, 0]
    train_X = torch.FloatTensor(train_X)
    train_y = torch.LongTensor(train_y)
    test_X = test_data
    test_y = test_data[:, 0]
    test_X = torch.FloatTensor(test_X)
    test_y = torch.LongTensor(test_y)
    len_train_data = len(train_X)
    len_test_data = len(test_X)
    train_dataset = []
    test_dataset = []
    for i in range(len(train_X)):
        train_dataset.append((train_X[i], train_y[i]))
    for i in range(len(test_X)):
        test_dataset.append((test_X[i], test_y[i]))
    train_iter = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_iter = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter, len_train_data, len_test_data


# In[ ]:


from torch import nn

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        if self.conv3:
            x = self.conv3(x)
        output = output + x
        output = self.relu(output)
        return output


# In[ ]:


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


# In[ ]:


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                      nn.BatchNorm2d(64), nn.ReLU(),
                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.block3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.block4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.block5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, 10))
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        output = self.head(x)
        return output


# In[ ]:


# 设置超参数
lr = 0.01
num_epochs = 10
batch_size = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("training on:{}".format(device))  # 显示训练设备

# 取数据
train_iter, test_iter, len_train_data, len_test_data = dataloader(batch_size)

# 实例化模型
model = ResNet()
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)


# In[ ]:


from tqdm import tqdm

# 开始训练
acc_list = [0.,]
for epoch in range(num_epochs):
    acc = 0
    with tqdm(train_iter, unit='batch') as tepoch:
        for data, label in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1} train")
            data = data.reshape(batch_size, 1, 28, 28)
            data = data.to(device)
            label = label.to(device)
            outputs = model(data)
            loss = loss_fn(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc += (outputs.argmax(1) == label).sum() / len_train_data
            tepoch.set_postfix(acc=acc.item())
    acc_list.append(acc.item())


# In[ ]:


import matplotlib.pyplot as plt

title_name = "Accuracy"
plt.plot(acc_list, label="ResNet")
plt.legend()
plt.xlabel("num_epochs")
plt.ylabel("Accuracy")
plt.title(title_name)
plt.show


# In[ ]:


#做预测输出
predictions = []
with tqdm(test_iter, unit='batch') as tepoch:
    for data, _ in tepoch:
        tepoch.set_description(f"Prediction")
        data = data.reshape(batch_size, 1, 28, 28)
        data = data.to(device)
        outputs = model(data)
        outputs_label = outputs.argmax(1).cpu().numpy()
        for label in outputs_label:
            predictions.append(label)


# In[ ]:


out_dict = {
    'id':list(np.arange(len_test_data)),
    'label':predictions
}
out = pd.DataFrame(out_dict)
out.to_csv('submission.csv',index=False)

