#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import torchvision
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
import joblib


# In[ ]:


features = pd.read_csv('../input/digit-recognizer/train.csv').drop('label',axis=1)
features.head(5)

scaler = StandardScaler()
df_x = torch.tensor(scaler.fit_transform(features))
joblib.dump(scaler,"scaler.pkl")


# In[ ]:


df_y = torch.tensor(pd.read_csv('../input/digit-recognizer/train.csv')['label'].values)
train_tensor = torch.utils.data.TensorDataset(df_x,df_y)
trainset = torch.utils.data.DataLoader(train_tensor,shuffle=True,batch_size=20)


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,64,dtype=torch.float64)
        self.fc2 = nn.Linear(64,64,dtype=torch.float64)
        self.fc3 = nn.Linear(64,64,dtype=torch.float64)
        self.fc4 = nn.Linear(64,10,dtype=torch.float64)

    def Forward_Prop(self,X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = F.log_softmax(self.fc4(X),dim=1)
        return X


# In[ ]:


net = Net()
optimizer = optim.Adam(net.parameters(),lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        X,Y = data
        # print(X.view((-1,28*28)))
        # print(Y)
        # break
        net.zero_grad()
        output = net.Forward_Prop(X.view((-1,28*28)))
        loss = F.nll_loss(output,Y)
        loss.backward()
        optimizer.step()
    print(loss)


# In[ ]:


scaler = joblib.load('scaler.pkl')
testset = scaler.transform(pd.read_csv('../input/digit-recognizer/test.csv'))
testset = torch.tensor(testset)


# In[ ]:


i = 78
import matplotlib.pyplot as plt
plt.imshow(testset[i].view(28,28))
plt.show()

print("The given Image is number : {}".format(torch.argmax(net.Forward_Prop(testset[i].view(-1,28*28))).item()))

