#!/usr/bin/env python
# coding: utf-8

# # Please, upvote [the notebook](https://www.kaggle.com/datasets/ilyaryabov/dataframe-for-educating-ml-model-in-google-ch-2022) if you use it

# ## The Pytorch model will correct the output of [GSDC2 - baseline submission](https://www.kaggle.com/code/saitodevel01/gsdc2-baseline-submission) model using additional information from device_gnss.csv

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


df = pd.read_csv('../input/dataframe-for-educating-ml-model-in-google-ch-2022/DataFrame_for_ML_model.csv')


# In[ ]:


df.sample(10)


# ## Short description of DataFrame:
# * **'lat' and 'lon' columns are the ground thruth**
# * **'LatitudeDegrees' and 'LongitudeDegrees' are predicted values**
# * **all the rest infomation is from device_gnss.csv**

# In[ ]:


df = df.fillna(0)


# In[ ]:


corr_table = abs(df.corr()[['lat','lon']]).sort_values(by = 'lon', ascending=False)
print(corr_table.T.columns[:10])
corr_table.head(10).drop(['lat', 'lon', 'UnixTimeMillis'])


# In[ ]:


learning_columns = ['LongitudeDegrees', 'WlsPositionYEcefMeters',
                       'WlsPositionXEcefMeters', 'LatitudeDegrees',
                       'WlsPositionZEcefMeters', 'FullBiasNanos',
                       'ReceivedSvTimeNanosSinceGpsEpoch'] # These 7 columns influence the ground truth latitude and longitude the most

N = len(learning_columns)

X_train = df[learning_columns].fillna(0).values
y_train = df[['lat','lon']].values


# In[ ]:


sc = MinMaxScaler()
sct = MinMaxScaler()

X_train=sc.fit_transform(X_train.reshape(-1,N))
y_train =sct.fit_transform(y_train.reshape(-1,2))


# In[ ]:


X_train = torch.from_numpy(X_train.astype(np.float32)).view(-1,N)
y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1,2)


# In[ ]:


y_train.shape


# In[ ]:


input_size = N
output_size = 2


# In[ ]:


class CorrectionModel(torch.nn.Module):

    def __init__(self):
        super(CorrectionModel, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, 50)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(50, output_size)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    
model = CorrectionModel()


# In[ ]:


learning_rate = 0.002
l = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


# In[ ]:


num_epochs = 1000
losses = []

for i in tqdm(range(num_epochs)):
    #forward feed
    y_pred = model(X_train.requires_grad_())

    #calculate the loss
    loss = l(y_pred, y_train)
    
    #backward propagation: calculate gradients
    loss.backward()

    #update the weights
    optimizer.step()

    #clear out the gradients from the last step loss.backward()
    optimizer.zero_grad()
    
    losses.append(loss.item())
    
    if divmod(i, 100)[1] == 0:
        print('epoch {}, loss {}'.format(i, loss.item()))
        
#print('epoch {}, loss {}'.format(epoch, loss.item()))


# In[ ]:


plt.figure(figsize = (16,8))
plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('MSE over epoches')
plt.show()


# In[ ]:




