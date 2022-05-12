#!/usr/bin/env python
# coding: utf-8

# ## Global parameters
# 

# In[ ]:


years = [2016,2017,2018]
zone = 'NW'

# How many stations we take to predict temperature in the area between them
around = 10 

# The initial station from which we start
STATION = 14066001

mindist = 10


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk(
    '/kaggle/input/meteonet/'+zone+\
    '_Ground_Stations/'+zone+\
    '_Ground_Stations/'+zone+\
    '_Ground_Stations_'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Getting and pre-processing data from kaggle datasets
# ## Step one

# In[ ]:


fname = '/kaggle/input/meteonet/'+zone+    '_Ground_Stations/'+zone+    '_Ground_Stations/'+zone+    '_Ground_Stations_'+str(2016)+'.csv'
df = pd.read_csv(fname)

# this dataframe we need to get closest stations from STATION
new_df = df.drop(['height_sta','date','dd','ff','precip','hu','td','t','psl'],axis=1)    .drop_duplicates('number_sta').reset_index()
    
lat = new_df.loc[0].lat
lon = new_df.loc[0].lon
    
    
# Arrays of closest stations
neighbours= np.zeros(around)
neighbours[0] = STATION
    
for station in range(1,around):
    for i in range(0,new_df.shape[0]):
        if new_df.loc[i]['number_sta'] not in neighbours:
            currdist = np.abs(lat - new_df.loc[i].lat) + np.abs(lon-new_df.loc[i].lon)
            if(mindist > currdist):
                mindist = currdist
                index = i
    neighbours[station]=new_df.loc[index]['number_sta']
    mindist=10
    
del new_df

del df


# In[ ]:


weather = pd.DataFrame()
for year in years:
    fname = '/kaggle/input/meteonet/'+zone+    '_Ground_Stations/'+zone+    '_Ground_Stations/'+zone+    '_Ground_Stations_'+str(year)+'.csv'
    df = pd.read_csv(fname)
    
    # Getting source data from stations in neighbours
    for i in range(0,around):
        dataframe = df[(df['number_sta'] == int(neighbours[i]))]
        weather = weather.append(dataframe,ignore_index = True)
    
    del df
    
    del dataframe


# In[ ]:


weather.head(10)


# ## Filling na by bfill groupped data by number_sta

# In[ ]:


weather.date = pd.to_datetime(weather.date)
weather = weather.sort_values('number_sta').reset_index().drop('index',axis=1)
weather = weather.fillna(method = 'bfill')


# In[ ]:


weather = weather.sort_values('date').reset_index().drop('index',axis=1)
print(weather.shape)


# In[ ]:


weather.dtypes


# ## taking part from dataset

# In[ ]:


dataset = weather[weather['date']<'2019-01-01']
print(dataset.shape)
print(dataset.head())


# Перенос координат в координаты нужной нам станции

# In[ ]:


lat = dataset[dataset['number_sta']==STATION]['lat'].unique()[0]
lon = dataset[dataset['number_sta']==STATION]['lon'].unique()[0]

dataset['lat'] = dataset['lat']-lat
dataset['lon'] = dataset['lon']-lon


# In[ ]:


dataset.head()


# ## Split date to year, yday, hour, minute

# In[ ]:


dataset['year'] = dataset.date.dt.year
dataset['yday'] = dataset.date.dt.dayofyear

dataset['hour'] = dataset.date.dt.hour
dataset['minute'] = dataset.date.dt.minute


# In[ ]:


dataset.head()


# ## Filling nan by bfill groupped data by date

# In[ ]:


dataset = dataset.set_index('date')


# Можно заполнить значения станции через среднее значений следущего и предыдущего временного этапа этой же станции.. А если и они nan, то по количеству и качеству каждой станции в округе( чем дальше она тем несущественнее ее результат сказывается)

# In[ ]:


# def fill_na_by_mean(dataset,param):
#     #print(dataset['number_sta'].unique())
#     stations = dataset['number_sta'].unique()
#     for station in stations:
#         for index in range(len(dataset[dataset['number_sta']==station])-5):
#             if np.isnan(dataset[dataset['number_sta']==station].iloc[index][param]):
#                 i=1
#                 numb=0
#                 while np.isnan(dataset[dataset['number_sta']==station].iloc[index-i][param]) and numb<5:
#                     i=+1
#                     numb+=1
#                 if numb<5:
#                     first = dataset[dataset['number_sta']==station].iloc[index-i][param]
#                 else:
#                     break
                
#                 i=1
#                 numb=0
#                 while np.isnan(dataset[dataset['number_sta']==station].iloc[index+i][param]) and numb<5:
#                     i=+1
#                     numb+=1
#                 if numb<5:
#                     second = dataset[dataset['number_sta']==station].iloc[index-i][param]
#                 else:
#                     break
        
#                 dataset[dataset['number_sta']==station].loc[index][param] = (first+second)/2
#     return dataset  

# fill_na_by_mean(dataset,'t')


# In[ ]:


dataset = dataset.fillna(method = 'bfill')


# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.shape


# ## Normalizating data...

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))


# In[ ]:


dataset['dd'] = scaler.fit_transform(dataset['dd'].values.reshape(-1,1))
dataset['ff'] = scaler.fit_transform(dataset['ff'].values.reshape(-1,1))
dataset['precip'] = scaler.fit_transform(dataset['precip'].values.reshape(-1,1))
dataset['hu'] = scaler.fit_transform(dataset['hu'].values.reshape(-1,1))
dataset['td'] = scaler.fit_transform(dataset['td'].values.reshape(-1,1))
dataset['psl'] = scaler.fit_transform(dataset['psl'].values.reshape(-1,1))
dataset['t'] = scaler.fit_transform(dataset['t'].values.reshape(-1,1))


# In[ ]:


dataset.head()


# In[ ]:


dataset.shape


# In[ ]:


import seaborn as sns


# In[ ]:


dataset.reset_index().drop(['date','year','minute','number_sta'],axis=1).corr()


# In[ ]:


sns.heatmap(data = dataset.reset_index().drop(['date','year','minute','number_sta'],axis = 1).corr())


# ## Подготавливаем датасет к обучению
# 
# ## Количество X может быть не равно y; Надо проверять есть ли за этот период запись в 'y' и только тогда добавлять X, чтобы их количество было равное

# In[ ]:


def make_tensors(table,predicted_par):
    
    data = []
    target = []

    periods = table.index.unique().astype('str')
    
    label = table[table.number_sta==STATION][predicted_par]
    target = label.values.tolist()
    
    #new_table = table.loc[table['number_sta'] != STATION].drop(['td','lat','lon','height_sta','precip','dd','ff','hu','psl','yday','year','hour','minute','number_sta'],axis=1)
    new_table = table.loc[table['number_sta'] != STATION].drop(['lat','lon','height_sta','year','yday','hour','minute','number_sta'],axis=1)
    
    for period in periods:
        if period in label:
            data.append(new_table[period:period].values.tolist())

    return data,target


# In[ ]:


X,y = make_tensors(dataset,'t')


# In[ ]:


len(X)


# In[ ]:


len(X[0][0])


# In[ ]:


len(y)


# ## Делаем Х_ полным для 9 станций по 5 зависимым столбцам каждый
# ## а остальные не берем в датасет

# In[ ]:


columns = len(X[0][0])


# In[ ]:


X_ = np.empty((0,(around-1)*columns),dtype='f')
index = []
for i in range(len(X)):
    if len([a for b in X[i] for a in b]) == (around-1)*columns and i!=len(X)-1:
        X_= np.append(X_,[[a for b in X[i] for a in b]],axis=0)
    else:
        index.append(i)

index.pop(len(index)-1)

y=np.delete(y,index,None)
y = np.delete(y,len(y)-1,None)


# In[ ]:


len(X_)


# In[ ]:


len(y)


# In[ ]:


y_ = [[] for k in range(len(y))]
for i in range(len(y)):
    y_[i] = [y[i]]
print(len(y_))
y = y_


# In[ ]:


del dataset


# In[ ]:


del X


# ## Разделение датасета на тестовую и тренировочную части

# In[ ]:


X_train = X_[0:int(len(X_)*0.9)]
X_test = X_[int(len(X_)*0.9):]

y_train = y[0:int(len(y)*0.9)]
y_test = y[int(len(y)*0.9):]


# In[ ]:


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2, random_state=39)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


len(y_train)


# In[ ]:


len(y_test)


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


X_train = torch.from_numpy(X_train).type(torch.Tensor)
X_test = torch.from_numpy(X_test).type(torch.Tensor)


# In[ ]:


y_train = torch.from_numpy(np.array(y_train)).type(torch.Tensor)
y_test = torch.from_numpy(np.array(y_test)).type(torch.Tensor)


# ## Само обучение**

# In[ ]:


class MLP(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super().__init__() 
 
        self.input_fc = nn.Linear(input_dim, 40) 
        self.hidden_fc = nn.Linear(40, 20) 
        self.output_fc = nn.Linear(20, output_dim) 
 
    def forward(self, x): 
 
        # x = [batch size, height, width] 
 
        batch_size = x.shape[0] 
 
        x = x.view(batch_size, -1) 
 
        # x = [batch size, height * width] 
 
        h_1 = torch.tanh(self.input_fc(x)) 
 
        # h_1 = [batch size, 5] 
 
        h_2 = torch.tanh(self.hidden_fc(h_1)) 
 
        # h_2 = [batch size, 3] 
 
        y_pred = self.output_fc(h_2) 
 
        # y_pred = [batch size, output dim] 
 
        return y_pred


# In[ ]:


model = MLP(input_dim=X_train[0].shape[0], output_dim=len(y_train[0]))
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.03)


# In[ ]:


num_epochs = 100


# In[ ]:


import time
hist = np.zeros(num_epochs)
start_time = time.time()

for epoch in range(num_epochs):
    y_train_pred = model(X_train)
    #print(y_train_pred)
    loss = criterion(y_train_pred, y_train)
    print("Epoch ", epoch, "MSE: ", loss.item(), " Time: ",time.time()-start_time)
    hist[epoch] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

ax = sns.lineplot(data=hist, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Training Loss", size = 14, fontweight='bold')
fig.set_figheight(6)
fig.set_figwidth(16)


# In[ ]:


y_test_pred = model(X_test)


# In[ ]:


y_test_predict = scaler.inverse_transform(y_test_pred.detach().numpy().reshape(-1, 1))
y_test_target = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))


# In[ ]:


y_test_predict = pd.DataFrame(y_test_predict)


# In[ ]:


y_test_target = pd.DataFrame(y_test_target)


# In[ ]:


y_test_predict.nunique()


# In[ ]:


fig = plt.figure()
ax = sns.lineplot(x = y_test_target.index, y = y_test_target[0], label="Data", color='tomato')
ax = sns.lineplot(x = y_test_predict.index, y = y_test_predict[0], label="Prediction", color='royalblue')
ax.set_title('Temperature', size = 14, fontweight='bold')
ax.set_xlabel("Records", size = 14)
ax.set_ylabel("Kelvins", size = 14)
fig.set_figheight(6)
fig.set_figwidth(16)
#ax.set_xticklabels('', size=10)


# In[ ]:


import plotly.express as px
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=y_test_target.index, y=y_test_target[0],
                    mode='lines',
                    name='Actual Value')))
fig.add_trace(go.Scatter(x=y_test_predict.index, y=y_test_predict[0],
                    mode='lines',
                    name='Test prediction'))

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Temperature (Kelvins)',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template = 'plotly_dark'

)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Results (MLP)',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()


# ## Metrics

# In[ ]:


y_train_predict = scaler.inverse_transform(y_train_pred.detach().numpy().reshape(-1, 1))
y_train_target = scaler.inverse_transform(y_train.detach().numpy().reshape(-1, 1))


# In[ ]:


import math
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


# In[ ]:


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train_target, y_train_predict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test_target, y_test_predict))
print('Test Score: %.2f RMSE' % (testScore))

trainScore = mean_absolute_percentage_error(y_train_target, y_train_predict)
print('Train Score: %.2f MAPE' % (trainScore))
testScore = mean_absolute_percentage_error(y_test_target, y_test_predict)
print('Test Score: %.2f MAPE' % (testScore))


# ## Вторая модель

# In[ ]:


class MLP(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super().__init__() 
 
        self.input_fc = nn.Linear(input_dim, 45) 
        self.hidden_fc = nn.Linear(45, 15) 
        self.output_fc = nn.Linear(15, output_dim) 
 
    def forward(self, x): 
 
        # x = [batch size, height, width] 
 
        batch_size = x.shape[0] 
 
        x = x.view(batch_size, -1) 
 
        # x = [batch size, height * width] 
 
        h_1 = torch.tanh(self.input_fc(x)) 
 
        # h_1 = [batch size, 5] 
 
        h_2 = torch.tanh(self.hidden_fc(h_1)) 
 
        # h_2 = [batch size, 3] 
 
        y_pred = self.output_fc(h_2) 
 
        # y_pred = [batch size, output dim] 
 
        return y_pred

model = MLP(input_dim=X_train[0].shape[0], output_dim=len(y_train[0]))
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.03)

num_epochs = 100


hist = np.zeros(num_epochs)
start_time = time.time()

for epoch in range(num_epochs):
    y_train_pred = model(X_train)
    #print(y_train_pred)
    loss = criterion(y_train_pred, y_train)
    print("Epoch ", epoch, "MSE: ", loss.item(), " Time: ",time.time()-start_time)
    hist[epoch] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))

y_test_pred = model(X_test)

y_test_predict = scaler.inverse_transform(y_test_pred.detach().numpy().reshape(-1, 1))
y_test_target = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))

y_test_predict = pd.DataFrame(y_test_predict)
y_test_target = pd.DataFrame(y_test_target)


fig = plt.figure()
ax = sns.lineplot(x = y_test_target.index, y = y_test_target[0], label="Data", color='tomato')
ax = sns.lineplot(x = y_test_predict.index, y = y_test_predict[0], label="Prediction", color='royalblue')
ax.set_title('Temperature', size = 14, fontweight='bold')
ax.set_xlabel("Records", size = 14)
ax.set_ylabel("Kelvins", size = 14)
fig.set_figheight(6)
fig.set_figwidth(16)
#ax.set_xticklabels('', size=10)


fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=y_test_target.index, y=y_test_target[0],
                    mode='lines',
                    name='Actual Value')))
fig.add_trace(go.Scatter(x=y_test_predict.index, y=y_test_predict[0],
                    mode='lines',
                    name='Test prediction'))

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Temperature (Kelvins)',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template = 'plotly_dark'

)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Results (MLP)',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()


y_train_predict = scaler.inverse_transform(y_train_pred.detach().numpy().reshape(-1, 1))
y_train_target = scaler.inverse_transform(y_train.detach().numpy().reshape(-1, 1))

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train_target, y_train_predict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test_target, y_test_predict))
print('Test Score: %.2f RMSE' % (testScore))

trainScore = mean_absolute_percentage_error(y_train_target, y_train_predict)
print('Train Score: %.2f MAPE' % (trainScore))
testScore = mean_absolute_percentage_error(y_test_target, y_test_predict)
print('Test Score: %.2f MAPE' % (testScore))


# ##  Третья модель

# In[ ]:


class MLP(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super().__init__() 
 
        self.input_fc = nn.Linear(input_dim, 45) 
        self.hidden_fc1 = nn.Linear(45, 30) 
        self.hidden_fc2 = nn.Linear(30, 15) 
        self.output_fc = nn.Linear(15, output_dim) 
 
    def forward(self, x): 
 
        # x = [batch size, height, width] 
 
        batch_size = x.shape[0] 
 
        x = x.view(batch_size, -1) 
 
        # x = [batch size, height * width] 
 
        h_1 = torch.tanh(self.input_fc(x)) 
 
        # h_1 = [batch size, 5] 
 
        h_2 = torch.tanh(self.hidden_fc1(h_1)) 
    
        h_3 = torch.tanh(self.hidden_fc2(h_2)) 
 
        # h_2 = [batch size, 3] 
 
        y_pred = self.output_fc(h_3) 
 
        # y_pred = [batch size, output dim] 
 
        return y_pred

model = MLP(input_dim=X_train[0].shape[0], output_dim=len(y_train[0]))
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.03)

num_epochs = 100


hist = np.zeros(num_epochs)
start_time = time.time()

for epoch in range(num_epochs):
    y_train_pred = model(X_train)
    #print(y_train_pred)
    loss = criterion(y_train_pred, y_train)
    print("Epoch ", epoch, "MSE: ", loss.item(), " Time: ",time.time()-start_time)
    hist[epoch] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))

y_test_pred = model(X_test)

y_test_predict = scaler.inverse_transform(y_test_pred.detach().numpy().reshape(-1, 1))
y_test_target = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))

y_test_predict = pd.DataFrame(y_test_predict)
y_test_target = pd.DataFrame(y_test_target)


fig = plt.figure()
ax = sns.lineplot(x = y_test_target.index, y = y_test_target[0], label="Data", color='tomato')
ax = sns.lineplot(x = y_test_predict.index, y = y_test_predict[0], label="Prediction", color='royalblue')
ax.set_title('Temperature', size = 14, fontweight='bold')
ax.set_xlabel("Records", size = 14)
ax.set_ylabel("Kelvins", size = 14)
fig.set_figheight(6)
fig.set_figwidth(16)
#ax.set_xticklabels('', size=10)


fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=y_test_target.index, y=y_test_target[0],
                    mode='lines',
                    name='Actual Value')))
fig.add_trace(go.Scatter(x=y_test_predict.index, y=y_test_predict[0],
                    mode='lines',
                    name='Test prediction'))

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Temperature (Kelvins)',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template = 'plotly_dark'

)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Results (MLP)',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()


y_train_predict = scaler.inverse_transform(y_train_pred.detach().numpy().reshape(-1, 1))
y_train_target = scaler.inverse_transform(y_train.detach().numpy().reshape(-1, 1))

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train_target, y_train_predict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test_target, y_test_predict))
print('Test Score: %.2f RMSE' % (testScore))

trainScore = mean_absolute_percentage_error(y_train_target, y_train_predict)
print('Train Score: %.2f MAPE' % (trainScore))
testScore = mean_absolute_percentage_error(y_test_target, y_test_predict)
print('Test Score: %.2f MAPE' % (testScore))


# ## Четвертая модель

# In[ ]:


class MLP(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super().__init__() 
 
        self.input_fc = nn.Linear(input_dim, 50) 
        self.hidden_fc1 = nn.Linear(50, 30) 
        self.hidden_fc2 = nn.Linear(30, 20)
        self.hidden_fc3 = nn.Linear(20, 10)
        self.output_fc = nn.Linear(10, output_dim) 
 
    def forward(self, x): 
 
        # x = [batch size, height, width] 
 
        batch_size = x.shape[0] 
 
        x = x.view(batch_size, -1) 
 
        # x = [batch size, height * width] 
 
        h_1 = torch.tanh(self.input_fc(x)) 
 
        # h_1 = [batch size, 5] 
 
        h_2 = torch.tanh(self.hidden_fc1(h_1)) 
    
        h_3 = torch.tanh(self.hidden_fc2(h_2)) 
        
        h_4 = torch.tanh(self.hidden_fc3(h_3))
 
        # h_2 = [batch size, 3] 
 
        y_pred = self.output_fc(h_4) 
 
        # y_pred = [batch size, output dim] 
 
        return y_pred

model = MLP(input_dim=X_train[0].shape[0], output_dim=len(y_train[0]))
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.03)

num_epochs = 100


hist = np.zeros(num_epochs)
start_time = time.time()

for epoch in range(num_epochs):
    y_train_pred = model(X_train)
    #print(y_train_pred)
    loss = criterion(y_train_pred, y_train)
    print("Epoch ", epoch, "MSE: ", loss.item(), " Time: ",time.time()-start_time)
    hist[epoch] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))

y_test_pred = model(X_test)

y_test_predict = scaler.inverse_transform(y_test_pred.detach().numpy().reshape(-1, 1))
y_test_target = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))

y_test_predict = pd.DataFrame(y_test_predict)
y_test_target = pd.DataFrame(y_test_target)


fig = plt.figure()
ax = sns.lineplot(x = y_test_target.index, y = y_test_target[0], label="Data", color='tomato')
ax = sns.lineplot(x = y_test_predict.index, y = y_test_predict[0], label="Prediction", color='royalblue')
ax.set_title('Temperature', size = 14, fontweight='bold')
ax.set_xlabel("Records", size = 14)
ax.set_ylabel("Kelvins", size = 14)
fig.set_figheight(6)
fig.set_figwidth(16)
#ax.set_xticklabels('', size=10)


fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=y_test_target.index, y=y_test_target[0],
                    mode='lines',
                    name='Actual Value')))
fig.add_trace(go.Scatter(x=y_test_predict.index, y=y_test_predict[0],
                    mode='lines',
                    name='Test prediction'))

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Temperature (Kelvins)',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template = 'plotly_dark'

)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Results (MLP)',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()


y_train_predict = scaler.inverse_transform(y_train_pred.detach().numpy().reshape(-1, 1))
y_train_target = scaler.inverse_transform(y_train.detach().numpy().reshape(-1, 1))

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train_target, y_train_predict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test_target, y_test_predict))
print('Test Score: %.2f RMSE' % (testScore))

trainScore = mean_absolute_percentage_error(y_train_target, y_train_predict)
print('Train Score: %.2f MAPE' % (trainScore))
testScore = mean_absolute_percentage_error(y_test_target, y_test_predict)
print('Test Score: %.2f MAPE' % (testScore))


# ## Пятая модель

# In[ ]:


class MLP(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super().__init__() 
 
        self.input_fc = nn.Linear(input_dim, 48) 
        self.hidden_fc1 = nn.Linear(48, 37) 
        self.hidden_fc2 = nn.Linear(37, 26)
        self.hidden_fc3 = nn.Linear(26, 19)
        
        self.hidden_fc4 = nn.Linear(19, 8)
        self.output_fc = nn.Linear(8, output_dim) 
 
    def forward(self, x): 
 
        # x = [batch size, height, width] 
 
        batch_size = x.shape[0] 
 
        x = x.view(batch_size, -1) 
 
        # x = [batch size, height * width] 
 
        h_1 = torch.tanh(self.input_fc(x)) 
 
        # h_1 = [batch size, 5] 
 
        h_2 = torch.tanh(self.hidden_fc1(h_1)) 
    
        h_3 = torch.tanh(self.hidden_fc2(h_2)) 
        
        h_4 = torch.tanh(self.hidden_fc3(h_3))
        
        h_5 = torch.tanh(self.hidden_fc4(h_4))
 
        # h_2 = [batch size, 3] 
 
        y_pred = self.output_fc(h_5) 
 
        # y_pred = [batch size, output dim] 
 
        return y_pred

model = MLP(input_dim=X_train[0].shape[0], output_dim=len(y_train[0]))
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.03)

num_epochs = 100


hist = np.zeros(num_epochs)
start_time = time.time()

for epoch in range(num_epochs):
    y_train_pred = model(X_train)
    #print(y_train_pred)
    loss = criterion(y_train_pred, y_train)
    print("Epoch ", epoch, "MSE: ", loss.item(), " Time: ",time.time()-start_time)
    hist[epoch] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))

y_test_pred = model(X_test)

y_test_predict = scaler.inverse_transform(y_test_pred.detach().numpy().reshape(-1, 1))
y_test_target = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))

y_test_predict = pd.DataFrame(y_test_predict)
y_test_target = pd.DataFrame(y_test_target)


fig = plt.figure()
ax = sns.lineplot(x = y_test_target.index, y = y_test_target[0], label="Data", color='tomato')
ax = sns.lineplot(x = y_test_predict.index, y = y_test_predict[0], label="Prediction", color='royalblue')
ax.set_title('Temperature', size = 14, fontweight='bold')
ax.set_xlabel("Records", size = 14)
ax.set_ylabel("Kelvins", size = 14)
fig.set_figheight(6)
fig.set_figwidth(16)
#ax.set_xticklabels('', size=10)


fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=y_test_target.index, y=y_test_target[0],
                    mode='lines',
                    name='Actual Value')))
fig.add_trace(go.Scatter(x=y_test_predict.index, y=y_test_predict[0],
                    mode='lines',
                    name='Test prediction'))

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Temperature (Kelvins)',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template = 'plotly_dark'

)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Results (MLP)',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()


y_train_predict = scaler.inverse_transform(y_train_pred.detach().numpy().reshape(-1, 1))
y_train_target = scaler.inverse_transform(y_train.detach().numpy().reshape(-1, 1))

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train_target, y_train_predict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test_target, y_test_predict))
print('Test Score: %.2f RMSE' % (testScore))

trainScore = mean_absolute_percentage_error(y_train_target, y_train_predict)
print('Train Score: %.2f MAPE' % (trainScore))
testScore = mean_absolute_percentage_error(y_test_target, y_test_predict)
print('Test Score: %.2f MAPE' % (testScore))


# ## Шестая модель

# In[ ]:


class MLP(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super().__init__() 
 
        self.input_fc = nn.Linear(input_dim, 40) 
        self.hidden_fc1 = nn.Linear(40, 30) 
        self.hidden_fc2 = nn.Linear(30, 15)
        self.output_fc = nn.Linear(15, output_dim) 
 
    def forward(self, x): 
 
        # x = [batch size, height, width] 
 
        batch_size = x.shape[0] 
 
        x = x.view(batch_size, -1) 
 
        # x = [batch size, height * width] 
 
        h_1 = torch.tanh(self.input_fc(x)) 
 
        # h_1 = [batch size, 5] 
 
        h_2 = torch.tanh(self.hidden_fc1(h_1)) 
    
        h_3 = torch.tanh(self.hidden_fc2(h_2)) 
        
        # h_2 = [batch size, 3] 
 
        y_pred = self.output_fc(h_3) 
 
        # y_pred = [batch size, output dim] 
 
        return y_pred

model = MLP(input_dim=X_train[0].shape[0], output_dim=len(y_train[0]))
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.03)

num_epochs = 100


hist = np.zeros(num_epochs)
start_time = time.time()

for epoch in range(num_epochs):
    y_train_pred = model(X_train)
    #print(y_train_pred)
    loss = criterion(y_train_pred, y_train)
    print("Epoch ", epoch, "MSE: ", loss.item(), " Time: ",time.time()-start_time)
    hist[epoch] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))

y_test_pred = model(X_test)

y_test_predict = scaler.inverse_transform(y_test_pred.detach().numpy().reshape(-1, 1))
y_test_target = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))

y_test_predict = pd.DataFrame(y_test_predict)
y_test_target = pd.DataFrame(y_test_target)


fig = plt.figure()
ax = sns.lineplot(x = y_test_target.index, y = y_test_target[0], label="Data", color='tomato')
ax = sns.lineplot(x = y_test_predict.index, y = y_test_predict[0], label="Prediction", color='royalblue')
ax.set_title('Temperature', size = 14, fontweight='bold')
ax.set_xlabel("Records", size = 14)
ax.set_ylabel("Kelvins", size = 14)
fig.set_figheight(6)
fig.set_figwidth(16)
#ax.set_xticklabels('', size=10)


fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=y_test_target.index, y=y_test_target[0],
                    mode='lines',
                    name='Actual Value')))
fig.add_trace(go.Scatter(x=y_test_predict.index, y=y_test_predict[0],
                    mode='lines',
                    name='Test prediction'))

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Temperature (Kelvins)',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template = 'plotly_dark'

)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Results (MLP)',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()


y_train_predict = scaler.inverse_transform(y_train_pred.detach().numpy().reshape(-1, 1))
y_train_target = scaler.inverse_transform(y_train.detach().numpy().reshape(-1, 1))

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train_target, y_train_predict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test_target, y_test_predict))
print('Test Score: %.2f RMSE' % (testScore))

trainScore = mean_absolute_percentage_error(y_train_target, y_train_predict)
print('Train Score: %.2f MAPE' % (trainScore))
testScore = mean_absolute_percentage_error(y_test_target, y_test_predict)
print('Test Score: %.2f MAPE' % (testScore))


# ## Седьмая модель
# 

# In[ ]:


class MLP(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super().__init__() 
 
        self.input_fc = nn.Linear(input_dim, 50) 
        self.hidden_fc1 = nn.Linear(50, 40) 
        self.hidden_fc2 = nn.Linear(40, 30)
        
        self.hidden_fc3 = nn.Linear(30, 20)
        
        self.hidden_fc4 = nn.Linear(20, 10)
        self.output_fc = nn.Linear(10, output_dim) 
 
    def forward(self, x): 
 
        # x = [batch size, height, width] 
 
        batch_size = x.shape[0] 
 
        x = x.view(batch_size, -1) 
 
        # x = [batch size, height * width] 
 
        h_1 = torch.tanh(self.input_fc(x)) 
 
        # h_1 = [batch size, 5] 
 
        h_2 = torch.tanh(self.hidden_fc1(h_1)) 
    
        h_3 = torch.tanh(self.hidden_fc2(h_2)) 
        
        h_4 = torch.tanh(self.hidden_fc3(h_3)) 
    
        h_5 = torch.tanh(self.hidden_fc4(h_4)) 
        
        # h_2 = [batch size, 3] 
 
        y_pred = self.output_fc(h_5) 
 
        # y_pred = [batch size, output dim] 
 
        return y_pred

model = MLP(input_dim=X_train[0].shape[0], output_dim=len(y_train[0]))
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.03)

num_epochs = 100


hist = np.zeros(num_epochs)
start_time = time.time()

for epoch in range(num_epochs):
    y_train_pred = model(X_train)
    #print(y_train_pred)
    loss = criterion(y_train_pred, y_train)
    print("Epoch ", epoch, "MSE: ", loss.item(), " Time: ",time.time()-start_time)
    hist[epoch] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))

y_test_pred = model(X_test)

y_test_predict = scaler.inverse_transform(y_test_pred.detach().numpy().reshape(-1, 1))
y_test_target = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))

y_test_predict = pd.DataFrame(y_test_predict)
y_test_target = pd.DataFrame(y_test_target)


fig = plt.figure()
ax = sns.lineplot(x = y_test_target.index, y = y_test_target[0], label="Data", color='tomato')
ax = sns.lineplot(x = y_test_predict.index, y = y_test_predict[0], label="Prediction", color='royalblue')
ax.set_title('Temperature', size = 14, fontweight='bold')
ax.set_xlabel("Records", size = 14)
ax.set_ylabel("Kelvins", size = 14)
fig.set_figheight(6)
fig.set_figwidth(16)
#ax.set_xticklabels('', size=10)


fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=y_test_target.index, y=y_test_target[0],
                    mode='lines',
                    name='Actual Value')))
fig.add_trace(go.Scatter(x=y_test_predict.index, y=y_test_predict[0],
                    mode='lines',
                    name='Test prediction'))

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Temperature (Kelvins)',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template = 'plotly_dark'

)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Results (MLP)',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()


y_train_predict = scaler.inverse_transform(y_train_pred.detach().numpy().reshape(-1, 1))
y_train_target = scaler.inverse_transform(y_train.detach().numpy().reshape(-1, 1))

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train_target, y_train_predict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test_target, y_test_predict))
print('Test Score: %.2f RMSE' % (testScore))

trainScore = mean_absolute_percentage_error(y_train_target, y_train_predict)
print('Train Score: %.2f MAPE' % (trainScore))
testScore = mean_absolute_percentage_error(y_test_target, y_test_predict)
print('Test Score: %.2f MAPE' % (testScore))


# ## Восьмая модель

# In[ ]:


class MLP(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super().__init__() 
 
        self.input_fc = nn.Linear(input_dim, 35) 
        self.hidden_fc1 = nn.Linear(35, 15) 

        self.output_fc = nn.Linear(15, output_dim) 
 
    def forward(self, x): 
 
        # x = [batch size, height, width] 
 
        batch_size = x.shape[0] 
 
        x = x.view(batch_size, -1) 
 
        # x = [batch size, height * width] 
 
        h_1 = torch.tanh(self.input_fc(x)) 
 
        # h_1 = [batch size, 5] 
 
        h_2 = torch.tanh(self.hidden_fc1(h_1)) 
    
        
        # h_2 = [batch size, 3] 
 
        y_pred = self.output_fc(h_2) 
 
        # y_pred = [batch size, output dim] 
 
        return y_pred

model = MLP(input_dim=X_train[0].shape[0], output_dim=len(y_train[0]))
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.03)

num_epochs = 100


hist = np.zeros(num_epochs)
start_time = time.time()

for epoch in range(num_epochs):
    y_train_pred = model(X_train)
    #print(y_train_pred)
    loss = criterion(y_train_pred, y_train)
    print("Epoch ", epoch, "MSE: ", loss.item(), " Time: ",time.time()-start_time)
    hist[epoch] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))

y_test_pred = model(X_test)

y_test_predict = scaler.inverse_transform(y_test_pred.detach().numpy().reshape(-1, 1))
y_test_target = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))

y_test_predict = pd.DataFrame(y_test_predict)
y_test_target = pd.DataFrame(y_test_target)


fig = plt.figure()
ax = sns.lineplot(x = y_test_target.index, y = y_test_target[0], label="Data", color='tomato')
ax = sns.lineplot(x = y_test_predict.index, y = y_test_predict[0], label="Prediction", color='royalblue')
ax.set_title('Temperature', size = 14, fontweight='bold')
ax.set_xlabel("Records", size = 14)
ax.set_ylabel("Kelvins", size = 14)
fig.set_figheight(6)
fig.set_figwidth(16)
#ax.set_xticklabels('', size=10)


fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=y_test_target.index, y=y_test_target[0],
                    mode='lines',
                    name='Actual Value')))
fig.add_trace(go.Scatter(x=y_test_predict.index, y=y_test_predict[0],
                    mode='lines',
                    name='Test prediction'))

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='white',
        linewidth=2
    ),
    yaxis=dict(
        title_text='Temperature (Kelvins)',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='white',
        ),
    ),
    showlegend=True,
    template = 'plotly_dark'

)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Results (MLP)',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()


y_train_predict = scaler.inverse_transform(y_train_pred.detach().numpy().reshape(-1, 1))
y_train_target = scaler.inverse_transform(y_train.detach().numpy().reshape(-1, 1))

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train_target, y_train_predict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test_target, y_test_predict))
print('Test Score: %.2f RMSE' % (testScore))

trainScore = mean_absolute_percentage_error(y_train_target, y_train_predict)
print('Train Score: %.2f MAPE' % (trainScore))
testScore = mean_absolute_percentage_error(y_test_target, y_test_predict)
print('Test Score: %.2f MAPE' % (testScore))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




