#!/usr/bin/env python
# coding: utf-8

#  # Gold/Silver and Yamana comparison

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# # Data preparation

# In[ ]:


dataG0=pd.read_csv('../input/daily-gold-price-historical-data/gold.csv')
dataG=dataG0[['Date','Close']]
display(dataG)


# In[ ]:


dataS0=pd.read_csv('../input/daily-silver-price-historical-data/silver.csv')
dataS=dataS0[['Date','Close']]
display(dataS)


# In[ ]:


get_ipython().system('pip install openpyxl')
import openpyxl


# In[ ]:


dataY0=pd.read_excel('../input/yamana-gold-inc-stock-price/Yamana_Gold_Inc._AUY.csv.xlsx')
dataY=dataY0[['Date','Close']]
dataY['Date']=dataY['Date'].astype(str)
display(dataY)


# In[ ]:


data=pd.merge(dataG,dataS,on='Date',how='left')
data=pd.merge(data,dataY,on='Date',how='left')
data.columns=['Date','Gold_Close','Silver_Close','Yamana_Close']
data=data.dropna()[-850:]
data=data.reset_index(drop=True)
display(data)


# In[ ]:


data['Gold_Close MA20'] = data['Gold_Close'].rolling(window=20).mean()
data['Gold_Close MA20 shift year']=data['Gold_Close MA20'].shift(252)
data['Gold Yearly Growth']=(data['Gold_Close MA20']-data['Gold_Close MA20 shift year'])*100/data['Gold_Close MA20 shift year']

data['Silver_Close MA20'] = data['Silver_Close'].rolling(window=20).mean()
data['Silver_Close MA20 shift year']=data['Silver_Close MA20'].shift(252)
data['Silver Yearly Growth']=(data['Silver_Close MA20']-data['Silver_Close MA20 shift year'])*100/data['Silver_Close MA20 shift year']

data['Yamana_Close MA20'] = data['Yamana_Close'].rolling(window=20).mean()
data['Yamana_Close MA20 shift year']=data['Yamana_Close MA20'].shift(252)
data['Yamana Yearly Growth']=(data['Yamana_Close MA20']-data['Yamana_Close MA20 shift year'])*100/data['Yamana_Close MA20 shift year']


# In[ ]:


fig=make_subplots(specs=[[{"secondary_y":True}]])
fig.add_trace(go.Scatter(x=data['Date'],y=data['Gold_Close MA20'],name='Gold'),secondary_y=False)
fig.add_trace(go.Scatter(x=data['Date'],y=data['Silver_Close MA20'],name='Silver'),secondary_y=True,)
fig.add_trace(go.Scatter(x=data['Date'],y=data['Yamana_Close MA20'],name='Yamana'),secondary_y=True,)
fig.update_layout(autosize=False,width=800,height=500,title_text='MA20 Close Change')
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text='Gold USD',secondary_y=False)
fig.update_yaxes(title_text='Silver, Yamana USD',secondary_y=True)
fig.show()


# In[ ]:


fig=make_subplots(specs=[[{"secondary_y":False}]])
fig.add_trace(go.Scatter(x=data['Date'][252:],y=data['Gold Yearly Growth'][252:],name='Gold'),secondary_y=False,)
fig.add_trace(go.Scatter(x=data['Date'][252:],y=data['Silver Yearly Growth'][252:],name='Silver'),secondary_y=False,)
fig.add_trace(go.Scatter(x=data['Date'][252:],y=data['Yamana Yearly Growth'][252:],name='Yamana'),secondary_y=False,)

fig.update_layout(autosize=False,width=800,height=500,title_text='Yearly Growth %')
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Yearly Growth %",secondary_y=False)
fig.show()


# In[ ]:


import math
from sklearn.linear_model import LinearRegression
mod = LinearRegression()

def modscore(x,y):
    R2=mod.score(x,y)
    return R2


# ### Gold vs Silver

# In[ ]:


df_x=pd.DataFrame(data['Gold_Close'])
df_y=pd.DataFrame(data['Silver_Close'])
df_z=pd.DataFrame(data['Yamana_Close'])
mod_lin = mod.fit(df_x, df_y)
y_lin_fit = mod_lin.predict(df_x)
r2_lin = mod.score(df_x, df_y)
score=modscore(df_x,df_y)
print('R2='+str(round(score,4)))


# In[ ]:


fig,ax = plt.subplots(figsize=(6,6))
ax.set_title('Gold vs Silver',fontsize=20)
ax.set_xlabel('Gold',fontsize=12)
ax.set_ylabel('Silver',fontsize=12)
ax.scatter(df_x,df_y)
plt.plot(df_x, y_lin_fit, color='black', linewidth=0.5)
plt.show()
print('R2='+str(round(score,4)))


# ### Gold vs Yamana

# In[ ]:


mod_lin = mod.fit(df_x, df_z)
y_lin_fit = mod_lin.predict(df_x)
r2_lin = mod.score(df_x, df_z)
score=modscore(df_x,df_z)
print('R2='+str(round(score,4)))


# In[ ]:


fig,ax = plt.subplots(figsize=(6,6))
ax.set_title('Gold vs Yamana',fontsize=20)
ax.set_xlabel('Gold',fontsize=12)
ax.set_ylabel('Yamana',fontsize=12)
ax.scatter(df_x,df_z)
plt.plot(df_x, y_lin_fit, color='black', linewidth=0.5)
plt.show()
print('R2='+str(round(score,4)))


# ### Silver vs Yamana

# In[ ]:


mod_lin = mod.fit(df_y, df_z)
y_lin_fit = mod_lin.predict(df_y)
r2_lin = mod.score(df_y, df_z)
score=modscore(df_y,df_z)
print('R2='+str(round(score,4)))


# In[ ]:


fig,ax = plt.subplots(figsize=(6,6))
ax.set_title('Silver vs Yamana',fontsize=20)
ax.set_xlabel('Silver',fontsize=12)
ax.set_ylabel('Yamana',fontsize=12)
ax.scatter(df_y,df_z)
plt.plot(df_y, y_lin_fit, color='black', linewidth=0.5)
plt.show()
print('R2='+str(round(score,4)))


# In[ ]:




