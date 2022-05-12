#!/usr/bin/env python
# coding: utf-8

# # Economic Index Similarity

# In[ ]:


import numpy as np
import pandas as pd
import os


# In[ ]:


data0=pd.read_csv('../input/yfinance-ep-data/combined_data.csv')
names0=data0.columns.tolist()
names1=[]
for item in names0:
    if '_Close' in item:
        names1+=[item]
print(names1)
print(len(names1))


# In[ ]:


data=data0[['Date']+names1]
data=data.dropna()
data=data[-120:]
data=data.reset_index(drop=True)
display(data)


# In[ ]:


import math
from sklearn.linear_model import LinearRegression
mod = LinearRegression()

def modscore(x,y):
    R2=mod.score(x,y)
    return R2


# In[ ]:


LIST1=pd.DataFrame(columns=['x','y','R2'],index=list(range(len(names1)**2)))


# In[ ]:


t=0
for x,itemX in enumerate(names1):
    for y,itemY in enumerate(names1):
        df_x=pd.DataFrame(data[itemX])
        df_y=pd.DataFrame(data[itemY])
        mod_lin = mod.fit(df_x, df_y)
        y_lin_fit = mod_lin.predict(df_x)
        r2_lin = mod.score(df_x, df_y)
        score=modscore(df_x,df_y)
        LIST1.loc[t,'x']=x
        LIST1.loc[t,'y']=y
        LIST1.loc[t,'R2']=round(score,3) 
        t+=1


# In[ ]:


LIST1['name1']=LIST1['x'].apply(lambda x:names1[x].split('_')[0])
LIST1['name2']=LIST1['y'].apply(lambda y:names1[y].split('_')[0])
display(LIST1)


# In[ ]:


LIST1[LIST1['name1']=='NIKKEI'].sort_values('R2',ascending=False)


# In[ ]:


LIST1[LIST1['name1']=='USDJPY'].sort_values('R2',ascending=False)


# In[ ]:


LIST1[LIST1['name1']=='SP500'].sort_values('R2',ascending=False)


# In[ ]:


LIST1[LIST1['name1']=='NASDAQ'].sort_values('R2',ascending=False)


# In[ ]:





# In[ ]:




