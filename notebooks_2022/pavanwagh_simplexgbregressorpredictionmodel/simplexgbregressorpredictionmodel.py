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


traindata = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
testdata = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


traindata.head()


# In[ ]:


lst=[]
for i in traindata.columns:
    if traindata[i].dtype == 'int64':
        lst.append(str(i))


# In[ ]:


def clean(data):
    #data=data.drop(["Id"],axis=1)
    for i in lst:
        data[i].fillna(data[i].median())
    data=data.drop(["Id"],axis=1)
    lst.remove("Id")
    return data
traindata=clean(traindata)


# In[ ]:


data=traindata[lst]


# In[ ]:


def cleanTest(data):
    #data=data.drop(["Id"],axis=1)
    lst.remove("SalePrice")
    for i in lst:
        data[i].fillna(data[i].median())
    #data=data.drop(["Id"],axis=1)
    #lst.remove("Id")
    return data
testdata=cleanTest(testdata)
testdata.head()


# In[ ]:


xtrain = data.drop(["SalePrice"],axis=1)
ytrain = data["SalePrice"]


# In[ ]:


import xgboost
clf = xgboost.XGBRegressor()
clf.fit(xtrain,ytrain)


# In[ ]:


tdata = testdata[lst]


# In[ ]:


pred = clf.predict(tdata)


# In[ ]:


testdata.head()


# In[ ]:


TD = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


TD.head()


# In[ ]:


predi = pd.DataFrame(pred)
dat = pd.concat([TD["Id"],predi],axis=1)


# In[ ]:


dat.to_csv("submission.csv")


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
df = pd.DataFrame(dat)
create_download_link(df)


# In[ ]:




