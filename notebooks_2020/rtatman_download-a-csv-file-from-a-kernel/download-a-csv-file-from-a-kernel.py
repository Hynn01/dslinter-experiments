#!/usr/bin/env python
# coding: utf-8

# Someone asked [on the forums](https://www.kaggle.com/product-feedback/61487#post359375) whether it was possible to download a file directly from the kernel without having to commit it. I did some poking around and found a nifty example [Polong Lin](http://www.polonglin.com) that allows you to do just that by creating a download link for a data file. 
# 
# The download link will work in both the editor and kernel viewer. 

# In[ ]:


# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))

# create a link to download the dataframe
create_download_link(df)

# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 

