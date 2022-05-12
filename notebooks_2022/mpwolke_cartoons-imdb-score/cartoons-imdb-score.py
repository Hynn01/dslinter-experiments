#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

import plotly
plotly.offline.init_notebook_mode(connected=True)

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv("/kaggle/input/cartoons/cartoons.csv", delimiter=',', encoding='utf8')
pd.set_option('display.max_columns', None)
df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


bear = df[(df['cartoon_names']=='The Hair Bear Bunch')].reset_index(drop=True)
bear.head()


# ![](https://i.ytimg.com/vi/QFqFPWsBXzA/maxresdefault.jpg)youtube.com

# In[ ]:


df["cartoon_names"].value_counts()


# In[ ]:


df["genre"].value_counts()


# In[ ]:


df["release_date"].value_counts()


# In[ ]:


##Code by Taha07  https://www.kaggle.com/taha07/data-scientists-jobs-analysis-visualization/notebook

from wordcloud import WordCloud
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'Red',
                      height =2000,
                      width = 2000
                     ).generate(str(df["cartoon_names"]))
plt.rcParams['figure.figsize'] = (12,12)
plt.axis("off")
plt.imshow(wordcloud)
plt.title("Cartoon Names")
plt.show()


# In[ ]:


#Code by Siti K https://www.kaggle.com/khotijahs1/2020-indonesia-university-rank/comments

#The Best 20 Cartoons by IMDb Score
top_cartoon = df.sort_values(by='imdb_score', ascending=False)[:20]
figure = plt.figure(figsize=(10,6))
sns.barplot(y=top_cartoon.cartoon_names, x=top_cartoon.imdb_score)
plt.xticks()
plt.xlabel('Imdb Score')
plt.ylabel('Cartoon')
plt.title('Top 20 Cartoons by Imdb Score')
plt.show()


# In[ ]:


#Correlation map to see how features are correlated with each other and with target
corrmat = df.corr(method='kendall')
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True);


# #Feed only the DataBase
# 
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQUGMu41sJT-LbKnpEeAmTFlDZCIiD0UsRmeQ&usqp=CAU)intaninbase.com
