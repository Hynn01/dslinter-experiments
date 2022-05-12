#!/usr/bin/env python
# coding: utf-8

# **Content**
# 1. [Importing Libraries And Loading Datasets](#1)
# 1. [Exploratory Data Analysis (EDA)](#2)
# 1. [Data Visualization](#3)
# 1. [Pandas Profiling](#4)
# 1. [Ending](#5)

# <a id="1"></a> <br>
# ## 1. Importing Libraries And Loading Datasets

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.express as px


# In[ ]:


df_train = pd.read_csv('../input/learn-together/train.csv',index_col='Id')
df_test  = pd.read_csv('../input/learn-together/test.csv',index_col='Id')


# <a id="2"></a> <br>
# ## 2. Exploratory Data Analysis (EDA)

# In[ ]:


df_train.head()


# In[ ]:


print("Train dataset shape: "+ str(df_train.shape))
print("Test dataset shape:  "+ str(df_test.shape))


# In[ ]:


df_train.info()


# There are total 55 columns in train dataset (54 in test dataset except the last column that is y), All of them are not null.

# In[ ]:


df_train.describe().T


# All Wilderness_Area and Soil_Type columns have values in the range of 0 and 1. Quite likely these columns are categorical and consist of 0 and 1. To validate this i'm checking distinct values of following columns:

# In[ ]:


print(df_train.iloc[:,10:-1].columns)


# In[ ]:


pd.unique(df_train.iloc[:,10:-1].values.ravel())


# Yes all wilderness area and soil type columns consist of 0 and 1. In other words they are categorical. So i'm convering these columns to categorical ones.

# In[ ]:


df_train.iloc[:,10:-1] = df_train.iloc[:,10:-1].astype("category")
df_test.iloc[:,10:] = df_test.iloc[:,10:].astype("category")


# I'm trying to find out correlation between columns with heatmap in this step. 

# In[ ]:


f,ax = plt.subplots(figsize=(8,6))
sns.heatmap(df_train.corr(),annot=True, 
            linewidths=.5, fmt='.1f', ax=ax) 

plt.show()


# It seems the most important correlations are between "Horizontal Distance To Hydrology" and "Vertical Distance To Hydrology" with 70%;  between "Aspect" and "Hillshade 3pm" with 60%;  between "Hillshade Noon" and "Hillshade 3pm" with %60;  between "Elevation" and "Horizontal Distance To Roadways" with %60. Let's see how they are looking.

# <a id="3"></a> <br>
# ## 3. Data Visualization

# In[ ]:


df_train.plot(kind='scatter', x='Vertical_Distance_To_Hydrology', 
              y='Horizontal_Distance_To_Hydrology', alpha=0.5, 
              color='darkblue', figsize = (12,9))

plt.title('Vertical And Horizontal Distance To Hydrology')
plt.xlabel("Vertical Distance")
plt.ylabel("Horizontal Distance")

plt.show()


# In[ ]:


df_train.plot(kind='scatter', x='Aspect', y='Hillshade_3pm', 
              alpha=0.5, color='maroon', figsize = (12,9))

plt.title('Aspect and Hillshade 3pm Relation')
plt.xlabel("Aspect")
plt.ylabel("Hillshade 3pm")

plt.show()


# In[ ]:


df_train.plot(kind='scatter', x='Hillshade_Noon', y='Hillshade_3pm', 
              alpha=0.5, color='purple', 
              figsize = (12,9))

plt.title('Hillshade Noon and Hillshade 3pm Relation')
plt.xlabel("Hillshade_Noon")
plt.ylabel("Hillshade 3pm")

plt.show()


# There are obvious patterns if we ignore to outliers. And with this patterns, our model will learn.
# 
# Boxplot can use to see outliers. For a better visualization i will use plotly this time.

# In[ ]:


trace1 = go.Box(
    y=df_train["Vertical_Distance_To_Hydrology"],
    name = 'Vertical Distance',
    marker = dict(color = 'rgb(0,145,119)')
)

trace2 = go.Box(
    y=df_train["Horizontal_Distance_To_Hydrology"],
    name = 'Horizontal Distance',
    marker = dict(color = 'rgb(5, 79, 174)')
)

data = [trace1, trace2]
layout = dict(autosize=False, width=700,height=500, title='Distance To Hydrology', paper_bgcolor='rgb(243, 243, 243)', 
              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))

fig = dict(data=data, layout=layout)
iplot(fig)


# In[ ]:


trace1 = go.Box(
    y=df_train["Hillshade_Noon"],
    name = 'Hillshade Noon',
    marker = dict(color = 'rgb(255,111,145)')
)

trace2 = go.Box(
    y=df_train["Hillshade_3pm"],
    name = 'Hillshade 3pm',
    marker = dict(color = 'rgb(132,94,194)')
)

data = [trace1, trace2]
layout = dict(autosize=False, width=700,height=500, title='Hillshade 3pm and Noon', paper_bgcolor='rgb(243, 243, 243)', 
              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))

fig = dict(data=data, layout=layout)
iplot(fig)


# This time i'll compare vertical and horizontal distance to hydrology with histogram.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,7))
df_train.Vertical_Distance_To_Hydrology.plot.hist(ax=ax[0],bins=30,
                                                  edgecolor='black',color='crimson') 
                                       
ax[0].set_title('Vertical Distance To Hydrology')
x1=list(range(-150,350,50))
ax[0].set_xticks(x1)

df_train.Horizontal_Distance_To_Hydrology.plot.hist(ax=ax[1],bins=30,
                                                    edgecolor='black',color='darkmagenta') 
                                                                                                        
ax[1].set_title('Horizontal Distance To Hydrology')
x2=list(range(0,1000,100))
ax[1].set_xticks(x2)

plt.show


# i'll take a look our categorical categorical variables soil types and wilderness areas.

# In[ ]:


soil_types = df_train.iloc[:,14:-1].sum(axis=0)

plt.figure(figsize=(18,9))
sns.barplot(x=soil_types.index, y=soil_types.values, 
            palette="rocket")

plt.xticks(rotation= 75)
plt.ylabel('Total')
plt.title('Count of Soil Types With Value 1',
          color = 'darkred',fontsize=12)

plt.show()


# Type 7, Type 8, Type 15 and Type 25 have either no or too few values. Must examine carefully before create a model.

# In[ ]:


wilderness_areas = df_train.iloc[:,10:14].sum(axis=0)

plt.figure(figsize=(7,5))
sns.barplot(x=wilderness_areas.index, y=wilderness_areas.values, 
            palette="Blues_d")

plt.xticks(rotation=90)
plt.title('Wilderness Areas',color = 'darkred',fontsize=12)
plt.ylabel('Total')

plt.show()


# I wonder how many (y) labels we have in each class. I'll take a look the last column (cover type) for this.

# In[ ]:


cover_type = df_train["Cover_Type"].value_counts()
df_cover_type = pd.DataFrame({'CoverType': cover_type.index, 'Total':cover_type.values})

fig = px.bar(df_cover_type, x='CoverType', y='Total', 
             height=400, width=650)

fig.show()


# There are same amount of data for each class exactly...
# 
# In terms of horizontal distance to x point, distribution of class charts following...

# In[ ]:


f,ax=plt.subplots(1,3,figsize=(21,7))
df_train.plot.scatter(ax=ax[0],x='Cover_Type', y='Horizontal_Distance_To_Fire_Points', 
                      alpha=0.5, color='purple')

ax[0].set_title('Horizontal Distance To Fire Points')
x1=list(range(1,8,1))
ax[0].set_ylabel("")
ax[0].set_xlabel("Cover Type")
df_train.plot.scatter(ax=ax[1],x='Cover_Type', y='Horizontal_Distance_To_Roadways', 
                      alpha=0.5, color='purple')

ax[1].set_title('Horizontal Distance To Roadways')
x2=list(range(1,8,1))
ax[1].set_ylabel("")
ax[1].set_xlabel("Cover Type")
df_train.plot.scatter(ax=ax[2],x='Cover_Type', y='Horizontal_Distance_To_Hydrology', 
                      alpha=0.5, color='purple')

ax[2].set_title('Horizontal Distance To Hydrology')
x2=list(range(1,8,1))
ax[2].set_ylabel("")
ax[2].set_xlabel("Cover Type")

plt.show()


# All of them have a similar pattern, haven't it?

# <a id="4"></a> <br>
# ## 4. Pandas Profiling

# Actually there is a faster way for exploratory data analysis. Pandas provides you powerful HTML profiling reports with pandas-profiling. It's like a magic! You can click "Overview", "Variables" etc tabs for a quick run.

# In[ ]:


report = pp.ProfileReport(df_train)

report.to_file("report.html")

report


# <a id="5"></a> <br>
# ## 5. Ending

# ### To be continued... If you like, Please upvote.
