#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from wordcloud import WordCloud


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
#         print(os.path.join(dirname, filename))
        
        if filename == "train.csv":
            train_data = pd.read_csv(os.path.join(dirname, filename))
    

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Checking the data ✔️

# In[ ]:


train_data


# ## Confusion 😵
# ## discourse_id -> Id of each discourse/paragraph/topic.
# ## id -> Text document id

# In[ ]:


print(len(train_data))

## Sanity check
print(len(list(train_data.id.unique())))
print(len(os.listdir(os.path.join("/kaggle/input/feedback-prize-2021", "train")))) 


# In[ ]:


for row in train_data[train_data["id"]=="423A1CA112E2"].iterrows():
    print(row[1]["discourse_type"])
    print(row[1]["discourse_text"])
    print(row[1]["discourse_id"])
# train_data.loc[0,"predictionstring"]


# In[ ]:


with open(os.path.join("/kaggle/input/feedback-prize-2021", "train","423A1CA112E2"+".txt")) as f:
    print(f.read())


# In[ ]:


discourse_categories = list(train_data.discourse_type.unique())
discourse_categories


# In[ ]:


list(train_data.discourse_type_num.unique())


# # Data Analysis 📈

# ## 1. In how many documents does each discourse_cat occurs out of total documents in training data

# In[ ]:


grouped_data = train_data.groupby(["discourse_type"]).nunique()
print(grouped_data["id"].values)


# In[ ]:


grouped_data


# In[ ]:


print("No. of unique id present in train data")
len(list(train_data.id.unique()))


# In[ ]:



unique_id_count = grouped_data["id"].values
discourse_categories_idx = [i for i, _ in enumerate(discourse_categories)]

# plt.bar(discourse_categories_idx, unique_id_count, color="pink")
# plt.xlabel("Discourse Type")
# plt.ylabel("No. of documents(unique doc id)")
# plt.title("Distribution of discourse type by documents")
# plt.xticks(discourse_categories_idx,discourse_categories, rotation=45)


# In[ ]:


import plotly.express as px

fig = px.bar(discourse_categories_idx, y = unique_id_count, width=800, height=400, labels={
                     "index":"Discourse Type",
                     "y":"No. of documents(unique doc id)",
                 },
                title="Distribution of discourse type by documents")
fig.update_traces(marker_color='pink')
fig.update_layout(
    margin=dict(l=20, r=20, t=30, b=20),
    paper_bgcolor="#17becf",
    xaxis = dict(
        tickmode = 'array',
        tickvals = discourse_categories_idx,
        ticktext = discourse_categories
    )
)

fig.show()


# In[ ]:


# fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
# plt.hist(train_data["discourse_type"], bins=list(train_data.discourse_type.unique()), rwidth=0.3, align="left")
# plt.xticks(rotation=45)

# train_data["id"].hist(train_data)


# ## 2. How many times each discourse type can occur in a document?

# In[ ]:


categories = list(train_data.discourse_type.unique())
# print(categories)
for cat in categories:
    filtered_data = train_data[train_data["discourse_type"]==cat]
    print(cat)
    value, counts = np.unique(filtered_data["id"], return_counts=True)
    print(f"{cat} can occur {np.unique(counts)} these many times ")
    fig = px.histogram(counts, nbins=12, width=800, height=400)
    fig.update_traces(marker_color='purple')
#     plt.hist(counts, bins=range(12), rwidth=0.5, align="left")
    fig.show()
#     plt.xticks(np.unique(counts))
## Trying to how many times Lead is occuring in one document.


# ## 3. Analysing the length of text for each discourse_type

# In[ ]:


agg_data = train_data.groupby("discourse_type").agg(['min','max',np.mean])


# In[ ]:


agg_data.sort_values(by=[('discourse_start', 'mean')])


# In[ ]:


agg_data[('discourse', 'len_min')] = agg_data[('discourse_end', 'min')] - agg_data[('discourse_start', 'min')]
agg_data[('discourse', 'len_max')] = agg_data[('discourse_end', 'max')] - agg_data[('discourse_start', 'max')]
agg_data[('discourse', 'len_mean')] = agg_data[('discourse_end', 'mean')] - agg_data[('discourse_start', 'mean')]


# In[ ]:


agg_data


# In[ ]:


discouse_meanlen = agg_data[('discourse', 'len_mean')]

print(discouse_meanlen)

discourse_type = list(agg_data.index)

# print(discourse_type)

discourse_type_idx = [i for i, _ in enumerate(discourse_type)]

fig = px.bar(discourse_type_idx, np.array(discouse_meanlen), width=800, height=400, labels = {'x': "Mean length of discourse type", 'index': "Discourse Type" }, title= "Mean length Discourse wise")
fig.update_traces(marker_color='orange')
fig.update_layout(
    yaxis = dict(
        tickmode = 'array',
        tickvals = discourse_categories_idx,
        ticktext = discourse_categories
    )
)

fig.show()


# ## 4. Analysing the word occurence for each discourse_type

# In[ ]:


for cat in discourse_categories:
    
    

    text = ' '.join(train_data[train_data["discourse_type"] == cat]["discourse_text"].tolist())
    wordcloud = WordCloud (
                        background_color = 'white',
                        width = 512,
                        height = 384
                            ).generate(text)
    plt.imshow(wordcloud) # image show
    plt.axis('off') # to off the axis of x and y
    plt.title(cat)
    plt.savefig('Plotly-World_Cloud.png')
    plt.show()


# In[ ]:


## Relative min and max length of discourse type
## using preconditionindex
## What is discourse_type_num


# In[ ]:




