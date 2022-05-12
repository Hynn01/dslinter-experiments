#!/usr/bin/env python
# coding: utf-8

# # Basic data science in Python
# Please see its companion notebook, [doing stuff in R](https://www.kaggle.com/code/bennettgillig/doing-stuff-in-r)

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from functools import reduce
data = pd.read_csv("../input/students-adaptability-level-in-online-education/students_adaptability_level_online_education.csv")
data.set_axis((col.replace(" ", "") for col in data.columns), axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


# I can't separately color the dots and the line it seems (since idk how to plot a regression line without a scatterplot)
x = [reduce(lambda a,b:(a+b)/2, (int(i) for i in re.findall(r"\d+", age))) for age in data.Age]
y = [{"Poor": 0, "Mid": 1, "Rich": 2}[stat] for stat in data.FinancialCondition]

sns.regplot(x=x, y=y, color=(0, 0, 1)).set(xlim=(0, 30), ylim=(0.6, 1.2)) and None # shut it up


# In[ ]:


unique = [*data.EducationLevel.unique()]
colors = ((0.91, 0.67, 1), (0.67, 0.91, 1), (1, 1, 0.67))

cooldata = [reduce(lambda a,b:(a+b)/2, (int(i) for i in re.findall(r"\d+", age))) for age in data.Age]
max_freq = max((max(data[data.EducationLevel == col].Age.value_counts()) for col in unique))

graph = sns.histplot(data.assign(Age=cooldata), x="Age", hue="EducationLevel", palette=colors)
[graph.bar_label(c, fmt="black lab") for c in graph.containers]
graph.set(ylabel="Freq", title="heyo") and None # shut it up


# In[ ]:




