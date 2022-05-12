#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Library

import numpy as np
import pandas as pd
import pandas_profiling

from sklearn import tree

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


# dataset of seaborn

dataset_names = sns.get_dataset_names()
print(dataset_names)


# In[ ]:


anagrams = sns.load_dataset('anagrams')
print(anagrams.head())
print()
print(anagrams.info())


# In[ ]:


anscombe = sns.load_dataset('anscombe')
print(anscombe.head())
print()
print(anscombe.info())


# In[ ]:


fmri = sns.load_dataset('fmri')
print(fmri.head())
print()
print(fmri.info())

# Functional magnetic resonance imaging or functional MRI(fMRI) measures brain activity using a strong, static magnetic field 
# to detect changes associated with blood flow. When an area of the brain is in use, blood flow to that region also increases.


# In[ ]:


tips = sns.load_dataset('tips')
print(tips.head())
print()
print(tips.info())
# memory usage: 7.4 KB


# In[ ]:


# for small data memory
pandas_profiling.ProfileReport(tips)


# In[ ]:


diamonds = sns.load_dataset('diamonds')
print(diamonds.head())
print()
print(diamonds.info())


# In[ ]:


# for a big data memory
pandas_profiling.ProfileReport(diamonds, minimal=True)


# In[ ]:


titanic = sns.load_dataset('titanic')
print(titanic.describe())
print()
titanic.info()


# In[ ]:


pandas_profiling.ProfileReport(titanic, minimal=True)


# In[ ]:


tips.head()


# In[ ]:


sns.relplot(data=tips, x="total_bill", y="tip", hue="day")
plt.show()


# In[ ]:


# relational plots : replot
# style - whitegrid,darkgrid,dark,white,ticks
# palette - deep, muted, pastel, bright, dark, colorblind

sns.set_style('whitegrid', {"grid.color": "0.8", "grid.linestyle": "-"})
sns.color_palette('pastel')

ax = sns.relplot(data=tips, kind='scatter', x='total_bill', y='tip', hue='smoker', col='day', height=4)
ax.set(xlabel="total bill")

plt.show()


# In[ ]:


# scatterplot() (with kind="scatter"; the default)
# lineplot() (with kind="line")

sns.set_style('whitegrid', {"grid.color": ".8", "grid.linestyle": "-"})
sns.color_palette('pastel')

ax = sns.relplot(data=tips, kind='scatter', x='total_bill',y='tip',hue='smoker',col='time', height=4, style='smoker', size='size', row='sex')
ax.set(xlabel="total bill")

plt.show()


# In[ ]:


sns.relplot(kind='scatter', x='total_bill', y='tip', data=tips, hue='smoker',col='size',height=3, col_wrap=3, palette="ch:.9,rot=.1,dark=.2")
plt.show()


# In[ ]:


fmri.head()


# In[ ]:


sns.relplot(data=fmri, kind='line', x='timepoint', y='signal', hue='event',sort=False)
plt.show()


# In[ ]:


sns.relplot(data=fmri, kind='line', x='timepoint',y='signal', hue='event',sort=True)
plt.show()


# In[ ]:


sns.relplot(data=fmri, kind='line', x='timepoint',y='signal', hue='event',sort=True,height=4,style='region',col='subject',
            col_wrap=5,ci=False,markers=True)
plt.show()


# In[ ]:


sns.relplot(data=fmri.query("subject=='s0' or subject=='s1'"), x='timepoint',y='signal', , hue='event',kind='line',sort=True)
plt.show()


# In[ ]:


# Line Plot and Scatter plot
tips.head()


# In[ ]:


sns.lineplot(data=fmri, x='timepoint',y='signal', size='subject',sizes=(3,5),hue='subject',palette="ch:.7,-.3,d=.1_r",ci=False)
plt.show()


# In[ ]:


iris = sns.load_dataset('iris')


# In[ ]:


iris.head()


# In[ ]:


sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species',marker='o')
plt.show()


# In[ ]:


# Categorical plot - catplot
tips.sample(5)


# In[ ]:


sns.catplot(data=tips,kind='strip', x='day',y='tip',)
plt.show()


# In[ ]:


sns.catplot(x='day',y='tip',data=tips, hue='smoker',col='size',col_wrap=3,height=3)
plt.show()


# In[ ]:


sns.catplot(y='sex', x='tip', data=tips, kind='swarm', hue='size',s=3)
plt.show()


# In[ ]:


sns.catplot(x='sex', y='tip', data=tips, kind='swarm', hue='smoker',s=5,col='size',col_wrap=3, height=4)
plt.show()


# In[ ]:


sns.catplot(x='day',y='tip', data=tips, kind='box')
plt.show()


# In[ ]:


sns.catplot(x='day',y='tip', data=tips, kind='box', col='sex',hue='sex')
plt.show()


# In[ ]:


sns.catplot(x='day',y='tip', data=tips, kind='box', hue='sex')
plt.show()


# In[ ]:


sns.catplot(x='day',y='tip', data=tips, kind='box', hue='sex', dodge=False)
plt.show()


# In[ ]:


diamond.head()


# In[ ]:


sns.catplot(x='color', y='price', data=diamond, kind='boxen')
plt.show()


# In[ ]:


sns.catplot(x='color', y='price', data=diamond, kind='boxen', col='cut')
plt.show()


# In[ ]:


sns.catplot(x='size', y='tip',data=tips, kind='boxen', hue='sex',col='sex',height=4)
plt.show()


# In[ ]:


tips.head()


# In[ ]:


sns.catplot(x='day', y='tip', data=tips, kind = 'violin',col = 'sex')
plt.show()


# In[ ]:


sns.catplot(y='day', x='tip', data=tips, kind = 'violin')
plt.show()


# In[ ]:


sns.catplot(y='tip', x='day', data=tips, kind='violin', hue='sex')
plt.show()


# In[ ]:


sns.catplot(y='tip', x='day', data=tips, kind='violin', hue='sex', split=True)
plt.show()


# In[ ]:


sns.catplot(y='tip', x='day', data=tips, kind='violin', hue='sex', split=True, inner='stick')
plt.show()


# In[ ]:


titanic.head()


# In[ ]:


sns.catplot(data=titanic, kind='bar', x='sex', y='survived', hue='pclass', palette="bright")
plt.show()


# In[ ]:


pd.value_counts(titanic['class'])


# In[ ]:


sns.catplot(x='class', kind='count', data=titanic, palette="ch:0.2")
plt.show()


# In[ ]:


# 5 Data distribution
sns.displot(x = fmri['signal'], bins=20, color='b',kde=True)
plt.show()


# In[ ]:


sns.displot(x=fmri['signal'],  rug=True, bins=25, color='#ff0099', kde=True, kind='hist', col=fmri['event'])
plt.show()


# In[ ]:


sns.displot(x = fmri['signal'], color='green',kind='kde',col=fmri['subject'],
            col_wrap=5, height=4, rug=True)
plt.show()


# In[ ]:


sns.kdeplot(y=fmri['signal'], shade=True, hue=fmri['region'], gridsize=1000)
plt.show()


# In[ ]:


sns.kdeplot(x=fmri['signal'], shade=True, hue=fmri['subject'], gridsize=1000, palette="ch:.7,-.3,d=.1_r")
plt.show()


# In[ ]:


tips.head()


# In[ ]:


sns.jointplot(x=tips['total_bill'],y=tips['tip'],kind='scatter',hue=tips['smoker'])
plt.show()


# In[ ]:


sequential_colors = sns.color_palette("RdPu", 2)
sns.jointplot(x=tips['total_bill'],y=tips['tip'], kind='hex', palette=sequential_colors)
plt.show()


# In[ ]:


sns.jointplot(x=tips['total_bill'],y=tips['tip'], kind='hex', color='green', marginal_ticks=True)
plt.show()


# In[ ]:


iris.head()


# In[ ]:


sns.pairplot(data=iris,hue='species')
plt.show()


# In[ ]:


sns.pairplot(data=iris,hue='species', kind='kde')
plt.show()


# In[ ]:


# 6 Linear Regression
sns.regplot(x=tips['total_bill'], y=tips['tip'],color='blue')
plt.show()


# In[ ]:


# robust - It will remove the outlier in the dataset

sns.lmplot(data=tips, x='total_bill', y='tip',  hue='sex',robust=True, col='sex',ci=90, markers='o',
           height=4, row='time')
plt.show()


# In[ ]:


# Heat map
iris.sample(4)


# In[ ]:


a=pd.DataFrame.corr(iris)
sns.heatmap(a,annot=True,vmin=-0.5, vmax=0.5,linewidths=.5)
plt.show()

