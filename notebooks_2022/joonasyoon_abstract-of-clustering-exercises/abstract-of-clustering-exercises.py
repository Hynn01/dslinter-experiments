#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as n
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

csv_files = []

for dirname, _, filenames in os.walk('../input/clustering-exercises'):
    for filename in sorted(filenames):
        print(filename)
        csv_files.append(os.path.join(dirname, filename))

print(len(csv_files), 'file(s)')


# In[ ]:


COLORS = ['#F23D3A', '#3A3AF2', '#006E7F', '#F8CB2E', '#AB46D2', '#FF6FB5', '#55D8C1', '#FCF69C', '#733C3C', '#B22727', '#B4ECE3', '#FFF8D5', '#FFBD35', '#3FA796', '#8267BE', '#502064', '#FF5D9E', '#8F71FF', '#82ACFF', '#8BFFFF', '#3E3838', '#AE7C7C', '#6CBBB3', '#EFE784', '#072227', '#35858B', '#4FBDBA', '#AEFEFF', '#4B5D67', '#322F3D', '#59405C', '#87556F']


# In[ ]:


fig, axes = plt.subplots(5, 6, figsize=(14, 13))
fig.suptitle('Scatter plot', fontsize=32)
plt.tight_layout()
plt.subplots_adjust(top=0.9, wspace=0.5, hspace=0.5)

for i, csv in enumerate(csv_files):
    df = pd.read_csv(csv)
    color_cnt = len(df['color'].unique())
    r, c = i // 6, i % 6
    fname = csv.split('/')[-1].replace('.csv', '')
    ax = sns.scatterplot(data=df, x='x', y='y', hue='color', ax=axes[r][c], palette=COLORS[:color_cnt], s=5)
    ax.set_title('{}({} colors)'.format(fname, color_cnt))

for ax in axes.flatten():
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend().set_visible(False)
    
plt.show()


# In[ ]:


fig, axes = plt.subplots(5, 6, figsize=(14, 13))
fig.suptitle('Histogram plot', fontsize=32)
plt.tight_layout()
plt.subplots_adjust(top=0.9, wspace=0.5, hspace=0.5)

for i, csv in enumerate(csv_files):
    df = pd.read_csv(csv)
    color_cnt = len(df['color'].unique())
    r, c = i // 6, i % 6
    fname = csv.split('/')[-1].replace('.csv', '')
    ax = sns.histplot(data=df, x='x', y='y', hue='color', ax=axes[r][c], palette=COLORS[:color_cnt])
    ax.set_title('{}({} colors)'.format(fname, color_cnt))

for ax in axes.flatten():
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend().set_visible(False)
    
plt.show()


# In[ ]:


fig, axes = plt.subplots(5, 6, figsize=(14, 13))
fig.suptitle('Histogram of colors', fontsize=32)
plt.tight_layout()
plt.subplots_adjust(top=0.9, wspace=0.5, hspace=0.5)

for i, csv in enumerate(csv_files):
    df = pd.read_csv(csv)
    color_cnt = len(df['color'].unique())
    r, c = i // 6, i % 6
    fname = csv.split('/')[-1].replace('.csv', '')
    df2 = df.groupby(by='color').count()['x']
    ax = sns.barplot(x=df2.index, y=df2.values, palette=COLORS[:color_cnt], ax=axes[r][c])
    ax.set_title(fname)

for ax in axes.flatten():
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend().set_visible(False)
    
plt.show()


# In[ ]:


def scaled_df(df):
    d = df.copy()
    for color in set(df['color']):
        color_df = df[df['color'] == color]
        color_df = (color_df - color_df.min()) / (color_df.max() - color_df.min())
        d.loc[color_df.index, ['x', 'y']] = color_df[['x', 'y']]
    return d


# In[ ]:


fig, axes = plt.subplots(5, 6, figsize=(14, 13))
fig.suptitle('Relative positions for colors', fontsize=32)
plt.tight_layout()
plt.subplots_adjust(top=0.9, wspace=0.5, hspace=0.5)

for i, csv in enumerate(csv_files):
    df = pd.read_csv(csv)
    color_cnt = len(df['color'].unique())
    r, c = i // 6, i % 6
    fname = csv.split('/')[-1].replace('.csv', '')
    ax = sns.histplot(
       data=scaled_df(df), x='x', y='y', hue='color',
       fill=True, common_norm=False, palette='rocket',
       alpha=1/color_cnt, linewidth=0, ax=axes[r][c]
    )
    ax.set_title(fname)

for ax in axes.flatten():
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend().set_visible(False)
    
plt.show()


# In[ ]:


fig, axes = plt.subplots(5, 6, figsize=(14, 13))
fig.suptitle('Relative position x for colors', fontsize=32)
plt.tight_layout()
plt.subplots_adjust(top=0.9, wspace=0.5, hspace=0.5)

for i, csv in enumerate(csv_files):
    df = pd.read_csv(csv)
    color_cnt = len(df['color'].unique())
    r, c = i // 6, i % 6
    fname = csv.split('/')[-1].replace('.csv', '')
    ax = sns.kdeplot(
       data=scaled_df(df), x='x', hue='color',
       fill=True, common_norm=False, palette=COLORS[:color_cnt],
       alpha=1/color_cnt, linewidth=0, ax=axes[r][c]
    )
    ax.set_title(fname)

for ax in axes.flatten():
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend().set_visible(False)
    
plt.show()

