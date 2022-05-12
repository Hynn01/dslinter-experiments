#!/usr/bin/env python
# coding: utf-8

# Looking at the most important factors, "standard deviation of TARGET", "mean of TARGET" and "score by stock".

# # Definition

# In[ ]:


import os
import gc
import warnings

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None
warnings.simplefilter('ignore')
pd.set_option('max_column', None)
sns.set_style("darkgrid")
colors = sns.color_palette('Set2')
warnings.filterwarnings("ignore")

TRAIN_DIR = "../input/jpx-tokyo-stock-exchange-prediction/train_files"


# # load csv
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndf_prices = pd.read_csv(os.path.join(TRAIN_DIR, \'stock_prices.csv\'))\n# df_prices2 = pd.read_csv(os.path.join(TRAIN_DIR, \'secondary_stock_prices.csv\'))\n# df_fins = pd.read_csv(os.path.join(TRAIN_DIR, \'financials.csv\'))\n# df_opts = pd.read_csv(os.path.join(TRAIN_DIR, \'options.csv\'))\n# df_trades = pd.read_csv(os.path.join(TRAIN_DIR, \'trades.csv\'))\ndf_stock_list = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")')


# # Calculate score by stock
# `TargetStd` `TargetMean` `TargetScore`  
# I will focus on the above and see. Rank makes it easier to interpret.    
# I thought std was important, but it turns out that mean is more dominant for scores by stocks.    
# Short seems to be more important for std than long, so it would be better to choose long and short differently.  

# In[ ]:


_df_stock_list = df_stock_list[['Name', 'SecuritiesCode', 'Section/Products','NewMarketSegment','33SectorName', 'NewIndexSeriesSize']]
target_std = df_prices.groupby('SecuritiesCode')['Target'].std().rename('TargetStd')
df_stocks = pd.merge(_df_stock_list, target_std, on='SecuritiesCode')
target_mean = df_prices.groupby('SecuritiesCode')['Target'].mean().rename('TargetMean')
df_stocks = pd.merge(df_stocks, target_mean, on='SecuritiesCode')
target_score = (target_mean / target_std).rename('TargetScore')
df_stocks = pd.merge(df_stocks, target_score, on='SecuritiesCode')
df_stocks['TargetStdRank'] = df_stocks['TargetStd'].rank(axis=0, ascending=True).astype(int)
df_stocks['TargetMeanRank'] = df_stocks['TargetMean'].rank(axis=0, ascending=False).astype(int)
df_stocks['TargetScoreRank'] = df_stocks['TargetScore'].rank(axis=0, ascending=False).astype(int)


# In[ ]:


print('long')
display(df_stocks.sort_values('TargetScore', ascending=False).head(10))
print('short')
display(df_stocks.sort_values('TargetScore', ascending=True).head(10))


# # Percentage of 33Sector in "Scores by stock"
# 
# In the long, Information & Communication, Service, and Electric Appliances are strong.  
# In the short, Banks, Retail Trade, and Information & Communication are strong.  
# There is a clear difference between the top 100 and the top 500-600.  

# In[ ]:


print('long top100')
df_stock_long = df_stocks.sort_values('TargetScore', ascending=False).head(100)
df_stock_long['33SectorName'].value_counts().plot.pie()
plt.show()

print('long top500-600')
df_stock_long = df_stocks.sort_values('TargetScore', ascending=False).head(600).tail(100)
df_stock_long['33SectorName'].value_counts().plot.pie()
plt.show()

print('short top100')
df_stock_short = df_stocks.sort_values('TargetScore', ascending=True).head(100)
df_stock_short['33SectorName'].value_counts().plot.pie()
plt.show()

print('short top500-600')
df_stock_short = df_stocks.sort_values('TargetScore', ascending=True).head(600).tail(100)
df_stock_short['33SectorName'].value_counts().plot.pie()
plt.show()


# # TargetScore
# 
# A clear difference in trend is evident between LONG and SHORT.   
# In addition, there is a sense of heterogeneity near the topmost stocks.  
# 

# In[ ]:


# long
print('TargetScore')
df = df_stocks.sort_values('TargetScore', ascending=False).head(200)
df['TargetScore'].plot.bar(figsize=(20,3))
plt.show()
print('TargetStdRank')
df['TargetStdRank'].plot.bar(figsize=(20,3))
plt.show()
print('TargetMeanRank')
df['TargetMeanRank'].plot.bar(figsize=(20,3))
plt.show()


# In[ ]:


# short
print('TargetScore')
df = df_stocks.sort_values('TargetScore', ascending=True).head(200)
df['TargetScore'].plot.bar(figsize=(20,3))
plt.show()
print('TargetStdRank')
df['TargetStdRank'].plot.bar(figsize=(20,3))
plt.show()
print('TargetMeanRank')
df['TargetMeanRank'].plot.bar(figsize=(20,3))
plt.show()


# # Top Stocks by Fiscal Year

# In[ ]:


def add_feature_by_year(df_stocks, year):
    df = df_prices.query(f'"{year}-01-01" <= Date < "{year+1}-01-01"')
    std = df.groupby('SecuritiesCode')['Target'].std().rename(f'{year}TargetStd')
    mean = df.groupby('SecuritiesCode')['Target'].mean().rename(f'{year}TargetMean')
    score = (mean / std).rename(f'{year}TargetScore')
    std_rank = std.rank(axis=0, ascending=True).astype(int).rename(f'{year}TargetStdRank')
    mean_rank = mean.rank(axis=0, ascending=False).astype(int).rename(f'{year}TargetMeanRank')
    score_rank = score.rank(axis=0, ascending=False).astype(int).rename(f'{year}TargetScoreRank')
    df_stocks = pd.merge(df_stocks, std, on='SecuritiesCode', how='left')
    df_stocks = pd.merge(df_stocks, mean, on='SecuritiesCode', how='left')
    df_stocks = pd.merge(df_stocks, score, on='SecuritiesCode', how='left')
    df_stocks = pd.merge(df_stocks, std_rank, on='SecuritiesCode', how='left')
    df_stocks = pd.merge(df_stocks, mean_rank, on='SecuritiesCode', how='left')
    df_stocks = pd.merge(df_stocks, score_rank, on='SecuritiesCode', how='left')
    return df_stocks

df_stocks = add_feature_by_year(df_stocks, 2017)
df_stocks = add_feature_by_year(df_stocks, 2018)
df_stocks = add_feature_by_year(df_stocks, 2019)
df_stocks = add_feature_by_year(df_stocks, 2020)
df_stocks = add_feature_by_year(df_stocks, 2021)


# In[ ]:


df = df_stocks.sort_values('TargetScore', ascending=False)
print('std')
display(df[['SecuritiesCode', 'TargetStd', 'TargetMean', 'TargetScore', 'TargetStdRank', 'TargetMeanRank', 'TargetScoreRank', '2017TargetStdRank', '2018TargetStdRank', '2019TargetStdRank', '2020TargetStdRank', '2021TargetStdRank']].head(10))
print('mean')
display(df[['SecuritiesCode', 'TargetStd', 'TargetMean', 'TargetScore', 'TargetStdRank', 'TargetMeanRank', 'TargetScoreRank', '2017TargetMeanRank', '2018TargetMeanRank', '2019TargetMeanRank', '2020TargetMeanRank', '2021TargetMeanRank']].head(10))
print('score')
display(df[['SecuritiesCode', 'TargetStd', 'TargetMean', 'TargetScore', 'TargetStdRank', 'TargetMeanRank', 'TargetScoreRank', '2017TargetScoreRank', '2018TargetScoreRank', '2019TargetScoreRank', '2020TargetScoreRank', '2021TargetScoreRank']].head(10))


# The stocks that were TOP throughout the entire year also did not perform that well when looking at the most recent year.  
# Since there are no new stocks added in the LB, it may be better to omit the first year and compare scores.  
