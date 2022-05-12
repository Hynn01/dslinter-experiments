#!/usr/bin/env python
# coding: utf-8

# # <b>1 <span style='color:#3f4d63'>|</span> Introduction</b>
# 
# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>1.1 | About this competition</b></p>
# </div>
# 
# Success in any financial market requires one to identify solid investments. When a stock or derivative is undervalued, it makes sense to buy. If it's overvalued, perhaps it's time to sell. While these finance decisions were historically made manually by professionals, technology has ushered in new opportunities for retail investors. Data scientists, specifically, may be interested to explore quantitative trading, where decisions are executed programmatically based on predictions from trained models.
# 
# There are plenty of existing quantitative trading efforts used to analyze financial markets and formulate investment strategies. To create and execute such a strategy requires both historical and real-time data, which is difficult to obtain especially for retail investors. This competition will provide financial data for the Japanese market, allowing retail investors to analyze the market to the fullest extent.
# 
# Japan Exchange Group, Inc. (JPX) is a holding company operating one of the largest stock exchanges in the world, Tokyo Stock Exchange (TSE), and derivatives exchanges Osaka Exchange (OSE) and Tokyo Commodity Exchange (TOCOM). JPX is hosting this competition and is supported by AI technology company AlpacaJapan Co.,Ltd.
# 
# ![](https://news.utexas.edu/wp-content/uploads/2020/03/Stock-Market-pic-1200x800-c-default.jpg)
# 
# This competition will compare your models against real future returns after the training phase is complete. The competition will involve building portfolios from the stocks eligible for predictions (around 2,000 stocks). Specifically, each participant ranks the stocks from highest to lowest expected returns and is evaluated on the difference in returns between the top and bottom 200 stocks. You'll have access to financial data from the Japanese market, such as stock information and historical stock prices to train and test your model.
# 
# All winning models will be made public so that other participants can learn from the outstanding models. Excellent models also may increase the interest in the market among retail investors, including those who want to practice quantitative trading. At the same time, you'll gain your own insights into programmatic investment methods and portfolio analysis―and you may even discover you have an affinity for the Japanese market.
# 
# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>1.2 | References</b></p>
# </div>
# 
# Before starting, just mention some astonishing works from which I got some new ideas and inspiration. **Take a look on them !**
# 
# - [Easy to understand the competition.](https://www.kaggle.com/code/chumajin/easy-to-understand-the-competition)
# - [JPX - Detailed EDA.](https://www.kaggle.com/code/abaojiang/jpx-detailed-eda)
# - [Useful features in Predicting Stock Prices](https://www.kaggle.com/code/riteshsinha/useful-features-in-predicting-stock-prices#Analyzing-SMA-(Simple-Moving-Averages)-and-EMA-(Exponential-Moving-Averages))
# 
# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>1.3 | Importing Packages</b></p>
# </div>

# In[ ]:


from IPython.display import clear_output
import os
import warnings
from pathlib import Path

# Basic libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_profiling as pp
import seaborn as sns

# Clustering
from sklearn.cluster import KMeans

# Principal Component Analysis (PCA)
from sklearn.decomposition import PCA

#Mutual Information
from sklearn.feature_selection import mutual_info_regression

# Cross Validation
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, learning_curve, train_test_split

# Encoding
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from category_encoders import MEstimateEncoder
from category_encoders import MEstimateEncoder

# Algorithms
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Optuna - Bayesian Optimization 
import optuna
from optuna.samplers import TPESampler

# Plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.offline as offline
import plotly.graph_objs as go

# Spaceship Titanic Metric
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

def load_data():
    data_dir = Path("../input/jpx-tokyo-stock-exchange-prediction")
    train_dir = Path("../input/jpx-tokyo-stock-exchange-prediction/train_files")
    stock_prices = pd.read_csv(train_dir / "stock_prices.csv")
    financials = pd.read_csv(train_dir / "financials.csv")
    options = pd.read_csv(train_dir / "options.csv")
    secondary_stock_prices = pd.read_csv(train_dir / "secondary_stock_prices.csv")
    stock_list = pd.read_csv('../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv')    
    return stock_prices, financials, options, secondary_stock_prices, stock_list

def plot_feature_importance(importance,names,model_type):
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    fi_df = fi_df[fi_df.feature_importance > 0]
    fig = px.bar(fi_df, x='feature_names', y='feature_importance', color="feature_importance",
             color_continuous_scale='Blugrn')
    # General Styling
    fig.update_layout(height=400, bargap=0.2,
                  margin=dict(b=50,r=30,l=100,t=100),
                  title = "<span style='font-size:36px; font-family:Times New Roman'>Feature Importance Analysis</span>",                  
                  plot_bgcolor='rgb(242,242,242)',
                  paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
    fig.show()
    
"""
function annotation_helper(...)

Helper for annotations in plotly. While reducing the amount of code to create an annotation, it also:
- Allows us to provide the text into an array of 
  strings(one for each new line) instead of one really long <br> separated text param
- Provides basic functionality for individual line spacing(s) between each line
- Custom annotation rectangle
- Basic debugging for annotation positioning
"""

def annotation_helper(fig, texts, x, y, line_spacing, align="left", bgcolor="rgba(0,0,0,0)", borderpad=0, ref="axes", xref="x", yref="y", width=100, debug = False):
    
    is_line_spacing_list = isinstance(line_spacing, list)
    total_spacing = 0
    
    for index, text in enumerate(texts):
        if is_line_spacing_list and index!= len(line_spacing):
            current_line_spacing = line_spacing[index]
        elif not is_line_spacing_list:
            current_line_spacing = line_spacing
        
        fig.add_annotation(dict(
            x= x,
            y= y - total_spacing,
            width = width,
            showarrow=False,
            text= text,
            align= align,
            borderpad=4 if debug == False else 0, # doesn't work with new background box implementation :S
            xref= "paper" if ref=="paper" else xref,
            yref= "paper" if ref=="paper" else yref,
            
            bordercolor= "#222",
            borderwidth= 2 if debug == True else 0 # shows the actual borders of the annotation box
        ))
        
        total_spacing  += current_line_spacing
    
    if bgcolor != "rgba(0,0,0,0)":
        fig.add_shape(type="rect",
            xref= "paper" if ref=="paper" else xref,
            yref= "paper" if ref=="paper" else yref,
            xanchor = x, xsizemode = "pixel", 
            x0=-width/2, x1= +width/2, y0=y + line_spacing[-1], y1=y -total_spacing,
            fillcolor= bgcolor,
            line = dict(width=0))  
      
    if debug == True:
        handle_annot_debug(fig, x, y, ref)

stock_prices, financials, options, secondary_stock_prices, stock_list = load_data()
clear_output()


# # <b>2 <span style='color:#3f4d63'>|</span> Exploratory Data Analysis</b>
# 
# We'll examine the **stock_list file** and the different **.csv files the train files folder** has. Those files are: 
# 
# * `financials.csv`
# * `options.csv`
# * `secondary_stock_prices.csv`
# * `stock_prices.csv`
# * `trades.csv`
# 
# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.1| Stock Prices File</b></p>
# </div>
# 
# **File Description:**
# The core file of interest. Includes the daily closing price for each stock and the target column. Below, I have added the definition of each of the features. These explanations have been taken from **data_specifications/stock_price_spce.csv**
# * `RowId`: unique ID of price records, the combination of `Date` and `SecuritiesCode`.
# * `Date`: trade date.
# * `SecuritiesCode`: local securities code.
# * `Open`: first traded price on a day.
# * `High`: highest traded price on a day.
# * `Low`: lowest traded price on a day.
# * `Close`: last traded price on a day.
# * `Volume`: number of traded stocks on a day.
# * `AdjustmentFactor`: to calculate theoretical price/volume when split/reverse-split happens (NOT including dividend/allot...)
# * `SupervisionFlag`: Flag of securities under supervision and securities to be delisted -> [info](https://www.jpx.co.jp/english/listing/market-alerts/supervision/00-archives/index.html).
# * `ExpectedDividend`: Expected dividend value for ex-right date. This value is recorded 2 business days before ex-dividend...
# * `Target`: Change ratio of adjusted closing price between t+2 and t+1 where t+0 is TradeDate.

# In[ ]:


pp.ProfileReport(stock_prices)


# **Early Insights**
# - Numeric Features: 9; Categorical: 2; Boolean: 1
# - ExpectedDividend has plenty of missing values. 
# - Open, Close, High, Low features are hihgly correlated. 
# - Target has 3.8% of zero values. 
# 
# Hereafter, we are going to start doing some EDA stuff of some features. 
# 
# <div style="color:white;display:fill;
#             background-color:#235f83;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.1.1 |Date Feature</b></p>
# </div>
# 
# First thing we notice when examining this feature is that data is recorded on working days. Thus, on Saturdays and Sundays there is no stock data recorded. That's the explanation of the blank spaces we'll appreciate in the following chart.

# In[ ]:


stock_prices['Date'] = pd.to_datetime(stock_prices['Date'])
fig = px.bar(stock_prices['Date'].value_counts(), color_discrete_sequence = [px.colors.sequential.Blugrn[6]])

# General Styling
fig.update_layout(height=400, bargap=0.2,
                  margin=dict(r=30,l=100, t=100, b=0),
                  plot_bgcolor='rgb(242,242,242)',
                  #paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
#Title
text = [
    "<span style='font-size:35px; font-family:Times New Roman'>Stock Records per Date Analysis</span>"
]
annotation_helper(fig, text, -0.043, 1.45, [0.095,0.095,0.065],ref="paper", width=1300)
#Subtitle
text = [
    "<span style='font-size:14px; font-family:Helvetica'> Number of stock records per date. As we can appreciate, not all 2000 stocks have data depending on the date. Minimum amount of stocks recorded is 1865, while 2000 is maximum.</span>",
    "<span style='font-size:14px; font-family:Helvetica'> We can see some blank spaces between some pairs of date. For example, in 2019, between 27th of April and 7th of May there are no stock data recorded. </span>",
]
annotation_helper(fig, text, -0.043, 1.25, [0.075,0.095,0.065],ref="paper", width=1300)
fig.show()


# 📌 **Interpret:** from the above chart we can conclude that not all stocks were recorded since first day of recording. As we can observe, number of stocks per day increases over time. As the days go by, new stocks are being introduced into the recording. 
# 
# <div style="color:white;display:fill;
#             background-color:#235f83;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.1.2 |SecuritiesCode Feature</b></p>
# </div>
# 
# We are going to explore how many times each stock appears in our dataset. With this, we'll obtain the amount of recorded data we have for a certain stock.

# In[ ]:


stock_prices['SecuritiesCode'].value_counts()


# 📌 **Interpret:** as we can appreciate there is a huge difference between most recorded stocks and less ones. Concretely this difference is around one thousand dates. Let's now study the percentage of stocks that have fewer data recorded. As we show below, this percentage is almost insignificant.

# In[ ]:


tmp = pd.DataFrame(stock_prices['SecuritiesCode'].value_counts())
print('Percentage of stocks with fewer records than 500: ', tmp[tmp.SecuritiesCode < 500].shape[0] / tmp.shape[0])


# <div style="color:white;display:fill;
#             background-color:#235f83;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.1.3 |Target Feature</b></p>
# </div>
# 
# <div style="color:white;display:fill;
#             background-color:lightgrey;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>How is target calculated ?</b></p>
# </div>
# 
# To start with, we'll introduce **how the target feature is calculated**. In order to understand it properly, it's neccesary to understand first what's the meaning of close value of a stock in one day. Target feature refers to the change ratio of the closing value of a stock in the following two dates.  Its mathematical formula is the following: 
# 
# 
# $$
# \begin{equation}
# \large{r_{k,t} = \frac{C_{k,t+2} - C_{k,t+1}}{C_{k,t+1}}}
# \end{equation}
# $$
# 
# where **k is the concrete stock**, and **t is the current date**.

# In[ ]:


tmp = stock_prices[stock_prices.SecuritiesCode == 1301].copy()
# New features with closing value of the stock on the following two dates
tmp["CloseShift1"] = tmp["Close"].shift(-1)
tmp["CloseShift2"] = tmp["Close"].shift(-2)

tmp["rate"] = (tmp["CloseShift2"] - tmp["CloseShift1"]) / tmp["CloseShift1"]
tmp[['Date','SecuritiesCode','Close','Target','CloseShift1','CloseShift2','rate']].head()


# <div style="color:white;display:fill;
#             background-color:lightgrey;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>Target Statistical Analysis</b></p>
# </div>
# 
# Let's begin by knowing some statistical information about the target feature. Concretely, we are going to start by showing its kurtosis and skew value. After it, we'll show its distribution.
# 
# **Skewness:** Let's now check skewness. Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. The skewness value can be positive, zero, negative, or undefined.
# 
# For a unimodal distribution, negative skew commonly indicates that the tail is on the left side of the distribution, and positive skew indicates that the tail is on the right. In cases where one tail is long but the other tail is fat, skewness does not obey a simple rule. For example, a zero value means that the tails on both sides of the mean balance out overall; this is the case for a symmetric distribution, but can also be true for an asymmetric distribution where one tail is long and thin, and the other is short but fat. A skewness greater than 1 is generally judged to be skewed, so check mainly those greater than 1.

# In[ ]:


from scipy.stats import skew, kurtosis
print('Skewness: ', skew(stock_prices[stock_prices.Target.isnull() == False]['Target']))


# **Kurtosis:** In probability theory and statistics, kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random variable. Like skewness, kurtosis describes the shape of a probability distribution and there are different ways of quantifying it for a theoretical distribution and corresponding ways of estimating it from a sample from a population. Different measures of kurtosis may have different interpretations.
# 
# The standard measure of a distribution's kurtosis, originating with Karl Pearson, is a scaled version of the fourth moment of the distribution. This number is related to the tails of the distribution, not its peak; hence, the sometimes-seen characterization of kurtosis as "peakedness" is incorrect. For this measure, higher kurtosis corresponds to greater extremity of deviations (or outliers), and not the configuration of data near the mean.

# In[ ]:


print('Kurtosis: ', kurtosis(stock_prices[stock_prices.Target.isnull() == False]['Target']))


# In[ ]:


fig = px.histogram(stock_prices[stock_prices.Target.isnull() == False], x='Target', nbins = 1000, color_discrete_sequence = [px.colors.sequential.Viridis[5]])
fig.add_vline(x=stock_prices[stock_prices.Target.isnull() == False]['Target'].mean(), line_width=3, line_dash="dash", line_color = px.colors.qualitative.Plotly[1])

fig.update_xaxes(range = [-0.3,0.3])

# General Styling
fig.update_layout(height=400, bargap=0.2,                  
                  margin=dict(r=30,l=100, t=100, b=0),
                  plot_bgcolor='rgb(242,242,242)',
                  #paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
#Title
text = [
    "<span style='font-size:35px; font-family:Times New Roman'>Target Distribution</span>"
]
annotation_helper(fig, text, -0.043, 1.4, [0.095,0.095,0.065],ref="paper", width=1300)

#Subtitle
text = [
    "<span style='font-size:14px; font-family:Helvetica'> Target seems to have a <b>simmetric</b> distribution. Also, it has a large kurtosis. For investors, high kurtosis of the return distribution implies the investor will experience occasional extreme</span>",
    "<span style='font-size:14px; font-family:Helvetica'> returns (either positive or negative). This phenomenon is known as <b>kurtosis risk</b>.</span>",
]
annotation_helper(fig, text, -0.043, 1.225, [0.075,0.05,0.065],ref="paper", width=1300)

fig.show()


# Now, we are going to make a brief study of the target feature. We'll begin by plotting an OHLCV chart for one of the stocks of the dataset. For a decent understanding of these charts, it is preferable to look at them in depth. In the following, therefore, we will see how to interpret them without making any mistakes. 
# 
# **Detail of candlestick charts**: In order to start creating and interpreting a candlestick chart, the first thing to know is that the data contains price highs, lows, opens and closes. The empty and coloured portions of the bars, are called the "body". The long thin lines above and below the body represent the upward or downward ranges and are generally referred to as "shadows", "highlights" or "tails". If the lines are placed at the top of the body, this will indicate the rising and closing price, while the line at the bottom of the chart will indicate the falling and closing price at that position. 
# 
# ![](https://www.avatrade.es/wp-content/uploads/2017/04/candlestick_1.png)
# 
# The colours of the body of the candlestick vary depending on the broker, showing green or blue if the price is rising, and red if the price is falling. Longer versus shorter bodies will indicate buying or selling pressure among traders. Short bodies represent that little movement is taking place. One of the most reliable and popular patterns in chart analysis is the head and shoulders pattern. This is a pattern that indicates a change in trajectory. That is to say that its formation will signal that the current trend is soon to register a change, either from upward to downward or vice versa. There are two versions of the head and shoulders pattern:
# 
# ![](https://www.avatrade.es/wp-content/uploads/2017/04/head-shoulders.png)
# 
# * **Head and shoulders top.** Usually formed at the peak of an uptrend and indicates that the price of the asset is ready to fall once the pattern is completed.
# * **Head and shoulders base (or head and shoulders reversal)**. Usually formed during a downtrend, it indicates that the price of the asset is ready to move upwards.
# 
# This following chart function was taken from [JPX - Detailed EDA.](https://www.kaggle.com/code/abaojiang/jpx-detailed-eda)

# In[ ]:


def plot_candle_with_target(stock_code, prime=True):    
    df_ = stock_prices.copy() if prime else df_prices_sec.copy()
    df_ = df_[df_['SecuritiesCode'] == stock_code]
    dates = df_['Date'].values
    ohlc = {
        'open': df_['Open'].values, 
        'high': df_['High'].values, 
        'low': df_['Low'].values, 
        'close': df_['Close'].values
    }
    vol = df_['Volume'].values
    target = df_['Target'].values
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, x_title='Date')
    fig.add_trace(go.Candlestick(x=dates, name='OHLC', **ohlc),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=dates, y=vol, name='Volume'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=target, name='Target', marker = dict(color = px.colors.sequential.Viridis[5])),
                  row=3, col=1)
    
    fig.update(layout_xaxis_rangeslider_visible=False)
    
    # General Styling
    fig.update_layout(height=600, bargap=0.2,
                      margin=dict(r=30,l=100, t=150, b=0),
                      plot_bgcolor='rgb(242,242,242)',
                      #paper_bgcolor = 'rgb(242,242,242)',
                      font=dict(family="Times New Roman", size= 14),
                      hoverlabel=dict(font_color="floralwhite"),
                      showlegend=False)
    #Title
    text = [
        "<span style='font-size:35px; font-family:Times New Roman'>CandleStick Chart with target series</span>"
    ]
    annotation_helper(fig, text, -0.043, 1.35, [0.095,0.095,0.065],ref="paper", width=1300)
    #Subtitle
    text = [
        "<span style='font-size:14px; font-family:Helvetica'> CandleStick chart includes several key pieces of information per each data point recorded in the chart. Vertical lines of both illustrate the price ranges of a trading period, while the body</span>",
        "<span style='font-size:14px; font-family:Helvetica'> of the candlestick is used in different colours to represent the changes in the market over that time period. Stock exchange trading volume is the number of shares traded over a given</span>",
        "<span style='font-size:14px; font-family:Helvetica'> period of time, or in other words the number of traded stocks over that period of time.</span>",
    ]
    annotation_helper(fig, text, -0.043, 1.225, [0.05,0.05,0.065],ref="paper", width=1300)
    
    fig.show()
    
plot_candle_with_target(stock_prices['SecuritiesCode'].unique()[240])


# 📌 **Interpret:** as we can appreciate above the more drastic changes in stock values are, the higher values of volume we have. Let's study **some statistical target information depending on each of the stocks** now, for each of the available stocks. 
# 
# <div style="color:white;display:fill;
#             background-color:lightgrey;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>Target per Stock</b></p>
# </div>
# 

# In[ ]:


target_mean_per_stock = stock_prices.groupby(['SecuritiesCode'])['Target'].mean()
target_mean_mean = target_mean_per_stock.mean()
print('Skewness: ', skew(target_mean_per_stock.values))
print('Kurtosis: ', kurtosis(target_mean_per_stock.values))


# In[ ]:


target_mean = pd.DataFrame(target_mean_per_stock)
fig = px.histogram(target_mean, x='Target', nbins = 1000, color_discrete_sequence = [px.colors.sequential.Viridis[5]])
fig.add_vline(x=target_mean_mean, line_width=3, line_dash="dash", line_color = px.colors.qualitative.Plotly[1])
# General Styling
fig.update_layout(height=400, bargap=0.2,                  
                  margin=dict(r=30,l=100, t=100, b=0),
                  plot_bgcolor='rgb(242,242,242)',
                  #paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
#Title
text = [
    "<span style='font-size:35px; font-family:Times New Roman'>Target mean per stock Distribution</span>"
]
annotation_helper(fig, text, -0.043, 1.4, [0.095,0.095,0.065],ref="paper", width=1300)

#Subtitle
text = [
    "<span style='font-size:14px; font-family:Helvetica'> Target mean is a <b>right-skewed</b> distribution. Also, it has a large kurtosis. For investors, high kurtosis of the return distribution implies the investor will experience occasional extreme</span>",
    "<span style='font-size:14px; font-family:Helvetica'> returns (either positive or negative). This phenomenon is known as <b>kurtosis risk</b>.</span>",
]
annotation_helper(fig, text, -0.043, 1.225, [0.075,0.05,0.065],ref="paper", width=1300)

fig.show()


# Hereafter, we'll focus on analysing **target standard desviation**. This has the aim of studying how dispersed are the values of the target.

# In[ ]:


target_std_per_stock = stock_prices.groupby(['SecuritiesCode'])['Target'].std()
target_std_mean = target_std_per_stock.mean()
print('Skewness: ', skew(target_std_per_stock.values))
print('Kurtosis: ', kurtosis(target_std_per_stock.values))


# In[ ]:


target_std = pd.DataFrame(target_std_per_stock)
fig = px.histogram(target_std, x='Target', nbins = 1000, color_discrete_sequence = [px.colors.sequential.Viridis[5]])
fig.add_vline(x=target_std_mean, line_width=3, line_dash="dash", line_color = px.colors.qualitative.Plotly[1])
# General Styling
fig.update_layout(height=400, bargap=0.2,                  
                  margin=dict(r=30,l=100, t=100, b=0),
                  plot_bgcolor='rgb(242,242,242)',
                  #paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
#Title
text = [
    "<span style='font-size:35px; font-family:Times New Roman'>Target std per stock Distribution</span>"
]
annotation_helper(fig, text, -0.043, 1.4, [0.095,0.095,0.065],ref="paper", width=1300)

#Subtitle
text = [
    "<span style='font-size:14px; font-family:Helvetica'> Target std is a <b>right-skewed</b> distribution. Also, it has a large kurtosis. For investors, high kurtosis of the return distribution implies the investor will experience occasional extreme</span>",
    "<span style='font-size:14px; font-family:Helvetica'> returns (either positive or negative). This phenomenon is known as <b>kurtosis risk</b>.</span>",
]
annotation_helper(fig, text, -0.043, 1.225, [0.075,0.05,0.065],ref="paper", width=1300)

fig.show()


# In[ ]:


aux = target_mean.sort_values(by='Target',ascending=False)
print('Stock ', aux.index[0], ' has the maximum target mean. Its value is: ', aux.values[0])
print('Stock ', aux.index[-1], ' has the minimum target mean. Its value is: ', aux.values[-1])
aux = target_std.sort_values(by='Target',ascending=False)
print('Stock ', aux.index[0], ' has the maximum target std. Its value is: ', aux.values[0])


# In[ ]:


plot_candle_with_target(4169)


# In[ ]:


plot_candle_with_target(4883)


# 📌 **Interpret:** we can observe that **stock 4169 value** has experienced an **incremental rise** over recording time. In contrast, **stock 4883**, having the minimum target mean, has been losing value. In fact, we can see some **big volume values at the beginning of the recording time, due to the collapse in the value of this**. August 28th presented a difference of ¡400! between opening and closing value. 
# 
# * Stock 4169 belongs to **Enechange Ltd**. This, is a Japan-based company mainly engaged in the energy platform business and the energy data business. For more info -> [info](https://enechange.co.jp/en/). 
# * Stock 4883 belongs to **Modalis Therapeutics Corp**. More info -> [info](https://www.modalistx.com/en/)
# 
# <div style="color:white;display:fill;
#             background-color:lightgrey;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>Target per Date</b></p>
# </div>
# 
# We have checked distributions of both target standard desviation and mean, depending on each of the stocks. Now, we'll analyse them for each recording date.

# In[ ]:


target_mean_per_date = stock_prices.groupby(['Date'])['Target'].mean()
target_mean_mean = target_mean_per_date.mean()

from scipy.stats import skew, kurtosis
print('Skewness: ', skew(target_mean_per_date.values))
print('Kurtosis: ', kurtosis(target_mean_per_date.values))


# In[ ]:


target_mean = pd.DataFrame(target_mean_per_date)
fig = px.histogram(target_mean, x='Target', nbins = 1000, color_discrete_sequence = [px.colors.sequential.Viridis[5]])
fig.add_vline(x=target_mean_mean, line_width=3, line_dash="dash", line_color = px.colors.qualitative.Plotly[1])
# General Styling
fig.update_layout(height=400, bargap=0.2,                  
                  margin=dict(r=30,l=100, t=100, b=0),
                  plot_bgcolor='rgb(242,242,242)',
                  #paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
#Title
text = [
    "<span style='font-size:35px; font-family:Times New Roman'>Target mean per Date Distribution</span>"
]
annotation_helper(fig, text, -0.043, 1.4, [0.095,0.095,0.065],ref="paper", width=1300)

#Subtitle
text = [
    "<span style='font-size:14px; font-family:Helvetica'> Target mean per <b> date</b> seems to be <b>almost simmetric</b> distributed. However, it again has a large kurtosis. Hence, outliers should be carefully handled. </span>",
]
annotation_helper(fig, text, -0.043, 1.2, [0.075,0.05,0.065],ref="paper", width=1300)

fig.show()


# In[ ]:


aux = target_mean.sort_values(by='Target',ascending=False)
print(aux.index[0], ' has the maximum target mean. Its value is: ', aux.values[0])
print(aux.index[-1], ' has the minimum target mean. Its value is: ', aux.values[-1])
target_std_per_date = stock_prices.groupby(['Date'])['Target'].std()
target_std = pd.DataFrame(target_std_per_date)
aux = target_std.sort_values(by='Target',ascending=False)
print(aux.index[0], ' has the maximum target std. Its value is: ', aux.values[0])


# Let's find out which days were these. Let's begin with the trading date having the highest value. This is `2018-12-25`, a date that is part of the Japanese Christmas season. The possible reason fir this could be the **biggest point increase in history of Wall Street**
# 
# Talking about `2020-03-05`, this date coincides with the beginning of the covid-19 pandemic. Japan, like the rest of the world, began to experience a significant rise in covid-19 cases in its population. This would lead to the IOC announcing the postponement of the Olympics a couple of dates later. It was concretely, in `2020-03-17` when stocks values suffer a major falling. Hence, we can go to see what happened in `2020-03-5`, the $T+2$ and $T+1$ trading dates, as those dates are the ones taking into account for calculating the target. First, we are going to see what is the **percentage of stocks which suffered this failing**, in terms of its closing prices.

# In[ ]:


import datetime
major_failing_date = stock_prices[stock_prices.Date == datetime.datetime(2020,3,5)]
next_date = stock_prices[stock_prices.Date == datetime.datetime(2020,3,6)]
tmp = pd.DataFrame({'SecuritiesCode':next_date['SecuritiesCode'].values, 'Current Date Close': major_failing_date['Close'].values, 'Next Date Close': next_date['Close'].values,
                    'Close Difference':next_date['Close'].values - major_failing_date['Close'].values})
print('Percentage of stocks with Close Price failing between 2020-3-5 and T+1: ', len(tmp[tmp['Close Difference'] < 0])/tmp['Close Difference'].shape[0])

next_date2 = stock_prices[(stock_prices.Date == datetime.datetime(2020,3,9)) & (stock_prices.SecuritiesCode.isin(tmp.SecuritiesCode))]
tmp2 = pd.DataFrame({'SecuritiesCode':next_date['SecuritiesCode'].values, 'Current Date Close': next_date['Close'].values, 'Next Date Close': next_date2['Close'].values,
                    'Close Difference':next_date2['Close'].values - next_date['Close'].values})
print('Percentage of stocks with Close Price failing between 2020-3-6 and T+1: ', len(tmp2[tmp2['Close Difference'] < 0])/tmp2['Close Difference'].shape[0])


# In[ ]:


fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing = 0.05)
fig.add_trace(go.Bar(x=tmp[tmp['Close Difference'] >= 0].index, y = tmp[tmp['Close Difference'] >= 0]['Close Difference'], marker = dict(color = px.colors.sequential.Blugrn[6])), row=1, col=1)
fig.add_trace(go.Bar(x=tmp[tmp['Close Difference'] < 0].index, y = tmp[tmp['Close Difference'] < 0]['Close Difference'], marker = dict(color = 'red')), row=1, col=1)

fig.add_trace(go.Bar(x=tmp2[tmp2['Close Difference'] >= 0].index, y = tmp2[tmp2['Close Difference'] >= 0]['Close Difference'], marker = dict(color = px.colors.sequential.Blugrn[6])), row=2, col=1)
fig.add_trace(go.Bar(x=tmp2[tmp2['Close Difference'] < 0].index, y = tmp2[tmp2['Close Difference'] < 0]['Close Difference'], marker = dict(color = 'red')), row=2, col=1)
# General Styling
fig.update_layout(height=600, bargap=0.1,                  
                  margin=dict(r=30,l=100, t=100, b=0),
                  plot_bgcolor='rgb(242,242,242)',
                  #paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
#Title
text = [
    "<span style='font-size:35px; font-family:Times New Roman'>Closing Price Difference per Stocks</span>"
]
annotation_helper(fig, text, -0.043, 1.225, [0.095,0.095,0.065],ref="paper", width=1300)

#Subtitle
text = [
    "<span style='font-size:14px; font-family:Helvetica'> We are plotting difference between closing price values in <b>2020-3-5</b>, and <b>T+1, T+2 trading dates</b>. We do this to examine how stocks suffered Covid-19 pandemic beginning. </span>",
    "<span style='font-size:14px; font-family:Helvetica'> As plotly allows to make interactive charts, for a better analysis zoom this parts of interest. </span>",
    
]
annotation_helper(fig, text, -0.043, 1.135, [0.04,0.05,0.065],ref="paper", width=1300)

fig.show()


# 📌 **Interpret:** as we can appreciate closing prices suffered an outrageous decreasing. For example, check the middle stock that appears in both charts. Between $T$ and $T+1$ its value decreases a lot. But, this difference is even much greater betwen $T+1$ and $T+2$. On the other hand, there are very few stocks that increased his closing values. Let's examine both, stocks with the major failings and the ones that increase theis closing prices. 

# In[ ]:


tmp2['Close Difference'] = tmp2['Close Difference'] + tmp['Close Difference']
tmp2.sort_values(by='Close Difference').head()


# * `6273`: This stock belongs to **SMC Corporation (SMC 株式会社, SMC Kabushiki-gaisha)**. This is a Japanese TOPIX Large 70 company founded in 1959 as Sintered Metal Corporation, which **specializes in pneumatic control engineering** to support industrial automation. SMC develops a broad range of control systems and equipment, such as directional control valves, actuators, and air line equipment, to support diverse applications. SMC's head office is located in Sotokanda, Chiyoda-ku, Tokyo, Japan. The company has a global engineering network, with technical facilities in the United States, Europe and China, as well as Japan. Key production facilities are located in China and Singapore, and local production facilities are in United States, Mexico, Brazil, Europe, India, Korea and Australia.
# 
# * `6861`: this stocks belongs to **Keyence Corporation (キーエンス, Kīensu)**. It is a **direct sales organization** that develops and manufactures automation sensors, vision systems, barcode readers, laser markers, measuring instruments, and digital microscopes. Keyence is fabless - although it is a manufacturer; it specializes solely in product planning and development and does not manufacture the final products. Keyence products are manufactured at qualified contract manufacturing companies.

# In[ ]:


stock_list[stock_list.SecuritiesCode.isin([6273,4488,4880,6861,6146])].style.set_properties(subset=['Name'], **{'background-color': 'lightgrey'})


# <div style="color:white;display:fill;
#             background-color:lightgrey;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>SMA (Simple Moving Average) and EMA (Exponential Moving Average)</b></p>
# </div>
# 
# This part was taken from [Useful features in Predicting Stock Prices](https://www.kaggle.com/code/riteshsinha/useful-features-in-predicting-stock-prices#Analyzing-SMA-(Simple-Moving-Averages)-and-EMA-(Exponential-Moving-Averages)). SMA and EMA are some well know pointers when it comes to Price Tracking and making decisions based on them. These methods help in identifying trends related to stock prices. While as the name suggests, SMA are jsut the average of a period where as EMA attach weights to the calculation and sensitive to recent price movements. Let's make some charts in order to see how they perfom. We are gonna plot moving averages for 20, 50, and 100 period-days

# In[ ]:


tmp = stock_prices[stock_prices.SecuritiesCode == stock_prices.SecuritiesCode.unique()[0]].copy()
tmp['SMA-20'] = tmp['Close'].rolling(window = 20).mean()
tmp['SMA-50'] = tmp['Close'].rolling(window = 50).mean()
tmp['SMA-100'] = tmp['Close'].rolling(window = 100).mean()
#EMA
tmp['EMA-20'] = tmp['Close'].ewm(span = 20, adjust = False).mean()
tmp['EMA-50'] = tmp['Close'].ewm(span = 50, adjust = False).mean()
tmp['EMA-100'] = tmp['Close'].ewm(span = 100, adjust = False).mean()

fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Scatter(x=tmp['Date'].values, y=tmp['Close'], mode='lines', name='Close'), row=1, col=1)
fig.add_trace(go.Scatter(x=tmp['Date'].values, y=tmp['SMA-20'], mode='lines', name='SMA-20'), row=1, col=1)
fig.add_trace(go.Scatter(x=tmp['Date'].values, y=tmp['SMA-50'], mode='lines', name='SMA-50'), row=1, col=1)
fig.add_trace(go.Scatter(x=tmp['Date'].values, y=tmp['SMA-100'], mode='lines', name='SMA-100'), row=1, col=1)

fig.add_trace(go.Scatter(x=tmp['Date'].values, y=tmp['Close'], mode='lines', name='Close'), row=1, col=2)
fig.add_trace(go.Scatter(x=tmp['Date'].values, y=tmp['EMA-20'], mode='lines', name='EMA-20'), row=1, col=2)
fig.add_trace(go.Scatter(x=tmp['Date'].values, y=tmp['EMA-50'], mode='lines', name='EMA-50'), row=1, col=2)
fig.add_trace(go.Scatter(x=tmp['Date'].values, y=tmp['EMA-100'], mode='lines', name='EMA-100'), row=1, col=2)

# General Styling
fig.update_layout(height=400, bargap=0.1,                  
                  margin=dict(r=30,l=100, t=100, b=0),
                  plot_bgcolor='rgb(242,242,242)',
                  legend = dict(bgcolor = 'rgb(242,242,242)'),
                  #paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=True)
#Title
text = [
    "<span style='font-size:35px; font-family:Times New Roman'>SMA and EMA Comparison</span>"
]
annotation_helper(fig, text, -0.043, 1.4, [0.095,0.095,0.065],ref="paper", width=1300)

#Subtitle
text = [
    "<span style='font-size:14px; font-family:Helvetica'> SMA and EMA are some well know pointers when it comes to Price Tracking and making decisions based on them. These methods help in identifying trends related to stock prices. </span>", 
    "<span style='font-size:14px; font-family:Helvetica'> While as the name suggests, SMA are jsut the average of a period where as EMA attach weights to the calculation and sensitive to recent price movements. </span>",    
]
annotation_helper(fig, text, -0.043, 1.25, [0.065,0.05,0.065],ref="paper", width=1300)

fig.show()


# <div style="color:white;display:fill;
#             background-color:lightgrey;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>Highest Correlated Stocks</b></p>
# </div>
# 
# On this following section we'll analyse which are the **most correlated stocks**. Concretely, we'll show the **top3 pairs** of stocks. As we can appreciate below, **stocks 8306 - 8316 are the most correlated ones**. They are followed by 9008 - 9007, and 5401 - 5411 having almost the same correlation value as the second one.

# In[ ]:


tmp = stock_prices[['Date','SecuritiesCode','Target']]
corr = tmp.set_index(['SecuritiesCode','Date']).unstack('SecuritiesCode').corr()
corr = corr[(corr < 1) & (abs(corr) > 0.75)].stack()
corr.sort_values(by='Target',ascending=False).head(5)


# Let's take a brief insight on which stocks are the ones belonging to the top3 pairs seen above.

# In[ ]:


stock_list[(stock_list.SecuritiesCode == 8306) | (stock_list.SecuritiesCode == 8316) | (stock_list.SecuritiesCode == 9008) | (stock_list.SecuritiesCode == 9007)].style.set_properties(subset=['Name','Section/Products','NewMarketSegment'], **{'background-color': 'lightgrey'})


# <div style="color:white;display:fill;
#             background-color:lightgrey;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>Time Series Analysis</b></p>
# </div>
# 
# In this following section, we'll study **how time series components of the stocks 8306 - 8316 have evolved during the recorded time**. If we assume an additive decomposition, then we can write $y_t=S_t+T_t+R_t$, where $y_t$ is the target, $S_t$ is the seasonal component, $T_t$ is the trend-cycle component and $R_t$ is the residual component, all at period $t$. Also,for a multiplicative decomposition, we have $y_t=S_t*T_t*R_t$.
# 
# The additive decomposition is the most appropriate if the magnitude of the seasonal fluctuations, or the variation around the trend-cycle, does not vary with the level of the time series. When the variation in the seasonal pattern, or the variation around the trend-cycle, appears to be proportional to the level of the time series, then a multiplicative decomposition is more appropriate. Multiplicative decompositions are common with economic time series.
# 
# #### **Trend**
# 
# The trend component of a time series represents a persistent, long-term change in the mean of the series. The trend is the slowest-moving part of a series, the part representing the largest time scale of importance. In a time series of product sales, an increasing trend might be the effect of a market expansion as more people become aware of the product year by year.
# 
# #### **Seasonality**
# 
# We say that a time series exhibits seasonality whenever there is a regular, periodic change in the mean of the series. Seasonal changes generally follow the clock and calendar - repetitions over a day, a week, or a year are common. Seasonality is often driven by the cycles of the natural world over days and years or by conventions of social behavior surrounding dates and times.
# 
# #### **Decomposition**

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
get_ipython().run_line_magic('matplotlib', 'inline')
tmp1 = stock_prices[stock_prices.SecuritiesCode == 8306]
tmp2 = stock_prices[stock_prices.SecuritiesCode == 8316]

decomp1 = seasonal_decompose(tmp1['Target'], period=365, model='additive', extrapolate_trend='freq')
decomp2 = seasonal_decompose(tmp2['Target'], period=365, model='additive', extrapolate_trend='freq')

fig, ax = plt.subplots(ncols=4, nrows=4, sharex=True, figsize=(22,16))
ax[0,0].set_title('Observed values for Target', fontsize=16)
decomp1.observed.plot(ax = ax[0,0], legend=False, color=px.colors.sequential.Viridis[4])
decomp2.observed.plot(ax = ax[0,0], legend=False, color=px.colors.qualitative.Plotly[4])

ax[0,1].set_title('Target Trend', fontsize=16)
decomp1.trend.plot(ax = ax[0,1],legend=False, color=px.colors.sequential.Viridis[4])
decomp2.trend.plot(ax = ax[0,1],legend=False, color=px.colors.qualitative.Plotly[4])

ax[0,2].set_title('Target Seasonality', fontsize=16)
decomp1.seasonal.plot(ax = ax[0,2],legend=False, color=px.colors.sequential.Viridis[4])
decomp2.seasonal.plot(ax = ax[0,2],legend=False, color=px.colors.qualitative.Plotly[4])

ax[0,3].set_title('Target Noise', fontsize=16)
decomp1.resid.plot(ax = ax[0,3],legend=False, color=px.colors.sequential.Viridis[4])
decomp2.resid.plot(ax = ax[0,3],legend=False, color=px.colors.qualitative.Plotly[4])

decomp1 = seasonal_decompose(tmp1[tmp1.Open.isnull() == False]['Open'], period=365, model='additive', extrapolate_trend='freq')
decomp2 = seasonal_decompose(tmp2[tmp2.Open.isnull() == False]['Open'], period=365, model='additive', extrapolate_trend='freq')

ax[1,0].set_title('Observed values for Open', fontsize=16)
decomp1.observed.plot(ax = ax[1,0], legend=False, color=px.colors.sequential.Viridis[4])
decomp2.observed.plot(ax = ax[1,0], legend=False, color=px.colors.qualitative.Plotly[4])

ax[1,1].set_title('Open Trend', fontsize=16)
decomp1.trend.plot(ax = ax[1,1],legend=False, color=px.colors.sequential.Viridis[4])
decomp2.trend.plot(ax = ax[1,1],legend=False, color=px.colors.qualitative.Plotly[4])

ax[1,2].set_title('Open Seasonality', fontsize=16)
decomp1.seasonal.plot(ax = ax[1,2],legend=False, color=px.colors.sequential.Viridis[4])
decomp2.seasonal.plot(ax = ax[1,2],legend=False, color=px.colors.qualitative.Plotly[4])

ax[1,3].set_title('Open Noise', fontsize=16)
decomp1.resid.plot(ax = ax[1,3],legend=False, color=px.colors.sequential.Viridis[4])
decomp2.resid.plot(ax = ax[1,3],legend=False, color=px.colors.qualitative.Plotly[4])

decomp1 = seasonal_decompose(tmp1[tmp1.Close.isnull() == False]['Close'], period=365, model='additive', extrapolate_trend='freq')
decomp2 = seasonal_decompose(tmp2[tmp2.Close.isnull() == False]['Close'], period=365, model='additive', extrapolate_trend='freq')

ax[2,0].set_title('Observed values for Close', fontsize=16)
decomp1.observed.plot(ax = ax[2,0], legend=False, color=px.colors.sequential.Viridis[4])
decomp2.observed.plot(ax = ax[2,0], legend=False, color=px.colors.qualitative.Plotly[4])

ax[2,1].set_title('Close Trend', fontsize=16)
decomp1.trend.plot(ax = ax[2,1],legend=False, color=px.colors.sequential.Viridis[4])
decomp2.trend.plot(ax = ax[2,1],legend=False, color=px.colors.qualitative.Plotly[4])

ax[2,2].set_title('Close Seasonality', fontsize=16)
decomp1.seasonal.plot(ax = ax[2,2],legend=False, color=px.colors.sequential.Viridis[4])
decomp2.seasonal.plot(ax = ax[2,2],legend=False, color=px.colors.qualitative.Plotly[4])

ax[2,3].set_title('Close Noise', fontsize=16)
decomp1.resid.plot(ax = ax[2,3],legend=False, color=px.colors.sequential.Viridis[4])
decomp2.resid.plot(ax = ax[2,3],legend=False, color=px.colors.qualitative.Plotly[4])

decomp1 = seasonal_decompose(tmp1[tmp1.Volume.isnull() == False]['Volume'], period=365, model='additive', extrapolate_trend='freq')
decomp2 = seasonal_decompose(tmp2[tmp2.Volume.isnull() == False]['Volume'], period=365, model='additive', extrapolate_trend='freq')

ax[3,0].set_title('Observed values for Volume', fontsize=16)
decomp1.observed.plot(ax = ax[3,0], legend=False, color=px.colors.sequential.Viridis[4])
decomp2.observed.plot(ax = ax[3,0], legend=False, color=px.colors.qualitative.Plotly[4])

ax[3,1].set_title('Volume Trend', fontsize=16)
decomp1.trend.plot(ax = ax[3,1],legend=False, color=px.colors.sequential.Viridis[4])
decomp2.trend.plot(ax = ax[3,1],legend=False, color=px.colors.qualitative.Plotly[4])

ax[3,2].set_title('Volume Seasonality', fontsize=16)
decomp1.seasonal.plot(ax = ax[3,2],legend=False, color=px.colors.sequential.Viridis[4])
decomp2.seasonal.plot(ax = ax[3,2],legend=False, color=px.colors.qualitative.Plotly[4])

ax[3,3].set_title('Volume Noise', fontsize=16)
decomp1.resid.plot(ax = ax[3,3],legend=False, color=px.colors.sequential.Viridis[4])
decomp2.resid.plot(ax = ax[3,3],legend=False, color=px.colors.qualitative.Plotly[4])


# 📌 **Interpret:** beginning with the target feature, we can observe how they overlap almost perfectly. That's the reason of such a high value of correlation between them. Another interesting finding is that they both have a ridiculously big fall of stock value at the same period. In this same period, volume values are pretty much higher (obviously due to the sale of shares due to loss of value).

# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.2| Secondary Stock Prices File</b></p>
# </div>
# 
# 📌 **File Description:**
# The core dataset contains on the 2,000 most commonly traded equities but many less liquid securities are also traded on the Tokyo market. This file contains data for those securities, which aren't scored but may be of interest for assessing the market as a whole. These explanations have been taken from **data_specifications/stock_price_spce.csv**
# * `RowId`: unique ID of price records, the combination of `Date` and `SecuritiesCode`.
# * `Date`: trade date.
# * `SecuritiesCode`: local securities code.
# * `Open`: first traded price on a day.
# * `High`: highest traded price on a day.
# * `Low`: lowest traded price on a day.
# * `Close`: last traded price on a day.
# * `Volume`: number of traded stocks on a day.
# * `AdjustmentFactor`: to calculate theoretical price/volume when split/reverse-split happens (NOT including dividend/allot...)
# * `SupervisionFlag`: Flag of securities under supervision and securities to be delisted -> [info](https://www.jpx.co.jp/english/listing/market-alerts/supervision/00-archives/index.html).
# * `ExpectedDividend`: Expected dividend value for ex-right date. This value is recorded 2 business days before ex-dividend...
# * `Target`: Change ratio of adjusted closing price between t+2 and t+1 where t+0 is TradeDate.
# 
# 📌 **Liquidity**: The concept of liquidity is closely linked to the world of trading and the stock market. Liquidity is the ease with which an asset can be bought or sold in the market without affecting its price. We can also speak of market liquidity. When an asset is in high demand it is said to have high liquidity, and therefore it will be easier to get someone to buy or sell it.
# 
# 

# In[ ]:


pp.ProfileReport(secondary_stock_prices)


# <div style="color:white;display:fill;
#             background-color:#235f83;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.2.1 | SecuritiesCode</b></p>
# </div>
# 
# We'll start the analysis of this dataset by examinating whether are repeated stocks in this file and the previous and main one. We'll also show how many unique stocks we have, and whether the dataset has missing values or not. 

# In[ ]:


import numba
print('Number of unique stocks in this file: ', len(secondary_stock_prices['SecuritiesCode'].unique()))
def repeated_stocks():
    repeated = 0
    for c in secondary_stock_prices['SecuritiesCode'].unique():
        if c in stock_prices['SecuritiesCode'].unique():
            print('Repeated Stock')
            repeated = 1
    if repeated == 0:
        print('No repeated stocks on both files.')
        
repeated_stocks()
secondary_stock_prices.isnull().sum()


# <div style="color:white;display:fill;
#             background-color:#235f83;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.2.2 | Target Feature</b></p>
# </div>
# <div style="color:white;display:fill;
#             background-color:lightgrey;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>Target Statistical Analysis</b></p>
# </div>
# 
# In this section we'll repeat the same process we did for the stock_prices dataset. This means that we are starting with its statistical information. Both kurtosis and skewness, and its distribution plot. As we can observe below, kurtosis feature value is ridiculously high. 

# In[ ]:


print('Skewness: ', skew(secondary_stock_prices[secondary_stock_prices.Target.isnull() == False]['Target'].values))
print('Kurtosis: ', kurtosis(secondary_stock_prices[secondary_stock_prices.Target.isnull() == False]['Target'].values))


# In[ ]:


fig = px.histogram(secondary_stock_prices[secondary_stock_prices.Target.isnull() == False], x='Target', nbins = 10000, color_discrete_sequence = [px.colors.sequential.Viridis[5]])
fig.add_vline(x=secondary_stock_prices[secondary_stock_prices.Target.isnull() == False]['Target'].mean(), line_width=3, line_dash="dash", line_color = px.colors.qualitative.Plotly[1])

fig.update_xaxes(range = [-0.3,0.3])

# General Styling
fig.update_layout(height=400, bargap=0.2,                  
                  margin=dict(r=30,l=100, t=100, b=0),
                  plot_bgcolor='rgb(242,242,242)',
                  #paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
#Title
text = [
    "<span style='font-size:35px; font-family:Times New Roman'>Target Distribution</span>"
]
annotation_helper(fig, text, -0.043, 1.4, [0.095,0.095,0.065],ref="paper", width=1300)

#Subtitle
text = [
    "<span style='font-size:14px; font-family:Helvetica'> Target has <b>right-skewed</b> distribution. Also, it has a large kurtosis. For investors, high kurtosis of the return distribution implies the investor will experience occasional extreme</span>",
    "<span style='font-size:14px; font-family:Helvetica'> returns (either positive or negative). This phenomenon is known as <b>kurtosis risk</b>.</span>",
]
annotation_helper(fig, text, -0.043, 1.225, [0.075,0.05,0.065],ref="paper", width=1300)

fig.show()


# <div style="color:white;display:fill;
#             background-color:lightgrey;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>Target per Stock Analysis</b></p>
# </div>

# In[ ]:


target_mean_per_stock = secondary_stock_prices.groupby(['SecuritiesCode'])['Target'].mean()
target_mean_mean = target_mean_per_stock.mean()
print('Skewness: ', skew(target_mean_per_stock.values))
print('Kurtosis: ', kurtosis(target_mean_per_stock.values))


# In[ ]:


target_mean = pd.DataFrame(target_mean_per_stock)
fig = px.histogram(target_mean, x='Target', nbins = 1000, color_discrete_sequence = [px.colors.sequential.Viridis[5]])
fig.add_vline(x=target_mean_mean, line_width=3, line_dash="dash", line_color = px.colors.qualitative.Plotly[1])

# General Styling
fig.update_layout(height=400, bargap=0.2,                  
                  margin=dict(r=30,l=100, t=100, b=0),
                  plot_bgcolor='rgb(242,242,242)',
                  #paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
#Title
text = [
    "<span style='font-size:35px; font-family:Times New Roman'>Target mean per stock Distribution</span>"
]
annotation_helper(fig, text, -0.043, 1.4, [0.095,0.095,0.065],ref="paper", width=1300)

#Subtitle
text = [
    "<span style='font-size:14px; font-family:Helvetica'> Although target distribution seems to be simmetric, it is not. We just have to look at the skewness value shown before. This is caused by the large kurtosis value this distribution has.</span>",
]
annotation_helper(fig, text, -0.043, 1.225, [0.075,0.05,0.065],ref="paper", width=1300)

fig.show()


# In[ ]:


target_std_per_stock = secondary_stock_prices.groupby(['SecuritiesCode'])['Target'].std()
target_std_mean = target_std_per_stock.mean()
print('Skewness: ', skew(target_std_per_stock.values))
print('Kurtosis: ', kurtosis(target_std_per_stock.values))


# In[ ]:


target_std = pd.DataFrame(target_std_per_stock)
fig = px.histogram(target_std, x='Target', nbins = 1000, color_discrete_sequence = [px.colors.sequential.Viridis[5]])
fig.add_vline(x=target_std_mean, line_width=3, line_dash="dash", line_color = px.colors.qualitative.Plotly[1])
# General Styling
fig.update_layout(height=400, bargap=0.2,                  
                  margin=dict(r=30,l=100, t=100, b=0),
                  plot_bgcolor='rgb(242,242,242)',
                  #paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
#Title
text = [
    "<span style='font-size:35px; font-family:Times New Roman'>Target std per stock Distribution</span>"
]
annotation_helper(fig, text, -0.043, 1.4, [0.095,0.095,0.065],ref="paper", width=1300)

#Subtitle
text = [
    "<span style='font-size:14px; font-family:Helvetica'> Target std is a really <b>right-skewed</b> distribution. As the previous distributions, its kurtosis value keeps being large. But not as much as before. </span>",
]
annotation_helper(fig, text, -0.043, 1.225, [0.075,0.05,0.065],ref="paper", width=1300)

fig.show()


# <div style="color:white;display:fill;
#             background-color:lightgrey;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>Target per Date Analysis</b></p>
# </div>
# 
# We have checked distributions of both target standard desviation and mean, depending on each of the stocks. Now, we'll analyse them for each recording date.

# In[ ]:


target_mean_per_date = secondary_stock_prices.groupby(['Date'])['Target'].mean()
target_mean_mean = target_mean_per_date.mean()

from scipy.stats import skew, kurtosis
print('Skewness: ', skew(target_mean_per_date.values))
print('Kurtosis: ', kurtosis(target_mean_per_date.values))


# In[ ]:


target_mean = pd.DataFrame(target_mean_per_date)
fig = px.histogram(target_mean, x='Target', nbins = 1000, color_discrete_sequence = [px.colors.sequential.Viridis[5]])
fig.add_vline(x=target_mean_mean, line_width=3, line_dash="dash", line_color = px.colors.qualitative.Plotly[1])
# General Styling
fig.update_layout(height=400, bargap=0.2,                  
                  margin=dict(r=30,l=100, t=100, b=0),
                  plot_bgcolor='rgb(242,242,242)',
                  #paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
#Title
text = [
    "<span style='font-size:35px; font-family:Times New Roman'>Target mean per Date Distribution</span>"
]
annotation_helper(fig, text, -0.043, 1.4, [0.095,0.095,0.065],ref="paper", width=1300)

#Subtitle
text = [
    "<span style='font-size:14px; font-family:Helvetica'> Target mean per <b> date</b> seems to be <b>almost simmetric</b> distributed. However, it again has a large kurtosis. Hence, outliers should be carefully handled. </span>",
]
annotation_helper(fig, text, -0.043, 1.2, [0.075,0.05,0.065],ref="paper", width=1300)

fig.show()


# In[ ]:


aux = target_mean.sort_values(by='Target',ascending=False)
print(aux.index[0], ' has the maximum target mean. Its value is: ', aux.values[0])
print(aux.index[-1], ' has the minimum target mean. Its value is: ', aux.values[-1])
target_std_per_date = secondary_stock_prices.groupby(['Date'])['Target'].std()
target_std = pd.DataFrame(target_std_per_date)
aux = target_std.sort_values(by='Target',ascending=False)
print(aux.index[0], ' has the maximum target std. Its value is: ', aux.values[0])


# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.3| Stock List File</b></p>
# </div>
# 
# **File Description:**
# Mapping between the `SecuritiesCode` and company names, plus general information about which industry the company is in. Below, I have added the definition of each of the features. These explanations have been taken from **data_specifications/stock_list_spce.csv**
# * `SecuritiesCode`: local securities code.
# * `EffectiveDate`: the effective date.
# * `Name`: Name of security.
# * `Section/Products`: Section/Product.
# * `NewMarketSegment`: New market segment effective from 2022-04-04 (as of 15:30 JST on Mar 11 2022) [ref.](https://www.jpx.c).
# 
# **Concepts that should be known in order to understand properly this feature:** 
# > The **primary market or issuance market** is the financial market in which marketable securities are issued and in which securities are thus transferred for the first time. Securities markets are divided into primary and secondary markets, separating the securities issuance phase and the subsequent trading phase. It is called primary market when the financial assets exchanged are newly created, the suppliers of securities in the market are the entities in need of financial resources and that come to this market to issue their securities, on the side of the demanders are the investors, who with surplus financial resources come to these markets to acquire securities. The securities are issued in the primary market, which is used to raise savings and therefore raises new financing, and the securities already acquired are subsequently traded in the secondary market, which is a second-hand or resale market. **The placing of shares on the primary market always involves an increase in a company's share capital. Secondary market shares, on the other hand, are securities that already exist in the company and do not represent an increase in the company's share capital.**
# 
# * `33SectorCode`: 33 Sector Name [ref](https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_13_sector.pdf).
# * `33SectorName`: 33 Sector Name [ref](https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_13_sector.pdf)
# * `17SectorCode`: 17 Sector Code [ref](https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_13_sector.pdf).
# * `17SectorName`: 17 Sector Name [ref](https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_13_sector.pdf).
# * `NewIndexSeriesSizeCode`: TOPIX New Index Series code [ref](https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_).
# * `NewIndexSeriesSize`: TOPIX New Index Series Name [ref](https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_).
# * `TradeDate`: Trade date to calculate MarketCapitalization.
# * `Close`: Close price to calculate MarketCapitalization.
# * `IssuedShares`: Issued shares.
# * `MarketCapitalization`: Market capitalization on Dec 3 2021.
# * `Universe0`: a flag of prediction target universe (top 2000 stocks by market capitalization).

# In[ ]:


pp.ProfileReport(stock_list)


# **Early Insights:** analysing the previous report we are able to conclude the following: 
# 
# * More than a half of the stocks belongs to companies from **First Sector**. Concretely, the ones selling **domestic** products account for **52% of the total stocks**
# * 90.9% of companies having stocks in our dataset offer **domestic** products.
# * Most stocks belong to the **1st Market**. Standard Market follows. 
# 
# <div style="color:white;display:fill;
#             background-color:#235f83;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.3.1 | 33SectorName Feature</b></p>
# </div>

# In[ ]:


fig = make_subplots(rows = 1, cols = 2, specs=[[{'type':'pie'},{'type':'pie'}]], subplot_titles=['17 Sector Name','17 Sector Code'])

tmp = pd.DataFrame(stock_list['17SectorName'].value_counts()).reset_index()
fig.add_trace(go.Pie(values=tmp['17SectorName'], labels=tmp['index']), row=1, col=1)

tmp = pd.DataFrame(stock_list['17SectorName'].value_counts()).reset_index()
fig.add_trace(go.Pie(values=tmp['17SectorName'], labels=tmp['index']), row=1, col=2)

fig.update_traces(textposition='inside', textinfo='percent+label')
# General Styling
fig.update_layout(height=400, bargap=0.2,                  
                  margin=dict(r=30,l=100, b=0),
                  plot_bgcolor='rgb(242,242,242)',
                  #paper_bgcolor = 'rgb(242,242,242)',
                  font=dict(family="Times New Roman", size= 14),
                  hoverlabel=dict(font_color="floralwhite"),
                  showlegend=False)
#Title
text = [
    "<span style='font-size:35px; font-family:Times New Roman'>Sector Name - Sector Code</span>"
]
annotation_helper(fig, text, -0.043, 1.35, [0.095,0.095,0.065],ref="paper", width=1300)

fig.show()


# # <b>3 <span style='color:#3f4d63'>|</span> Feature Engineering</b>
# 
# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.1 | Local CV Scoring Function</b></p>
# </div>

# **Still working on it ...**
# 
# * More EDA
# * Some FE stuff
# * Some Feature Selection Techniques
# 
# Keep an eye on !!!

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




