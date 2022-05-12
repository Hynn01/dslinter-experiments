#!/usr/bin/env python
# coding: utf-8

# <h2 style='background-color:#6A5ACD; border:0; color:black'><center><br>The Rise and Fall of Dogecoin(DOGE)?</center><h2>
# 
# 

# ![](https://images.unsplash.com/photo-1622618760546-8e443f8a909b?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2072&q=80)

# <a id="top"></a>
# 
# <div class="list-group" id="list-tab" role="tablist">
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='background-color:#6A5ACD; border:0' role="tab" aria-controls="home" color=black><center><br>Quick navigation</center></h3>
# 
# * [1. What is Dogecoin(DOGE)](#1)
# * [2. Technical analysis of DOGE](#2)
# * [3. The Celebrity coin](#3)
# * [4. The Competition ](#4)   
# * [5. The Future prospects of DOGE](#5)
# * [6. The Community](#6)
# * [7. The Rise and Fall - Summary](#7)
# * [8. Related works](#8)
# * [9. References](#9)
# 
# 
# 

# 
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:100%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#             text-align:center">
# <br>
# <h3><center><b>DOGECOIN (DOGE)<b></center></h3>
#     <br>
# <p><center><b>Dogecoin is a cryptocurrency invented by software engineers Billy Markus and Jackson Palmer, who decided to create a payment system that is instant, fun, and free from traditional banking fees.It is considered both the first "meme coin", and, more specifically, the first "dog coin". Despite its satirical nature, some consider it a legitimate investment prospect. Dogecoin features the face of the Shiba Inu dog from the "doge" meme as its logo and namesake.</b> <center><p>
# 
# <br>
# <h3><b> General Info:</b></h3>
# <p><center><b>Original author(s):</b> Billy Markus, Jackson Palmer</center></p>
# <p><center><b>Initial release:</b> December 6, 2013; 7 years ago</center></p>
# <p><center><b>Website:</b> https://dogechain.info/</center></p>
# <p><center><b>Block reward:</b> 10,000 Dogecoins</center></p>
# <p><center><b>Supply limit:</b> Unlimited Exactly five billion Dogecoins will enter circulation each year.</center></p>
# 
# <br>
# <br>
# <p><b>&nbsp;&nbsp;Here’s a list of few interesting facts about Dogecoin:</b>
# <br>
# <p>&nbsp;&nbsp;  1. Dogecoin started as a joke created by Jackson Palmer and Billy Marcus in November 2013. Marcus recently claimed that he sold all of his DOGE in 2015.</p>
# <p>&nbsp;&nbsp;  2. Billy Markus(Co-Founder) tweeted this when a user asked him how to make the coin more efficient. He said that he made Dogecoin in 2 hours and didn’t consider anything.</p>
# <p>&nbsp;&nbsp;  3. There are 128,264,356,384 DOGE coins in circulation at this moment, compared to 18.5 million bitcoins.</p>
# <p>&nbsp;&nbsp;  4. Dogecoin hosts one of the largest communities in the crypto space.</p>
# <p>&nbsp;&nbsp;  5. In 2014, the Dogecoin community raised $55,000 to sponsor NASCAR driver Josh Wise and covered his car entirely in Dogecoin and Reddit alien images.</p>
# <p>&nbsp;&nbsp;  6. SpaceX Accepts Dogecoin as a Payment to Launch a Mission to the Moon in 2022 </p>
# 
# <br>
# 
# <h6> Sources: https://en.wikipedia.org/wiki/Dogecoin<br>
# https://dogechain.info/<br>
# https://www.coindesk.com/learn/want-to-buy-dogecoin-read-this-first/<br>
# https://nooor.io/blog/10-interesting-facts-about-dogecoin-you-need-to-know/</h6><br>
# </div>
#     
# 

# <a id="2"></a>
# <h2 style='background-color:#6A5ACD; border:0; color:black'><center><br>Technical Analysis of DOGE</center><h2>
# 
# 

# <h2 style='background-color:#6A5ACD; border:0; color:black'><center><br>Required Libraries</center><h2>

# In[ ]:


import pandas as pd

import pandas as pd
import numpy as np 
import missingno as mno
import pickle 
import json
import time 
import gc
import random
import sklearn

#For Data Visualization
import matplotlib.pyplot as plt
#%matplotlib inline 
#output of plotting commands is displayed inline within frontends like the Jupyter notebook,
#directly below the code cell that produced it. The resulting plots will then also be stored in the notebook document.

import seaborn as sns
sns.set(rc={'figure.figsize':(10,6)})
custom_colors = ["#4e89ae", "#c56183","#ed6663","#ffa372"]

#NetworkX
import networkx as nx
import plotly.express as px 
import plotly.graph_objects as go #To construct network graphs
from plotly.subplots import make_subplots #To make multiple plots

#To avoid printing of un necessary Deprecation warning and future warnings!
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from plotly.offline import init_notebook_mode, iplot
from IPython.core.display import display, HTML #To display html content in a code cell
init_notebook_mode(connected=True)

#Colorama
from colorama import Fore, Back, Style # For text colors
y_= Fore.CYAN
m_= Fore.BLACK

gc.collect()


# In[ ]:


#Seaborn plot viz of the missing values in the data 
def missing_values(data):
    #import seaborn as sns
    missed = pd.DataFrame()
    missed['column'] = data.columns

    missed['percent'] = [round(100* data[col].isnull().sum() / len(data), 2) for col in data.columns]
    missed = missed.sort_values('percent',ascending=False)
    missed = missed[missed['percent']>0]

    fig = sns.barplot(
        x=missed['percent'], 
        y=missed["column"], 
        orientation='horizontal',palette="winter"
    ).set_title('Missed values percent for every column')


def triple_plot(x, title,c): # Use triple plot for numeric and important key features 
    fig, ax = plt.subplots(3,1,figsize=(20,10),sharex=True)
    sns.distplot(x, ax=ax[0],color=c)
    ax[0].set(xlabel=None)
    ax[0].set_title('Histogram + KDE')
    sns.boxplot(x, ax=ax[1],color=c)
    ax[1].set(xlabel=None)
    ax[1].set_title('Boxplot')
    sns.violinplot(x, ax=ax[2],color=c)
    ax[2].set(xlabel=None)
    ax[2].set_title('Violin plot')
    #fig.suptitle(title, fontsize=30)
    #plt.tight_layout(pad=3.0)
    plt.show()

#Info, missing values and describe of the dataframe along with the triple plot of integer variables
def data_understand(df):
    display(HTML('<div class="alert alert-info"><h4><center>Data Information</center></h4></div><br>'))
    print(f"{m_}Total records:{y_}{doge.shape}\n")
    print(f"{m_}The dataset has the prices of DOGE from :{y_}{doge['Date'].min()} {m_}{'to'} {y_}{doge['Date'].max()}\n")

    print(f"{m_}{df.info()}")
    display(HTML('<br><div class="alert alert-info"><h4><center>Data Description</center></h4></div><br>'))
    req_cols=df.select_dtypes(include=np.number).columns.to_list()[0:6]
    print(pd.DataFrame(round(df[req_cols].describe(),2)))
    try: missing_values(df)
    except: pass;
    display(HTML('<br><div class="alert alert-info"><h4><center>Missing values</center></h4></div><br>'))
    #print('\n\nNA Values statistics')
    print(df.isna().sum())
    print('\nNo missing values were present! 💯')
    print('\n\n')
    #df.hist(bins=10,figsize=(20,15)) 
    #For smaller data use smaller bin sizes(5 to 20), increase accordingly if the data size increases
    plt.show()
    clrs=0
    display(HTML('<br><div class="alert alert-info"><h4><center>Distribution of OHLC Values of Dogecoin</center></h4></div><br>'))
    for i in df.select_dtypes(include=np.number).columns.to_list()[0:5]:
        if i!='Volume': display(HTML('<div style="color:white;display:fill;border-radius:5px;background-color:#5642C5;font-size:150%;font-family:Verdana;letter-spacing:0.5px;text-align:center"><p style="padding: 10px;color:white;"><center> DOGE {} Price<center></p></div>'.format(i)))
        else: display(HTML('<div style="color:white;display:fill;border-radius:5px;background-color:#5642C5;font-size:150%;font-family:Verdana;letter-spacing:0.5px;text-align:center"><p style="padding: 10px;color:white;"><center> DOGE trading {} <center></p></div>'.format(i)))
        triple_plot(df[i], str(i).upper(),custom_colors[random.choice([0,1,2,3])])
        print('\n')
    display(HTML('<br><div class="alert alert-info"><h4><center>The Distributions of OHLC(Opening, Highest, Lowest, and Closing price) follow a log-normal distribution. Due to the very reason using mean as a central tendency measure wouldnt be ideal in this case. Since the distribution is log-normal, the mean values could be skewed. Therefore, for that very reason, we will be using median values as they are less prone to skewness.</center></h4></div><br>'))


# In[ ]:


doge=pd.read_csv('../input/top-10-cryptocurrencies-historical-dataset/Top 100 Crypto Coins/dogecoin.csv')

# Coverting the date column to a datetime format and sorting the dataframe by date
doge['Date'] =  pd.to_datetime(doge['Date'],infer_datetime_format=True,format='%y-%m-%d')
doge.sort_values(by='Date',inplace=True)

data_understand(doge)


# In[ ]:


def candle_stick(df,name):
    data=df
    
    fig = go.Figure(data=go.Ohlc(x=data['Date'],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close']))

    fig.update_layout(template='plotly_dark',
        title='{} Price'.format(name),
        xaxis_title="Date",
        yaxis_title='{} Price'.format(name),title_x=0.5,
        )
    fig.update_yaxes( # the y-axis is in dollars
        tickprefix="$", showgrid=False
    )

    fig.update_xaxes(
         showgrid=True
    )
    
    low=[data['Open'].min(),data['High'].min(),data['Low'].min(),data['Close'].min()]
    high=[data['Open'].max(),data['High'].max(),data['Low'].max(),data['Close'].max()]
    
    fig.show()
    
    display(HTML('<div style="color:white;display:fill;border-radius:5px;background-color:#5642C5;font-size:100%;font-family:Verdana;letter-spacing:0.5px;text-align:center"><p style="padding: 10px;color:white;"><center><b> {} <b>Summary</b> :<center></b></p><br><b>Median price:<b> {}<br><b>Mean price:</b> {}<br><b>Highest price:</b> {}<br><b>Lowest price:</b> {}<br></div>'.format(name,data['Close'].median(),data['Close'].mean(),max(high),min(low))))


candle_stick(doge,'DOGE')


# <div class='alert alert-info'>
# <h4> DOGE saw a sudden rise in price in the early months of 2021 causing a lot of retail and institutional investors to jump on this train. The sudden rise was mainly due to strong positive advocacy towards dogecoin by influential people like Mark cuban, and Elon musk.</h4>
# </div>

# In[ ]:


def sma_and_ema(df,name):
    df['SMA5'] = df.Close.rolling(5).mean()
    df['SMA20'] = df.Close.rolling(20).mean()
    df['SMA50'] = df.Close.rolling(50).mean()
    df['SMA200'] = df.Close.rolling(200).mean()
    df['SMA500'] = df.Close.rolling(500).mean()

    fig = go.Figure(data=[go.Ohlc(x=df['Date'],open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'], name = "OHLC"),
                          go.Scatter(x=df.Date, y=df.SMA5, line=dict(color='orange', width=1), name="SMA5"),
                          go.Scatter(x=df.Date, y=df.SMA20, line=dict(color='green', width=1), name="SMA20"),
                          go.Scatter(x=df.Date, y=df.SMA50, line=dict(color='blue', width=1), name="SMA50"),
                          go.Scatter(x=df.Date, y=df.SMA200, line=dict(color='violet', width=1), name="SMA200"),
                          go.Scatter(x=df.Date, y=df.SMA500, line=dict(color='purple', width=1), name="SMA500")])
    
    fig.update_layout(template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Price',
        title='SMA of {}'.format(name),
        hovermode="x",title_x=0.5
    )
    fig.update_yaxes( # the y-axis is in dollars
        tickprefix="$", showgrid=True
    )

    fig.update_xaxes(
         showgrid=True
    )
    
    fig.show()


    df['EMA5'] = df.Close.ewm(span=5, adjust=False).mean()
    df['EMA20'] = df.Close.ewm(span=20, adjust=False).mean()
    df['EMA50'] = df.Close.ewm(span=50, adjust=False).mean()
    df['EMA200'] = df.Close.ewm(span=200, adjust=False).mean()
    df['EMA500'] = df.Close.ewm(span=500, adjust=False).mean()

    fig = go.Figure(data=[go.Ohlc(x=df['Date'],
                                  open=df['Open'],
                                  high=df['High'],
                                  low=df['Low'],
                                  close=df['Close'], name = "OHLC"),
                          go.Scatter(x=df.Date, y=df.EMA5, line=dict(color='orange', width=1), name="EMA5"),
                          go.Scatter(x=df.Date, y=df.EMA20, line=dict(color='green', width=1), name="EMA20"),
                          go.Scatter(x=df.Date, y=df.EMA50, line=dict(color='blue', width=1), name="EMA50"),
                          go.Scatter(x=df.Date, y=df.EMA200, line=dict(color='violet', width=1), name="EMA200"),
                          go.Scatter(x=df.Date, y=df.EMA500, line=dict(color='purple', width=1), name="EMA500")])
    fig.update_layout(template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Price',
        title='EMA of {}'.format(name),
        hovermode="x",title_x=0.5
    )
    fig.update_yaxes( # the y-axis is in dollars
        tickprefix="$", showgrid=True
    )

    fig.update_xaxes(
         showgrid=True
    )
    
    fig.show()


# In[ ]:


sma_and_ema(doge,'Dogecoin(DOGE)')


# In[ ]:


import datetime
def mom_mean_median(final_df):
    year_df=final_df[(final_df['Year']==2021)]
    group_yr_df=year_df.groupby(['Month']).agg({'Close':np.mean}).reset_index()

    def mom(m1,m2):
        return round(((m2-m1)/m1)*100,2)

    mom_list=[]
    
    df=group_yr_df
    for j in range(0,len(df)):
        if df['Month'].iloc[j]!=1:
            mom_list.append(mom(df['Close'].iloc[j-1],df['Close'].iloc[j]))
        else:
            mom_list.append(0)

    group_yr_df['mom']=mom_list

    group_yr_df['mom']=round(group_yr_df['mom'],2)
    group_yr_df['Coin']='DOGE Mean '
    df2=group_yr_df.pivot_table(index='Coin',columns='Month',values='mom')
    
    
    ##Median values 
    year_df=final_df[(final_df['Year']==2021)]
    group_yr_df=year_df.groupby(['Month']).agg({'Close':np.median}).reset_index()

    def mom(m1,m2):
        return round(((m2-m1)/m1)*100,2)

    mom_list=[]
    
    df=group_yr_df
    for j in range(0,len(df)):
        if df['Month'].iloc[j]!=1:
            mom_list.append(mom(df['Close'].iloc[j-1],df['Close'].iloc[j]))
        else:
            mom_list.append(0)

    group_yr_df['mom']=mom_list

    group_yr_df['mom']=round(group_yr_df['mom'],2)
    group_yr_df['Coin']='DOGE Median '
    df3=group_yr_df.pivot_table(index='Coin',columns='Month',values='mom')
    df2=pd.concat([df2,df3],axis=0)
    month_changer=list(df2.columns)
    new_month_name=[datetime.datetime.strptime(str(i), "%m").strftime("%b") for i in month_changer]
    
    df2.columns=new_month_name
    

    def style_negative(v, props=''):
        return props if v < 0 else None
    s2 = df2.style.applymap(style_negative, props='color:red;')                  .applymap(lambda v: 'opacity: 20%;' if (v < 0.3) and (v > -0) else None)

    def highlight_max(s, props=''):
        return np.where(s == np.nanmax(s.values), props, '')
    s2.apply(highlight_max, props='color:white;background-color:darkblue', axis=1)

    s2.set_caption("Mean/Median Price Month on Month (MOM) Growth rate - 2021")     .set_table_styles([{
         'selector': 'caption',
         'props': 'caption-side: top; font-size:2.50em;'
     }], overwrite=False)
    
    
    return s2
doge['Date']=pd.to_datetime(doge['Date'],format='%Y-%m-%d')
doge['Year']=doge['Date'].dt.year
doge['Month']=doge['Date'].dt.month



# In[ ]:


mom_mean_median(doge)


# <div class='alert alert-info'>
# <h4> Month on month growth rate rose significantly during the first few few months of 2021 causing a lot of speculation and volatility in the market.</h4>
#     
# <ol>
# <li>Dogecoin's adoption in the U.S. supersedes Bitcoin and Ethereum with 30.6% of crypto owners saying they own Dogecoin. That's 1.6 times the global average adoption rate of 19.2%.
#     
# <li>A recently published survey from the web portal gamblerspick.com suggests one out of every four Americans believe dogecoin is the future.
#     (source: https://news.bitcoin.com/survey-1-in-4-american-investors-believe-dogecoin-is-the-future/)
# </div>

# ![](https://static.news.bitcoin.com/wp-content/uploads/2021/05/dogecoin-or-lottery-ticket-png-39b4c4c22ba252f0432baaf3a397e125-1024x563.png)

# <a id="3"></a>
# <h2 style='background-color:#6A5ACD; border:0; color:black'><center><br>The Celebrity Coin</center><h2>
# 
# 

# <div class='alert alert-warning'>
#     <h3><center> <b>Tweet by Mark cuban On 14th April 2021</b>, American billionaire entrepreneur,owner of the National Basketball Association's (NBA) Dallas Mavericks<center></h3>
# <br><h4>FYI, the Mavs sales in @dogecoin have increased 550pct over the past month. We have now sold more than 122k Doge in merchandise ! We will never sell 1 single Doge ever. So keep buying</h4> 
# </div>

# <div style="text-align:center"><img src="https://images.news18.com/ibnlive/uploads/2021/04/1618473964_untitled-design-3.jpg?impolicy=website&width=534&height=356" height="300" width="600"></div>

# <div class='alert alert-info'>
#     <h4> Elon musk's tweet about Dogecoin on Apr 15, 2021, which said <b>Doge barking at the Moon</b> received around 20.8k comments, 52.3k re-tweets, and 314.1K likes</h4>
#     </div>
#     

# In[ ]:


req_cols=['Date','Open','High','Low','Close','Volume']
doge[(doge['Date'].dt.year==2021)&(doge['Month']==4)&(doge['Date'].dt.day>=13)&(doge['Date'].dt.day<17)][req_cols]


# <div class='alert alert-info'>
# 
# <h4><center><b>Just look at the numbers!</b></center></h4>
#     
# <p><center>On 13th april 1 dogecoin was trading at 0.09$</center><br></p>
# <p><center>On 14th april 1 dogecoin went upto to a high price of 0.12 USD 📈(Mark cuban tweets about doge)</center><br></p>
# <p><center>On 15th april 1 dogecoin went upto to a high price of 0.18 USD 📈(Elon musk tweets about doge)</center><br></p>
# <p><center>On 16th april 1 dogecoin ended with an all time high of 0.36 USD 📈 , which is twice the jump compared to the previous day</center><br></p>
# 
# <br>
# 
# <p>1. As of 11:10 a.m. Friday(16th april), the value of Dogecoin had jumped 203% in just the past 24 hours to 0.44(USD), according to Coinbase, giving the cryptocurrency a market cap value of $52.2 billion.</p>
# 
# <p>2. Over the past week, Dogecoin’s value has more than quintupled in value.</p>
# 
# <p>3. Dogecoin was the seventh largest cryptocurrency in terms of market cap</p>
# 
# </div>
# 

# <div class='alert alert-info'>
# 
# <h4><center>If you think the above numbers were suprising then you are wrong. Things changed pretty quick once elon musk posted this tweet and multiple companies accpeting doge as a payment method</center></h4>
#     
# </div>

# <div style="text-align:center"><img src="https://www.thesun.co.uk/wp-content/uploads/2021/04/doge.png" height="300" width="600"></div>

# <div class='alert alert-info'>
# <h4>1. You could see that the tweet by elon musk saying that he is gonna be a part of the Saturday day night live(May 8) titled THE DOGEFATHER at CNBC with miley cyrus took the dogecoin price to new heights of 0.344$</h4>
# 
# <h4>2. Speculation about a possible cryptocurrency sketch pushed Doge past the $0.50 mark for the first time. It peaked at 0.66 on May 5.</h4>
# 
# <h4>3. Dogecoin rose 140%, from 0.2747 on April 27 to 0.6618 on May 5.</h4>
# </div>

# In[ ]:


doge[(doge['Date'].dt.year==2021)&(doge['Month']==5)&(doge['Date'].dt.day>=1)&(doge['Date'].dt.day<7)][req_cols]


# <div class='alert alert-info'>
# <h4>In the past one week Dogecoin closing price on 1st May was around 0.39USD and it significantly increased to 0.58 USD on May 6th.</h3>
# 
# <h4> 1. People are buying heavily to make sure that they hold dogecoin before the SNL on MAY 8</h3>
# <h4> 2. Huge volatility has been observed, where we can see certain amount of whales playing a bear game</h3>
# </div>

# <div class='alert alert-info'>
# 
# <h4>A recently published survey from the web portal gamblerspick.com suggests one out of every four Americans believe dogecoin is the future.The survey also asked the participants how do they feel about elon musk promoting dogecoin? (See below)<br>
# <h6>source: https://news.bitcoin.com/survey-1-in-4-american-investors-believe-dogecoin-is-the-future/)</h6>
# </div>

# ![](https://static.news.bitcoin.com/wp-content/uploads/2021/05/edgdssjjdjd5677723.jpg)

# <div class='alert alert-info'>
# 
# <h4> <center>List of celebrities supporting dogecoin</center></h4>
# 
# 1. Elon Musk
# 2. Snoop Dogg
# 3. Mark Cuban
# 4. Kevin Jonas
# 5. Gene Simmons
# 6. Lil' Yachty
# 7. Vicky-Lee Valentino
# 8. Angela White
# 9. Mia Khalifa
# 10. Jake paul
# 11. Ben Phillips
# 12. Marques Brownlee
# 
# <br>
# Read more at : https://www.ibtimes.sg/which-celebrities-have-invested-dogecoin-cryptocurrency-heres-complete-list-57221
# </div>

# <a id="4"></a>
# <h2 style='background-color:#6A5ACD; border:0; color:black'><center><br>The Competition</center><h2>
# 
# 

# 
# <div style="text-align:center;background-color:black;">
# <img src="https://bitcoinist.com/wp-content/uploads/2021/11/Shiba-inu-VS-Dogecoin.jpeg" width=800>
# </div>

# <div class='alert alert-info'>
#     <h3><center><b>The Competition</b></center></h3>
# 
# Both DOGE and SHIB were created as a joke, and they have the same dog, a Shiba Inu, for a mascot. However, this is where the similarities end.
# 
# <h5><b><center>Shiba Inu was created in August 2020, dubbing itself the 'Dogecoin killer'.</center></b></h5>
# 
# 
# <h4><b>What’s fuelling memecoins?</b></h4>
# <p>With celebrity shoutouts, social media frenzy and millennial and Gen Z investors boosting their prices, memecoins seem to have entered the mainstream.<p>
# <p>Looking to capitalize on the trend, new memecoins are flooding the market. For now, DOGE and SHIB are still the “top dogs,” and despite the recent run-up in their prices, the two coins appear far from done.</p>
# 
# <br>
# <h4><b>All bark, no bite?</b></h4>
# 
# <p>Unlike bitcoin, ethereum, USD coin and other popular crypto tokens, memecoins don’t have any intrinsic utility, and their values are largely driven by speculation and investor enthusiasm.</p>
# 
# <br>
# <h6>Source: https://www.moneysense.ca/save/investing/crypto/shiba-inu-vs-dogecoin-are-memecoins-a-good-investment/#:~:text=While%20far%20smaller%20in%20unit,1%2C%202022.</h6>
# </div>

# 
# <div style="color:white;
#            display:fill;
#            border-radius:5px;
#            background-color:#5642C5;
#            font-size:100%;
#            font-family:Verdana;
#            letter-spacing:0.5px;
#             text-align:center">
# <br>
# <h3><center><b>Shiba Inu token (SHIB)<b></center></h3>
#     <br>
# <h4>SHIB is a decentralized cryptocurrency created in August 2020 by an anonymous person or group known as "Ryoshi". It is named after the Shiba Inu (柴犬), a Japanese breed of dog originating in the Chūbu region, the same breed that is depicted in Dogecoin's symbol, itself originally a satirical cryptocurrency based on the Doge meme. Shiba Inu has been characterized as a "meme coin" and a pump and dump scheme.There have also been concerns about the concentration of the coin with a single "whale" wallet controlling billions of dollars' worth of the token, and frenzied buying by retail investors motivated by fear of missing out (FOMO).</h4>
# <br><br>
# <h4>Shiba Inu was created in August 2020, dubbing itself the 'Dogecoin killer'. On 13 May, Vitalik Buterin donated more than 50 trillion SHIB (worth over 1 billion USD at the time) to the India COVID-Crypto Relief Fund.</h4>
# <br><br>
# <h4>The exchange price of the cryptocurrency notably surged in early October 2021. Its value increased 240% over the week.However, at the beginning of November the price dropped and continued to fall, ending the month having lost approximately 55% of its value.</h4>
# <br>
# </div>
#     
# 

# In[ ]:


shib=pd.read_csv('../input/top-10-cryptocurrencies-historical-dataset/Top 100 Crypto Coins/SHIBA INU.csv')
shib['Date']=pd.to_datetime(shib['Date'],format='%Y-%m-%d')
shib['Year']=shib['Date'].dt.year
shib['Month']=shib['Date'].dt.month


# In[ ]:


doge_vol=pd.DataFrame(doge[doge['Year']==2021].groupby(['Month']).agg({'Volume':np.mean}).reset_index())
shib_vol=pd.DataFrame(shib[shib['Year']==2021].groupby(['Month']).agg({'Volume':np.mean}).reset_index())

plt_df=doge_vol[['Month','Volume']].merge(shib_vol[['Month','Volume']],on='Month',how='left')[0:-1]
plt_df.rename(columns={'Volume_x':'Dogecoin','Volume_y':'Shiba Inu'},inplace=True)

plt_df['Month']=plt_df['Month'].apply(lambda x: datetime.datetime.strptime(str(x), "%m").strftime("%b"))


fig = px.line(plt_df, x='Month', y=['Dogecoin','Shiba Inu'])
fig.update_layout(template='plotly_dark',
    xaxis_title='Month',
    yaxis_title='Mean trading volume',
    title='Mean trading volume of DOGE and SHIB in 2021',
    hovermode="x",title_x=0.5
)
fig.update_yaxes( # the y-axis is in dollars
    tickprefix="$", showgrid=True
)

fig.update_xaxes(
     showgrid=True
)
fig.show()


# In[ ]:


doge_vol=pd.DataFrame(doge[doge['Year']==2022].groupby(['Month']).agg({'Volume':np.mean}).reset_index())
shib_vol=pd.DataFrame(shib[shib['Year']==2022].groupby(['Month']).agg({'Volume':np.mean}).reset_index())

plt_df=doge_vol[['Month','Volume']].merge(shib_vol[['Month','Volume']],on='Month',how='left')[0:-1]
plt_df.rename(columns={'Volume_x':'Dogecoin','Volume_y':'Shiba Inu'},inplace=True)
plt_df['Month']=plt_df['Month'].apply(lambda x: datetime.datetime.strptime(str(x), "%m").strftime("%b"))

fig = px.line(plt_df, x='Month', y=['Dogecoin','Shiba Inu'])
fig.update_layout(template='plotly_dark',
    xaxis_title='Month',
    yaxis_title='Mean trading volume',
    title='Mean trading volume of DOGE and SHIB in 2022',
    hovermode="x",title_x=0.5
)
fig.update_yaxes( # the y-axis is in dollars
    tickprefix="$", showgrid=True
)

fig.update_xaxes(
     showgrid=True
)
fig.show()


# <div class='alert alert-info'>
# <h4> In my view, initial few months was the golden period for DOGE as it rose to unexpected levels causing FOMO(Fear of Missing Out) among investors and therefore causing huge volatility in the market</h4>
#     
# <br>
# <h4>But this uptrend didnt last longer, as DOGE prices were mainly fueled by social media tweets and endorsements from celebrities. So the downfall was evident for the coin as it crashed to severe lows from 0.72 USD in May 7th to 0.17 USD by the end of Dec 2021.</h4>
# 
# </div>

# <div class='alert alert-info'>
# 
# <h3><center><b>40,000 Holders Exit Dogecoin</b></center></h3>
# 
# <br>
# <h5>Dogecoin has been losing ground in terms of its price over the past year(2021) and this has been bleeding out into its investors. As such, investors have been exiting the meme coin en masse. The most recent batch of this exodus consisted of 40,000 DOGE holders who have now left the cryptocurrency.</h5>
# 
# <br>
# 
# <h5>This happened over a period of ten days, following the news that Dogecoin had lost over 700,000 investors. It is a direct consequence of an ever-declining price with no end or reprieve in sight. After hitting its all-time high of 0.7 USD last year going off the hype from billionaire Elon Musk, the meme coin has had a hard time holding on to its gains. This has resulted in the loss of over 70% of its all-time high in the space of a year and continues to decline.</h5>
# 
# <br>
# 
# <h5> Source: https://bitcoinist.com/dark-times-for-dogecoin-as-another-40k-holders-exit/
# 
# 
# </div>

# <a id="5"></a>
# <h2 style='background-color:#6A5ACD; border:0; color:black'><center><br>Future Prospects of Dogecoin</center><h2>
# 
# 

# <div class='alert alert-info'>
#     
# <h4>The Dogecoin Foundation, a nonprofit organization that aims to support the development of the meme coin through advocacy, has launched its first-ever road map in its eight-year history detailing a number of new projects.The foundation announced a dogecoin “trail map” that features eight projects, including the launch of LibDogecoin and GigaWallet.
# <br><br>
#     
# <ol>
# <li>In August 2021, the foundation signed the Dogecoin Manifesto, which explained the goal of DOGE and allowed the fans of the Shiba Inu-inspired cryptocurrency to also sign the manifesto capturing feedback and what the community wanted from the project.<br>
# 
# <li>The Dogecoin Foundation, boasts some well-known board members and advisers, including Ethereum co-founder Vitalik Buterin.<br>
# <li>In its road map, the foundation said it is working with Buterin on “crafting a uniquely Doge proposal for a ‘community staking’ version of proof-of-stake (PoS) that will allow everyone, not just the big players to participate in a way that rewards them for their contribution to running the network.”
# <li>The foundation goes on to say it has “some influential friends” on its side and a growing group of people who are getting ready to contribute development time to these open-source projects.
# <li>In February, Elon Musk suggested in a tweet that dogecoin might be “the future currency of earth.” Musk’s involvement in the DOGE token tribe has helped send the cryptocurrency “mooooning” (his word), along with other alternative cryptocurrencies.
# <li>DOGE started as a joke in 2013, and is now the 12th most valuable cryptocurrency by market value, according to CoinMarketCap.(As of May 2022)
# <li>In July, dogecoin founder Jackson Palmer said that he would not be returning to cryptocurrency as it is “is an inherently right-wing, hyper-capitalistic technology built primarily to amplify the wealth of its proponents through a combination of tax avoidance, diminished regulatory oversight and artificially enforced scarcity.”
# </ol>
# 
# <h6>Source: https://www.coindesk.com/business/2021/12/24/planning-for-a-better-breed-of-doge-dogecoin-foundation-lays-out-first-ever-roadmap/</h6>
# </div>

# <a id="6"></a>
# <h2 style='background-color:#6A5ACD; border:0; color:black'><center><br>The Community</center><h2>
# 
# 

# <div class='alert alert-info'>
# <h4> The Dogecoin Community in Reddit played a huge role in the Dogecoin Project. It acted as a hub where many and many Doge HODLers came together as a community and spoke out about their predictions and speculations on DOGE. It currently has around 2.3 Million Subscribers(Subshibers)</h4>
# 
# <br>
# <h5>DOGE Reddit : https://www.reddit.com/r/dogecoin/</h5> 
# <h5> Twitter Dogecoin page: https://twitter.com/dogecoin</h5>
#     
# <h5> Below are some of the memes, and contents that were available in the reddit community</h5>
# </div>

# 
# <div style="text-align:center;background-color:black;">
# <img src="https://preview.redd.it/eyzf6owvg8171.jpg?auto=webp&s=e2d3c366c53f0998ad0d5d6464bf94ff7685e2d5" width=500>
# </div>
# 
# 

# 
# <div style="text-align:center;background-color:white;">
# <img src="https://preview.redd.it/xmhl14ynr0p11.png?auto=webp&s=90a3f26d2694c3163ed4a3cec5550475d74f3ca9" width=500>
# </div>
# 
# 

# 
# <div style="text-align:center;background-color:white;">
# <img src="https://external-preview.redd.it/aR1oEi39nXhQEirsD0A_Ks4GiwtRbuBwEPBx6i3UX78.jpg?auto=webp&s=12608f6b24e546aeb7dd6c02211ab8c158417fbf" width=300>
# </div>
# 
# 

# 
# <div style="text-align:center;background-color:white;">
# <img src="https://preview.redd.it/dhse9gu37ox81.jpg?width=640&crop=smart&auto=webp&s=85658330dfc6555d8100ad7158627b77a8a45d25" width=300>
# </div>
# 
# 

# <a id="7"></a>
# <h2 style='background-color:#6A5ACD; border:0; color:black'><center><br>The Rise and Fall of DOGE 🐕 - Summary</center><h2>
# 
# <h3><center><b>The Rise📈</b></center></h3>
# 
# * The hype about dogecoin from twitter tweets by influential people and from the reddit community fueled the prices of DOGE by creating a FOMO among the investors.
# * The trading volume reached an whooping total of around 20 Billion USD in Feb -2021
# * The coin reached an all time high of 0.72 USD dollars on May 7th, 2021
# * According to Google Trends’ “Year in Search 2021,” Dogecoin was the fourth-most popular news search term on Google in 2021, both globally and separately in the United States.
# 
# 
# <h3><center><b>The fall📉</b></center></h3>
#     
# * The social media craze on DOGE slowly diluted and made the coin to loss over 70% of its all-time high in the space of a year and continues to decline.
# * Coins like SHIBA INU (SHIB) proved to be a fiere competitor as it bagged the highest trading volume comapred to DOGE in the months of Feb, and March 2022.
# * Dogecoin has been losing ground in terms of its price over the past year(2021) and this has been bleeding out into its investors.  The most recent batch of this exodus consisted of 40,000 DOGE holders who have now left the cryptocurrency in a span of 10 days followed by the news of Dogecoin lossing over 700,000 investors.
# 
# 
# 
# 
# 

# <a id="8"></a>
# <h2 style='background-color:#6A5ACD; border:0; color:black'><center><br>Related Works</center><h2>
# 
# 

# <div style="text-align:center"><img src="https://static.news.bitcoin.com/wp-content/uploads/2019/10/5aY6rsB5-crypto-learning.jpg" height="300" width="600"></div>

# <div class='alert alert-info'>
# <center><h4>- If you are interested in Bitcoin, I would suggest you other related resources of mine for you to explore</h4></center>
# </div>
# 
# * [Bitcoin's  price prediction using the facebook prophet model](https://www.kaggle.com/kaushiksuresh147/bitcoin-prices-eda-and-prediction-r2-0-99)
# * [Bitcoin Tweets Exploratory data analysis](https://www.kaggle.com/kaushiksuresh147/bitcoin-prices-eda-and-prediction-r2-0-99)
# * [Bitcoin Tweets Dataset updated on a weekly basis](https://www.kaggle.com/kaushiksuresh147/bitcoin-tweets)
# 
# <div class='alert alert-info'>
# <center><h4>- If you would like to know more about cryptocurrency and Blockchain, pls refer to the below notebooks</h4></center>
# </div>
# 
# * [what, why, where, and how-of-blockchain?](https://www.kaggle.com/kaushiksuresh147/what-why-where-and-how-of-blockchain)
# * [What is cryptocurrency?](https://www.kaggle.com/kaushiksuresh147/what-is-cryptocurrency/edit/run/78333005)
# 
# 

# 
# <div class='alert alert-info'>
# <center><h3>Other Cryptocurrency related resources</h3></center>
# 
# <h4><center> Datasets 📚</center></h4>
# </div>
# 
# 1. [Bitcoin Tweets](https://www.kaggle.com/kaushiksuresh147/bitcoin-tweets)
# 2. [Top 10 Crytocurrency historical dataset](https://www.kaggle.com/kaushiksuresh147/top-10-cryptocurrencies-historical-dataset)
# 3. [Ethereum historical dataset](https://www.kaggle.com/kaushiksuresh147/ethereum-cryptocurrency-historical-dataset)
# 4. [Polygon(Matic) Historical dataset](https://www.kaggle.com/kaushiksuresh147/maticpolygon-crytocurrency-historical-dataset)
# 5. [Solana Historical dataset](https://www.kaggle.com/kaushiksuresh147/solana-cryptocurrency-historical-dataset)
# 6. [Metaverse Crypto Tokens Historical data 📊](https://www.kaggle.com/datasets/kaushiksuresh147/metaverse-cryptos-historical-data)
# 7. [India wants Crypto movement tweets](https://www.kaggle.com/kaushiksuresh147/india-wants-crypto-tweets)
# 
# 
# <div class='alert alert-info'>
# <h4><center>Notebooks 📓</center></h4>
# </div>
# 
#     
# 1. [Bitcoin EDA AND Prediction](https://www.kaggle.com/kaushiksuresh147/bitcoin-prices-eda-and-prediction-r2-0-99)
# 2. [People reaction on India proposed Crypto ban](https://www.kaggle.com/kaushiksuresh147/people-s-reaction-on-india-s-proposed-crypto-ban)
# 3. [Ethereum EDA and Prediction using Facebook prophet](https://www.kaggle.com/kaushiksuresh147/ethereum-eda-and-prediction-using-prophet)
# 4. [Dogecoin EDA and prediction ](https://www.kaggle.com/kaushiksuresh147/doge-coin-to-moon-eda-and-prediction)
# 5. [what, why, where, and how-of-blockchain?](https://www.kaggle.com/kaushiksuresh147/what-why-where-and-how-of-blockchain)
# 6. [Bitcoin Volatility Analysis with Interactive Viz📊](https://www.kaggle.com/code/kaushiksuresh147/bitcoin-volatility-analysis-with-interactive-viz)
# 7. [Dogecoin Vs Shiba Inu 🐕 : A Dog Fight 🤺](https://www.kaggle.com/code/kaushiksuresh147/dogecoin-vs-shiba-inu-a-dog-fight)
# 
# 

# <a id="9"></a>
# <h2 style='background-color:#6A5ACD; border:0; color:black'><center><br>Resources</center><h2>

# <div class='alert alert-info'>
# 
# * [Top 100 Cryptocurrencies Historical Dataset](https://www.kaggle.com/datasets/kaushiksuresh147/top-10-cryptocurrencies-historical-dataset) was used inorder to perform the above analysis
# 
# - The dataset consists of the historical prices of nearly 100 Crypto coins and is being updated an a weekly basis.
# </div>
