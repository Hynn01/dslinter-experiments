#!/usr/bin/env python
# coding: utf-8

# # Betting in the Play
# > ## Basketball Live Prediction & Analytics with Play-by-play data

# ![intro_pic](https://www.legitgamblingsites.com/wp-content/uploads/2019/02/College-Concepts.png)

# # Introduction

# > ### All you need to know about in-play betting in basketball

# In-play betting, a.k.a live betting or in-play wagering, is simply wagering on a game while it’s happening. In-pkay betting explodes in U.S. lately as it is a really fun way that make bettors continue to be engaged in the game, and it gives chance bettors who misses placing a bet before the games begin. <sup>[1]</sup> In basketball, the in-play betting is mostly in the forms of betting in final scores, in spread points <sup>[2]</sup>, or in winning teams. 
# 
# However, in-play betting still has a strong potential to growth as it only takes less than 20% of the market in Nevada sportsbooks, <sup>[1]</sup>  which consits of most of legal betting sites in the state. One important reason is that a lot of sportsbook operators haven't acquire the technology to deplore it. 
# 
# [1]: https://www.thelines.com/betting/in-play/
# [2]: https://www.thelines.com/betting/point-spread/
# 

# >### Explore in-play betting with March Madness Data

# Here comes the opportunity for data science. In this notebook, I'm going to :
# > 1. Construct live prediction models minute by minute, utilizing linear regression models with L1 and L2 norm, to **predict the game final scores** with March Madness Men's play-by-play data.     
# jump to conclusion : [live prediction models](#models)
#       
# > 2. **Explore the "game changing" events** and the best team strategy, e.g. when to make a technical foul or what is a good winning strategy in the last 10 minutes of the game,based on the regression results.     
# jump to conclusion : [game changing events](#game-changing)
# 
# > * and here is how I re-construct the play-by-play data into game-based data minute-by minute:       
#     jump to data aggragation : [data pre-processing](#data-preprocessing)

# In[ ]:


import numpy as np
from time import time
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import statsmodels.api as stat
import sys
import warnings  

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import (Exchangeable,
    Independence,Autoregressive)
from statsmodels.genmod.families import Poisson

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# # Part 1 - Data Description

# > ### Exploring 2015-2019 March Madness men's play-by-play data

# As we are building up a live prediction model, every single play (i.e., miss a 2-point shoot, fouled by the other team, or a player is substitued by another player) matters to the team performance, thus affect the final scores and the winning odds. Therefore, play-by-play data is of great use in building up the model. 
# 
# Luckily, we have the March Madness MEvents & WEvents data from 2015-2019 that list the play-by-play event logs for more than 99.5% of games from that season<sup>[3]</sup> . Each event is assigned to either a team or a single one of the team's players.
# 
# To start with, let's get a breif description of 2015-2019 March Madness Men's play-by-play data, which is the **tarining data** I'm going to use in latter model construction . 
# 
# [3]: https://www.kaggle.com/c/march-madness-analytics-2020/data
# 

# MEvents data consists of each play, events including the event types and the sub event types. For example, if a player performs a defensive rebound, the it is represented as : `EventType-reb` with the `SubEventType - def` (the second row of the data represents this specific play). 

# In[ ]:


mens_events = []
MENS_DIR = '/kaggle/input/march-madness-analytics-2020/MPlayByPlay_Stage2'
for year in [2015, 2016, 2017, 2018, 2019]:
    mens_events.append(pd.read_csv(f'{MENS_DIR}/MEvents{year}.csv'))
MEvents = pd.concat(mens_events)
#print("2015-2019 Mevents data volume - {} columns, {} rows".format(MEvents.shape[1],MEvents.shape[0]))
MEvents.head()


# here is a brief description of all variables in the MEvents data, it consists of 13 million data entries 17 attributes spreading from 2014-15 season to 2018-19 season.     
# Except for `EventSubType`, all data point is complete without missing value. As for the `EventSubType`, this is because the more and more sub events are added in the latter seasons. I'll fill all the NAs to 0 in the cumu-count aggregation process.   

# In[ ]:


def brief(data,columns):
    print("-------------------------------------------------------------------------")
    print('data has {} rows {} attributes'.format(len(data),len(data.columns)))
    print("-------------------------------------------------------------------------") 
    print(data.columns)
#     print("-------------------------------------------------------------------------") 
#     print('data dtypes:\n')
#     print(data.dtypes)
    print("-------------------------------------------------------------------------")
    describe = data[columns].describe().transpose().reset_index()
    describe.rename(columns={'index':'Attribute_name'},inplace=True)
    #print([data[col].nunique() for col in data.columns]) 
    pd.options.display.float_format = '{:,.2f}'.format
    pd.options.display.width = 200
    print(pd.DataFrame(describe[['Attribute_name','mean','std','min','max']]))
    print("-------------------------------------------------------------------------") 
    print("Unique values:")
    print([(col,data[col].nunique())for col in data.columns])
    print("-------------------------------------------------------------------------") 
    print("missing values:")
    print( "; ".join(["-".join(x) for x in zip(data.columns,data.isna().sum(axis=0).astype(str).values)]))
    print("-------------------------------------------------------------------------")
    print("Most common value top 3:")
    print([(col,str(data[col].value_counts().head(3).index).            strip('Int64Index([\',\'], dtype=\'int64\'))').strip('Float64Index([\',\'], dtype=\'floa"\'))')            .strip('], dtype=\'objec")'))            for col in data.columns])
    
columns = ['Season', 'DayNum','WFinalScore', 'LFinalScore', 
           'WCurrentScore', 'LCurrentScore', 'ElapsedSeconds',
           'X', 'Y', 'Area']
brief(MEvents,columns)


# In 2015-2019 Men play-by-play data, there are altogether 16 events, followed by 52 subevents (and one `SubEventType` as nan). 

# In[ ]:


def categorical(data):
    for col in data.columns:
        unique = pd.unique(data[col])
        print("there are {} unique value of variable {}:".format(len(unique),col))
        print(unique)
categorical(MEvents[['EventType','EventSubType']])


# Here are the top-10 most frequent `EventType` and `EventSubType` in MEvents as a glimpse to the common events happening in the basketball games: 

# In[ ]:


def plot_most_freq(Events,col):
    plt.style.use('fivethirtyeight')
    Events['counter'] = 1
    Events.groupby(col)['counter']     .sum()     .sort_values(ascending=False).iloc[:10]     .plot(kind='bar',
         #color=mypal[2],
        color='#ed6663',alpha = 0.7,label=col+' frequency')
    
    plt.xticks(rotation=1)
    
#fig = plt.figure(1)
# plt.rcParams['xtick.labelsize']=18
# plt.rcParams['ytick.labelsize']=18
plt.style.use('fivethirtyeight')
f, axs = plt.subplots(1,1,figsize=(20,10))
plot_most_freq(MEvents,'EventType')
plt.title('TOP 10 Most Freqent Events',fontsize = 24)
plt.xlabel('Event Type',fontsize=20)
plt.ylabel('Counts',fontsize=20)
plt.legend(fontsize=20)
axs.tick_params(axis='both', which='major', labelsize=20)
plt.style.use('fivethirtyeight')
f, axs = plt.subplots(1,1,figsize=(20,10))
plot_most_freq(MEvents,'EventSubType')
plt.title('TOP 10 Most Freqent Sub Events',fontsize=24)
plt.xlabel('Sub Event Type',fontsize=20)
plt.ylabel('Counts',fontsize=20)
axs.tick_params(axis='both', which='major', labelsize=20)
plt.legend(fontsize=20)
plt.show()


# Also, as time is a very important variable in the model construction, let's explore more about the `ElapsedSeconds` variable in the dataset. `ElapsedSeconds` represents the number of seconds that have elapsed from the start of the game until the event occurred. With a 20-minute half, that means that an `ElapsedSeconds` value from 0 to 1200 represents an event in the first half, a value from 1200 to 2400 represents an event in the second half, and a value above 2400 represents an event in overtime. For example, since overtime periods are five minutes long (that's 300 seconds), a value of 2699 would represent one second left in the first overtime. <sup>[3]</sup>     
# 
# Unlike normal basketball games that consist of 4 quarters, college basketball games consist of 2 halves, thus the main game time is on average 40 mins. As shown in the histogram, the major recorded `ElapsedSeconds` are distributed before 2400 seconds. The time between 2401 and 3600 seconds is the overtime game, and we can see a significant data-size shrink in overtime. 
# 
# [3]:https://www.kaggle.com/c/march-madness-analytics-2020/data

# In[ ]:


fig = plt.figure(figsize = (20,10))
plt.hist(MEvents['ElapsedSeconds'],bins=60,
         label="data frequency",
         color = '#0f4c81',
         alpha = 0.3,density = False)

plt.title('Distribution of Elapsed Seconds',fontsize=24)
plt.xlabel('Elapsed Seconds',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.style.use('fivethirtyeight')
plt.legend(fontsize=20)
plt.show()


# # Part 2 - Data Pre-processing

# > ### Game-based aggregation in every minute

# <div id='data-preprocessing'>
# As we are going to predict the game-by-game result, the first step of data engineering is transforming the play-by-play data to game-based data.
# Here I used the frequency as the aggregation method, calculating the cumulative frequency for each `EventType` and `SubEventType` grouped by `ElapsedMinutes` in each game. 
# </div>
# 

# To aggregate the game-based data in each minute for both the winning team and losing team in the game, we can: 
#  
# > step 1 : aggregate the frequency of each Event & subevent grouping by season, date, winning team id losing team id, event team id and elapsed time in minutes    
#      
# > step 2 : merge two team's data based on season, day number, winning team id losing team id    
#       
# > step 3 : then, each game will be represented by 2 data rows, one for winning team and one for losing team. The `FinalScore` is the dependent variable that we'd like to predict.     
#      
# > step 4 : in each data entry, events with "_x" represent the current team and its id is `EventTeamID_x`; events with "_y" represents the opposite team in the game with the id as `EvenTeamID_y`     

# In[ ]:


def get_game_based_dat(event_df,gp_para,type_para):
    '''
    create game-based event data
    '''
    ## for EventType
    event_df['count'] = 1
    type_para.remove('EventSubType')
    events = event_df[type_para+['count']].groupby(gp_para+['EventType']).sum()
    events = events.unstack(level=-1).fillna(0)
    events = pd.DataFrame(events.to_records())
    events.columns = [hdr.replace("('count', ", "").replace(")", "").replace('\'','')                          for hdr in events.columns]

    ## for EventSubType
    type_para.remove('EventType')
    type_para.append('EventSubType')
    sub_events = event_df[type_para+['count']].groupby(gp_para+['EventSubType']).count()
    sub_events = sub_events.unstack(level=-1).fillna(0)
    sub_events = pd.DataFrame(sub_events.to_records())
    sub_events.columns = [hdr.replace("('count', ", "sub_",).replace(")", "").replace('\'','')                          for hdr in sub_events.columns]
    
    # merge events and subevents
    events = events.merge(sub_events)

    winevent = events[events['WTeamID']==events['EventTeamID']]
    losevent = events[events['LTeamID']==events['EventTeamID']]

    gp_para.remove('EventTeamID')
    win_merge = winevent.merge(losevent,left_on = gp_para,right_on = gp_para)
    win_merge['victory'] = 1
    los_merge = losevent.merge(winevent,left_on = gp_para,right_on = gp_para)
    los_merge['victory'] = 0
    gameEvents = pd.concat([win_merge,los_merge],axis = 0)

    return gameEvents

def get_final_scores(df,gp_para,target_para):
    type_para = gp_para + target_para
    final_score = df[type_para].groupby(gp_para).max()
    final_score = pd.DataFrame(final_score.to_records())
      
    final_score['victory'] = 1
    w_final_score = final_score[gp_para+['victory',target_para[0]]].    rename(columns={target_para[0]:target_para[0][1:]})
    
    l_final_score = final_score
    l_final_score['victory'] = 0
    l_final_score = final_score[gp_para+['victory',target_para[1]]].    rename(columns={target_para[1]:target_para[1][1:]})
    
    final_score = pd.concat([w_final_score,l_final_score],axis = 0)
    
    return final_score        

def get_current_scores(df,gp_para,target_para):
    type_para = gp_para + target_para
    final_score = df[type_para].groupby(gp_para).max()
    final_score = pd.DataFrame(final_score.to_records())
     
    final_score['victory'] = 1
    w_final_score = final_score.    rename(columns={target_para[0]:target_para[0][1:]+'_x',target_para[1]:target_para[1][1:]+'_y'})
    l_final_score = final_score
    l_final_score['victory'] = 0
    l_final_score = final_score.    rename(columns={target_para[0]:target_para[0][1:]+'_y',target_para[1]:target_para[1][1:]+'_x'})
    final_score = pd.concat([w_final_score,l_final_score],axis = 0)

    return final_score
    

def get_timebucket():
    time_bucket = {min:[i for i in range((min-1)*60+1,min*60+1)] for min in range(1,61)}
    time_bucket[1].append(0)
    return time_bucket

def find_bucket(x,time_bucket):
    for key in time_bucket.keys():
        if x in time_bucket[key]:
            return key
    return None

def pre_processing(events,gp_para,type_para):
    time_bucket = get_timebucket()
    events['ElapsedMinutes'] = events['ElapsedSeconds'].apply(lambda x: find_bucket(x,time_bucket))
    gameEventsMin = get_game_based_dat(events,gp_para,type_para)
    cumu_cols = set(gameEventsMin.columns)-{'EventTeamID_x','EventTeamID_y','ElapsedMinutes'}
    cumuEventsMin = gameEventsMin[cumu_cols].groupby(['Season','DayNum','WTeamID','LTeamID','victory']).cumsum()
    
    EventsMin = pd.concat([gameEventsMin[['Season','DayNum',
              'WTeamID','LTeamID','EventTeamID_x',
              'EventTeamID_y','ElapsedMinutes','victory']],cumuEventsMin],axis = 1)
    
    final_score = get_final_scores(events,
                             gp_para = ['Season','DayNum','WTeamID','LTeamID'],
                             target_para = ['WFinalScore','LFinalScore'])
    current_score = get_current_scores(events,
                               gp_para = ['Season','DayNum','WTeamID','LTeamID','ElapsedMinutes'],
                             target_para = ['WCurrentScore','LCurrentScore'])
    
    EventsMin = EventsMin.merge(final_score)
    EventsMin = EventsMin.merge(current_score)
     
    return EventsMin


def get_column_info(gameEvents):
    g_info = gameEvents.columns[:4].values
    x_info = [i for i in gameEvents.columns.values if '_x' in i]
    x_info.remove('CurrentScore_x')
    x_info.remove('EventTeamID_x')
    y_info = [i for i in gameEvents.columns.values if '_y' in i]
    y_info.remove('CurrentScore_y')
    y_info.remove('EventTeamID_y')
    return x_info,y_info


# After data pre-processing, our data structure changed to the frame below, consist of 2.2 million data entries.    
# With the aggregation process, each game-based data row consists of both winning team play-by-play information and losing team play-by-play information, as well as their current scores.  

# In[ ]:


full_mins_dat = []
for season in [2015,2016,2017,2018,2019]:
    gp_para = ['Season','DayNum','WTeamID','LTeamID','EventTeamID','ElapsedMinutes']
    type_para = gp_para + ['EventType','EventSubType']
    game_season = MEvents[MEvents['Season'] == season]
    #print(game_season.shape)
    game_min_dat = pre_processing(game_season,gp_para,type_para)
    full_mins_dat.append(game_min_dat)

Events_Min = pd.concat(full_mins_dat,axis = 0)
    


# In[ ]:


print("the aggregated event shape is :", Events_Min.shape)
x_info,y_info = get_column_info(Events_Min)
print("game info: \n",gp_para+['EventTeamID_y','EventTeamID_x'])
print("team x info: \n",x_info)
print("team y info: \n",y_info)
y_labels = list(set(Events_Min.columns)-set(x_info)-set(y_info)                -set(gp_para)-{'EventTeamID_y','EventTeamID_x'})
print("dependent variables: \n",y_labels)


# Here is the data distribution in minutes, just as shown in the second distribution before, major data points regarding minute spread in the area before 40 minutes as a full game is approximately 40-minute long.  Events that happened between 41 minutes to 60 minutes are considered to be in the overtime game. But with a significant drop in data sample size, I will **focus on the main game prediction** in this report.   

# In[ ]:


fig = plt.figure(figsize=(20,10))
plt.hist(Events_Min['ElapsedMinutes'],bins=60,
         label="data frenquency",
         color = '#0f4c81',
         alpha = 0.5)
plt.title('Data Distribution in Minutes',fontsize=24)
plt.xlabel('Elapsed Minutes',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.style.use('fivethirtyeight')
plt.legend(fontsize=20)
plt.show()


# # Part 3 : In-play Live Modeling

# >### Modeling with Linear Regression

# <div id="models">
# First of, with all the data aggregated, let's construct linear regression models with L1 and L2 norm.  
# Instead of generating a single regression model, I generated 40 models using the cumulative aggregated data in each minute of the main game with Lasso, Ridge regression models, compared with default models that use the average as the prediction. We are expecting the model prediction is more and more accurate when the game keeps going on.    
# </div>

# In[ ]:


# cross validation
def cv_split(data,kfold):
    dat = data.sample(frac=1,random_state=888).reset_index()
    dat = dat.iloc[:,1:]
    fsize = len(dat)//kfold
    lsize = len(dat)%kfold
    fold_label = []
    for k in range(kfold-1):
        fold_label += [k for i in range(fsize)]
    fold_label += [kfold-1 for i in range(fsize+lsize)]
    dat['kfold'] = fold_label
    return dat


# Ridge Regression
def get_pred_ridge(train,test,penalty=None):
    lr = Ridge()
    lr.fit(X=train.iloc[:,:-1],y=train.iloc[:,-1])
    predict = pd.DataFrame(data= lr.predict(test.iloc[:, :-1]),columns=['prediction'])
    predict['true output'] = np.array(test.iloc[:,-1]).reshape(len(test),1)
    return predict


# Lasso Regression
def get_pred_lasso(train,test,penalty=None):
    lr = Lasso()
    lr.fit(X=train.iloc[:,:-1],y=train.iloc[:,-1])
    predict = pd.DataFrame(data= lr.predict(test.iloc[:, :-1]),columns=['prediction'])
    predict['true output'] = np.array(test.iloc[:,-1]).reshape(len(test),1)
    return predict

def cv_mse(cv_data,model,para):
    agg_pre = pd.DataFrame(columns=['prediction','true output'])
    for k in cv_data.iloc[:,-1].unique():
        test = cv_data[cv_data.iloc[:,-1]==k].iloc[:,:-1]
        #print(test.iloc[:4,-1])
        train = cv_data[cv_data.iloc[:,-1]!=k].iloc[:,:-1]
        #print(train.iloc[:4,-1])
        pred = model(train,test,para)
        agg_pre = pd.concat([agg_pre,pred],axis = 0)
    mse = np.mean((agg_pre['prediction'] - agg_pre['true output'])**2)
    return mse


def lr_by_time(df,train_label,model,interval,kfold):
    iteration = df['ElapsedMinutes'].max()//interval
    #print(iteration)
    mse_df = []
    for i in range(1,iteration+1):
        train_dat = df[df['ElapsedMinutes']==i*interval][train_label]
        #print("data szie in {} minutes : {}".format(i*interval,train_dat.shape[0]))
        cv_dat = cv_split(train_dat,kfold)
        mse = cv_mse(cv_dat,model,None)
        mse_df.append([i*interval,mse,train_dat.shape[0]])
    mse_df = pd.DataFrame(mse_df,columns = ['ElapsedMinutes','MSE','DSize'])
    return mse_df        


def get_default_mode(df,interval):
    iteration = df['ElapsedMinutes'].max()//interval
    mse_df = []
    for i in range(1,iteration+1):
        train_dat = df[df['ElapsedMinutes']==i*interval][train_label]
        predict_val = train_dat['FinalScore'].mean()
        mse = np.mean((train_dat['FinalScore'] - predict_val)**2)
        mse_df.append([i*interval,mse])
    mse_df = pd.DataFrame(mse_df,columns = ['ElapsedMinutes','MSE'])
    return mse_df
        

cols = ['#1b262c','#0f4c81','#888888']
def get_reg_fig(df_list,model_name,cols):
    plt.style.use('fivethirtyeight')
    i = 0
    f, axs = plt.subplots(1,1,figsize=(20,10))
    for df in df_list:
        df_main = df[df['ElapsedMinutes']<=40]  
        plt.plot(df_main['ElapsedMinutes'],df_main['MSE'].apply(np.sqrt),
                 label=model_name[i],color = cols[i],alpha=0.8)
        i += 1
    plt.title('Models RMSE in Main Game',fontsize = 24)
    plt.xlabel('Elapsed Minutes',fontsize=20)
    plt.ylabel('RMSE',fontsize=20)
    axs.tick_params(axis='both', which='major', labelsize=20) 
    plt.legend(fontsize=20)
    '''
    f, axs = plt.subplots(1,1,figsize=(20,7))
    i = 0    
    for df in df_list:
        df_over = df[df['ElapsedMinutes']>40]
        plt.plot(df_over['ElapsedMinutes'],df_over['MSE'].apply(np.sqrt),
                 label=model_name[i],color = cols[i],alpha=0.8)
        plt.title('Models RMSE in Overtime Game')
        i += 1
    plt.legend()
    '''


# The model prediction result follows our expectation, that with more data aggregated when the game marching forward.   
# According to the RMSE result, the is likely to make a [-10,+10] prediction mistake to the final scores in the first 10 minutes. Later on, the prediction mistake interval decreased to approximately [-8,+8] with the event data of the first half. The prediction mistake interval keeps decreasing to approximately [-6,+6] before the final betting time for most of the live wagering - 5 minutes before the game ends. 

# In[ ]:


Events_Min.shape
Events_Min = Events_Min.fillna(0)

x_info,y_info = get_column_info(Events_Min)
train_label = x_info + y_info +['FinalScore']        

mse_df_ridge = lr_by_time(Events_Min[Events_Min['ElapsedMinutes']<=40],train_label,get_pred_ridge,1,20)
mse_df_lasso = lr_by_time(Events_Min[Events_Min['ElapsedMinutes']<=40],train_label,get_pred_lasso,1,20)
mse_df_default = get_default_mode(Events_Min[Events_Min['ElapsedMinutes']<=40],1)
df_list = [mse_df_ridge,mse_df_lasso,mse_df_default]

# default model regression 
model_name = ['Ridge','Lasso','default']
get_reg_fig(df_list,model_name,cols)


# Now it is time for validating the model in 2020 men's play-by-play data. 
# The result turns out to be pretty good, with no over-fitting in the testing data. The reason for the low error rate for the 2020 validation data could be the cancelation of the 2020 season. Compare the 2 default models (take the average as prediction), we can see that the 2020 `FinalScore` mean is lower than the mean of 2015-2019. This could result from that the 2020 season got canceled in the middle due to the pandemic, thus the data distribution is different from the previous seasons.  

# In[ ]:


def validation_mse(df,test,train_label,m_instance,interval):
    iteration = df['ElapsedMinutes'].max()//interval
    #print(iteration)
    mse_ls = []
    for i in range(1,iteration+1):
        train_dat = df[df['ElapsedMinutes']==i*interval][train_label]
        test_dat =  test[test['ElapsedMinutes']==i*interval][train_label]
        lr = m_instance
        X = train_dat.iloc[:,:-1]
        y = train_dat.iloc[:,-1]
        lr.fit(X,y)
        pred = pd.DataFrame(data= lr.predict(test_dat.iloc[:, :-1]),columns=['prediction'])
        pred['true output'] = np.array(test_dat.iloc[:,-1]).reshape(len(test_dat),1)
        mse = np.mean((pred['true output']-pred['prediction'])**2)
        mse_ls.append([i*interval,mse])
    mse_df = pd.DataFrame(mse_ls,columns=['ElapsedMinutes','MSE'])
    return mse_df      

gp_para = ['Season','DayNum','WTeamID','LTeamID','EventTeamID','ElapsedMinutes']
type_para = gp_para + ['EventType','EventSubType']
# get 2020 data
year = '2020'
Events_20 = pd.read_csv(f'{MENS_DIR}/MEvents{year}.csv')
Events_20_Min = pre_processing(Events_20,gp_para,type_para)

x_info,y_info = get_column_info(Events_Min)
train_label = x_info + y_info +['FinalScore']

mse_validation = validation_mse(Events_Min[Events_Min['ElapsedMinutes']<=40],
                                  Events_20_Min[Events_20_Min['ElapsedMinutes']<=40],train_label,Ridge(),1)

cols = ['#1b262c','#f79071','#888888','#ffc38b']
#rmse_validation['MSE'] = rmse_validation['RMSE'].apply(lambda x: x**2)
mse_df_default_20 = get_default_mode(Events_20_Min[Events_20_Min['ElapsedMinutes']<=40],1)
df_list = [mse_df_ridge,mse_validation,mse_df_default,mse_df_default_20]
model_name = ['Training set - 2015~19','Testing set - 2020','default with 2015~19 data',
              'default with 2020 data']
get_reg_fig(df_list,model_name,cols)


# # Part 4 : Go Further from Linear Regression

# >### What are the game-changing moments?

# <div id="game-changing">
# The next question is, what are the most important play-by-play events that influenced the final score?   
# Next, by further examing the coefficients in the models, let's explore what are the game-changing moments in the game.  
# </div>

# In[ ]:


def critical_vars(df,train_label,m_instance,interval):
    iteration = df['ElapsedMinutes'].max()//interval
    #print(iteration)
    coef = []
    for i in range(1,iteration+1):
        train_dat = df[df['ElapsedMinutes']==i*interval][train_label]
        #print("data szie in {} minutes : {}".format(i*interval,train_dat.shape[0]))
        lr = m_instance
        X = train_dat.iloc[:,:-1]
        y = train_dat.iloc[:,-1]
        lr.fit(X,y)
        c_df = pd.DataFrame(zip([i*interval for n in range(len(X))],X.columns, lr.coef_),
                        columns=['ElapsedMinutes','variables','coef']).\
        sort_values('coef',ascending=False).dropna()
        c_df = pd.concat([c_df[:5],c_df[-5:]],axis = 0)
        #print(c_df)
        coef.append(c_df)
    #coef_df = pd.concat(coef,axis = 1)
    return coef


# Let's look into the Ridge regression result for the main game.        
#         
# The table below displays the top 5 events that have the most positive effect for a team's final score during the main game, as well as top 5 events that have the most negative effect for a team's final score, according to the coefficients of the Ridge Regressions from 10-mins data, 20-mins data, 30-mins data, and 40-mins data.     
#     
#     
# Altogether, 24 events can be considered as **game-changing** during the main game. With the time-to-time coefficient change information given by the model, there are a lot of interesting facts that we can implicate from the data, most of them are strongly related to how to adjust team strategy over time and of the great importance of winning the game. 

# In[ ]:


x_info,y_info = get_column_info(Events_Min)
train_label = x_info + y_info +['FinalScore'] 

coef = critical_vars(Events_Min.query('ElapsedMinutes<=40'),train_label,Ridge(),10)
coef_df = pd.concat(coef,axis =1).fillna('na').reset_index().drop('index',axis = 1)
full_coef = critical_vars(Events_Min.query('ElapsedMinutes<=40'),train_label,Ridge(),1)
full_coef_df = pd.concat(full_coef,axis =0).fillna('na')
coef_df


# Here are the interesting and important facts that we can implicate from the data:
# > **1. It is very important to make a 3-point shot in the beginning!**    
# The more time elapsed, the less important a 3 points shot is, as the `coef.` drops from **3.33** to **2.69** while a game is playing.   
# While 2-point shots are worth their values during the middle of the game; and a 1-point shot (mostly for a free shot) is vital at the last minute of the game, occurs only once in the top-5 board in the 40-minute regression model with the coefficient of **0.85**.  

# In[ ]:


cols = ['#00a8cc','#0f4c81','#ed6663','#f79071','#1b262c']
def get_coef_fig(var_names,coef_df,cols):
    plt.style.use('fivethirtyeight')
    i = 0
    f, axs = plt.subplots(1,1,figsize=(20,10))
    for var in var_names:
        tmp = coef_df[coef_df['variables']==var]
        if len(tmp)>1:
            plt.plot(tmp['ElapsedMinutes'],tmp['coef'],
                 label=var,color = cols[i],alpha=0.8)
            i += 1
       
        elif len(tmp)==1:
            plt.plot(tmp['ElapsedMinutes'],tmp['coef'],
                 label=var,color = cols[i],alpha=0.8, marker='o', markersize=10)
            i += 1
    plt.title('Coefficients for '+ ','.join(var_names),fontsize=24)
    plt.xlabel('Elapsed Minutes',fontsize=20)
    plt.ylabel('Coefficient',fontsize=20)
    plt.legend(fontsize = 20)
    axs.tick_params(axis='both', which='major', labelsize=20)
    plt.show()


made = ['made1_x','made2_x','made3_x']
get_coef_fig(made,full_coef_df,cols)
#tmp = full_coef_df[full_coef_df['variables'].isin(made)]
#fig, ax = plt.subplots(figsize=(15,7))
#tmp.groupby('variables').plot('ElapsedTime','coef')


# > **2. Don't make the 5-second violation in the beginning.**     
# The effect of making a 5-second violation (`sub_5sec_x`)in the first 5 minutes is disastrous, which has the potential to 
# cause more than 6 points lost to the final score.      
# However, in the second half of the game, a 5-second violation seems to have a positive effect on the final score.  

# In[ ]:


sec = ['sub_5sec_x']
get_coef_fig(sec,full_coef_df,cols = ['#0f4c81'])


# > **3. A foul act could be harmful, even for technical ones.**    
# A bench technical foul (identified as `sub_bente_x`) is beneficial in the middle of the game. with a `coef.` of 2.63.    
# However, a team should be careful about the foul act, even if it is a technical foul (identified as `sub_admte_x`), as the administrative-technical foul is harmful with the `coef.` of -1.63 in the first half of the game.   

# In[ ]:


foul = ['sub_bente_x','sub_admte_x']
get_coef_fig(foul,full_coef_df,cols=['#0f4c81','#ed6663'])


# > **4. It is okay to lose the jump balls when the game approaches the end.**   
#     The side effect of losing a jump ball (`sub_lost_x`) is mitigated as the game gets more and more intense in the end.   

# In[ ]:


jumpball = ['sub_lost_x']
get_coef_fig(jumpball,full_coef_df,cols=['#0f4c81','#ed6663'])


# > **5. Don't give a 3-free-throw chance to the other team in the last 10 minutes**   
# In the last 10 minutes, even if your opponent misses a free shot out of 3 free throws, your final score will still get effected by -1 score. The benefit of your opponent missing 3 out of 3 to your team's final score decreases in the last 10 minutes.

# In[ ]:


missfree = ['sub_1of3_y','sub_2of3_y','sub_3of3_y']
get_coef_fig(missfree,full_coef_df,cols=['#00a8cc','#0f4c81','#ed6663'])


# > References:     
# [1] : What Is In Play Sports Betting?
# https://www.thelines.com/betting/in-play/    
# [2] :  What Is Point Spread Betting?
# https://www.thelines.com/betting/point-spread/    
# [3] : Google Cloud & NCAA® March Madness Analytics - Data Description
# https://www.kaggle.com/c/march-madness-analytics-2020/data
