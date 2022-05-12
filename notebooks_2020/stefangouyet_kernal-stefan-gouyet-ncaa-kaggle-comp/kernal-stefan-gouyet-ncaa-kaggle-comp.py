#!/usr/bin/env python
# coding: utf-8

# # A Trend of March Madness
# 
# ## Contributors:
# * Stefan Gouyet 
# * Alexander Gouyet
# 
# ![March Madness](http://thefederalist.com/wp-content/uploads/2020/03/2347392515_3753b32db8_o-998x749.jpg)

# > “In computer science unnecessary comparisons are always bad, a waste of time and effort. But in sports that’s far from the case. In many respects, after all, the games themselves are the point. “
# 
# *Algorithms to Live By*
# 

# In[ ]:


import numpy as np
import pandas as pd
import os
import gc
from datetime import datetime, timedelta, date
import time
import os


# In[ ]:


##need pytrends for Google Trends analysis (https://pypi.org/project/pytrends/)
#!pip install pytrends
get_ipython().system('pip install --upgrade --user git+https://github.com/GeneralMills/pytrends')
import pytrends


# ## An Exploratory Analysis of March Madness Data, with Google Trends

# # Notebook Chapters:
# 
# ### #1. Gather Data and Data Manipulation
# * *Get all relevant data from Kaggle dataset*
# 
# ### #2. Google Trends Data
# * *Use Pytrends package and Google Trends to obtain interest about team prior to game day*
# 
# ### #3. Findings
# * Is there a relationship between Google Trends interest and the winner?
# 

# # Part 1. Combine Data Sources

# # Data Sources:
# 
# ### Reference Data:
# 1. Teams ID (MTeams.csv)
# 2. Season Data (MSeasons.csv)
# 3. Team Coaches (MTeamCoaches.csv)
# 
# ### Main Data:
# 4. Regular Season Data (MRegularSeasonCompactResults.csv)
# 5. Tournament Data (MTournamentCompactResults.csv)
# 
# 
# ### Additional (External) Data:
# 6. Google Trends (via Python pytrends package)
# 
# 

# In[ ]:


MAIN_DIR = '../input/march-madness-analytics-2020/'


# In[ ]:


#Dataset 1: Teams ID
MTeams = pd.read_csv(f'{MAIN_DIR}/MDataFiles_Stage2/MTeams.csv')
MTeams.columns

MTeams = MTeams[['TeamID','TeamName','FirstD1Season']]
MTeams.sort_values('FirstD1Season', ascending=False).head()


# In[ ]:


#Dataset 2: Seasons Data
MSeasons = pd.read_csv(f'{MAIN_DIR}/MDataFiles_Stage2/MSeasons.csv')

##Change date format to YYYY-mm-dd for Google Trends data
MSeasons['DayZero'] = MSeasons.apply(lambda row: datetime.strptime(row['DayZero'],                                                                                '%m/%d/%Y').strftime('%Y-%m-%d'),axis=1)

MSeasons.head()


# In[ ]:


#Dataset 3: Coaches Data
MCoaches = pd.read_csv(f'{MAIN_DIR}/MDataFiles_Stage2/MTeamCoaches.csv')
MCoaches.head()

##drop duplicates by (Season, TeamID); only keep coach who was in the dataset at March Madness time
MCoaches.drop_duplicates(subset = ['Season','TeamID'], keep = 'last',inplace=True)

##change CoachName format to {FirstName LastName} for Google Trends analysis
MCoaches['CoachName'] = MCoaches.apply(lambda row: row['CoachName'].replace('_',' '),axis=1)


# In[ ]:


MCoaches.tail()


# ## **Dataset 4: Regular Season Results**
# 

# In[ ]:


#Dataset 4: Get Regular Season Results
MRegularSeasonCompactResults = pd.read_csv(f'{MAIN_DIR}/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')

##Keep only 2004-present to match Google Trends data
MRegularSeasonCompactResults = MRegularSeasonCompactResults.loc[MRegularSeasonCompactResults['Season'] >= 2003]


# In[ ]:


MRegularSeasonCompactResults.shape[0]


# In[ ]:


MRegularSeasonCompactResults.head()


# In[ ]:


MRegularSeasonCompactResults_Melted = pd.melt(MRegularSeasonCompactResults, 
        id_vars=['Season','DayNum','WScore','LScore','WLoc','NumOT'],
        value_vars=['WTeamID','LTeamID'],
        var_name='Result', value_name='TeamID')

MRegularSeasonCompactResults_Melted['Result'] = MRegularSeasonCompactResults_Melted.apply(lambda row: 1 if row['Result'] == 'WTeamID'                                                                                         else 0,axis=1)


# In[ ]:


MRegularSeasonCompactResults_Melted.sort_values(['Season','DayNum','WScore','LScore']).head(10)


# In[ ]:


MRegularSeasonCompactResults_Melted_Wins = MRegularSeasonCompactResults_Melted.groupby(['Season', 'TeamID']).agg({'Result':"sum"}).rename(columns={'Result':'Reg_Season_Wins_Num'}).reset_index()


# In[ ]:


MRegularSeasonCompactResults_Melted_Wins.sort_values(['Reg_Season_Wins_Num'],ascending=False)


# ## Main Dataset: Historical Tournament Results

# In[ ]:


#Dataset 5: Tourney Compact Results
MTournamentCompactResults = pd.read_csv(f'{MAIN_DIR}/MDataFiles_Stage2/MNCAATourneyCompactResults.csv')

##Keep only tournament results 2004 -present
MTournamentCompactResults = MTournamentCompactResults.loc[MTournamentCompactResults['Season'] > 2003]


# In[ ]:


MTournamentCompactResults.head()


# In[ ]:


##merge tournament  results with reg season wins count
MTournamentCompactResults =  MTournamentCompactResults.merge(MRegularSeasonCompactResults_Melted_Wins,
                                                                   left_on=['Season','WTeamID'],
                                                                   right_on=['Season','TeamID'],
                                                                   validate='many_to_one').drop('TeamID', axis=1)\
                                                            .rename(columns={'Reg_Season_Wins_Num': 'WTeam_Reg_Season_Wins'}) \
                                                            .merge(MRegularSeasonCompactResults_Melted_Wins,
                                                                   left_on=['Season','LTeamID'],
                                                                   right_on=['Season','TeamID']).drop('TeamID', axis=1)\
                                                            .rename(columns={'Reg_Season_Wins_Num': 'LTeam_Reg_Season_Wins'})


# In[ ]:


MTournamentCompactResults.head()


# In[ ]:


##Get game date (DayZero + DayNum)
MTournamentCompactResults['GameDate']= MTournamentCompactResults.apply(lambda row: (datetime.strptime(MSeasons.loc[MSeasons['Season']==row['Season'],
                                                                                                     'DayZero'].item(),'%Y-%m-%d') \
                                                                      + timedelta(days=row['DayNum'])),axis=1)


# In[ ]:


MTournamentCompactResults.GameDate.head()


# In[ ]:


##merge regular season results with teamID
MTournamentCompactResults =  MTournamentCompactResults.merge(MTeams[['TeamName', 'TeamID']],
                                                                   left_on='WTeamID',
                                                                   right_on='TeamID',
                                                                   validate='many_to_one').drop('TeamID', axis=1) \
                                                            .rename(columns={'TeamName': 'WTeam'}) \
                                                            .merge(MTeams[['TeamName', 'TeamID']],
                                                                   left_on='LTeamID',
                                                                   right_on='TeamID') \
                                                            .drop('TeamID', axis=1) \
                                                            .rename(columns={'TeamName': 'LTeam'})


# In[ ]:


MCoaches.head()


# In[ ]:


##merge tournament results with Team Coach
MTournamentCompactResults =  MTournamentCompactResults.merge(MCoaches[['Season', 'TeamID', 'CoachName']],
                                                                   left_on=['Season','WTeamID'],
                                                                   right_on=['Season','TeamID'],
                                                                   validate='many_to_one').drop('TeamID', axis=1) \
                                                            .rename(columns={'CoachName': 'WTeamCoach'}) \
                                                            .merge(MCoaches[['Season', 'TeamID', 'CoachName']],
                                                                   left_on=['Season','LTeamID'],
                                                                   right_on=['Season','TeamID'],) \
                                                            .drop('TeamID', axis=1) \
                                                            .rename(columns={'CoachName': 'LTeamCoach'})


# In[ ]:


MTournamentCompactResults.head(10)


# # Part 2. Add Google Trends data

# ## Methodology:
# ### #1. Look at Google Trends interest for both teams
# ### #2. Use {TeamName} + '  basketball' to prevent possible inaccuracy of search
# ### #2. Use data from day prior to the {GameDate}

# ## Because pytrends is slow, I have already ran the Google Trends data and will import a new CSV (./MTournamentCompactResults_with_trends.csv) with the data after pytrends from our input/ folder. 
# 
# ### The below code displays the function get_g_trends_data(), which takes two keywords/phrases and a date, returning Google Trends interest data for both keywords/phrases on the day prior to the date paramater.
# 
# ** note: I chose to analyze date prior to the game date to reduce the possibility of gathering data after the game has taken place (i.e. gathering data at 11:00pm EST after game X has already occured).
# 
# 

# In[ ]:


##adding in data after running get_g_trends_data()
MTournamentCompactResults_with_trends = pd.read_csv('../input/mtournamentcompactresults-with-trends1/MTournamentCompactResults_with_trends.csv')


# ```
# 
# ##Adding in Google Trends
# from pytrends.request import TrendReq
# pytrend = TrendReq()
# 
# def get_g_trends_data(val1,val2,game_date):
#     """
#     Get Google Trends data using pytrends, for two keywords/phrases at specific date.
# 
#     Args:
#         val1,val2: keywords/phrases to analyze
#         game_date: date of event
# 
#     Returns:
#         A Pandas Series with two values: Google Trends interest at (date - 1) for two keywords/phrases 
# 
#     """
#     
#     #pytrend.build_payload(kw_list=list(team1,team2), timeframe='2016-12-14 2017-01-25', geo = 'US')
#     if not datetime.strptime(game_date, '%Y-%m-%d'):
#         raise ValueError("Issue with game_date: need in format %Y-%m-%d")
#     
#     day_before_game_date = (datetime.strptime(game_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d') 
#     print('using date {}'.format(day_before_game_date))
#     
#     try:
#         pytrend.build_payload(kw_list=[val1+ str(' basketball'),
#                                        val2+ str(' basketball')], timeframe= '{} {}'.format(day_before_game_date,
#                                                                                day_before_game_date), geo = 'US')
#         interest_over_time_df = pytrend.interest_over_time()
#         val1_interest = interest_over_time_df.iloc[0,0]
#         val2_interest = interest_over_time_df.iloc[0,1]
#         print(val1 + ' interest: ' + str(val1_interest))
#         print(val2 + ' interest: ' + str(val2_interest))
#     except Exception as e:
#         print('Error: {}'.format(e))
#         val1_interest = None
#         val2_interest = None
# 
#     
#     time.sleep(1)
#     return pd.Series([val1_interest, val2_interest])
#     
# ```

# ```
# MTournamentCompactResults[['team_1_interest','team_2_interest']] = MTournamentCompactResults.apply(lambda row: get_g_trends_data(val1 = row['WTeam'],
#                                                                                                                          val2 = row['LTeam'],
#                                                                                                                          game_date = str(row['GameDate']).split(' 00:')[0]),
#                                                                                                                          axis=1)
#                                                                                                                          
# MTournamentCompactResults.to_csv('MTournamentCompactResults_with_trends.csv',index=False)                                                                                                               
#                                                                                                                
# ```

# In[ ]:


MTournamentCompactResults_with_trends.head()


# ## Now that we have the data: what relationship can we see between Interest in the two teams, and who won the game?

# In[ ]:


MTournamentCompactResults_with_trends.info()


# In[ ]:


MTournamentCompactResults_with_trends.loc[: , "team_1_interest":"team_2_interest"].mean()


# In[ ]:


MTournamentCompactResults_with_trends.loc[: , "team_1_interest":"team_2_interest"].median()


# In[ ]:


MTournamentCompactResults_with_trends.boxplot(column=['team_1_interest', 'team_2_interest'])


# ## Very interesting! This is some simple and preliminary analysis, but it points to a possible relationship between Google Trends interest and the result of a March Madness game.

# ### Thanks for reading!
# 
# -Stefan and Alexander

# In[ ]:




