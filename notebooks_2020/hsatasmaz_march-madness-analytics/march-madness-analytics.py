#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os, sys, gc
import matplotlib.pyplot as plt


# ### READ FILES

# In[ ]:


num_files = 0
#files = {}

last_folders = []
file_list = []
file_short = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    last_folder = str(dirname).split('/')[-1]
    
    for filename in filenames:
        file = str(filename).split('.')[0]
        
        last_folders.append(last_folder)
        
        file_curr = 'DF_' + last_folder.replace('-', '_') + '_' + file        
        file_list.append(file_curr)
        file_short.append(file)
        
        df = pd.read_csv(os.path.join(dirname, filename), encoding='Latin5')
        exec(file_curr + " = " + str('df'))
        num_files += 1        
        
print('Num files read: ', num_files)


# In[ ]:


DF_LIST = pd.DataFrame(file_list)
DF_LIST.columns = ['DF_NAME']

DF_LIST['FOLDER'] = last_folders
DF_LIST['FILE'] = file_short

# Remove trails (-) and replace with underscore (_)
DF_LIST['FOLDER'] = DF_LIST['FOLDER'].apply(lambda f: f.replace('-', '_'))


# Add descriptive file attributes

# In[ ]:


DF_LIST['FILE_TYPE'] = DF_LIST['FILE'].apply(lambda f: 'Gender specific' if f[0] in ['W', 'M'] else 'Common')

DF_LIST['FILE_COMMON'] = DF_LIST['FILE'].apply(lambda f: f[1:] if f[0] in ['W', 'M'] else f).apply(
                                               lambda f: 'W' if f.find('Womens')>0 else f).apply(
                                               lambda f: 'M' if f.find('Mens')>0 else f)

DF_LIST['FOLDER_COMMON'] = DF_LIST['FOLDER'].apply(lambda f: f[1:] if f[0] in ['W', 'M'] else f).apply(
                                            lambda f: f.replace('_Womens', '')).apply(
                                            lambda f: f.replace('_Mens', ''))

DF_LIST['GENDER'] = DF_LIST[['FILE', 'FOLDER']].apply(
                                            lambda f: f.FILE[0] if f.FILE[0] in ['W', 'M'] else f.FOLDER, 
                                            axis=1).apply(
                                            lambda f: 'M' if f.find('Mens')>0 or f[0] =='M' else f).apply(
                                            lambda f: 'W' if f.find('Womens')>0 or f[0] == 'W' else f)


# Combine men and women's data in order to reduce df number and be able to analyze male and female plays together

# In[ ]:


def append_men_women_df(df_men=None, df_women=None):
    
    if not df_men.empty and not df_women.empty:
        df_men['GENDER']   = 'M'
        df_women['GENDER'] = 'W'
        return(df_men.append(df_women, ignore_index=True, sort=False))
    
    elif df_men.empty and not df_women.empty:
        df_women['GENDER'] = 'W'
        return(df_women)
    
    elif not df_men.empty and df_women.empty:
        df_men['GENDER']   = 'M'
        return(df_men)    


# In[ ]:


merged_df = []
merged_df_counter = 0
default_df = pd.DataFrame() 

file_common = DF_LIST.query('FILE_TYPE=="Gender specific"').FILE_COMMON.unique()

for f in file_common:
    df_common_1 = DF_LIST.query('FILE_COMMON=="{}"'.format(f))
    folder_common = df_common_1.FOLDER_COMMON.unique()
    
    if f.find('Events') >= 0:
        continue
    
    for f2 in folder_common:        
        df_name_new = 'DF_' + f2 + '_' + f        
        
        df_common = df_common_1.query('FOLDER_COMMON=="{}"'.format(f2))
        
        df_men = df_common.query('GENDER=="M"').DF_NAME.values[0] if df_common.query('GENDER=="M"').shape[0]>0 else 'default_df'
        df_women = df_common.query('GENDER=="W"').DF_NAME.values[0] if df_common.query('GENDER=="W"').shape[0]>0 else 'default_df'
        
        df_new = append_men_women_df(eval(df_men), eval(df_women))        
        exec(df_name_new + " = " + str('df_new'))
        
        merged_df_counter = merged_df_counter + 1 if df_men != 'default_df' else merged_df_counter
        merged_df_counter = merged_df_counter + 1 if df_women != 'default_df' else merged_df_counter
        
        merged_df.append(df_name_new)

print('Number of df merged', merged_df_counter)


# In[ ]:


merged_df2 = [] 

file_common = DF_LIST.query('FILE_TYPE!="Gender specific"').FILE_COMMON.unique()

for f in file_common:
    df_common_1 = DF_LIST.query('FILE_COMMON=="{}"'.format(f))
    folder_common = df_common_1.FOLDER_COMMON.unique()
    
    for f2 in folder_common:        
        df_name_new = 'DF_' + f2 + '_' + f        
        
        df_common = df_common_1.query('FOLDER_COMMON=="{}"'.format(f2))
        
        df_men = df_common.query('GENDER=="M"').DF_NAME.values[0] if df_common.query('GENDER=="M"').shape[0]>0 else 'default_df'
        df_women = df_common.query('GENDER=="W"').DF_NAME.values[0] if df_common.query('GENDER=="W"').shape[0]>0 else 'default_df'
        
        df_new = append_men_women_df(eval(df_men), eval(df_women))        
        exec(df_name_new + " = " + str('df_new'))
        
        merged_df2.append(df_name_new)

print('Number of df merged', len(merged_df2))


# In[ ]:


DF_Cities      = DF_DataFiles_Stage2_Cities.append(DF_DataFiles_Stage1_Cities).drop_duplicates()
DF_Conferences = DF_DataFiles_Stage2_Conferences.append(DF_DataFiles_Stage1_Conferences).drop_duplicates()


# ### CONTROL 
# 
# Check whether dataframes are correctly merged or not

# In[ ]:


def check_sum():
    
    total_rows = 0
    
    print('Checking merged data with the original...')
    
    # exclude df which are not merged
    df_exc = DF_LIST.query('(DF_NAME.str.contains("Cities|Conferences") and FILE_TYPE =="Common") or                              DF_NAME.str.contains("Events")', engine='python').DF_NAME.unique()
    df_ = [d for d in DF_LIST.DF_NAME if d not in df_exc]

    print('Excluding Events, Cities and Conferences which are not gender specific')

    for d in df_:
        total_rows += eval(d).shape[0]

    print(total_rows, ': Total Rows before dataframes merged')

    total_rows_check = 0

    for d in merged_df:
        total_rows_check += eval(d).shape[0]

    print(total_rows_check, ': Total Rows after dataframes merged')

    print('\nTotal rows match: ', total_rows == total_rows_check)


# In[ ]:


check_sum()


# ### ANALYSIS OF EVENTS IN DETAIL

# Requires internet connection

# In[ ]:


def plot_court(version=1):
    
    court_image1 = 'https://developer.geniussports.com/warehouse/rest/basketball_coords.png'
    court_image2 = 'https://developer.geniussports.com/warehouse/rest/basketball_courtmap.png'
    
    court_image  = eval('court_image' + str(version))
    
    img = plt.imread(court_image)
    fig, ax = plt.subplots(figsize=(24,8))
    ax.imshow(img)
    plt.axis('off');


# In[ ]:


#plot_court(2)


# In[ ]:


sample_pid = DF_MPlayByPlay_Stage2_MPlayers.PlayerID.sample(1).values[0]

sample1 = DF_MPlayByPlay_Stage2_MPlayers.query('PlayerID=={}'.format(sample_pid))
sample2 = DF_2020_Mens_Data_MPlayers.query('PlayerID=={}'.format(sample_pid))

sample1.append(sample2)


# In[ ]:


sample_eid = DF_MPlayByPlay_Stage2_MEvents2015.EventID.sample(1).values[0]

sample1 = DF_MPlayByPlay_Stage2_MEvents2015.query('EventID=={}'.format(sample_eid))
sample2 = DF_2020_Mens_Data_MEvents2015.query('EventID=={}'.format(sample_eid))

sample1.append(sample2)


# ### FINAL DATAFRAMES TO ANALYZE
# In order to reduce analysis effort and memory usage, only Stage 2 files are considered and the remaining are deleted

# In[ ]:


FINAL_DF_LIST = [f for f in merged_df if f.find('Stage1')<0]


# In[ ]:


FINAL_DF_LIST_EVENTS = DF_LIST.query('DF_NAME.str.contains("Stage2") and FILE_COMMON.str.contains("Events")', engine='python').DF_NAME.unique()


# Delete unused dataframes, all df excluding "Events"

# In[ ]:


DF_2_DEL = [df for df in DF_LIST.DF_NAME.unique() if df not in FINAL_DF_LIST_EVENTS]

for df_name in DF_2_DEL:
    exec('del ' + str(df_name))
    
gc.collect()


# ### TEAM and PLAYER MODELLING
# 
# Explore different player and team profiles based on event performance

# In[ ]:


def time_period_obsolete(elapsed_seconds):
    if elapsed_seconds <= 1200:
        i  = elapsed_seconds // 300
        tp = 'First Half {}-{} min'.format(str(i*5), str((i+1)*5))
    elif elapsed_seconds <= 2400:
        i  = (elapsed_seconds - 1200) // 300
        tp = 'Second Half {}-{} min'.format(str(i*5), str((i+1)*5))
    else:
        # trailing part is added to include 300th or 60th seconds in the previous period not in the next
        ot = (elapsed_seconds - 2400) // 300.00001
        i  = (elapsed_seconds - 2400 - ot * 300) // 60.00001   
        tp = 'Overtime ({}) {}-{} min'.format(str(int(ot) + 1), str(int(i)*1), str((int(i)+1)*1))
    
    return(tp)


# In[ ]:


def time_period(elapsed_seconds):
    
    if elapsed_seconds <= 1200:
        tp = 'FirstHalf'
    elif elapsed_seconds <= 2400:
        tp = 'SecondHalf'
    elif elapsed_seconds <= 2700:
        tp = 'OverTime1'
    elif elapsed_seconds <= 3000:
        tp = 'OverTime2'
    elif elapsed_seconds <= 3300:
        tp = 'OverTime3'
    elif elapsed_seconds <= 3600:
        tp = 'OverTime4'
    else:
        tp = 'OverTime4P'
    
    return(tp)


# Add time period to all events dataframes

# In[ ]:


for df_name in FINAL_DF_LIST_EVENTS:
    df = eval(df_name)
    
    df['TimePeriod'] = df['ElapsedSeconds'].apply(time_period)
    df['WinnerFlag'] = (df['EventTeamID'] == df['WTeamID'])*1


# Define unique key to each play 

# In[ ]:


def add_keys(df, cols_key):
    
    vals = df[cols_key].values
    keys = [str(int(a)) + '_' + str(int(b)) + '_' + str(int(c)) + '_' + str(int(d)) for [a,b,c,d] in vals]
    df['Key'] = keys
    
    cols_new = [c for c in df.columns if c not in cols_key]
    
    return(df[cols_new])


# Prepare team event statistics

# In[ ]:


group_cols = ['EventTeamID', 'Season', 'DayNum', 'WTeamID', 'LTeamID', 'WinnerFlag', 'EventType']
key_cols   = ['Season', 'DayNum', 'WTeamID', 'LTeamID']
first = True

print('Preparing team event statistics')

for df_name in FINAL_DF_LIST_EVENTS:
    df = eval(df_name)
    stat_temp = df.groupby(group_cols)['EventID'].count().reset_index().rename(columns={'EventID':'EventCount'})
    stat_det_temp = df.groupby(group_cols + ['TimePeriod'])['EventID'].count().reset_index().rename(columns={'EventID':'EventCount'})
    
    team_event_stats1 = stat_temp if first else team_event_stats1.append(stat_temp)
    team_event_detailed_stats1 = stat_det_temp if first else team_event_detailed_stats1.append(stat_det_temp)
    
    if first:
        first = False

# ADD COACH NAME TO TEAM STATS DATAFRAMES
print('Adding coach information to teams')

team_event_stats = team_event_stats1.merge(DF_DataFiles_Stage2_TeamCoaches, left_on=['Season', 'EventTeamID'],
                                           right_on=['Season', 'TeamID'], how='left').query(
                                           'DayNum>=FirstDayNum and DayNum<=LastDayNum').drop(
                                           ['TeamID', 'FirstDayNum', 'LastDayNum', 'GENDER'], axis=1)

team_event_detailed_stats = team_event_detailed_stats1.merge(DF_DataFiles_Stage2_TeamCoaches, left_on=['Season', 'EventTeamID'],
                                           right_on=['Season', 'TeamID'], how='left').query(
                                           'DayNum>=FirstDayNum and DayNum<=LastDayNum').drop(
                                           ['TeamID', 'FirstDayNum', 'LastDayNum', 'GENDER'], axis=1)


team_event_stats = add_keys(team_event_stats, key_cols)
team_event_detailed_stats = add_keys(team_event_detailed_stats, key_cols)    


# Checkpoint - Each team has only one coach on a specific day

# In[ ]:


team_event_stats.groupby(['EventTeamID', 'Season', 'DayNum'])['CoachName'].nunique().reset_index().sort_values('CoachName').tail(1)


# Add result type: NCAA, secondary, regular to team stats

# In[ ]:


DF_DataFiles_Stage2_GameCities = add_keys(DF_DataFiles_Stage2_GameCities, key_cols)


# In[ ]:


team_event_stats_final = team_event_stats.merge(DF_DataFiles_Stage2_GameCities, on='Key', how='left')


# Drop rows with NULL result types which is very small in percentage

# In[ ]:


team_event_stats_final.CRType.isna().sum() / team_event_stats_1.shape[0]


# In[ ]:


team_event_stats_final.dropna(axis=0, inplace=True)


# Prepare player event statistics

# In[ ]:


group_cols = ['EventPlayerID', 'Season', 'DayNum', 'WTeamID', 'LTeamID', 'EventType'] #'WinnerFlag'
key_cols   = ['Season', 'DayNum', 'WTeamID', 'LTeamID']
first = True

for df_name in FINAL_DF_LIST_EVENTS:
    df = eval(df_name)
    stat_temp = df.groupby(group_cols)['EventID'].count().reset_index().rename(columns={'EventID':'EventCount'})
    #stat_det_temp = df.groupby(group_cols + ['TimePeriod'])['EventID'].count().reset_index().rename(columns={'EventID':'EventCount'})
    
    player_event_stats = stat_temp if first else player_event_stats.append(stat_temp)
    player_event_detailed_stats = stat_det_temp if first else player_event_detailed_stats.append(stat_det_temp)
    
    if first:
        first = False
    
player_event_stats = add_keys(player_event_stats, key_cols)
#player_event_detailed_stats = add_keys(player_event_detailed_stats, key_cols)    


# Add gender field to player stats

# In[ ]:


team_gender = DF_DataFiles_Stage2_Teams[['TeamID', 'GENDER']].values

gender_dict = {int(t):g for t,g in team_gender}


# In[ ]:


teamids = player_event_stats['Key'].apply(lambda k: k.split('_')[3]).apply(int).values
gender  = [gender_dict[t] for t in teamids]
player_event_stats['Gender'] = gender


# Some players may not be actively playing during the whole match which is a bias towards those who played longer. This bias is ignored due to:
# 
# - Each player has the same chance to play longer or shorter time in a match
# - Data is accumulated from 5 years of plays which randomly includes longer/shorter durations of activity
# - Mean of event counts occured among all plays is considered

# In[ ]:


player_event_mean = player_event_stats.pivot_table(index='EventPlayerID', columns='EventType', values='EventCount', 
                                                   aggfunc=np.mean).reset_index().fillna(0)


# In[ ]:


player_event_mean_mens = player_event_stats.query('Gender=="M"').pivot_table(index='EventPlayerID', columns='EventType', values='EventCount', 
                                                   aggfunc=np.mean).reset_index().fillna(0)


# In[ ]:


player_event_mean_womens = player_event_stats.query('Gender=="W"').pivot_table(index='EventPlayerID', columns='EventType', values='EventCount', 
                                                   aggfunc=np.mean).reset_index().fillna(0)


# ### IDENTIFYING DIFFERENT PLAYER PROFILES
# 
# K-Means algorithm is used to explore player weakness/strength and profiles
# 2 sub cluster is built based on attribute groups
# 
# 1. Based on shots, i.e misses and mades
# 2. Offensive/defensive/other attributes

# In[ ]:


from sklearn.cluster import KMeans 

import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


def cluster_stats(df, cols_cluster, col_profile):
    
    count   = df.groupby(col_profile)[cols_cluster[0]].count().reset_index().rename(columns={cols_cluster[0]:'count'})
    mean    = df.groupby(col_profile)[cols_cluster].mean().reset_index()
    median  = df.groupby(col_profile)[cols_cluster].mean().reset_index()
    
    mean.columns   = [col_profile] + [c + '_mean' for c in cols_cluster]
    median.columns = [col_profile] + [c + '_median' for c in cols_cluster]
    
    stats   = count.merge(mean, on=col_profile, how='inner'
                         ).merge(median, on=col_profile, how='inner')
    
    return(stats)


# In[ ]:


def make_cluster(df, cols_cluster, col_profile, n_cluster=4):

    data = df[cols_cluster].values

    km = KMeans(n_clusters=n_cluster, random_state=1111)
    km.fit(data)
    
    pred = km.predict(data)
    
    df[col_profile] = pred
    
    clust_stats = cluster_stats(df, cols_cluster, col_profile)
    clust_stats_compact = df[[col_profile] + cols_cluster].melt(col_profile)
    
    return df, clust_stats, clust_stats_compact


# In[ ]:


def plot_counts(data, col_profile):
    
    sns.set_palette("Paired")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title('Number of records in each cluster - {}'.format(col_profile))

    sns.barplot(x=col_profile, y="count", data=data, ax=ax)    ;


# In[ ]:


def plot_cluster(data, cols_cluster, col_profile, title):
    
    sns.set_palette("Paired")
    
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_title(title)

    # cut off outlier values
    data_plot = data.query('value<=10')

    sns.boxplot(x=col_profile, y="value", hue="EventType", data=data_plot, ax=ax);


# **1st sub-cluster**

# Firstly cluster only male players

# In[ ]:


cols_cluster = ['made1', 'made2', 'made3', 'miss1', 'miss2', 'miss3']
col_profile  = 'PROFILE_1'
title        = 'Male Player profiles based on shots made or missed'

player_event_mean_mens, clust1_stats, clust1_stats_compact =  make_cluster(player_event_mean_mens, cols_cluster, col_profile, n_cluster=4)

plot_counts(clust1_stats, col_profile)
plot_cluster(clust1_stats_compact, cols_cluster, col_profile, title)


# Now cluster only female players

# In[ ]:


cols_cluster = ['made1', 'made2', 'made3', 'miss1', 'miss2', 'miss3']
col_profile  = 'PROFILE_1'
title        = 'Female Player profiles based on shots made or missed'

player_event_mean_womens, clust1_stats, clust1_stats_compact =  make_cluster(player_event_mean_womens, cols_cluster, col_profile, n_cluster=4)

plot_counts(clust1_stats, col_profile)
plot_cluster(clust1_stats_compact, cols_cluster, col_profile, title)


# Comparing Male and Female Player profiles:
# - They have simillar groupings in terms of shots made or missed
#     - Female Profile 0 - Men Profile 2 look simliar
#     - Female Profile 3 - Men Profile 1 look simliar
#     - Female Profile 1 - Men Profile 0 look simliar
#     - Female Profile 2 - Men Profile 3 look simliar
# 
# - Next step: Cluster male and female players together since they are simillar

# Finally, male and female players clustered together

# In[ ]:


cols_cluster = ['made1', 'made2', 'made3', 'miss1', 'miss2', 'miss3']
col_profile  = 'PROFILE_1'
title        = 'Player profiles based on shots made or missed'

player_event_mean, clust1_stats, clust1_stats_compact =  make_cluster(player_event_mean, cols_cluster, col_profile, n_cluster=4)

plot_counts(clust1_stats, col_profile)
plot_cluster(clust1_stats_compact, cols_cluster, col_profile, title)


# <div style='font-family:"Calibri"; border: 1px solid gray; padding:8px;'>
# <br/>
# **FROM THE GRAPH ABOVE:**
# 
# <br/>
# - There is not any sub group who is totally successful or totally failure, i.e. Scoring high mades and low misses, or scoring low mades and high misses
# 
# - Some of the distinguishing attributes of profiles:
# <ul>
#     <li> PROFILE 3 - BEST SHORT DISTANCE SHOOTTERS: They are far better at making 1 or 2 points, however they equally miss 2 points </li>
#     <li> PROFILE 2 - AVERAGE SHOOTTER FAILING LONG DISTANCE: Although they are good at 1-2 point shots, they missed 3 points the most </li>
#     <li> PROFILE 0 - AVERAGE SHOOTTER </li>
#     <li> PROFILE 1 - NOT A SHOOTHER </li>
# </ul>  
# </div>

# Recode sub-cluster - 1

# In[ ]:


shooter_dict = {3: 'BEST_SHORT_DISTANCE_SHOOTER', 2: 'AVERAGE_SHOOTER_FAILING_LONG_DISTANCE', 0: 'AVERAGE_SHOOTER', 1: 'NOT_SHOOTER'}


# In[ ]:


player_event_mean['SHOOTER_PROFILE'] = player_event_mean['PROFILE_1'].apply(lambda x: shooter_dict[x])


# **2nd sub-cluster**

# In[ ]:


cols_cluster = ['assist', 'block', 'reb', 'steal']
col_profile  = 'PROFILE_2'
title        = 'Player profiles based on offensive & defensive traits'

player_event_mean, clust1_stats, clust1_stats_compact =  make_cluster(player_event_mean, cols_cluster, col_profile, n_cluster=4)

plot_counts(clust1_stats, col_profile)
plot_cluster(clust1_stats_compact, cols_cluster, col_profile, title)


# **PROFILE 2: PROFILES BASED ON OFFENSIVE & DEFENSIVE CHARACTERISTICS**
# <ul style='font-family:"Calibri"; border: 1px solid gray; padding:8px;'>
#     <li> PROFILE 1 - BEST REBOUNDERS : Players who are best at rebounds </li>
#     <li> PROFILE 0 - GOOD REBOUNDERS : Players who are above average at rebounds </li>
#     <li> PROFILE 3 - AVERAGE ASSISTER REBOUNDER      : Players who are best at assisting but also a good rebounder </li>
#     <li> PROFILE 2 - NOT_ASSISTER_REBOUNDER        : Players who do not have distinguishing offensive or defensive trait </li>
# </ul>

# RECODE PROFILE - 2

# In[ ]:


off_def_dict = {1: 'BEST_REBOUNDERS', 0: 'GOOD_REBOUNDERS', 3: 'AVERAGE_ASSISTER_REBOUNDER', 2: 'NOT_ASSISTER_REBOUNDER'}


# In[ ]:


player_event_mean['OFFENSIVE_DEFENSIVE_PROFILE'] = player_event_mean['PROFILE_2'].apply(lambda x: off_def_dict[x])


# Let's visualize and try to understand different profiles

# In[ ]:


player_event_mean.groupby(['SHOOTER_PROFILE', 'OFFENSIVE_DEFENSIVE_PROFILE'])['EventPlayerID'].count().reset_index().pivot(
            'SHOOTER_PROFILE', 'OFFENSIVE_DEFENSIVE_PROFILE', 'EventPlayerID')


# From the matrix above, it can be concluded that:
#     - Best short distance shooters are also best rebounders
#     - Average shooters are also average/good rebounders
#     - Players who are not a good shooter are mostly not very successful at assisting and rebounds

# #### ADD PLAYER PROFILES TO PLAYS

# In[ ]:


player_profiles_1 = player_event_stats[['Key', 'EventPlayerID', 'Gender']].drop_duplicates()


# In[ ]:


player_profiles_2 = player_event_mean[['EventPlayerID', 'OFFENSIVE_DEFENSIVE_PROFILE', 'SHOOTER_PROFILE']].drop_duplicates()


# In[ ]:


player_profiles = player_profiles_1.merge(player_profiles_2, on=['EventPlayerID'], how='inner')


# ### COACH PROFILES

# Data is duplicate due to event type column, so firstly make it unique for each key

# In[ ]:


coach_stats = team_event_stats_final[['Key', 'WinnerFlag', 'CoachName']].pivot_table(
                index='CoachName', columns='WinnerFlag', values='Key', aggfunc='count').reset_index()

coach_stats.columns = ['CoachName', 'LoosingPlays', 'WinningPlays']


# In[ ]:


coach_stats['TotalPlays'] = coach_stats['WinningPlays'] + coach_stats['LoosingPlays']
coach_stats['WinRatio'] = coach_stats['WinningPlays'] * 100 / coach_stats['TotalPlays']


# In[ ]:


coach_stats.TotalPlays.hist();


# In[ ]:


coach_stats.WinRatio.hist();


# Total plays clearly indicate experience and winning ratio in total plays is the measure of success. So both information should be used in order to profile coaches correctly
# 
# Simple ranking will be used based on quantiles 

# In[ ]:


def experience_rank(totalplays, lower, upper):
    if totalplays <= lower:
        return('LeastExperienced')
    elif totalplays <= upper:
        return('Experienced')
    else:
        return('MostExperienced')


# In[ ]:


def success_rank(totalplays, lower, upper):
    if totalplays <= lower:
        return('LeastSuccessful')
    elif totalplays <= upper:
        return('Successful')
    else:
        return('MostSuccessful')


# In[ ]:


lower = coach_stats.TotalPlays.quantile(.33)
upper = coach_stats.TotalPlays.quantile(.67)

coach_stats['ExperienceRank'] = coach_stats['TotalPlays'].apply(lambda x: experience_rank(x, lower, upper))


# In[ ]:


lower = coach_stats.WinRatio.quantile(.33)
upper = coach_stats.WinRatio.quantile(.67)

coach_stats['SuccessRank'] = coach_stats['WinRatio'].apply(lambda x: success_rank(x, lower, upper))


# In[ ]:


coach_stats['CoachProfile'] = coach_stats[['ExperienceRank', 'SuccessRank']].apply(lambda x: x.ExperienceRank + '_' + x.SuccessRank, axis=1)


# Add coach profile to the team final data

# In[ ]:


coach_dict = {str(c):str(p) for c,p in coach_stats[['CoachName', 'CoachProfile']].values}


# In[ ]:


team_event_stats_final['CoachProfile'] = team_event_stats_final.CoachName.apply(lambda c: coach_dict[c])


# Add player profile to the team final data

# In[ ]:


player_profiles_offdef = player_profiles.pivot_table(index='Key', columns='OFFENSIVE_DEFENSIVE_PROFILE', values='EventPlayerID', 
                                                     aggfunc='count').reset_index().fillna(0)
player_profiles_shooter = player_profiles.pivot_table(index='Key', columns='SHOOTER_PROFILE', values='EventPlayerID', 
                                                     aggfunc='count').reset_index().fillna(0)


# In[ ]:


player_dict1 = {str(c):str(p) for c,p in player_profiles[['EventPlayerID', 'OFFENSIVE_DEFENSIVE_PROFILE']].values}
player_dict2 = {str(c):str(p) for c,p in player_profiles[['EventPlayerID', 'SHOOTER_PROFILE']].values}


# ### FACTORS DETERMINING THE WINNER TEAM
# 
# Coach and player profiles along with event statistics can be used to better quantify the factors affecting the result of plays and determining the winning team

# In[ ]:




