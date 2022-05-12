#!/usr/bin/env python
# coding: utf-8

# This notebook downloads episodes using Kaggle's GetEpisodeReplay API and the [Meta Kaggle](https://www.kaggle.com/kaggle/meta-kaggle) dataset.
# 
# **To run this notebook you WILL need to re-add the Meta Kaggle dataset. After opening your copy of the notebook, click "+ Add data" top right in the notebook editor.
# **
# 
# Meta Kaggle is refreshed daily, but sometimes misses daily refreshes for a few days.
# 
# Why download replays?
# - Train your ML/RL model
# - Inspect the performance of yours and others agents
# - To add to your ever growing json collection 
# 
# Only one scraping strategy is implemented: For each top scoring submission, download all missing matches, move on to next submission.
# 
# Other scraping strategies can be implemented, but not here. Like download max X matches per submission or per team per day, or ignore certain teams or ignore where some scores < X, or only download some teams.
# 
# Todo:
# - Add teamid's once meta kaggle add them. Edit: it's been a long time, it doesn't look like Kaggle is adding this.

# In[ ]:


import pandas as pd
import numpy as np
import os
import requests
import json
import datetime
import time
import glob
import collections


# In[ ]:


## You should configure these to your needs. Choose one of ...
# 'hungry-geese', 'rock-paper-scissors', santa-2020', 'halite', 'google-football'
COMP = 'kore-2022'
MAX_CALLS_PER_DAY = 300 # Kaggle says don't do more than 3600 per day and 1 per second
LOWEST_SCORE_THRESH = 1450


# In[ ]:


ROOT ="../working/"
META = "../input/meta-kaggle/"
MATCH_DIR = '../working/'
#base_url = "https://www.kaggle.com/requests/EpisodeService/"
base_url = "https://www.kaggle.com/api/i/competitions.EpisodeService/"
    
get_url = base_url + "GetEpisodeReplay"
BUFFER = 1
COMPETITIONS = {
    'kore-2022': 34419,
    'lux-ai-2021': 30067,
    'hungry-geese': 25401,
    'rock-paper-scissors': 22838,
    'santa-2020': 24539,
    'halite': 18011,
    'google-football': 21723
}


# In[ ]:


# Filter Episodes.csv
data = pd.read_csv(META + "Episodes.csv", chunksize=1e6)
df_list = [] 
for chunk in data:
    df_list.append(chunk[chunk['CompetitionId']==COMPETITIONS[COMP]])
episodes_df = pd.concat(df_list)
del data
del chunk
del df_list
print(f'Episodes.csv: {len(episodes_df)} rows after filtering for {COMP}.')


# In[ ]:


# Filter EpisodeAgents.csv
data = pd.read_csv(META + "EpisodeAgents.csv", chunksize=1e6)
df_list = [] 
for chunk in data:
    df_list.append(chunk[chunk.EpisodeId.isin(episodes_df.Id)])
epagents_df = pd.concat(df_list)
del data
del chunk
del df_list
print(f'EpisodeAgents.csv: {len(epagents_df)} rows after filtering for {COMP}.')


# In[ ]:


# Prepare dataframes

episodes_df = episodes_df.set_index(['Id'])
episodes_df['CreateTime'] = pd.to_datetime(episodes_df['CreateTime'])
episodes_df['EndTime'] = pd.to_datetime(episodes_df['EndTime'])

epagents_df.fillna(0, inplace=True)
epagents_df = epagents_df.sort_values(by=['Id'], ascending=False)


# In[ ]:


# Get top scoring submissions# Get top scoring submissions
max_df = (epagents_df.sort_values(by=['EpisodeId'], ascending=False).groupby('SubmissionId').head(1).drop_duplicates().reset_index(drop=True))
max_df = max_df[max_df.UpdatedScore>=LOWEST_SCORE_THRESH]
max_df = pd.merge(left=episodes_df, right=max_df, left_on='Id', right_on='EpisodeId')
sub_to_score_top = pd.Series(max_df.UpdatedScore.values,index=max_df.SubmissionId).to_dict()
print(f'{len(sub_to_score_top)} submissions with score over {LOWEST_SCORE_THRESH}')


# In[ ]:


# Get episodes for these submissions
sub_to_episodes = collections.defaultdict(list)
for key, value in sorted(sub_to_score_top.items(), key=lambda kv: kv[1], reverse=True):
    excl = []
    if key not in excl: # we can filter subs like this
        eps = sorted(epagents_df[epagents_df['SubmissionId'].isin([key])]['EpisodeId'].values,reverse=True)
        sub_to_episodes[key] = eps
candidates = len(set([item for sublist in sub_to_episodes.values() for item in sublist]))
print(f'{candidates} episodes for these {len(sub_to_score_top)} submissions')


# In[ ]:


global num_api_calls_today
num_api_calls_today = 0
all_files = []
for root, dirs, files in os.walk(MATCH_DIR, topdown=False):
    all_files.extend(files)
seen_episodes = [int(f.split('.')[0]) for f in all_files 
                      if '.' in f and f.split('.')[0].isdigit() and f.split('.')[1] == 'json']
remaining = np.setdiff1d([item for sublist in sub_to_episodes.values() for item in sublist],seen_episodes)
print(f'{len(remaining)} of these {candidates} episodes not yet saved')
print('Total of {} games in existing library'.format(len(seen_episodes)))


# In[ ]:


def create_info_json(epid):
    
    create_seconds = int((episodes_df[episodes_df.index == epid]['CreateTime'].values[0]).item()/1e9)
    end_seconds = int((episodes_df[episodes_df.index == epid]['CreateTime'].values[0]).item()/1e9)

    agents = []
    for index, row in epagents_df[epagents_df['EpisodeId'] == epid].sort_values(by=['Index']).iterrows():
        agent = {
            "id": int(row["Id"]),
            "state": int(row["State"]),
            "submissionId": int(row['SubmissionId']),
            "reward": int(row['Reward']),
            "index": int(row['Index']),
            "initialScore": float(row['InitialScore']),
            "initialConfidence": float(row['InitialConfidence']),
            "updatedScore": float(row['UpdatedScore']),
            "updatedConfidence": float(row['UpdatedConfidence']),
            "teamId": int(99999)
        }
        agents.append(agent)

    info = {
        "id": int(epid),
        "competitionId": int(COMPETITIONS[COMP]),
        "createTime": {
            "seconds": int(create_seconds)
        },
        "endTime": {
            "seconds": int(end_seconds)
        },
        "agents": agents
    }

    return info


# In[ ]:


def saveEpisode(epid):
    # request
    re = requests.post(get_url, json = {"episodeId": int(epid)})
        
    # save replay
    with open(MATCH_DIR + '{}.json'.format(epid), 'w') as f:
        f.write(re.json()['replay'])

    # save match info
    info = create_info_json(epid)
    with open(MATCH_DIR +  '{}_info.json'.format(epid), 'w') as f:
        json.dump(info, f)


# In[ ]:


r = BUFFER;

start_time = datetime.datetime.now()
se=0
for key, value in sorted(sub_to_score_top.items(), key=lambda kv: kv[1], reverse=True):
    if num_api_calls_today<=MAX_CALLS_PER_DAY:
        print('')
        remaining = sorted(np.setdiff1d(sub_to_episodes[key],seen_episodes), reverse=True)
        print(f'submission={key}, LB={"{:.0f}".format(value)}, matches={len(set(sub_to_episodes[key]))}, still to save={len(remaining)}')
        
        for epid in remaining:
            if epid not in seen_episodes and num_api_calls_today<=MAX_CALLS_PER_DAY:
                saveEpisode(epid); 
                r+=1;
                se+=1
                try:
                    size = os.path.getsize(MATCH_DIR+'{}.json'.format(epid)) / 1e6
                    print(str(num_api_calls_today) + f': saved episode #{epid}')
                    seen_episodes.append(epid)
                    num_api_calls_today+=1
                except:
                    print('  file {}.json did not seem to save'.format(epid))    
                if r > (datetime.datetime.now() - start_time).seconds:
                    time.sleep( r - (datetime.datetime.now() - start_time).seconds)
            if num_api_calls_today>(min(3600,MAX_CALLS_PER_DAY)):
                break
print('')
print(f'Episodes saved: {se}')


# In[ ]:




