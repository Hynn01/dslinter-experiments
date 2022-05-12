#!/usr/bin/env python
# coding: utf-8

# I am trying to replicate the Lessons from the [Soccermatics from Python](https://github.com/Friends-of-Tracking-Data-FoTD/SoccermaticsForPython) series. I am trying to make it self contained and to add some text to what I am learning. More content, including the lecture can be found [here](https://uppsala.instructure.com/courses/44338/pages/1-visualising-actions).
# 
# Enjoy it.

# ## Lecture 11
# 
# I am following the folliwing exercise: [Simulate matches](https://github.com/Friends-of-Tracking-Data-FoTD/SoccermaticsForPython/blob/master/11SimulateMatches.py).
# 
# This code is adapted from https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling/

# In[ ]:


### basic
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn

### for SQL dataset
import sqlite3

### stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson,skellam


# Let's use the Kaggle dataset: [European Soccer Dataset](https://www.kaggle.com/datasets/hugomathien/soccer). Notice that in the original exercise this dataset is downloaded from a website. __I choose__ to use the Kaggle dataset. The next cells shows how to query this SQL dataset.

# In[ ]:


conn = sqlite3.connect('../input/soccer/database.sqlite')


# Let's use the Premier League (2016/17) season, which corresponds to `id=country_id=1729`

# In[ ]:


query = """
SELECT Match.id,
    League.name AS league_name, 
    season, 
    HT.team_long_name AS  HomeTeam,
    AT.team_long_name AS AwayTeam,
    home_team_goal AS HomeGoals, 
    away_team_goal AS AwayGoals                                        
FROM Match
    JOIN Country on Country.id = Match.country_id
    JOIN League on League.id = Match.league_id
    LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
    LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
WHERE league_id=1729 AND season='2015/2016'
"""
df = pd.read_sql(query, conn)
df.head()


# In[ ]:


df.info()


# In[ ]:


goal_model_data = pd.concat([
    df[['HomeTeam','AwayTeam','HomeGoals']].assign(home=1).rename(columns={'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}),
    df[['AwayTeam','HomeTeam','AwayGoals']].assign(home=0).rename(columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})
])
goal_model_data


# Fit the model to the data. Home advantage included. Team and opponent as fixed effects.

# In[ ]:


poisson_model = smf.glm(formula="goals ~ home + team + opponent", 
                        data=goal_model_data, 
                        family=sm.families.Poisson()).fit()
poisson_model.summary()


# Predict for Arsenal vs. Manchester City

# In[ ]:


home_team='Manchester City'
away_team='Arsenal'


home_score_rate=poisson_model.predict(pd.DataFrame(data={'team': home_team, 'opponent': away_team, 'home':1},index=[1]))
away_score_rate=poisson_model.predict(pd.DataFrame(data={'team': away_team, 'opponent': home_team, 'home':0},index=[1]))
print(f'{home_team} against {away_team} expect to score: {round(home_score_rate.to_list()[0],2)}')
print(f'{away_team} against {home_team} expect to score: {round(away_score_rate.to_list()[0],2)}')


# Lets just get a result

# In[ ]:


home_goals=np.random.poisson(home_score_rate)
away_goals=np.random.poisson(away_score_rate)
print(home_team + ': ' + str(home_goals[0]))
print(away_team + ': '  + str(away_goals[0]))


# Code to calculate the goals for the match. 
# Here a probability mass function (pmf) is used due to the discrete nature of goals. 

# In[ ]:


def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 'opponent': awayTeam,'home':1}, index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 'opponent': homeTeam,'home':0}, index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))
    


# In[ ]:


max_goals=5
score_matrix=simulate_match(poisson_model, home_team, away_team,max_goals)


# Make 2d histogram of results

# In[ ]:


fig=plt.figure(figsize=(8,5))
ax=fig.add_subplot(1,1,1)

pos=ax.imshow(score_matrix, extent=[-0.5,max_goals+0.5,-0.5,max_goals+0.5], aspect='auto',cmap=plt.cm.Reds)
fig.colorbar(pos, ax=ax)
ax.set_title('Probability of outcome')
plt.xlim((-0.5,5.5))
plt.ylim((-0.5,5.5))
plt.tight_layout()
ax.set_xlabel('Goals scored by ' + away_team)
ax.set_ylabel('Goals scored by ' + home_team)
plt.show()


# Home, draw, away probabilities

# In[ ]:


homewin=np.sum(np.tril(score_matrix, -1))
draw=np.sum(np.diag(score_matrix))
awaywin=np.sum(np.triu(score_matrix, 1))
print(f'Probabilities \n{home_team} win: {round(homewin,2)}, draw: {round(draw,2)}, {away_team} win: {round(awaywin,2)}') 


# ### <- [Previous notebook](https://www.kaggle.com/code/aleespinosa/soccermatics-for-python-10)
# ### -> [Next notebook](https://www.kaggle.com/code/aleespinosa/soccermatics-for-python-12)
