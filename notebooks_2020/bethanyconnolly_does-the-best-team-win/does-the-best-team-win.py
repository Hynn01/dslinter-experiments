#!/usr/bin/env python
# coding: utf-8

# # **Does The Best Team Win?**
# Google Cloud & NCAA March Madness Analytics Report by Bethany Connolly

# ![](https://images.pexels.com/photos/1293265/pexels-photo-1293265.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260)
# Stock Image from Pexels.com

# ## Contents
# 
# **1.** Summary
# 
# **2.** Introduction
# 
# **3.** Data Loading
# 
# **4.** Measuring Success: Winning
# 
# **5.** Examining The Data Spread 
# 
# **6.** Regular Season Win-Count vs. March Madness Results 
# 
# **7.** A Different Measure of Success: Score
# 
# **8.** Regular Season Cumulative Score vs. March Madness Results 
# 
# **9.** Does The Most Successfull Team Win
# 
# **10.** Conclusions

# ## 1. Summary
# 
# The NCAA Division I Men's Basketball Tournament is one of the most famous and popular sport tournaments in the United States. 
# This single-elimination competition sees 68 of the country’s top college basketball teams compete for the national title. But to what extent does the outcome of this fast paced, drop out system reflect the overall rankings of the teams that compete?
# In this report I compare the success of teams throughout the regular season to their subsequent success in March Madness. Success is defined as the total matches won as well as the cumulative point score achieved over the regular season. By ranking teams in this way, their likelihood of progressing to the later stages is predicted. Based on this data analysis, it can be argued that a team's cumulative score over the regular season is a more consistent metric for ranking it than either its number of wins or its ranking in March Madness.

# ## 2. Introduction
# 
# What makes a great basketball team? Is it the team that wins the national championship? This would certainly be one measure of success, but how reliable is it? 
# 
# March Madness is a brutal tournament: over just three weeks, 68 of the country’s top teams compete in single elimination games until a champion team is crowned. Is this team the most consistently successful, or is a team’s likelihood of progressing through the competition determined by other factors? Cinderella teams are well known low seeded teams which make it further through the competition that expected. The madness of this tournament offers underdogs a chance of glory at the expense of otherwise successful opponents. Imagine that your favourite team gets paired against one of the top teams in the country which causes them to drop out in the first round. Does that mean they are one of the worst teams or were they just unlucky?
# 
# The point of a tournament is to separate out teams, the assumption being that the further you progress in the tournament, the better your team is. 
# 
# **In the case of March Madness, is this true?**
# 
# March Madness isn’t the only chance for teams to compete each year. In fact, this short 20-day competition is preceded by an extensive regular season (132 days long). This longer regular season may offer a much more thorough evaluation of the different team’s performance consistency. It isn’t single elimination; win or lose, teams can compete over and over again. The ultimate result is more combinations of different teams competing, more data from many more games and the chance to better understand a team’s long-term performance over many months. 
# 
# In this report I use the regular season results to evaluate a different metric for identifying the 'best' teams. Through exploratory data analysis of the 1985 - 2019 competitions, I compare the results of the regular season to those of the tournament with the aim of determining to what extent a team’s success in the regular season is reflected by its performance in the NCAA tournament.

# ## 3. Data Loading
# 
# For this report I explored the men’s regular season and tournament results for the years 1985 - 2019. These different competitions can be easily identified by the day of the competition calendar they fall on: 
# 
# `DayNum 0 - 132 => Regular Season`
# 
# `DayNum 134 - 154 => NCAA Tournament`

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np  

from scipy import stats  
from operator import itemgetter 
from pathlib import Path

# Load the datafiles
data_directory = Path('../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/')
regular_season = pd.read_csv(data_directory / "MRegularSeasonCompactResults.csv")
tourney_results = pd.read_csv(data_directory / "MNCAATourneyCompactResults.csv")
tourney_seeds = pd.read_csv(data_directory / "MNCAATourneySeeds.csv")


# ## 4. Measuring Success: Winning
# 
# **How can a team’s success through the regular season be measured? **
# 
# One obvious answer to this is the frequency of its wins - the more times a team competes and wins, surely the better the team.
# This metric is easily calculated by recording each instance of a team winning over the course of the pre-season and summing the wins to obtain a win-count.
# 
# But instead of just identifying the team with the most wins at the end of the regular season, let's first consider how a team’s cumulative win-count changes over this 4-month season. For example, the graph below shows the change in win-count for each of the 68 teams in the 2019 regular season.
# 
# **What can we learn from this data?**
# 
# At the early stages of the season, the lines of the different teams are all bunched together - there isn’t a wide spread of wins/losses.
# As more games take place, and the teams progress through the regular season, their total win-counts continue to rise. What differs amongst them is the rate of increase of their win-counts i.e. the slope of each team’s line.  Over the four months of the game season, the lines continue to diverge and by end of the regular season we see a wide spread of cumulative win-count. 
# 
# If the purpose of a tournament is to separate teams out and rank them, then this looks like a good metric.
# 
# Let’s now examine the spread of this data further...
# 
# 
# 

# In[ ]:


# Drop the year 2020 (cancelled)
regular_season = regular_season.drop(regular_season[regular_season.Season == 2020].index)

# Get the unique team IDs for each competition year
tourney_teams = tourney_seeds.groupby('Season').TeamID.unique()

# Make a dictionary of years containing how many times each team won a game in the regular season
year_team_dict = {}
for year, year_group in regular_season.groupby('Season'):
    year_team_dict[year] = {}
    for win_team, win_team_group in year_group.groupby('WTeamID'):
        if win_team in tourney_teams[year]:
            year_team_dict[year][win_team] = win_team_group.DayNum.values

# Plot the results of the cumulative win count for each team in the 2019 season
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
plt.figure(figsize = (10, 5))
for team, days in year_team_dict[2019].items():
    plt.plot(days, range(len(days)))
    plt.xlabel("Regular Season (Days)", fontsize = 12)
    plt.ylabel("Cumulative Wins", fontsize = 12)
    plt.title("Cumulative Wins Per Team Over Regular Season: 2019", fontsize = 14, fontweight = 'bold')  
plt.show()


# ## 5. Examining The Data Spread  
# 
# We saw in the above graph that totalling a team’s wins over the entire pre-season seems to be a good way of separating teams. But how well separated are these teams? Below are a series of histograms for each regular season from 1985 - 2019. These charts show the frequency of total win-counts achieved by the teams each year.
# 
# This looks pretty promising - there is a wide spread of results over the years but in most cases a reasonably normal distribution is achieved (black curve over the histogram).
# A normal distribution means that the majority of teams get a certain total win-count - these are the average teams. They aren’t bad, but they're not amazing either.
# By comparison, a smaller group of teams perform quite badly, winning very few games over the entire season. Based on their performance, we probably wouldn't expect these teams to do too well in the tournament.
# Likewise, another small subset of teams fares much better than the rest, they win loads of their matches! The shaded blue area on each histogram shows the upper quartile - the teams that fall in this region represent the top 25% of all teams that year. Since these teams win so much, we might expect them to win more often in the tournament too.
# 
# Next, let’s examine if this is what really happens!
# 

# In[ ]:


# Obtain the final number of matches won per team, per year
year_team_cumulative = {}
for year in year_team_dict:
    year_team_cumulative[year] = {}
    for team, days in year_team_dict[year].items():
        year_team_cumulative[year][team] = len(days) 


# Figure of histograms for each year 1985 - 2019        
plt.figure(figsize = (14, 16))
for num, year in enumerate(year_team_cumulative):

    # Color map for histogram
    cm = plt.cm.get_cmap('plasma')
    ytc = list(year_team_cumulative[year].values())
    
    # Make the hisogram
    plt.subplot(7,5,(num + 1))
    n, bins, patches = plt.hist(ytc, density = True, alpha = 0.9, edgecolor='black')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Set the bin color, based on bin value
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
        
    # Labels and title
    plt.xlabel("Regular Season Win Count", fontsize = 9)
    plt.ylabel("Frequency Normalized", fontsize = 9)
    plt.title(year, fontsize = 12, fontweight = 'bold')
    
    # Set the xticks 
    xt = plt.xticks()[0]  
    xmin, xmax = min(xt), max(xt)  
    lnspc = np.linspace(xmin, xmax, len(ytc))
    
    # Fit the normal curve
    m, s = stats.norm.fit(ytc)   
    pdf_g = stats.norm.pdf(lnspc, m, s) 
    plt.plot(lnspc, pdf_g, color = 'black')

    # Show top 25th percent
    q3 = np.percentile(ytc, 75)
    maxq = np.percentile(ytc, 100)
    plt.axvspan(q3, maxq, color = 'blue', alpha = 0.2)
   
    plt.subplots_adjust(hspace = 1, wspace=0.45)   
    
plt.show()


# ## 6. Regular Season Win-Count vs. March Madness Results 
# 
# If both the regular season and March Madness are good ways of identifying top performing teams, we might reasonably expect the same teams to come out on top in each. Let’s consider a set of top teams in the tournament, the 'sweet sixteen'. These teams make up the semi-finalists of the competition. By comparing the top 16 winning teams from the regular season to those of the 'sweet sixteen' we can determine to what extend consistently winning teams make it to the later stages of March Madness. For each year, the top 16 teams in each competition were identified and the number of teams which appeared in each list were counted. 'Wins Prediction Accuracy' was calculated as a fraction of correctly identified teams i.e. ‘sweet sixteen’ teams correctly predicted by regular season win-count divided by sixteen. The chart below shows this prediction accuracy plotted for each year of the competition. 
# 
# For reference, lets imagine that the semi-final teams were not selected by competing but instead chosen completely by chance from all 68 of the original teams. Each team has a 1/68 chance of being chosen so we would now expect only 24% of the top 16 teams from the regular season to make it to the semi-finals.
# 
# By using our metric, cumulative win-count over the regular season, on average half of the sweet 16 teams can be correctly predicted (Win Count Prediction Accuracy': mean = 0.50, standard deviation = 0.09). 
# 
# **What can we conclude from this?**
# * Progression through the competition isn’t random, teams which win more games in the regular season do have a better chance of getting to the later stages of the NCAA tournament.
# * Simply considering this one variable - how many times a team wins over the regular season - is a reasonably good way of predicting how far it will progress through the tournament - its about twice as good as random.
# * However... not all top performing teams make it to the later stages of the competition! Although these top teams have proven themselves through the whole regular season, for some reason when March Madness comes around, they can’t keep their winning streak going.
# So why might this be?
# 
# The nature of the tournament is single elimination - you lose once and you're out. 
# This is a pretty brutal way of thinning the herd and prevents unlucky teams from redeeming themselves later on. If two top performing teams are paired in early stages of the tournament, only one can continue through. This also means that teams which perform less consistently in the regular season can make a lucky win to jump ahead of higher ranked teams. These so called ‘Cinderella teams’ take the place of teams which would be expected to progress further.  
# 
# **Is there a different measure of success which would allow us to better predict the tournament results?**
# 

# In[ ]:


# Get a dictionary of top 16 teams from the preseason based on win count
regular_topteams = {}
n = 16
for year in year_team_cumulative:
    top_n = dict(sorted(year_team_cumulative[year].items(), key = itemgetter(1), reverse = True)[:n])
    regular_topteams[year] = {}
    for team, score in year_team_cumulative[year].items():
        regular_topteams[year] = list(top_n.keys()) 

# Get a dictionary of 'sweet 16' teams each year based on teams which win on days 138/139
tournament_top_teams = {}
for year, year_group in tourney_results.groupby('Season'):
    top_teams = year_group[(year_group['DayNum'] == 138) | (year_group['DayNum'] == 139)] 
    tournament_top_teams[year] = list(top_teams.WTeamID)
    
# Calculate accuracy score of regular season top 16 predictions from tournament matches
score = {}
for year in tournament_top_teams:
    correct_teams = []
    for team in tournament_top_teams[year]:
        if team in regular_topteams[year]:
            correct_teams.append(team)
    team_predict_score = len(correct_teams)/ n    
    score[year] = team_predict_score   
scores = list(score.values())

# Calcuate average prediction across all years
mean_score = sum(scores) / len(scores)

# Calculate mean error in the accuracy score
std = np.std(scores)

# Baseline calculated as random chance that any 16 teams make it to the semi-final
baseline = (16/68)

# Scatterplot of accuracy each year
plt.figure(figsize = (10, 5))
ax = plt.subplot()
colors = scores
plt.scatter(score.keys(), score.values(), cmap = 'plasma', c = colors, edgecolors = 'black', marker='o', s = 75)
plt.title("Semi-Finalists Prediction Accuracy: Win Count", fontsize = 14, fontweight = 'bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Win Count Prediction Accuracy", fontsize = 12)
plt.hlines(mean_score, xmin = 1985, xmax = 2019, color = 'red', label = 'Mean Prediction Accuracy', linestyles='dashed', alpha = 0.7)
plt.text(2009,0.95,'- Mean Prediction Accuracy',rotation=0, color = 'red', alpha = 0.8, fontsize = 11)
plt.text(2009,0.9,'- Random Prediction Accuracy',rotation=0, color = 'blue', alpha = 0.8, fontsize = 11)
plt.hlines(baseline, xmin = 1985, xmax = 2019, color = 'blue', label = 'Random Prediction Accuracy', linestyles='dashed', alpha = 0.7)
ax.set_ylim(ymin=0, ymax = 1)
plt.show()


# ## 7. A Different Measure of Success: Score
# 
# So, using our win-count metric , we predict 50% of the 'sweet sixteen' teams. But is there a better way of measuring a team’s potential?
# 
# Although a team winning a game is an excellent way of measuring success, it offers no indication of the magnitude of the win. Imagine a team that just barely wins its game by scraping an extra point in the last few seconds of play. Now imagine another team which dominates the opposition and wins by a huge margin. By our previous metric both teams had won one game and the victories were recorded as equal. Likewise, a team which just barely lost was ranked as low as a team which suffered a huge defeat. There must be a better way of comparing these results!
# 
# **Let’s re-examine our data in a different way...**
# 
# Instead of ranking the different team’s success through the regular season by the number of games they win, lets now rank them by the score they get per game. The cumulative score can be defined by the total score over all the games a team plays through the season. Now teams which consistently win by large margins will get higher cumulative scores by the end of the season than those which win the same number of games by tiny margins.
# 
# The cumulative win score was therefore calculated by summing the score for each team (win or lose) per year over the entire regular season. The spread of this data is presented again as a series of histograms which show the frequency of total scores obtained amongst the teams.
# 
# Just like the total win-count histograms we saw earlier; the cumulative score histograms often show normal distributions (black curve over histogram). We can see that there are many average scoring teams as well as a handful or poor and great ones. 
# 
# **Cumulative score also seems to be a good way of separating the teams based on performance!**

# In[ ]:


# List of scores (win and loose games) for each team in the regular season
year_team_dict2 = {}
for year, year_group in regular_season.groupby('Season'):
    year_team_dict2[year] = {}
    for team, team_group in year_group.groupby('WTeamID'):
        if team in tourney_teams[year]:
            year_team_dict2[year][team] = list(team_group.WScore.values)
    for team, team_group in year_group.groupby('LTeamID'):
        if team in tourney_teams[year]:
            year_team_dict2[year][team].extend(team_group.LScore)

# Calculate each teams cumulative win score each year 
year_team_cumulative_score = {}
for year in year_team_dict2:
    year_team_cumulative_score[year] = {}
    for team, score in year_team_dict2[year].items():
        year_team_cumulative_score[year][team] = sum(score)
        
# Figure of histograms for each year 1985 - 2019   
plt.figure(figsize = (14, 16))
for num, year in enumerate(year_team_cumulative):
     
    # Color map for histogram
    cm = plt.cm.get_cmap('plasma') #color map is plasma
    ytc = list(year_team_cumulative_score[year].values()) # list of scores
    
    # make the hisogram
    plt.subplot(7,5,(num+1))
    n, bins, patches = plt.hist(ytc, density = True, alpha = 0.9, edgecolor='black')#plot each histogram
    
    # Set the bin color, based on bin value
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
     
    # Labels and title
    plt.xlabel("Cumulative Score", fontsize = 9)
    plt.ylabel("Frequency Normalized", fontsize = 9)
    plt.title(year, fontsize = 12, fontweight = 'bold')
    
    # Set the xticks 
    #xmin, xmax = 0, 40 
    xt = plt.xticks()[0]  
    xmin, xmax = min(xt), max(xt)  
    lnspc = np.linspace(xmin, xmax, len(ytc))
    
    # Fit the normal curve.
    m, s = stats.norm.fit(ytc)   
    pdf_g = stats.norm.pdf(lnspc, m, s) 
    plt.plot(lnspc, pdf_g, color = 'black')

    # Show top 25th percent
    q3 = np.percentile(ytc, 75)
    maxq = np.percentile(ytc, 100)
    plt.axvspan(q3, maxq, color = 'blue', alpha = 0.2)
   
    plt.subplots_adjust(hspace = 1, wspace=0.6)   
    
plt.show()


# ## 8. Regular Season Cumulative Score vs. March Madness Results 
# 
# Now that we have a new metric for determining success of the teams, lets again compare our calculated top teams to the tournament semi-finalists.
# 
# The graph below shows the 'Total Score Prediction Accuracy' per year i.e. the ratio of top 16 scoring teams in the regular season found in the 'sweet sixteen'. Although this result ('Total Score Prediction Accuracy': mean = 0.42, standard deviation = 0.07) is again significantly better than guessing at random (prediction accuracy = 0.24), considering its mean error, it shows basically the same prediction accuracy as the earlier metric, win-count.

# In[ ]:


# Dictionary of top 16 teams in the regular season based on cumulative score
regular_topteams_score = {}
n = 16
for year in year_team_cumulative_score:
    top_n = dict(sorted(year_team_cumulative_score[year].items(), key = itemgetter(1), reverse = True)[:n])
    regular_topteams_score[year] = {}
    for team, score in year_team_cumulative_score[year].items():
        regular_topteams_score[year] = list(top_n.keys()) 

# Teams in the tournament semi_final 
tournament_top_teams_score = {}
for year, year_group in tourney_results.groupby('Season'):
    top_teams = year_group[(year_group['DayNum'] == 138) | (year_group['DayNum'] == 139)] 
    tournament_top_teams_score[year] = list(top_teams.WTeamID)

# Calculate accuracy score of cumulative score top 16 predictions from tournament matches
for year in tournament_top_teams_score:
    correct_teams = []
    for team in tournament_top_teams_score[year]:
        if team in regular_topteams_score[year]:
            correct_teams.append(team)
    team_predict_score = len(correct_teams)/16
    score2[year] = team_predict_score   

# Calcuate average prediction across all years
scores2 = list(score2.values())
mean_score2 = sum(scores2) / len(scores2)

# Calculate mean error in the accuracy score
std2 = np.std(scores)

# Baseline calculated as random chance that any 16 teams make it to the semi-final
baseline2 = (16/68)

# Scatterplot of cumulative score prediction accuracy each year
plt.figure(figsize = (10, 5))
ax = plt.subplot()
colors = scores2
plt.scatter(score2.keys(), score2.values(), cmap = 'plasma', c = colors, edgecolors = 'black', marker='o', s = 75)
plt.title("Semi-Finalists Prediction Accuracy: Cumulative Score", fontsize = 14, fontweight = 'bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Total Score Prediction Accuracy", fontsize=12)
plt.hlines(mean_score2, xmin = 1985, xmax = 2019, color = 'red', label = 'Mean Prediction Accuracy', linestyles='dashed', alpha = 0.7)
plt.text(2009,0.95,'- Mean Prediction Accuracy',rotation=0, color = 'red', alpha = 0.9, fontsize = 11)
plt.text(2009,0.9,'- Random Prediction Accuracy',rotation=0, color = 'blue', alpha = 0.9, fontsize = 11)
plt.hlines(baseline2, xmin = 1985, xmax = 2019, color = 'blue', label = 'Random Prediction Accuracy', linestyles='dashed', alpha = 0.7)
ax.set_ylim(ymin=0, ymax = 1)

plt.show()


# This is further highlighted by the two histograms below, which compare the frequency of prediction accuracy of each of the two metrics. 
# 
# In each case a spread of prediction accuracies is seen around the mean value of around 0.5.

# In[ ]:


# Histograms figure comparing frequency of prediction accuracy for each metric
fig = plt.subplots(1,2, figsize = (14, 6))

# Histogram for Semi-Finalists Prediction Frequency (Win Count)
cm = plt.cm.get_cmap('plasma')
ax = plt.subplot(1,2,1)
n, bins, patches = plt.hist(scores, bins = 5, density = True, edgecolor='black', alpha = 0.9)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))
plt.xlabel("Prediction Accuracy", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.title("Semi-Finalists Prediction Frequency: Win Count", fontsize = 13, fontweight = 'bold')
ax.set_xlim(xmin=0, xmax = 1)
plt.axvline(mean_score, color = 'black', label = 'Mean Prediction Accuracy', linestyle='dashed')
plt.text(0.62,3.85,'Mean Prediction Accuracy',rotation=0, color = 'black', fontsize = 9.5)
plt.text(0.62,3.68,'Random Prediction Accuracy',rotation=0, color = 'blue', fontsize = 9.5)
plt.axvline(baseline, color = 'blue', label = 'Random Prediction Accuracy', linestyle='dashed')
ax.set_xlim(xmin=0, xmax = 1)

# Histogram for Semi-Finalists Prediction Frequency (Cumulative Score)
cm = plt.cm.get_cmap('plasma')
ax2 = plt.subplot(1,2,2)
n, bins, patches = plt.hist(scores2, bins = 5, density = True, edgecolor='black', alpha = 0.9)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))
plt.xlabel("Prediction Accuracy", fontsize = 14)
plt.ylabel("Frequency", fontsize = 14)
plt.title("Semi-Finalists Prediction Frequency: Cumulative Score", fontsize = 13, fontweight = 'bold')
ax2.set_xlim(xmin=0, xmax = 1)
plt.axvline(mean_score2, color = 'black', label = 'Mean Prediction Accuracy', linestyle='dashed')
plt.text(0.6,9.15,'Mean Prediction Accuracy',rotation=0, color = 'black', fontsize = 9.5)
plt.text(0.6,8.75,'Random Prediction Accuracy',rotation=0, color = 'blue', fontsize = 9.5)
plt.axvline(baseline2, color = 'blue', label = 'Random Prediction Accuracy', linestyle='dashed')
ax2.set_xlim(xmin=0, xmax = 1)

plt.subplots_adjust(wspace=0.15)   

plt.show()


# ## 9. Does The Best Team Win?
# 
# To wrap up, let’s take one final look at the ability of our metrics to record success. 
# Even if we can’t predict all the top teams in the competition, maybe we can predict the champion team. Surely the team which wins the most games or scores the most points over the regular season would be expected to do well in the championship!
# 
# The top performing team in the regular season was identified by both win-count and cumulative score for each year. The years where these teams went on to be the top performing team are listed below.
# 
# Using win-count, the national champion team was identified three times over the last 35 years, an 8.7% success rate.
# 
# **Using cumulative score, the national champion team was identified 7 times over the last 35 years, a 20.0% success rate.**
# For a single easy to calculate metric that’s pretty good!
# 
# Let’s compare these results to standard team ranking systems:
# 
# Since 1985, the AP number 1 ranked team has won the championship only 4 times - an 11% success rate. 
# 
# **This means that our easy metric of cumulative score is about twice as good at predicting the winning team.**
# 
# If we were to choose a team at random, we would be correct only 1.5% of the time, so this new metric is much better than guessing. Win-count on the other hand is significantly worse than cumulative score which demonstrates the benefit of considering the magnitude of wins rather than just absolute win or lose values.

# In[ ]:


# List of annual tournament champions 
tournament_number1team = {}
for year, year_group in tourney_results.groupby('Season'):
    number1team = year_group[(year_group['DayNum'] == 154)]
    tournament_number1team[year] = list(number1team.WTeamID)

# Win Count: top team in the regular season each year
regular_number1team_count = {}
n = 1
for year in year_team_cumulative:
    top_n = dict(sorted(year_team_cumulative[year].items(), key = itemgetter(1), reverse = True)[:n])
    regular_number1team_count[year] = {}
    for team, score in year_team_cumulative[year].items():
        regular_number1team_count[year] = list(top_n.keys()) 

# List of teams and years where win count top team went on to win the championship
reg_tournament_match = {}
for year in regular_number1team_count:
    if tournament_number1team[year] == regular_number1team_count[year]:
        reg_tournament_match[year] = tournament_number1team[year]

# Calculate percentage of years with matching topteam
total_years = len(tournament_number1team)
predict_number1_percent_count = len(reg_tournament_match)/total_years * 100


# Cumulative Score: top team in the regular season each year
regular_number1team_score = {}
n = 1
for year in year_team_cumulative_score:
    top_n_score = dict(sorted(year_team_cumulative_score[year].items(), key = itemgetter(1), reverse = True)[:n])
    regular_number1team_score[year] = {}
    for team, score in year_team_cumulative_score[year].items():
        regular_number1team_score[year] = list(top_n_score.keys()) 

# List of teams and years where cumulative score top team went on to win the championship
reg_tournament_match_score = {}
for year in regular_number1team_score:
    if tournament_number1team[year] == regular_number1team_score[year]:
        reg_tournament_match_score[year] = tournament_number1team[year]

# Calculate percentage of years with matching topteam
predict_number1_percent_score = len(reg_tournament_match_score)/total_years * 100

print("Years where the team with the most number of wins in the regular season became the national champion:")
print(list(reg_tournament_match.keys()))
print()
print("Years where the team with the highest cumulative score over the regular season became the national champion:")
print(list(reg_tournament_match_score.keys()))


# ## 10. Conclusions
# 
# By looking at historical data, we compared to what extent the results of the regular season are reflected by the results of March Madness.
# 
# Using the number of matches won over the preseason as a metric for success, we found that 50% of the top 16 teams in the national championship could be identified.
# 
# Using cumulative match score over the preseason instead as a metric, a comparable success rate in identifying the top team was identified.
# 
# These results show that while top teams are likely to do better in the tournament, it doesn’t guarantee them comparable success. There is probably some element of luck due to the unforgiving single-elimination nature of the tournament. March Madness doesn’t identify the teams which win consistently over many months, but rather the teams that perform well on a particular day or over a three-week period.
# 
# To further demonstrate this, the top performing team over the regular season was predicted using these metrics for success. The team that won the most matches was found to be the national champion only 8.7% of the time while the team which scored the most overall went on to win 20% of the time. This metric of success is more than twice as reliable as the current AP metric which has only been correct in predicting the champion team 11% of the time.  
# 
# These results demonstrate that even if your favourite team doesn't win the national championships, it doesn't mean they aren't the best team that year... it just means they weren't the best team that day.
