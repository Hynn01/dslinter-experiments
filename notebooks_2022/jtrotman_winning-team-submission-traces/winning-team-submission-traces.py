#!/usr/bin/env python
# coding: utf-8

# # Winning Team Submission Traces
# 
# This notebook shows public/private submission scores over time for winning teams of all the main Kaggle competitions. (Excluded are: winners with only 1 or 2 submissions.)
# 
# Evaluation metrics vary over competitions and the direction of 'better' submissions changes - to indicate this, the peak score in each competition for both public & private scores are shown as a dotted line.
# 
# Public leaderboard scores are in blue and private in red; and the submissions that are used for the public and private LB are marked with a dot.
# 
# The final week of each competition is now highlighted.
# 
# So, the name of the game when Kaggling is to get the red line, the red dotted line, and the red point all to coincide at the same point, that is: to select your submission with the best private test set score.
# 
# See also [this notebook][1] that generates these traces for all competitions a user has entered - feel free to fork it and try it out on your own competition history!
# 
# You can find the [stories behind most of these winning team solutions in this extensive notebook][2].
# 
# ## Notes
# 
# Some plots like <a href="#santander-product-recommendation">Santander Product Recommendation</a> and <a href="#predicting-red-hat-business-value">Red Hat</a> are hard to see the fine details because many of the submissions are from [leaderboard probing to scrape more information about the test set][3].
# 
# <a href="#facebook-v-predicting-check-ins">Other plots</a> may look like this but are in fact "mini submissions" &mdash; with a metric like MAP you can submit only 10% of predictions and scale up your score to check your progress and iterate much faster (this can save *hours* of time when the the test set is very large.)
# 
# ## A Story
# 
# 
# As shared by [Giulio](https://www.kaggle.com/adjgiulio) on [Quora](https://www.quora.com/How-did-you-become-a-Kaggle-Master-and-what-are-the-steps-resources-you-used-to-get-there) about an early [Avito competition][4]:
# 
# <blockquote>
# In the Avito competition I won, my teammate and I had a very good approach from the get going. We were in and out of the top-5 for most of the time. Every time we’d go in the top 3, some of the other top teams would merge and surpass us. With two weeks to go, and one week to the merger deadline, we came across a novel approach based on semi-supervised learning that would have put us in 1st (based on cross validation data we had). If we had made that submission I have 0 doubts some of the other teams would have merged and eventually won. So, we intentionally held off our best submission, slowly improving by submitting submissions where part of the predictions were replaced by random numbers. And we waited, and waited, and fell and fell on the leaderboard. Then the merger deadline came, and the day after (and the whole week after) was glorious. We submitted our best submission and at that point it was all over.
# </blockquote>
# 
# Can we see this in the [plot][4]?
# 
# 
#  [1]: https://www.kaggle.com/jtrotman/user-competition-submission-traces
#  [2]: https://www.kaggle.com/jtrotman/high-ranking-solution-posts
#  [3]: https://www.kaggle.com/c/predicting-red-hat-business-value/discussion/23786
#  [4]: #avito-prohibited-content
#  

# In[1]:


import gc, os, sys, time
import pandas as pd, numpy as np
from unidecode import unidecode
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython.display import HTML, display

CSV_DIR = Path('..', 'input', 'meta-kaggle')
if not CSV_DIR.is_dir():
    CSV_DIR = Path('..', 'input')

def read_csv_filtered(csv, col, values):
    dfs = [df.loc[df[col].isin(values)]
           for df in pd.read_csv(CSV_DIR / csv, chunksize=100000, low_memory=False)]
    return pd.concat(dfs, axis=0)

comps = pd.read_csv(CSV_DIR / 'Competitions.csv').set_index('Id')
comps = comps.query("HostSegmentTitle != 'InClass'")
idx = comps.EvaluationAlgorithmName.isnull()
comps.loc[idx, 'EvaluationAlgorithmName'] = comps.loc[idx, 'EvaluationAlgorithmAbbreviation']

comps['EvaluationLabel'] = comps.EvaluationAlgorithmAbbreviation
idx = comps.EvaluationLabel.str.len() > 30
comps.loc[idx, 'EvaluationLabel'] = comps.loc[idx, 'EvaluationLabel'].str.replace(r'[^A-Z\d\-]', '', regex=True)

comps['DeadlineDate'] = pd.to_datetime(comps.DeadlineDate)
comps['EnabledDate'] = pd.to_datetime(comps.EnabledDate)
comps['DeadlineDateText'] = comps.DeadlineDate.dt.strftime('%c')
comps['EnabledDateText'] = comps.EnabledDate.dt.strftime('%c')
comps['Year'] = comps.DeadlineDate.dt.year
comps['RewardQuantity'].fillna('', inplace=True)
comps['Days'] = (comps.DeadlineDate - comps.EnabledDate) / pd.Timedelta(1, 'd')
comps['FinalWeek'] = (comps.DeadlineDate - pd.Timedelta(1, 'w'))

teams = read_csv_filtered('Teams.csv', 'CompetitionId', comps.index).set_index('Id')
# Just the winning teams
teams = teams.query('PrivateLeaderboardRank==1').copy()

tmemb = read_csv_filtered('TeamMemberships.csv', 'TeamId', teams.index).set_index('Id')
users = read_csv_filtered('Users.csv', 'Id', tmemb.UserId)
tmemb = tmemb.merge(users, left_on='UserId', right_on='Id')

# Submissions
subs = read_csv_filtered('Submissions.csv', 'TeamId', tmemb.TeamId)
subs = subs.rename(columns={'PublicScoreFullPrecision': 'Public'})
subs = subs.rename(columns={'PrivateScoreFullPrecision': 'Private'})
subs['SubmissionDate'] = pd.to_datetime(subs.SubmissionDate)

asfloats = ['PublicScoreLeaderboardDisplay',
            'Public',
            'PrivateScoreLeaderboardDisplay',
            'Private',]

subs[asfloats] = subs[asfloats].astype(float)
# subs.IsAfterDeadline.mean()

subs = subs.query('not IsAfterDeadline').copy()
subs['CompetitionId'] = subs.TeamId.map(teams.CompetitionId)
# subs['CompetitionId'].nunique()

# values some competitions use as invalid scores
for bad in [99, 999999]:
    for c in asfloats:
        idx = (subs[c] == bad)
        subs.loc[idx, c] = subs.loc[idx, c].replace({bad: np.nan})

# Display order: most recent competitions first
subs = subs.sort_values(['CompetitionId', 'Id'], ascending=[False, True])


# In[2]:


plt.rc("figure", figsize=(14, 6))
plt.rc("font", size=14)
plt.rc("axes", xmargin=0.01)
plt.rc("axes", edgecolor='#606060')


def find_range(scores):
    scores = sorted(scores)
    n = len(scores)
    max_i = n - 1
    for i in range(n // 2, n):
        best = scores[:i]
        if len(best):
            m = np.mean(best)
            s = np.std(best)
            if s != 0:
                z = (scores[i] - m) / s
                if abs(z) < 3:
                    max_i = i
    return scores[0], scores[max_i]


def get_range(df):
    comp_id = df.iloc[0].CompetitionId
    c = comps.loc[comp_id]

    mul = -1 if c.EvaluationAlgorithmIsMax else 1
    a, b = find_range(df.Public.dropna().values * mul)
    A, B = find_range(df.Private.dropna().values * mul)

    A = min(a, A) * mul
    B = max(b, B) * mul

    R = (B - A)
    B += R / 20
    A -= R / 20
    return min(A, B), max(A, B)


# In[3]:


COLORS = dict(Public='blue', Private='red')

for i, (comp_id, subs_df) in enumerate(subs.groupby('CompetitionId', sort=False)):

    if subs_df.shape[0] < 3:
        continue
    if subs_df.Public.count() < 1:
        continue
    if subs_df.Private.count() < 1:
        continue
    
    c = comps.loc[comp_id]
    df = subs_df.sort_values('Id').reset_index()
    team_id = df.iloc[0].TeamId
    f = 'max' if c.EvaluationAlgorithmIsMax else 'min'
        
    mcols = ['UserName', 'RequestDate', 'DisplayName', 'SubCount',
             'RegisterDate', 'PerformanceTier']
    tcols = ['TeamName', 'ScoreFirstSubmittedDate', 'LastSubmissionDate',
             'PublicLeaderboardRank', 'PrivateLeaderboardRank']

    team = teams.query(f'CompetitionId=={c.name}').iloc[0]
    members = tmemb.query(f'TeamId=={team_id}').copy()
    members['SubCount'] = members.UserId.map(df.SubmittedUserId.value_counts()).fillna(0)
    members = members[mcols].set_index('UserName')
    members = members.T.dropna(how='all').T
    members.columns = members.columns.str.replace(r'([a-z])([A-Z])', r'\1<br/>\2', regex=True)

    markup = (
        '<h1 id="{Slug}">{Title}</h1>'
        '<p>'
        'Type: {HostSegmentTitle} &mdash; <i>{Subtitle}</i>'
        '<br/>'
        '<a href="https://www.kaggle.com/c/{Slug}/leaderboard">Leaderboard</a>'
        '<br/>'
        'Dates: <b>{EnabledDateText}</b> &mdash; <b>{DeadlineDateText}</b>'
        '<br/>'
        '<b>{TotalTeams}</b> teams; <b>{TotalCompetitors}</b> competitors; '
        '<b>{TotalSubmissions}</b> submissions'
        '<br/>'
        'Leaderboard percentage: <b>{LeaderboardPercentage}</b>'
        '<br/>'
        'Evaluation: <a title="{EvaluationAlgorithmDescription}">{EvaluationAlgorithmName}</a>'
        '<br/>'
        'Reward: <b>{RewardType}</b> {RewardQuantity} [{NumPrizes} prizes]'
        '<br/>'
        ).format(**c)

    markup += f'<h3>Team Members</h3>'
    markup += members.to_html(index_names=False, notebook=True, escape=False, na_rep='')
    markup += f'<h3>Submissions</h3>'
    display(HTML(markup))
    
    title = c.Title
    title += (' "{TeamName}"'
              ' - [public {PublicLeaderboardRank:.0f} '
              '| private {PrivateLeaderboardRank:.0f}]').format(**team)
    
    for t in ['Public', 'Private']:
        ax = df[t].plot(legend=True, color=COLORS[t])

        ser = df.Id.isin(teams[f'{t}LeaderboardSubmissionId'])
        q = df.loc[ser]
        plt.scatter(np.where(ser)[0], q[t], color=COLORS[t])

        # dotted line of peak score
        xs = np.arange(df.shape[0])
        yb = np.ones(df.shape[0])
        plt.plot(xs, yb * df[t].apply(f), linestyle=':', color=COLORS[t])

    if c.Days > 7:
        last_week = (df['SubmissionDate'] >= c.FinalWeek)
        week_markers = np.where(last_week)[0]
        if len(week_markers):
            plt.axvspan(week_markers.min(), week_markers.max(), color='k', alpha=0.1)

    if df.shape[0] > 4:
        bottom, top = get_range(df)
        plt.ylim(bottom, top)
    plt.title(unidecode(title))
    plt.ylabel(c.EvaluationLabel)
    plt.xlabel('Submission Index')
    plt.xlim(-1, df.shape[0])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()


# In[4]:


_ = """
Re-run to include recently completed competitions:

    Slug:feedback-prize-2021
    Slug:tabular-playground-series-mar-2022
    Slug:mens-march-mania-2022
    Slug:womens-march-mania-2022
    Slug:kore-2022-beta
    Slug:happy-whale-and-dolphin


"""

