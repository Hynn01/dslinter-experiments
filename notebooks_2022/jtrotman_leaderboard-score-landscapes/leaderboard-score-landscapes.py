#!/usr/bin/env python
# coding: utf-8

# # Leaderboard Score Landscapes
# 
# This notebook shows traces of public/private submission scores for the whole leaderboard, for each Kaggle competition.
# 
# Leaderboards are based on ranks of scores, but the distribution of scores themselves can help show the nature of a competition.
# 
# #### Color Scheme
# 
#  - <font color=red>Private Scores</font> 
#  - <font color=blue>Public Scores</font>
#  - <font color=#3a3>Top Public Scores (a different ordering of teams)</font>
# 
# Medal zones are are marked with a dotted line. (When there is no public LB, or for recent kernel competitions - only the <font color=red>red</font> line of private LB scores is shown.)
# 
# Very often, the gold medal solutions will score very much better by the competition metric... A sharp gradient in the <font color=red>red</font> line.
# 
# A sharp gradient in the <font color=#3a3>green</font> line indicates heavy public LB overfitting (see [Mercedes-Benz Greener Manufacturing][13]).
# 
# When the different lines match up really well it tends to indicate a competition with very large test set (see [Avito Duplicate Ads Detection][14]).
# 
# When the red and blue lines go flat (horizontally) it indicates identical scores - generally from shared public submissions (public kernels, see [Google Analytics Customer Revenue Prediction][9] where the zero benchmark won medals!)
# 
# This is also yet another way to indicate **shake-up**: the amount the blue line dances around the green line! (For example, 100th place finisher's public score vs 100th best public score.)
# 
# 
# Some notable entries:
# 
# ### Strong Wins/Golds
#  - [Rossmann Store Sales][3] @gertjac's outstanding solo win with few submissions, from a bungalowpark :)
#  - [Expedia Hotel Recommendations][4] @idle_speculation's legendary solo win (with ***ONE*** submission!)
#  - [Porto Seguro’s Safe Driver Prediction][5] @mjahrer's de-noising auto encoder solution that is in a league of it's own.
#  - [Homesite Quote Conversion][6] obvious shelf where big teams pushed for gold places.
#  - [NFL Big Data Bowl 2020][11] Zoo win again, with a striking margin
#  - [Liverpool ION Switching][10] extreme outlier top score from team that found a flaw in the data preparation.
# 
# ### Public Kernels Winning Medals
#  - [Recruit Restaurant Visitor Forecasting][7] long trail of identical bronze scores.
#  - [TalkingData AdTracking Fraud Detection Challenge][8] last day share of silver-worthy CSV file.
# 
# 
# (This notebook is adapted from [Winning Team Submission Traces][1] which shows the submissions scores over time for the winning team in each competition.)
# 
# 
# ### Revisions
# 
# Version 9 creates two plots per competition: one for medalists and one for all teams. Some of the 'global' plots have outliers that squash the range, making most of the field look like one flat line. Excluding outliers is a balance between omitting some teams from view and keeping the plot interesting! This will be fixed later. It's useful to have this version to refer back to.
# 
#  [1]: https://www.kaggle.com/jtrotman/winning-team-submission-traces
#  [2]: https://www.kaggle.com/jtrotman/blender-medal-counts
#  [3]: #rossmann-store-sales
#  [4]: #expedia-hotel-recommendations
#  [5]: #porto-seguro-safe-driver-prediction
#  [6]: #homesite-quote-conversion
#  [7]: #recruit-restaurant-visitor-forecasting
#  [8]: #talkingdata-adtracking-fraud-detection
#  [9]: #ga-customer-revenue-prediction
#  [10]: #liverpool-ion-switching
#  [11]: #nfl-big-data-bowl-2020
#  [12]: #walmart-recruiting-trip-type-classification
#  [13]: #mercedes-benz-greener-manufacturing
#  [14]: #avito-duplicate-ads-detection
#  [15]: #planet-understanding-the-amazon-from-space
#  

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import gc, os, sys, time
import pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
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
comps['Year'] = comps.EnabledDate.dt.year
comps['RewardQuantity'].fillna('', inplace=True)

# Read teams - for LB ranks
teams = read_csv_filtered('Teams.csv', 'CompetitionId', comps.index).set_index('Id')
teams = teams.dropna(subset=['PublicLeaderboardSubmissionId', 'PrivateLeaderboardSubmissionId'])

# Read submissions - to get scores
subs = read_csv_filtered('Submissions.csv', 'TeamId', teams.index).set_index('Id')
subs['SubmissionDate'] = pd.to_datetime(subs.SubmissionDate)

asfloats = ['PublicScoreLeaderboardDisplay',
            'PublicScoreFullPrecision',
            'PrivateScoreLeaderboardDisplay',
            'PrivateScoreFullPrecision',]

subs[asfloats] = subs[asfloats].astype(float)
subs = subs.query('not IsAfterDeadline').copy()
subs['CompetitionId'] = subs.TeamId.map(teams.CompetitionId)

# values some competitions use as invalid scores
for bad in [99, 999999]:
    for c in asfloats:
        idx = (subs[c] == bad)
        subs.loc[idx, c] = subs.loc[idx, c].replace({bad: np.nan})

# Map scores to teams
# Beware: submission IDs are read as floats - should read as object & discard missing
teams['PublicScore'] = teams.PublicLeaderboardSubmissionId.map(subs.PublicScoreFullPrecision)
teams['PrivateScore'] = teams.PrivateLeaderboardSubmissionId.map(subs.PrivateScoreFullPrecision)
teams['Medal'].fillna(0, inplace=True)

score_cols = ['PublicScore', 'PrivateScore']

# The Random Number Grand Challenge looked like fun!
idx = teams.PublicScore > 1e99
teams.loc[idx, score_cols] = np.nan

# Mercedes-Benz Greener Manufacturing looked like fun!
idx = (teams.PublicScore < -7e7)
teams.loc[idx, ['PublicScore']] = np.nan

# Ordering for groupby
comp_id_order = comps.DeadlineDate.rank(method='first', ascending=False)
display_order = teams.CompetitionId.map(comp_id_order)


# In[2]:


plt.rc("figure", figsize=(18, 12))
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
    a, b = find_range(df.PublicScore.dropna().values * mul)
    A, B = find_range(df.PrivateScore.dropna().values * mul)

    A = min(a, A) * mul
    B = max(b, B) * mul

    R = (B - A)
    B += R / 20
    A -= R / 20
    return min(A, B), max(A, B)


# # Outlier Score Detection
# 
# This is work in progress...
# 
# If you want an explanation of this please comment and I'd be happy to write one :)

# In[3]:


NBUCKETS = 8
thresholds = np.arange(NBUCKETS + 1) / NBUCKETS

thresholds = {
    # EvaluationAlgorithmIsMax, best score on left
    True  : thresholds[::-1],
    # not EvaluationAlgorithmIsMax, best score on left
    False : thresholds
}

thresholds


# In[4]:


quantiles = {}
for cid, sub_df in teams.groupby('CompetitionId'):
    c = comps.loc[cid]
    thres = thresholds[c.EvaluationAlgorithmIsMax]
    vs = sub_df.PrivateScore.dropna().quantile(thres).values
    quantiles[cid] = (vs - vs.min()) / (vs.max() - vs.min())

quan_df = pd.Series(quantiles).apply(pd.Series).add_prefix('q')
quan_df.describe().T


# In[5]:


quan_df.index = comps.reindex(quan_df.index).Title
mag = (quan_df @ thresholds[True]) / quan_df.sum(1)
idx = np.argsort(mag)


# Hard to show all the labels with **Seaborn**...

# In[6]:


cmap = 'jet'
sns.heatmap(quan_df.iloc[idx], cmap=cmap);


# But **Pandas** can.

# In[7]:


with pd.option_context("display.max_rows", len(quan_df)):
    # need round(2) AND set_precision(2)
    display(quan_df.iloc[idx].round(2).style.background_gradient(axis=None, cmap=cmap).set_precision(2))


# # The Competitions

# In[8]:


medal_colors = ['Gold', 'Silver', 'Chocolate']

rank_cols = ['PublicLeaderboardRank', 'PrivateLeaderboardRank']

top_cols = [
    'TeamName', 'ScoreFirstSubmittedDate', 'LastSubmissionDate',
    'PublicLeaderboardRank', 'PublicScore', 'PrivateScore'
]


for i, (comp, sub_df) in enumerate(teams.groupby(display_order)):
        
    comp_id = sub_df.iloc[0].CompetitionId
    c = comps.loc[comp_id]

    ranked = sub_df.dropna(subset=rank_cols).copy()
    if ranked.shape[0] < 1:
        continue

    ranked[rank_cols] = ranked[rank_cols].astype(int)
    public_redundant = ranked.eval('PublicLeaderboardRank==PrivateLeaderboardRank').all()
    public_redundant = public_redundant or (ranked.PublicScore.var() == 0)
    abs_diff = ranked.eval('abs(PublicLeaderboardRank-PrivateLeaderboardRank)')
    shakeup = (abs_diff / len(ranked)).mean()
    
    # use LB rank to sort, then we don't even need to know EvaluationAlgorithmIsMax!
    df = ranked.set_index('PrivateLeaderboardRank')
    df = df.sort_index()

    if not public_redundant:
        pub = ranked.set_index('PublicLeaderboardRank')
        pub = pub.sort_index()
    else:
        pub = None

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
        '<br/>').format(**c)

    tmp = df.loc[[1, 2, 3, 4, 5], top_cols].copy()
    tmp.columns = tmp.columns.str.replace(r'([a-z])([A-Z])', r'\1<br/>\2', regex=True)
    markup += 'Top Five: '
    markup += tmp.to_html(index_names=False,
                          notebook=True,
                          escape=False,
                          na_rep='')
    display(HTML(markup))

    vc = df.Medal.value_counts()
    title = c.Title
    if str(c.Year) not in title:
        title += f' [{c.Year}]'
    if 1 in vc and 2 in vc and 3 in vc:
        title += f' - {vc[1]} gold; {vc[2]} silver; {vc[3]} bronze'
    if shakeup > 0:
        title += f' - {shakeup:.3f} shake-up'

    mthres = np.where(df.Medal.diff())[0]

    mdl = df.query('Medal!=0')
    if len(mdl) < 1:
        # no medals; show top 10%
        n = int(np.ceil(len(df) / 10))
        mdl = df.head(n)

    ############################## medalists
    plt.subplot(2, 1, 1)

    if pub is not None:
        pub.head(len(mdl))['PublicScore'].plot(color='Green',
                                               label='Best public')
        mdl['PublicScore'].plot(color='Blue', label='Public score', alpha=0.3)

    mdl['PrivateScore'].plot(color='Red', label='Private score')

    if len(mthres) == 4:
        xmin = 0.5
        for color, xval in zip(medal_colors, mthres[1:]):
            plt.axvspan(xmin, xmax=xval + 0.5, color=color, alpha=0.2)
            xmin = xval + 0.5

    plt.xlim(left=1)
    plt.title(title)
    plt.legend()
    plt.ylabel(c.EvaluationLabel)
    plt.grid(True, axis='x')

    ############################## global
    plt.subplot(2, 1, 2)

    if pub is not None:
        pub['PublicScore'].plot(color='Green', label='Best public')
        df['PublicScore'].plot(color='Blue', label='Public score', alpha=0.3)

    ax = df['PrivateScore'].plot(color='Red', label='Private score')

    if len(mthres) == 4:
        xmin = 0.5
        for color, xval in zip(medal_colors, mthres[1:]):
            plt.axvspan(xmin, xmax=xval + 0.5, color=color, alpha=0.2)
            xmin = xval + 0.5

    bottom, top = get_range(df)
    plt.ylim(bottom, top)
    plt.xlim(left=1)
    plt.legend()
    plt.ylabel(c.EvaluationLabel)
    plt.grid(True, axis='x')

    ############################## end
    plt.tight_layout()
    plt.show()


# ____
# 
# # Conclusions
# 
# Generally, the gold, silver, bronze thresholds do a really good job, it is rare that a publicly shared solution gets a medal.
# 
# Plotting the distribution of marathon running times shows spikes: people push harder to hit a new landmark time like sub three hours. It's the same here, the lure of a competition medal is strong!

# In[9]:


_ = """
Re-run to include recently completed competitions:

    Slug:feedback-prize-2021
    Slug:tabular-playground-series-mar-2022
    Slug:mens-march-mania-2022
    Slug:womens-march-mania-2022
    Slug:kore-2022-beta
    Slug:happy-whale-and-dolphin


"""

