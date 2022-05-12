#!/usr/bin/env python
# coding: utf-8

# **Exponentially Weighted Ensemble**
# 
# This model is an ensemble consisting of a weighted average of models from public notebooks. As such, it continues the long tradition of ensembling in Kaggle competitions. The technique chosen here is one that optimises a single parameter in order to determine the weights of the component models, and hence may be less vulnerable to overfitting to the public LB data than are regressions with maybe only one fewer parameter than models. Nonetheless, we give more weight to better models, so we may outperform a mean-based or median-based averaging approach. 
# 
# Each model represented in the ensemble is weighted by a factor that depends in an exponentially decaying manner on its public LB score. Specifically, each model is given an exponential weight according to
# 
# exp(b*(S-x))
# 
# where x is the public LB score of that model. Larger scores are worse, hence the weights get smaller as x increases and get larger as x decreases. Thus, the best models have the largest weights and make the largest contributions to the ensemble.
# 
# The parameter b is the one meaningfully adjustable parameter of the model, the larger b is then the faster the weights decay as the score gets worse. S is a calibration parameter defined such that if S is set to the best single model score then the highest unnormalised weight exp(b*(x-S)) is 1.0, which is convenient, but not essential.
# 
# The sum of the unnormalised weights is called q. Once all weights have been calculated, these weights are normalised by dividing them all by q.

# **Ask a friend**
# 
# This is like asking your friends each to predict the outcome of every game, and then scaling their guesses by how much you reckon each one knows about football. Here, however, we have the public LB scores as a guideline for how knowledgable each available notebook is. We give a higher weight to those we trust more.

# **Pundit or Prophet?**
# 
# We know that this competition requires 'predictions' to be made after the event. Thus, it is inherently vulnerable to both accidental leaks or deliberatev cheats. The hosts have explained why this is at it is, and if this were a Featured Competition with medals and larger prizes, it would be designed differently.
# 
# This notebook has from v16 onwards turned the internet off and does not look up past results from the training data.
# 
# In order to ensure that all predictions in a given row are for the same match, all files are sorted and ordered by 'id'. This does not imply use of information from later dates in the predictions.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **Notebooks included**
# 
# [TOP 5: Football prob prediction - LSTM_v02 (GPU)](https://www.kaggle.com/code/seraquevence/top-5-football-prob-prediction-lstm-v02-gpu) by @seraquevence (v38 0.99661)
# 
# [TOP 5: Football prob prediction - LSTM_v01](https://www.kaggle.com/code/seraquevence/top-5-football-prob-prediction-lstm-v01) by @seraquevence (v76 0.99945; v104 0.99695)
# 
# [LSTM and Feature Engineering-Top8](https://www.kaggle.com/code/ravi07bec/lstm-and-feature-engineering-top8) by @ravi07bec (v10 1.00523)
# 
# [Football Match Probability Prediction-LSTM starter](https://www.kaggle.com/code/igorkf/football-match-probability-prediction-lstm-starter) by @igorkf (v6 1.00598)
# 
# [Football Results Prediction](https://www.kaggle.com/code/henriqueweber/football-results-prediction) by @henriqueweber (v5 1.01021)
# 
# [Football Match Prediction XGBoost Baseline](https://www.kaggle.com/code/yousseftaoudi/football-match-prediction-xgboost-baseline) by @yousseftaoudi (v31 1.01122)
# 
# [Football-Match-Prediction](https://www.kaggle.com/code/shaggy11/football-match-prediction) by @shaggy11 (v5 1.01325)
# 
# [Football Probability Prediction](https://www.kaggle.com/code/purvansharora/football-probability-prediction) by @purvansharora (v11 1.01340)
# 
# [Logistic Regression Submission Example](https://www.kaggle.com/code/octosportio/logistic-regression-submission-example) by @octosportio (v1 1.01760)
# 
# [Investigation into rating feature octosport ](https://www.kaggle.com/code/curiosityquotient/investigation-into-rating-feature-octosport) by @curiosityquotient (v5 1.05541; v8 1.05272)
# 
# [Constant Probability Baseline ](https://www.kaggle.com/code/jbomitchell/constant-probability-baseline) by @jbomitchell (v3 1.0757)
# 
# [xG (expected goals) with simple sklearn models](https://www.kaggle.com/code/uzdavinys/xg-expected-goals-with-simple-sklearn-models) by @uzdavinys (v5 1.02505)
# 
# [football-match-probability-prediction Baseline](https://www.kaggle.com/code/hoangnguyen719/football-match-probability-prediction-baseline) by @hoangnguyen719 (v1 1.09861)
# 
# [Ensemble: Football prob prediction - LSTM+LGBM_v01](https://www.kaggle.com/code/seraquevence/ensemble-football-prob-prediction-lstm-lgbm-v01) by seraquevence (v15 0.99708)
# 
# [notebook931e348ab2](https://www.kaggle.com/code/arjunjanamatti/notebook931e348ab2) by @arjunjanamatti (v4 1.01931)

# In[ ]:


sub = pd.read_csv('../input/football-match-probability-prediction/sample_submission.csv')
sub.sort_values(by=['id'], inplace=True)
sub0 = pd.read_csv('../input/football-match-probability-prediction/sample_submission.csv')
sub0.sort_values(by=['id'], inplace=True)
sub1 = pd.read_csv('../input/football-results-prediction/099945_seraquevence_v76_submission.csv')
sub1.sort_values(by=['id'], inplace=True)
sub2 = pd.read_csv('../input/football-results-prediction/100523_ravi07bec_v10_submission.csv')
sub2.sort_values(by=['id'], inplace=True)
sub3 = pd.read_csv('../input/football-results-prediction/100598_igorkf_v6_submission.csv')
sub3.sort_values(by=['id'], inplace=True)
sub4 = pd.read_csv('../input/football-results-prediction/101021_henriqueweber_v5_submission.csv')
sub4.sort_values(by=['id'], inplace=True)
sub5 = pd.read_csv('../input/football-results-prediction/101122_yousseftaoudi_v31_submission.csv')
sub5.sort_values(by=['id'], inplace=True)
sub6 = pd.read_csv('../input/football-results-prediction/101325_shaggy11_v6_submission.csv')
sub6.sort_values(by=['id'], inplace=True)
sub7 = pd.read_csv('../input/football-results-prediction/101340_purvansharora_v11_submission.csv')
sub7.sort_values(by=['id'], inplace=True)
sub8 = pd.read_csv('../input/football-results-prediction/101760_octosportio_v1_submission.csv')
sub8.sort_values(by=['id'], inplace=True)
sub9 = pd.read_csv('../input/football-results-prediction/105541_curiosityquotient_v5_submission.csv')
sub9.sort_values(by=['id'], inplace=True)
sub10 = pd.read_csv('../input/football-results-prediction/107057_jbomitchell_v3_submission.csv')
sub10.sort_values(by=['id'], inplace=True)
sub11 = pd.read_csv('../input/football-results-prediction/099695_seraquevence_v104_submission.csv')
sub11.sort_values(by=['id'], inplace=True)
sub12 = pd.read_csv('../input/football-results-prediction/099661_seraquevence_v38_submission.csv')
sub12.sort_values(by=['id'], inplace=True)
sub13 = pd.read_csv('../input/football-results-prediction/102505_uzdavinys_v5_submission.csv')
sub13.sort_values(by=['id'], inplace=True)
sub14 = pd.read_csv('../input/football-results-prediction/109681_hoangnguyen719_v1_submission.csv')
sub14.sort_values(by=['id'], inplace=True)
sub15 = pd.read_csv('../input/football-results-prediction/105272_curiosityquotient_v8_submission.csv')
sub15.sort_values(by=['id'], inplace=True)
sub16 = pd.read_csv('../input/football-results-prediction/101931_arjunjanamatti_v4_submission.csv')
sub16.sort_values(by=['id'], inplace=True)
sub17 = pd.read_csv('../input/football-results-prediction/099708_seraquevence_v15_submission.csv')
sub17.sort_values(by=['id'], inplace=True)


# **Setting the parameters**
# 
# We can get an approximate feel for a suitable value of b by considering the difference in LB score for which we would want a second model to have a factor of e (~2.72) smaller contribution to the ensemble than the best available model. If we want the contribution to drop of by a factor of e for each 0.01 deterioration in score, then we would set b to 100.0. In practice, there is some trial and error involved in keeping b close to optimal as the ensemble progresses throughout the competition. 
# 
# S is set to the score of the best component model; this makes the maximum weight 1.0. Forgetting to reset S when a new best score is available will not have a significant impact.
# 
# q is the sum of the weights. We increment it as we add the contribution of each new model.

# In[ ]:


b = 490.0
S = 0.99661
q = 0.0


# **Iterating over component models to set 'home' probability**
# 
# We now iterate over each of the component models' 'home' probabilities, multiplying each one by the corresponding exponential weighting factor, and keeping track of the running sum and running sum of weights.
# 
# At the end of this process, we divide the final sum of weighted 'home' probabilities by the sum of weights to obtain a suitably scaled ensemble predicted 'home' probability.

# **Non-zero sum game**
# 
# Because 'home' probabilities are on average higher than away probabilites and more home than away fans attend a typical match, on average more fans see their team win than see their team lose. 

# In[ ]:


sub['home'] = sub1['home']*np.exp(b*(S-0.99945))
q = q + np.exp(b*(S-0.99945))
sub['home'] = sub['home'] + sub2['home']*np.exp(b*(S-1.00523))
q = q + np.exp(b*(S-1.00523))
sub['home'] = sub['home'] + sub3['home']*np.exp(b*(S-1.00598))
q = q + np.exp(b*(S-1.00598))
sub['home'] = sub['home'] + sub4['home']*np.exp(b*(S-1.01021))
q = q + np.exp(b*(S-1.01021))
sub['home'] = sub['home'] + sub5['home']*np.exp(b*(S-1.01122))
q = q + np.exp(b*(S-1.01122))
sub['home'] = sub['home'] + sub6['home']*np.exp(b*(S-1.01325))
q = q + np.exp(b*(S-1.01325))
sub['home'] = sub['home'] + sub7['home']*np.exp(b*(S-1.01340))
q = q + np.exp(b*(S-1.01340))
sub['home'] = sub['home'] + sub8['home']*np.exp(b*(S-1.01760))
q = q + np.exp(b*(S-1.01760))
sub['home'] = sub['home'] + sub9['home']*np.exp(b*(S-1.05541))
q = q + np.exp(b*(S-1.05541))
sub['home'] = sub['home'] + sub10['home']*np.exp(b*(S-1.07057))
q = q + np.exp(b*(S-1.07057))
sub['home'] = sub['home'] + sub11['home']*np.exp(b*(S-0.99695))
q = q + np.exp(b*(S-0.99695))
sub['home'] = sub['home'] + sub12['home']*np.exp(b*(S-0.99661))
q = q + np.exp(b*(S-0.99661))
sub['home'] = sub['home'] + sub13['home']*np.exp(b*(S-1.02505))
q = q + np.exp(b*(S-1.02505))
sub['home'] = sub['home'] + sub14['home']*np.exp(b*(S-1.09681))
q = q + np.exp(b*(S-1.09681))
sub['home'] = sub['home'] + sub15['home']*np.exp(b*(S-1.05272))
q = q + np.exp(b*(S-1.05272))
sub['home'] = sub['home'] + sub16['home']*np.exp(b*(S-1.01931))
q = q + np.exp(b*(S-1.01931))
sub['home'] = sub['home'] + sub17['home']*np.exp(b*(S-0.99708))
q = q + np.exp(b*(S-0.99708))
sub['home'] = sub['home']/q
print(q)


# Because we are going to reuse the same variable q when we iterate over the away probabilities, we must remember to reset q to 0.0.

# In[ ]:


q = 0.0


# **Iterating over component models to set 'away' probability**
# 
# We now iterate over each of the component models' 'away' probabilities, multiplying each one by the corresponding exponential weighting factor, and keeping track of the running sum and running sum of weights.
# 
# At the end of this process, we divide the final sum of weighted 'away' probabilities by the sum of weights to obtain a suitably scaled ensemble predicted 'away' probability.

# In[ ]:


sub['away'] = sub1['away']*np.exp(b*(S-0.99945))
q = q + np.exp(b*(S-0.99945))
sub['away'] = sub['away'] + sub2['away']*np.exp(b*(S-1.00523))
q = q + np.exp(b*(S-1.00523))
sub['away'] = sub['away'] + sub3['away']*np.exp(b*(S-1.00598))
q = q + np.exp(b*(S-1.00598))
sub['away'] = sub['away'] + sub4['away']*np.exp(b*(S-1.01021))
q = q + np.exp(b*(S-1.01021))
sub['away'] = sub['away'] + sub5['away']*np.exp(b*(S-1.01122))
q = q + np.exp(b*(S-1.01122))
sub['away'] = sub['away'] + sub6['away']*np.exp(b*(S-1.01325))
q = q + np.exp(b*(S-1.01325))
sub['away'] = sub['away'] + sub7['away']*np.exp(b*(S-1.01340))
q = q + np.exp(b*(S-1.01340))
sub['away'] = sub['away'] + sub8['away']*np.exp(b*(S-1.01760))
q = q + np.exp(b*(S-1.01760))
sub['away'] = sub['away'] + sub9['away']*np.exp(b*(S-1.05541))
q = q + np.exp(b*(S-1.05541))
sub['away'] = sub['away'] + sub10['away']*np.exp(b*(S-1.07057))
q = q + np.exp(b*(S-1.07057))
sub['away'] = sub['away'] + sub11['away']*np.exp(b*(S-0.99695))
q = q + np.exp(b*(S-0.99695))
sub['away'] = sub['away'] + sub12['away']*np.exp(b*(S-0.99661))
q = q + np.exp(b*(S-0.99661))
sub['away'] = sub['away'] + sub13['away']*np.exp(b*(S-1.02505))
q = q + np.exp(b*(S-1.02505))
sub['away'] = sub['away'] + sub14['away']*np.exp(b*(S-1.09681))
q = q + np.exp(b*(S-1.09681))
sub['away'] = sub['away'] + sub15['away']*np.exp(b*(S-1.05272))
q = q + np.exp(b*(S-1.05272))
sub['away'] = sub['away'] + sub16['away']*np.exp(b*(S-1.01931))
q = q + np.exp(b*(S-1.01931))
sub['away'] = sub['away'] + sub17['away']*np.exp(b*(S-0.99708))
q = q + np.exp(b*(S-0.99708))
sub['away'] = sub['away']/q
print(q)


# For tidiness, we will reset q to 0.0 again. Perhaps, we may wish to rewrite our code in a later version to calculate the 'draw' probability iteratively, and it would be easy to forget to reset q.

# In[ ]:


q = 0.0


# **The 'draw' probability**
# 
# The 'draw' probability is often the smallest of the three, and simple three-class classification models may therefore predict rather few draws. This isn't particularly significant here, as all our predictions are probabilistic.
# 
# Predicting draws was historically important in the UK because of the football pools, where life-changing amounts of money like £100,000 (in the 1970s, that was a fortune) could be won by picking eight score draws on the 'Treble Chance'. The football results would conclude with something like: "There were 31 home wins, 15 away wins and 9 draws, of which 8 were score draws. Telegram claims are required for 23 or 24 points and the dividend forecast is a possible jackpot."
# 
# Although we could compute the 'draw' probabilities iteratively as with 'home' and 'away', here we simply use the 'probabilities must sum to one' principle to infer them.

# In[ ]:


sub['draw'] = 1.0 - sub['home'] - sub['away']


# **Empirical Shift**
# 
# Here we test the effect of adding an empirical uniform shift to the predicted probabilities. This is designed to cover the possibility that there is a systematic non-optimality in the balance between the 'home', 'away', and 'draw' probabilities predicted by the models. In version 1 of this notebook, we found that transferring 0.001 probability from 'home' to 'draw' slightly improved the score on the public LB; version 9 showed an improvement when 0.001 was shifted from 'home' to 'away'. The downside is that there is a risk of overfitting.

# In[ ]:


sub['home'] = sub['home'] - 0.001
sub['draw'] = sub['draw'] + 0.001

sub['home'] = sub['home'] - 0.0015
sub['away'] = sub['away'] + 0.0015


# ** Clipping **
# 
# For the predictions, I make a couple of changes to the code.
# 
# Firstly, I clip the base home, away and draw probabilities to lie within the ranges given by:
# 
# sub['home'] = np.clip(sub['home'], 0.1, 0.95)
# 
# sub['away'] = np.clip(sub['away'], 0.05, 0.9)
# 
# sub['draw'] = np.clip(sub['draw'], 0.125, 0.325)
# 
# These values are based on the plots by @curiosityquotient and the tendency of log loss to punish predictions that are both extreme and wrong.
# 
# I also normalise the predictions such that the three final probabilities sum to one:
# 
# sub0['total'] = sub['home'] + sub['away'] + sub['draw']
# 
# sub['home'] = sub['home']/sub0['total']
# 
# sub['away'] = sub['away']/sub0['total']
# 
# sub['draw'] = sub['draw']/sub0['total']

# In[ ]:


sub['home'] = np.clip(sub['home'], 0.1, 0.95)

sub['away'] = np.clip(sub['away'], 0.05, 0.9)

sub['draw'] = np.clip(sub['draw'], 0.125, 0.325)

sub0['total'] = sub['home'] + sub['away'] + sub['draw']

sub['home'] = sub['home']/sub0['total']

sub['away'] = sub['away']/sub0['total']

sub['draw'] = sub['draw']/sub0['total']


# **Submission**
# 
# We write the predicted probabilities to a .csv file and print a few of them. It is important to check that these look sensible, are contained within the interval 0 -> 1, and sum to 1.0 for each match.

# In[ ]:


sub.to_csv('submission.csv', index=False)
sub.head(10)

