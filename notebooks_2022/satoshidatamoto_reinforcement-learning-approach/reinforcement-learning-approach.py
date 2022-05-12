#!/usr/bin/env python
# coding: utf-8

# >### Reinforcement Learning Starter
# >This is a simple starter notebook for Kaggle's Crypto Comp showing purged group timeseries KFold with extra data. There are many configuration variables below to allow you to experiment. Use either CPU or GPU. You can control which years are loaded, which neural networks are used, and whether to use feature engineering. You can experiment with different data preprocessing, model hyperparameters, loss, and number of seeds to ensemble. The extra datasets contain the full history of the assets at the same format of the competition, so you can input that into your model too.
# >

# # <span class="title-section w3-xxlarge" id="codebook">Reinforcement Learning Starter</span>
# This is a simple starter notebook for Kaggle's Crypto Comp showing the usage of reinforcement learning for crypto trading. There are many configuration variables below to allow you to experiment. Use a different stable-baseline agent. You can control which years are loaded, which models are used, and whether to use feature engineering. You can experiment with different data preprocessing, model hyperparameters, losses, and number of seeds to ensemble. The extra datasets contain the full history of the assets at the same format of the competition, so you can input that into your model too.
# 
# ____
# **Credits:**
# 
# Notebook this is baseline is based on: 
# - [Jane Street: Deep Reinforcement Learning Approach](https://www.kaggle.com/gogo827jz/jane-street-deep-reinforcement-learning-approach) by Yirun Zhang
# - [🤖 Deep RL: PPO2 with GPU Baseline 🦾](https://www.kaggle.com/metathesis/deep-rl-ppo2-with-gpu-baseline) by Jim Eric Skogman
# - [Triple Stratified Kfold with TFrecords](https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords) by Chris Deotte
# 

# # <span class="title-section w3-xxlarge" id="codebook">Why Reinforcement Learning?</span>
# 
# 
# > Credit: Huge credit to the original kaggle learn tutorial for providing the skeleton for this intro
# 
# 
# ### Introduction
# So far, on this competition, most notebooks only implemented supervised learning solutions. These solutions have relied on detailed information about how to trade the assets. 
# However, do to the nature of financial trading [specifically speaking: No "clearly defined" target]. A reinforcement learning approach might fit the problem of cryptocurrency trading well.
# 
# In this baseline notebook, I'll try and use reinforcement learning to build an intelligent agent without the use of a supervised target. Instead, we will gradually refine the agent's strategy over time, simply by trading the assets historicly  trying to maximize the the prediction of the target.
# 
# Since this is a starter notebook, we won't be able to explore this complex field in detail, but I hope to share about the big picture and explore some code that you can then use and improve yourself.
# 
# ### From Supervised Neural Networks to Deep Reinforcement Learning
# 
# It's difficult to come up with a perfect heuristic. Improving the heuristic generally entails trading the assets many times, to determine specific cases where the agent could have made better choices. And, it can prove challenging to interpret what exactly is going wrong, and ultimately to fix old mistakes without accidentally introducing new ones.
# 
# Wouldn't it be much easier if we had a more systematic way of improving the agent with trading experience?
# We'll just replace the supervised target with a neural network to estimate what should have been the target at the current observation state. 
# The network accepts the current market features as input. And, it outputs a value that is then used for trading.
# 
# This way, to encode a trading strategy, we need only amend the weights of the network so that for every possible market condition, it assigns higher probabilities to better trades.
# At least in theory, that's our goal. In practice, we won't actually check if that's the case -- since remember that there are infinite possibilies for market conditions.
# 
# 
# ### Reinforcement Learning
# There are many different reinforcement learning algorithms, such as DQN, A2C, and PPO, among others. All of these algorithms use a similar process to produce an agent:
# 
# Initially, the weights are set to random values.
# As the agent plays the game [Or trades the assets], the algorithm continually tries out new values for the weights, to see how the cumulative reward is affected, on average. Over time, after playing many games, we get a good idea of how the weights affect cumulative reward, and the algorithm settles towards weights that performed better.
# 
# Of course, we have glossed over the details here, and there's a lot of complexity involved in this process. For now, we focus on the big picture!
# This way, we'll end up with an agent that tries to win the game (so it gets the final reward of +1, and avoids the -1 and -10) and tries to make the game last as long as possible (so that it collects the 1/42 bonus as many times as it can).
# You might argue that it doesn't really make sense to want the game to last as long as possible -- this might result in a very inefficient agent that doesn't play obvious winning moves early in gameplay. And, your intuition would be correct -- this will make the agent take longer to play a winning move! The reason we include the 1/42 bonus is to help the algorithms we'll use to converge better. Further discussion is outside of the scope of this course, but you can learn more by reading about the "temporal credit assignment problem" and "reward shaping".
# In the next section, we'll use the Proximal Policy Optimization (PPO) algorithm to create an agent.
# 
# ### Code
# There are a lot of great implementations of reinforcement learning algorithms online. In this course, we'll use Stable Baselines.
# Currently, Stable Baselines is not yet compatible with TensorFlow 2.0. So, we begin by downgrading to TensorFlow 1.0.

# # <span class="title-section w3-xxlarge" id="install">Installations 💾</span>
# <hr>

# In[ ]:


get_ipython().system("pip install 'tensorflow==1.15.0'")
get_ipython().system('apt-get update')
get_ipython().system('apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev')
get_ipython().system('pip install "gym==0.19.0"')
get_ipython().system('pip install "stable-baselines[mpi]==2.9.0"')


# # <span class="title-section w3-xxlarge" id="outline">Libraries 📚</span>
# <hr>
# 
# #### Code starts here ⬇

# In[ ]:


import os
import json
import random
import datetime
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
import jpx_tokyo_market_prediction

import gym
from gym import spaces
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy


# In[ ]:


WINDOW_SIZE = 5


# # <span class="title-section w3-xxlarge" id="loading">Data Loading 🗃️</span>
# <hr>
# 
# The data organisation has already been done and saved to Kaggle datasets. Here we choose which years to load. We can use either 2017, 2018, 2019, 2020, 2021, Original, Supplement by changing the `INC2021`, `INC2020`, `INC2019`, `INC2018`, `INC2017`, `INCCOMP`, `INCSUPP` variables in the preceeding code section. These datasets are discussed [here][1].
# 
# [1]: https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/285726
# 

# In[ ]:


stock_list = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
stock_list = stock_list.loc[stock_list['SecuritiesCode'].isin(prices['SecuritiesCode'].unique())]
stock_name_dict = {stock_list['SecuritiesCode'].tolist()[idx]: stock_list['Name'].tolist()[idx] for idx in range(len(stock_list))}

def load_training_data(asset_id = None):
    prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
    supplemental_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
    df_train = pd.concat([prices, supplemental_prices]) if INCSUPP else prices
    df_train = pd.merge(df_train, stock_list[['SecuritiesCode', 'Name']], left_on = 'SecuritiesCode', right_on = 'SecuritiesCode', how = 'left')
    df_train['date'] = pd.to_datetime(df_train['Date'])
    df_train['year'] = df_train['date'].dt.year
    if not INC2022: df_train = df_train.loc[df_train['year'] != 2022]
    if not INC2021: df_train = df_train.loc[df_train['year'] != 2021]
    if not INC2020: df_train = df_train.loc[df_train['year'] != 2020]
    if not INC2019: df_train = df_train.loc[df_train['year'] != 2019]
    if not INC2018: df_train = df_train.loc[df_train['year'] != 2018]
    if not INC2017: df_train = df_train.loc[df_train['year'] != 2017]
    # asset_id = 1301 # Remove before flight
    if asset_id is not None: df_train = df_train.loc[df_train['SecuritiesCode'] == asset_id]
    # df_train = df_train[:1000] # Remove before flight
    return df_train


# In[ ]:


# WHICH YEARS TO INCLUDE? YES=1 NO=0
INC2022 = 1
INC2021 = 1
INC2020 = 1
INC2019 = 1
INC2018 = 1
INC2017 = 1
INCSUPP = 1

train = load_training_data()
print ("Loaded all data!")


# # <span class="title-section w3-xxlarge" id="features">Feature Engineering 🔬</span>
# <hr>
# 
# This notebook uses upper_shadow, lower_shadow, high_div_low, open_sub_close, seasonality/datetime features first shown in this notebook [here][1] and successfully used by julian3833 [here][2].
# 
# Additionally we can decide to use external data by changing the variables `INC2021`, `INC2020`, `INC2019`, `INC2018`, `INC2017`, `INCCOMP`, `INCSUPP` in the preceeding code section. These variables respectively indicate whether to load last year 2021 data and/or year 2020, 2019, 2018, 2017, the original, supplemented data. These datasets are discussed [here][3]
# 
# Consider experimenting with different feature engineering and/or external data. The code to extract features out of the dataset is taken from julian3833' notebook [here][2]. Thank you julian3833, this is great work.
# 
# [1]: https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition
# [2]: https://www.kaggle.com/julian3833
# [3]: TBD

# In[ ]:


# Two features from the competition tutorial
def upper_shadow(df): return df['High'] - np.maximum(df['Close'], df['Open'])
def lower_shadow(df): return np.minimum(df['Close'], df['Open']) - df['Low']

# A utility function to build features from the original df
def get_features(df):
    df_feat = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_feat['upper_Shadow'] = upper_shadow(df_feat)
    df_feat['lower_Shadow'] = lower_shadow(df_feat)
    df_feat["high_div_low"] = df_feat["High"] / df_feat["Low"]
    df_feat["open_sub_close"] = df_feat["Open"] - df_feat["Close"]
    return df_feat


# In[ ]:


def reduce_mem_usage(df,do_categoricals=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            if do_categoricals==True:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


train = train.sort_values('Date')
train.drop(columns = 'Date', inplace = True)
train.drop(columns = 'date', inplace = True)
target = train['Target'].copy()
train.drop(columns = 'Target', inplace = True)
train = reduce_mem_usage(train)


# In[ ]:


trian = get_features(train)
train = reduce_mem_usage(train)
train.head()


# In[ ]:


timestamp = datetime.datetime.now()
MODEL_ID = f"jpx_ppo_{timestamp.strftime('%s')}"


# # <span class="title-section w3-xxlarge" id="config">Defining the GYM Environment 🎚️</span>
# <hr >

# In[ ]:


class JPXEnv(gym.Env):

    def __init__(self, df, target, window_size = 5):
        super(JPXEnv, self).__init__()
        self.idx = 0
        self.df = df
        self.n_samples = df.shape[0]
        self.window_size = window_size
        self.target = np.nan_to_num(target.values)
        self.features = [col for col in list(self.df.select_dtypes('number').columns)]
        self.states = df[[col for col in list(self.df.select_dtypes('number').columns)]].values
        # Possible actions = target estimate
        self.action_space = spaces.Box( np.array([-1.0]), np.array([+1.0])) 
        # Prices contains the technical feature values for the last five prices
        self.observation_space = spaces.Box(low=-8.215050, high=5.872849e+01, shape=(df[self.features].shape[1], self.window_size))

    def _next_observation(self):         
        return np.nan_to_num(np.array([self.df[self.idx: self.idx + self.window_size][feature].values for feature in self.features]))

    def step(self, action):
        obs = self._next_observation()
        reward = -1.0 * (action - self.target[self.idx]) # Rewarding the agent inversely proportional to it's error
        self.idx += 1
        if self.idx >= self.n_samples - self.window_size:
            done = True
            self.idx = 0
        else: done = False
        return obs, reward, done, {}

    def reset(self):
        self.idx = 0
        return self._next_observation()

    def render(self):
        print(f'Step: {self.idx}')

        
class JPXPredictEnv(gym.Env):

    def __init__(self, df, target, window_size = 5):
        super(JPXPredictEnv, self).__init__()
        self.idx = 0
        self.df = df
        self.n_samples = df.shape[0]
        self.window_size = window_size
        self.target = target.values
        self.features = [col for col in list(self.df.select_dtypes('number').columns)]
        self.states = df[[col for col in list(self.df.select_dtypes('number').columns)]].values
        # Possible actions = target estimate
        self.action_space = spaces.Box( np.array([-1.0]), np.array([+1.0])) 
        # Prices contains the technical feature values for the last five prices
        self.observation_space = spaces.Box(low=-8.215050, high=5.872849e+01, shape=(df[self.features].shape[1], self.window_size))

    def _next_observation(self):         
        return np.array([self.df[self.idx: self.idx + self.window_size][feature].values for feature in self.features])

    def step(self, action):
        obs = self._next_observation()
        reward = -1.0 * (action - self.target[self.idx]) # Rewarding the agent inversely proportional to it's error
        self.idx += 1
        if self.idx >= self.n_samples - self.window_size:
            done = True
            self.idx = 0
        else: done = False
        return obs, reward, done, {}

    def reset(self):
        self.idx = 0
        return self._next_observation()

    def render(self):
        print(f'Step: {self.idx}')
        
features, window_size = [col for col in list(train.columns)], WINDOW_SIZE
env = DummyVecEnv([lambda: JPXEnv(train, target, window_size = WINDOW_SIZE)])


# # <span class="title-section w3-xxlarge" id="training">Reinforcement Learning 🏋️</span>
# <hr>

# ## Network architecture <a name="architecture"></a>
# 
# We are going to train a neural network, which is structured in the following way:
# 
# 
#                                 -------> Actor ---> Logits (𝜋)
#                               /
#     State (𝑠) ---> Encoder --- --------> Critic-1 ---> Value-1 (𝑉-1)
#                               \
#                                 -------> Critic-2 ---> Value-2 (𝑉-2)
# 
# 
# Each critic head predicts state-value $V_\theta(s)$ (estimate of discounted return from this point onwards), where $\theta$ stands for neural net parameters. Actor updates policy parameters for $\pi_\theta$, in the direction suggested by Critics.
# 
# I won't go into details here, for more reading visit this amazing notebook: https://www.kaggle.com/alexandersamarin/training-resnet-agent-from-scratch

# ## PPO algorithm explained <a name="ppo"></a>
# 
# Firstly, we will define policy gradient loss:
# 
# $$\mathcal{L}^{\text{PG}}(\theta) = \mathbb{E}[\log \pi_\theta(a|s) \hat{A}_\theta(s,a) ],$$
# 
# where first term $\log \pi_\theta(a|s)$ are log-probabilities from the output of policy network (actor head), and the second one is an estimate of `advantage function`, the relative value of selected action $a$. The value of $\hat{A}_\theta(s,a)$ is equal to `return` (or `discounted reward`) minus `baseline estimate`. Return at given time $t$ is calculated as follows:
# 
# $$ V_{\text{target}}(t) = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^\infty \gamma^k R_{t+k+1},$$
# 
# where $R^i_t$ is a reward at timestep $t$. Baseline estimate is the output of value network $V_\theta(s)$. Therefore,
# 
# $$ \hat{A}_\theta(t) = V_{\text{target}}(t) - V_\theta(s_t). $$
# 
# There also exists a generalized version of advantage estimation, that we are going to use:
# 
# $$\hat{A}_\theta(t) = \delta_t + (\gamma \lambda) \delta_{t+1} + \dots = \sum_{k=0}^\infty (\gamma \lambda)^k \delta_{t+k+1},$$
# $$ \text{where } \delta_t = R_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t),$$
# 
# which is reduced to previous equation when $\lambda = 1$.
# 
# Now, when $\hat{A}_\theta$ is positive, meaning that the action agent took resulted in a better than average return, we will increase probabilities of selecting it again in the future. On the other hand, if an advantage was negative, we will reduce the likelihood of selected actions.

# However, as PPO-paper quotes:
# 
# `While it is appealing to perform multiple steps of optimization on this loss using the same trajectory, doing so is not well-justified, and empirically it often leads to destructively large policy updates.`
# 
# In other words, we have to impose the constraint which won't allow our new policy to move too far away from an old one. Let’s denote the probability ratio between old and new policies as
# 
# $$r(\theta) = \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}. $$
# 
# Then, take a look at our new `surrogate` objective function:
# 
# $$\mathcal{L}^{\text{CPI}}(\theta) = \mathbb{E}[r(\theta) \hat{A}_\theta(s,a)].$$
# 
# It can be derived that maximimizing $\mathcal{L}^{\text{CPI}}(\theta)$ is identical to vanilla policy gradient method, but I'll bravely skip the proof. Now, we would like to insert the aforementioned constraint into this loss function. The main objective which PPO-parer proposes is the following.
# 
# $$J^{\text{CLIP}}(\theta) = \mathbb{E}[\min (r(\theta) \hat{A}_{\theta_{\text{old}}}(s,a), \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{\theta_{\text{old}}}(s,a)],$$
# 
# where $\epsilon$ is a `clip ratio` hyperparameter. The first term inside $min$ function, $r(\theta) \hat{A}_{\theta_{\text{old}}}(s,a)$ is a normal policy gradient objective. And the second one is its clipped version, which doesn't allow us to destroy our current policy based on a single estimate, because the value of $\hat{A}_{\theta_{\text{old}}}(s,a)$ is noisy (as it is based on an output of our network).
# 
# When applying PPO on the network architecture with shared parameters for both policy and value functions, in addition to the clipped reward, the objective function is augmented with an error term on the value estimation and an entropy term to encourage sufficient exploration. Final loss then becomes:
# 
# $$\mathcal{L}(\theta) = \mathbb{E}[-J(\theta) + c  (V_\theta(s) - V_{\text{target}})^2 - c_{\text{ent}}  H(s, \pi_{\theta}(\cdot))], $$
# 
# where $c$ and $c_{\text{ent}}$ are both hyperparameter constants.

# ## OpenAI's stable baselines
# To use this on the competition, we should implement it from scratch since we are not allowed to use internet on the final submissions. 
# Currently, to get things up and running: We are simply going to use OpenAI's stable-baselines. 
# As this is a work in progress: The next versions of this notebok will contain an ecapsulated implementation of the PPO agent. 

# ## Training and evaluation

# In[ ]:


def learn(timesteps=1000):
    model = PPO2(MlpLnLstmPolicy, env, verbose=1, nminibatches=1)
    model.learn(total_timesteps=timesteps)
    model.save(MODEL_ID)
    print(f"Saved model: {MODEL_ID}")
learn(timesteps = 1000)


# # <span class="title-section w3-xxlarge" id="submit">Submit To Kaggle 🇰</span>
# <hr>

# In[ ]:


all_df_test = []
predict_df = pd.DataFrame()
jpx = jpx_tokyo_market_prediction.make_env() # initialize the environment
iter_test = jpx.iter_test()       # an iterator which loops over the test set
model = PPO2.load(MODEL_ID)
for i, (df_test, options, financials, trades, secondary_prices, df_pred) in enumerate(iter_test):
    df_pred['Target'] = 0.0
    for j, row in df_test.iterrows():
        try:                        
            row_feats = get_features(row)
            predict_df.append(row_feats)
            if len(predict_df) > window_size:
                obs = np.array([predict_df[j: j + window_size][feature].values for feature in features])
                action, _states = model.predict(np.expand_dims(obs, axis = 0)) # _states are only useful when using LSTM policies
            else: action = 0.0
            y_pred = action             
        except: 
            y_pred = 0.0
            traceback.print_exc()        
        df_pred.iloc[j, df_pred.columns.get_loc('Target')] = y_pred    
    df_pred = df_pred.sort_values(by = "Target", ascending = False)
    df_pred['Rank'] = np.arange(0, 2000)
    df_pred = df_pred.sort_values(by = "SecuritiesCode", ascending = True)
    df_pred.drop(["Target"], axis = 1)
    submission = df_pred[["Date", "SecuritiesCode", "Rank"]]
    all_df_test.append(df_test)
    jpx.predict(submission)

