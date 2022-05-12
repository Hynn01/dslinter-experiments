#!/usr/bin/env python
# coding: utf-8

# >### Let's Talk Time Series Validation
# >Time Series Forecasting can be overwhelming. Especially if you are just getting startet. There are many different types of Time Series tasks each differ by the number of input or output sequences, the number of steps to predict, whether the input and/or the output sequence length is static or changing, and so on. In this notebook, we will experiment with **different types of Time Series Cross Validation Strategies** in order to become familiar with them and understand which works best for what case.
# >

# ____
# 
# # Time Series Forecasting
# 
# Time Series Forecasting can be overwhelming. Especially if you are just getting startet. There are many different types of Time Series tasks each differ by the number of input or output sequences, the number of steps to predict, whether the input and/or the output sequence length is static or changing, and so on. In this notebook, we will experiment with **different types of Time Series Cross Validation Strategies** in order to become familiar with them and understand which works best for what case. 
# 
# As written before, Time Series problems can be of different variations, so in order to get a deeper understanding we should explore each. 
# 
# <br>
# <font color='#EC7063'>
#     
# * **Variation I:** Number of Input / Output sequences
#     * (1.0) Single input and single output with input = output (Univariate with `in` = `out`)
#     * (1.1) Single input and single output with input $\neq$ output (Multivariate with `in` $\neq$  `out`)
#     * (1.2) Multiple inputs and multiple outputs with inputs = outputs (Multivariate with `in[N]` = `out[N]`)
#     * (1.3) Multiple inputs and multiple outputs with inputs $\neq$ outputs (Multivariate with `in[N]` $\neq$  `out[N]`)  
# 
# </font>
# 
# 
# <br>
# <font color='#5499C7'>
# 
# * **Variation II:** Length of Output sequences
#     * (2.0) Single step output sequence     
#     * (2.1) Multistep: Predict all steps at once (Single-shot) 
#     * (2.2) Multistep: Predict single step at a time and feedback to model to predict for multiple steps (Autoregressive) 
# 
# </font>
# 
# 
# <br>
# <font color='#45B39D'>
# 
# * **Variation III** Type of input sequence
#     * (3.0) Static length input sequence (Sliding Window) 
#     * (3.1) Variable length input sequence (Expanding Window) 
#   
# </font>
# <br>
# 
# Every different Time Series task have different combination of the above properties. For example, we could have:
# 
#     
# <br>
# <center>One sequence for which we are trying to <font color='#EC7063'>predict its values</font> <font color='#5499C7'>for the next 5 timesteps</font> <font color='#45B39D'>based on the previous 10 timesteps</font></center>
# <br>
# 
# This would make this task: 
# 
# <br>
# <center><b><font color='#EC7063'>Univariate</font> <font color='#5499C7'>multistep</font> time series forecasting with a <font color='#45B39D'>sliding window</font> (<font color='#45B39D'>3.0</font>, <font color='#5499C7'>2.1</font>, <font color='#EC7063'>1.0</font>)</b></center>
# <br>
# 
# Or we might need to 
# 
# <br>
# <center><b>Predict the amount of <font color='#5499C7'>snow for the next day</font> based on <font color='#45B39D'>all available past data</font> of <font color='#EC7063'>temperature and rain</font></b></center>
# <br>
# 
# [Take a moment to try and guess the problem types yourself..]
# 
# <br>
# 
# <center>
# This is a <b><font color='#EC7063'>Multivariate</font> <font color='#45B39D'>single step</font> time series forecasting with an <font color='#45B39D'>expanding window</font> (<font color='#5499C7'>2.0</font>, <font color='#45B39D'>3.1</font>, <font color='#EC7063'>1.3</font>).</b>
# </center>
# 
# <br>
# 
# 
# <hr> 
# 
# **Credits:** Some sections of this notebook (Including this intro) are heavily insipred by the great notebook made by Leonie: [Time Series Forecasting: Building Intuition](https://www.kaggle.com/iamleonie/time-series-forecasting-building-intuition). If you find this notebook useful, Please go upvote the original! 
# 
# <hr> 

# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import seed 
from datetime import datetime, date 
from IPython.display import display_html
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
seed(SEED)
np.random.seed(SEED)


# # Time-Series Toy Problems
# In order for us to get a better understanding, we will use a fictional time series so we can see the different types of time series problems in practice. There will be three features. `feature_1` is following a sine wave, `feature_2` is a linear function, and `feature_3` is a modulo function. The `feature_4` column is the result of a combination of the feature columns. The time series consists of 100 timesteps.

# In[ ]:


def split_sequences(features, targets, n_steps_in, n_steps_out, n_sliding_steps, window_type):
    X, y = list(), list()
    for i in range(0, len(features), n_sliding_steps):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(features): break
        if window_type == 'sliding': seq_x, seq_y = features[i:end_ix, :], targets[end_ix:out_end_ix, :]
        else: seq_x, seq_y = features[0:end_ix, :], targets[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
def plot_time_series_problem(X, y):
    fig, ax = plt.subplots(nrows=X.shape[0], ncols=1, figsize=(15, 2.5*X.shape[0]))
    for i in range(X.shape[0]):
        sns.lineplot(x=ts_df['timestamp'].values, y=ts_df['feature_1'].values, ax=ax[i], color='lightgrey', marker='o')
        if i < X.shape[0]-1:
            sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 1].astype(float), ax=ax[i], color='cornflowerblue', label='train', marker='o')
            sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 1].astype(float), ax=ax[i], color='orange', label='val', marker='o')
            ax[i].set_title(f"Training Sample {i}")
        else:
            sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 1].astype(float), ax=ax[i], color='mediumseagreen', label='in', marker='o')
            sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 1].astype(float), ax=ax[i], color='coral', label='pred', marker='o')
            ax[i].set_title(f"Testing")    
        ax[i].set_xlim([date(2021, 1, 1), date(2021, 4, 11)])
    plt.tight_layout()
    plt.show()  

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    plt.figure(1)
    for l in loss_list: plt.plot(epochs, history.history[l], 'cornflowerblue', label='Training loss')
    for l in val_loss_list: plt.plot(epochs, history.history[l], 'orange', label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
time = np.arange(0, 1100, 10)
ts_df = pd.DataFrame({'feature_1' : 15*np.sin(0.021*time+30)+13, 'feature_2' : (0.05*time)+20, 'feature_3' : (( time + 100 ) % 280)*0.1 + 50, })
ts_df['feature_4'] =  0.1 * ts_df.feature_1 * (ts_df.feature_2 + 5) - ts_df.feature_3.shift(10) + 100
ts_df = ts_df[10:110]
ts_df['timestamp'] = pd.date_range('2021-01-01', periods=100, freq='D')
ts_df.set_index('timestamp', inplace=True)
ts_df.reset_index(drop=False, inplace=True)
display(ts_df.head())


# **Visualizing the time series**

# In[ ]:


fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 10))
sns.lineplot(x=ts_df.timestamp, y=ts_df.feature_1, ax=ax[0], color='cornflowerblue', marker='o')
sns.lineplot(x=ts_df.timestamp, y=ts_df.feature_2, ax=ax[1], color='cornflowerblue', marker='o')
sns.lineplot(x=ts_df.timestamp, y=ts_df.feature_3, ax=ax[2], color='cornflowerblue', marker='o')
sns.lineplot(x=ts_df.timestamp, y=ts_df.feature_4, ax=ax[3], color='cornflowerblue', marker='o')
for i in range(4): ax[i].set_xlim([date(2021, 1, 1), date(2021, 4, 10)])
plt.tight_layout()
plt.show()


# The following function will help us build our lunch menu. It will take your input and/or output sequences and return you the training and testing data according to your order. You can specify the length of the input sequence, the length of the output sequence, and the step size of your window.

# ## Variation I: <font color='#EC7063'>Number of Input/Output Sequences</font>
# Let's look at the <font color='#EC7063'>number of input/output sequences</font>. To make things easier to understand, we will only <font color='#EC7063'>variate the number of input/output sequences </font>, an we will <font color='#45B39D'>keep the length of the output sequences</font> and the <font color='#45B39D'>type of input sequences</font> same for the following. 
# 
# We will use a <font color='#5499C7'>multistep output sequence of 5 steps</font> and the <font color='#45B39D'>sliding window setting with a stepsize of 20 for this.</font> 

# In[ ]:


### Help functions ###
def display_input_and_output_df(input_cols, output_cols):
    ts_df_styler1 = ts_df[input_cols].head().style.set_table_attributes("style='display:inline'").set_caption('Input (X)')
    ts_df_styler2 = ts_df[output_cols].head().style.set_table_attributes("style='display:inline'").set_caption('Output (y)')
    display_html(ts_df_styler1._repr_html_()+ts_df_styler2._repr_html_(), raw=True)


# ### Univariate: <font color='#EC7063'>Single input</font> and <font color='#EC7063'>single output</font> when input series == output series

# In[ ]:


input_cols = ['timestamp', 'feature_1']
output_cols = input_cols


# In[ ]:


display_input_and_output_df(input_cols, output_cols)

X, y = split_sequences(ts_df[input_cols].values, 
                       ts_df[output_cols].values, 
                       n_steps_in = 15, 
                       n_steps_out = 5, 
                       n_sliding_steps = 20, 
                       window_type='sliding')

fig, ax = plt.subplots(nrows=X.shape[0], ncols=1, figsize=(15, 2.5*X.shape[0]))
fig.suptitle('Univariate: Single input and single output with input = output')
for i in range(X.shape[0]):
    sns.lineplot(x=ts_df['timestamp'].values, y=ts_df['feature_1'].values, ax=ax[i], color='lightgrey', marker='o')

    if i < X.shape[0]-1:
        sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 1].astype(float), ax=ax[i], color='cornflowerblue', label='input', marker='o')
        sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 1].astype(float), ax=ax[i], color='orange', label='output', marker='o')
        ax[i].set_title(f"Training Sample {i}")
    else:
        sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 1].astype(float), ax=ax[i], color='mediumseagreen', label='input', marker='o')
        sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 1].astype(float), ax=ax[i], color='coral', label='predict', marker='o')
        ax[i].set_title(f"Testing")    
    ax[i].set_xlim([date(2021, 1, 1), date(2021, 4, 11)])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show() 


# ### Multivariate: <font color='#EC7063'>Multiple inputs</font> and <font color='#EC7063'>multiple outputs</font> with inputs series = outputs series

# In[ ]:


input_cols = ['timestamp', 'feature_1', 'feature_2']
output_cols = input_cols


# In[ ]:


display_input_and_output_df(input_cols, output_cols)

X, y = split_sequences(ts_df[input_cols].values, 
                       ts_df[output_cols].values, 
                       n_steps_in = 15, 
                       n_steps_out = 5, 
                       n_sliding_steps = 20, 
                       window_type='sliding')

fig, ax = plt.subplots(nrows=X.shape[0], ncols=2, figsize=(15, 2.5*X.shape[0]))
fig.suptitle(r'Multivariate: Multiple inputs and multiple outputs with input = output')

for i in range(X.shape[0]):
    sns.lineplot(x=ts_df['timestamp'].values, y=ts_df['feature_1'].values, ax=ax[i, 0], color='lightgrey', marker='o')
    sns.lineplot(x=ts_df['timestamp'].values, y=ts_df['feature_2'].values, ax=ax[i, 1], color='lightgrey', marker='o')

    if i < X.shape[0]-1:
        sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 1].astype(float), ax=ax[i, 0], color='cornflowerblue', marker='o')
        sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 1].astype(float), ax=ax[i, 0], color='orange', marker='o')
        
        sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 2].astype(float), ax=ax[i, 1], color='cornflowerblue', marker='o')
        sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 2].astype(float), ax=ax[i, 1], color='orange', marker='o')
        
        ax[i, 0].set_title(f"Training Sample {i}")
        ax[i, 1].set_title(f"Training Sample {i}")

    else:
        sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 1].astype(float), ax=ax[i, 0], color='mediumseagreen', marker='o')
        sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 1].astype(float), ax=ax[i, 0], color='coral', marker='o')
        
        sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 2].astype(float), ax=ax[i, 1], color='mediumseagreen', marker='o')
        sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 2].astype(float), ax=ax[i, 1], color='coral', marker='o')
        ax[i, 0].set_title(f"Testing Sample")    
        ax[i, 1].set_title(f"Testing Sample (Prediction)")    

    ax[i, 0].set_xlim([date(2021, 1, 1), date(2021, 4, 11)])
    ax[i, 1].set_xlim([date(2021, 1, 1), date(2021, 4, 11)])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show() 


# ### Multivariate: <font color='#EC7063'>Single input</font> and <font color='#EC7063'>single output</font> when input series $\neq$ output series

# In[ ]:


input_cols = ['timestamp', 'feature_1']
output_cols = ['timestamp', 'feature_4']


# In[ ]:


display_input_and_output_df(input_cols, output_cols)

X, y = split_sequences(ts_df[input_cols].values, 
                       ts_df[output_cols].values, 
                       n_steps_in = 15, 
                       n_steps_out = 5, 
                       n_sliding_steps = 20, 
                       window_type='sliding')

fig, ax = plt.subplots(nrows=X.shape[0], ncols=2, figsize=(15, 2.5*X.shape[0]))
fig.suptitle(r'Multivariate: Single input and single output with input $\neq$ output')

for i in range(X.shape[0]):
    sns.lineplot(x=ts_df['timestamp'].values, y=ts_df['feature_1'].values, ax=ax[i, 0], color='lightgrey', marker='o')
    sns.lineplot(x=ts_df['timestamp'].values, y=ts_df['feature_4'].values, ax=ax[i, 1], color='lightgrey', marker='o')

    if i < X.shape[0]-1:
        sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 1].astype(float), ax=ax[i, 0], color='cornflowerblue', marker='o')
        sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 1].astype(float), ax=ax[i, 1], color='orange', marker='o')
        ax[i, 0].set_title(f"Input Sequence of Training Sample {i}")
        ax[i, 1].set_title(f"Output Sequence of Training Sample {i}")

    else:
        sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 1].astype(float), ax=ax[i, 0], color='mediumseagreen', marker='o')
        sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 1].astype(float), ax=ax[i, 1], color='coral', marker='o')
        ax[i, 0].set_title(f"Input Sequence of Testing Sample")    
        ax[i, 1].set_title(f"Output Sequence of Testing Sample (Prediction)")    

    ax[i, 0].set_xlim([date(2021, 1, 1), date(2021, 4, 11)])
    ax[i, 1].set_xlim([date(2021, 1, 1), date(2021, 4, 11)])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show() 


# ### Multivariate: <font color='#EC7063'>Multiple inputs</font> and <font color='#EC7063'>multiple outputs</font> when inputs series ≠ outputs series

# In[ ]:


input_cols = ['timestamp', 'feature_1', 'feature_2']
output_cols = ['timestamp', 'feature_3', 'feature_4']


# In[ ]:


display_input_and_output_df(input_cols, output_cols)

X, y = split_sequences(ts_df[input_cols].values, 
                       ts_df[output_cols].values, 
                       n_steps_in = 15, 
                       n_steps_out = 5, 
                       n_sliding_steps = 20, 
                       window_type='sliding')

fig, ax = plt.subplots(nrows=X.shape[0]*2, ncols=2, figsize=(15, 2.5*X.shape[0]*2))
fig.suptitle(r'Multivariate: Single input and single output with input $\neq$ output')

for i in range(X.shape[0]):
    sns.lineplot(x=ts_df['timestamp'].values, y=ts_df['feature_1'].values, ax=ax[(i*2), 0], color='lightgrey', marker='o')
    sns.lineplot(x=ts_df['timestamp'].values, y=ts_df['feature_2'].values, ax=ax[(i*2)+1, 0], color='lightgrey', marker='o')

    sns.lineplot(x=ts_df['timestamp'].values, y=ts_df['feature_3'].values, ax=ax[(i*2), 1], color='lightgrey', marker='o')
    sns.lineplot(x=ts_df['timestamp'].values, y=ts_df['feature_4'].values, ax=ax[(i*2)+1, 1], color='lightgrey', marker='o')

    if i < X.shape[0]-1:
        sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 1].astype(float), ax=ax[(i*2), 0], color='cornflowerblue', marker='o')
        sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 2].astype(float), ax=ax[(i*2)+1, 0], color='cornflowerblue', marker='o')
        sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 1].astype(float), ax=ax[(i*2), 1], color='orange', marker='o')
        sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 2].astype(float), ax=ax[(i*2)+1, 1], color='orange', marker='o')
        ax[(i*2), 0].set_title(f"Input Sequence 1 of Training Sample {i}")
        ax[(i*2), 1].set_title(f"Output Sequence 1 of Training Sample {i}")

        ax[(i*2)+1, 0].set_title(f"Input Sequence 2 of Training Sample {i}")
        ax[(i*2)+1, 1].set_title(f"Output Sequence 2 of Training Sample {i}")

    else:
        #sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 1].astype(float), ax=ax[(i*2), 0], color='mediumseagreen', marker='o')
        #sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 1].astype(float), ax=ax[(i*2)+1, 1], color='coral', marker='o')
        
        sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 1].astype(float), ax=ax[(i*2), 0], color='mediumseagreen', marker='o')
        sns.lineplot(x=X[i][ :, 0], y=X[i][ :, 2].astype(float), ax=ax[(i*2)+1, 0], color='mediumseagreen', marker='o')
        sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 1].astype(float), ax=ax[(i*2), 1], color='coral', marker='o')
        sns.lineplot(x=y[i][ :, 0], y=y[i][ :, 2].astype(float), ax=ax[(i*2)+1, 1], color='coral', marker='o')
        
        ax[(i*2), 0].set_title(f"Input Sequence 1 of Testing Sample {i}")
        ax[(i*2), 1].set_title(f"Output Sequence 1 of Testing Sample {i} (Prediction)")

        ax[(i*2)+1, 0].set_title(f"Input Sequence 2 of Testing Sample {i}")
        ax[(i*2)+1, 1].set_title(f"Output Sequence 2 of Testing Sample {i} (Prediction)")

    ax[i, 0].set_xlim([date(2021, 1, 1), date(2021, 4, 11)])
    ax[i, 1].set_xlim([date(2021, 1, 1), date(2021, 4, 11)])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show() 


# ## Variation II: <font color='#5499C7'>Length of Output Sequences</font>
# Let's look at the length of output sequences (Variation II). We will will keep the <font color='#EC7063'>number of input/output sequences</font> and the <font color='#45B39D'>type of input sequences</font> same for the following. 
# 
# We will use a <font color='#EC7063'>univariate</font> problem with the <font color='#45B39D'>sliding window</font>  setting with a stepsize of 20 for this. 
# 
# ### <font color='#5499C7'>Single Step</font> Output Sequence 

# In[ ]:


input_cols = ['timestamp', 'feature_1']
output_cols = input_cols

X, y = split_sequences(ts_df[input_cols].values, 
                       ts_df[output_cols].values, 
                       n_steps_in = 19, 
                       n_steps_out = 1, 
                       n_sliding_steps = 20, 
                       window_type='sliding')

plot_time_series_problem(X, y)


# ### <font color='#5499C7'>Multistep</font> Output Sequence 

# In[ ]:


X, y = split_sequences(ts_df[input_cols].values, 
                       ts_df[output_cols].values, 
                       n_steps_in = 15, 
                       n_steps_out = 5, 
                       n_sliding_steps = 20, 
                       window_type='sliding')

plot_time_series_problem(X, y)


# ## Variation III: <font color='#45B39D'>Type of Input Sequences</font>
# 
# Now we will explore at the type of input sequences (<font color='#45B39D'>Variation III</font>). We will will keep the <font color='#EC7063'>number of input/output sequences</font> and <font color='#5499C7'>length of the output sequences</font> same for the following. 
# 
# We will use a <font color='#EC7063'>univariate</font> problem with a <font color='#5499C7'>multistep</font> output sequence of 5 steps for this. 
# 
# 
# ### Input Type: <font color='#45B39D'>Sliding Window</font>

# In[ ]:


X, y = split_sequences(ts_df[input_cols].values, 
                       ts_df[output_cols].values, 
                       n_steps_in = 15, 
                       n_steps_out = 5, 
                       n_sliding_steps = 20, 
                       window_type='sliding')

plot_time_series_problem(X, y)


# ### Input Type: <font color='#45B39D'>Expanding Window</font>

# In[ ]:


X, y = split_sequences(ts_df[input_cols].values, 
                       ts_df[output_cols].values, 
                       n_steps_in = 15, 
                       n_steps_out = 5, 
                       n_sliding_steps = 20, 
                       window_type='expanding')

plot_time_series_problem(X, y)


# _____
# 
# # Time Series Grouped Cross Validation
# ### How to validate a model on chronologically ordered data which also contains groups?
# 
# It is of highly importance to be able to 'locally' estimate an indication of our model's performance. 
# An estimate that is independent of the (time expensive and limited) submission API. This allows for much better tuning of hyper-parameters or other aspects of the model's training process.
# 
# On the next chaper we will a couple of techniques that can be used to do this. For every step we will see that there is a problem with using it for this particular competition. Fortunately the last cells provides a solution! 
# 
# 
# >**TL;DR: If you are not interested in an introduction in test and validation techniques, then skip to the bottom.** 
# 
# First up: train and test subsets.

# -----
# ### Credit:
# - Based on the great notebook: https://www.kaggle.com/jorijnsmit/found-the-holy-grail-grouptimeseriessplit
# -----

# In[ ]:





# ## Train and Test Subsets
# 
# The first obvious step is to set apart some data which the model never gets to see. After the model has been trained, we use that unseen data to verify our model's predicitions. Scikit-learn's [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) makes the process of splitting datasets easy for us. Let's load our training data and set aside a test set:

# In[ ]:


import os
import numpy as np
import pandas as pd
import datatable as dt


# In[ ]:


dtrain = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
dtrain.head()


# In[ ]:


dlabels = dtrain[['Target']]
dtrain = dtrain.drop(columns = 'Target')
dtrain['date'] = pd.to_datetime(dtrain['Date']).dt.date

print(dtrain.columns)
print(dlabels.columns)


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

dtrain = reduce_mem_usage(dtrain)


# In[ ]:


from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(
    dtrain,
    dlabels,
    test_size=.25,
    random_state=1,
    shuffle=False
)

print(x_train.shape, x_test.shape)
print(x_train.index)


# Use of `shuffle=False` is key here; since otherwise we would lose all chronological order.
# 
# However, this approach is problematic because by constantly verifying on the same data, we also slowly start to overfit on the test set ("leakage"). Splitting the test set again into a validation set could solve this: the model's hyper-parameters are tuned and verified on the validation set and once that is completely finished we test it (only once!) on the test set. The problem now becomes that either the test and validation sets become too small to be useful or that so much data is used to validate and test that nog enough data remains to train on.

# ## Cross-validation
# 
# In cross-validation (CV), multiple validation sets are derived from the training set. Every *fold* a new part of the training set is used as the vaildation set, and the data previously used for validation now becomes part of the training set again:
# 
# ![img](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)
# 
# (Source: [scikit-learn's User Guide, Ch. 3.1](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-evaluating-estimator-performance).)

# In[ ]:


from sklearn.model_selection import KFold

for train_idx, test_idx in KFold().split(x_train):
    #print(train, test)
    print(x_train.loc[train_idx, 'date'].index)
    print(x_train.loc[test_idx, 'date'].index)
    break


# ### TimeSeriesSplit
# 
# Note however, that in the first split shown above we are validating our *chronological* data on the past. We are training on trades starting from `358574` but testing on trades starting from `0`. In other words, our model has been trained using information which wasn't yet available at the time of the validation set. This is clear leakage; we are predicting the past with knowledge from the future. But our aim is to predict data in the future! This problem has already been addressed by scikit-learn in the form of [TimeSeriesSplit](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split):
# ![](https://scikit-learn.org/stable/_images/sphx_glr_plot_cv_indices_0101.png)
# But what is the problem this time? `TimeSeriesSplit` does not respect the groups available in the data. Although not clearly visible in this plot, we can imagine that a group can partially fall in the training set and partially in the test set.

# In[ ]:


from sklearn.model_selection import TimeSeriesSplit

for train_idx, test_idx in TimeSeriesSplit().split(x_train):
    print(x_train.loc[train_idx, 'date'].unique())
    print(x_train.loc[test_idx, 'date'].unique())
    break


# Already in the first split we can see data from day `44` present in both the training and test set. That would mean that we are training on half of the trades of a certain day, just to validate their performance on the other half of the trades of that day. What we of course want is to train on all trades of a particular day, and to validate them on the day that follows! Otherwise again leaking will occur.

# ### GroupKFold
# 
# ![](https://scikit-learn.org/stable/_images/sphx_glr_plot_cv_indices_0051.png)

# In[ ]:


from sklearn.model_selection import GroupKFold

for train_idx, test_idx in GroupKFold().split(x_train, groups=x_train['date']):
    print(x_train.loc[train_idx, 'date'].unique())
    print(x_train.loc[test_idx, 'date'].unique())
    break


# The [GroupKFold](https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold) iterator does respect groupings: no group will ever be part of two folds. Unfortunately, it is also clear that it mixes up the order completely and thus loses the temporal dimension again. What we need is a a crossover between `GroupKFold` and `TimeSeriesSplit`: `GroupTimesSeriesSplit`.

# ### GroupTimesSeriesSplit
# 
# OK, so this iterator does not exist yet in scikit-learn. However, a request for it has been documented on GitHub over a year ago ([Feature request: Group aware Time-based cross validation #14257](https://github.com/scikit-learn/scikit-learn/issues/14257)) and is almost ready for release. Thanks to open source we can take a sneak peek already! 
# 
# **Do note that this is not fully reviewed yet!!!** This might be the final code that it will make it into `sklearn`'s version `0.24` as a major feature, but there's also a chance of bugs still being present.
# 
# I did not write *any* of this but it did take me a good day of research and trying to write it myself. All credits go to [@getgaurav2](https://github.com/getgaurav2/).
# 
# Here are some more attempts at grouped cross-validation I encountered in my research:
# - https://stackoverflow.com/questions/51963713/cross-validation-for-grouped-time-series-panel-data
# - https://datascience.stackexchange.com/questions/77684/time-series-grouped-cross-validation
# - https://nander.cc/writing-custom-cross-validation-methods-grid-search

# In[ ]:


from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum size for a single training set.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupTimeSeriesSplit
    >>> groups = np.array(['a', 'a', 'a', 'a', 'a', 'a',\
                           'b', 'b', 'b', 'b', 'b',\
                           'c', 'c', 'c', 'c',\
                           'd', 'd', 'd'])
    >>> gtss = GroupTimeSeriesSplit(n_splits=3)
    >>> for train_idx, test_idx in gtss.split(groups, groups=groups):
    ...     print("TRAIN:", train_idx, "TEST:", test_idx)
    ...     print("TRAIN GROUP:", groups[train_idx],\
                  "TEST GROUP:", groups[test_idx])
    TRAIN: [0, 1, 2, 3, 4, 5] TEST: [6, 7, 8, 9, 10]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a']\
    TEST GROUP: ['b' 'b' 'b' 'b' 'b']
    TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] TEST: [11, 12, 13, 14]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b']\
    TEST GROUP: ['c' 'c' 'c' 'c']
    TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]\
    TEST: [15, 16, 17]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b' 'c' 'c' 'c' 'c']\
    TEST GROUP: ['d' 'd' 'd']
    """
    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_size=None
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))
        group_test_size = n_groups // n_folds
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []
            for train_group_idx in unique_groups[:group_test_start]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)
            train_end = train_array.size
            if self.max_train_size and self.max_train_size < train_end:
                train_array = train_array[train_end -
                                          self.max_train_size:train_end]
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)
            yield [int(i) for i in train_array], [int(i) for i in test_array]


# In[ ]:


"""
for idx, (train_idx, test_idx) in enumerate(GroupTimeSeriesSplit().split(x_train, groups=x_train['date'])):
    print('-' * 80)
    print('Fold: ', idx)
    print(x_train.loc[train_idx, 'date'].unique())
    print(x_train.loc[test_idx, 'date'].unique())
    print('-' * 80)
"""


# 
# # The Best Time Series Cross Validation
# 
# > "There are many different ways one can do cross-validation, and **it is the most critical step when building a good machine learning model** which is generalizable when it comes to unseen data."
# -- **Approaching (Almost) Any Machine Learning Problem**, by Abhishek Thakur
# 
# CV is the **first** step, but very few notebooks are talking about this. Here we look at "purged rolling time series CV" and actually apply it in hyperparameter tuning for a basic estimator. This notebook owes a debt of gratitude to the notebook ["Found the Holy Grail GroupTimeSeriesSplit"](https://www.kaggle.com/jorijnsmit/found-the-holy-grail-grouptimeseriessplit). That notebook is excellent and this solution is an extention of the quoted pending sklearn estimator. I modify that estimator to make it more suitable for the task at hand in this competition. The changes are
# 
# - you can specify a **gap** between each train and validation split. This is important because even though the **group** aspect keeps whole days together, we suspect that the anonymized features have some kind of lag or window calculations in them (which would be standard for financial features). By introducing a gap, we mitigate the risk that we leak information from train into validation
# - we can specify the size of the train and validation splits in terms of **number of days**. The ability to specify a validation set size is new and the the ability to specify days, as opposed to samples, is new.
# 
# The code for `PurgedTimeSeriesSplit` is below. I've hiden it becaused it is really meant to act as an imported class. If you want to see the code and copy for your work, click on the "Code" box.

# In[ ]:


import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples

# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size
 
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]
            
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]


# To show the general idea, we generate some simple grouped data. Imagine we have a dataset of 2,000 samples which below to 20 groups.

# In[ ]:


n_samples = 2000
n_groups = 20
assert n_samples % n_groups == 0

idx = np.linspace(0, n_samples-1, num=n_samples)
X_train = np.random.random(size=(n_samples, 5))
y_train = np.random.choice([0, 1], n_samples)
groups = np.repeat(np.linspace(0, n_groups-1, num=n_groups), n_samples/n_groups)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# this is code slightly modified from the sklearn docs here:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    
    cmap_cv = plt.cm.coolwarm

    jet = plt.cm.get_cmap('jet', 256)
    seq = np.linspace(0, 1, 256)
    _ = np.random.shuffle(seq)   # inplace
    cmap_data = ListedColormap(jet(seq))

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=plt.cm.Set3)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['target', 'day']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, len(y)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


# Let's again imagine we want to do
# - a rolling time series split
# - where we have a gap of 2 days between train and validation sets
# - and we make the maximum size of each train set to be 7 days
# 
# Here we specify the number of splits, the maximum number of groups in each train set, and the maximum number of groups in each valdiation set (sklearn has this convention where they call it the "test" set; I preserve that in the variable names, but prefer to call it the validation set).

# In[ ]:


fig, ax = plt.subplots()

cv = PurgedGroupTimeSeriesSplit(
    n_splits=5,
    max_train_group_size=7,
    group_gap=2,
    max_test_group_size=3
)

plot_cv_indices(cv, X_train, y_train, groups, ax, 5, lw=20);

