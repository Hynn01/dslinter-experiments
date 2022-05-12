#!/usr/bin/env python
# coding: utf-8

# >### Let's test a Transformer!
# >This is a simple starter notebook for Kaggle's Crypto JPX showing purged group timeseries KFold. Purged Times Series is explained [here][2]. There are many configuration variables below to allow you to experiment. Use either CPU or GPU. You can control which years are loaded, which neural networks are used, and whether to use feature engineering. You can experiment with different data preprocessing, model hyperparameters, loss, and number of seeds to ensemble. The extra datasets contain the full history of the assets at the same format of the competition, so you can input that into your model too.
# >
# >**NOTE:** this notebook lets you run a different experiment in each fold if you want to run lots of experiments. (Then it is like running multiple holdout validation experiments but in that case note that the overall CV score is meaningless because LB will be much different when the multiple experiments are ensembled to predict test). **If you want a proper CV with a reliable overall CV score you need to choose the same configuration for each fold.**
# >
# 
# [1]: TBD

# # <span class="title-section w3-xxlarge" id="codebook">Let's test a Transformer!</span>
# <hr>
# 
# This is a simple starter notebook for Kaggle's JPX Comp showing purged group timeseries KFold with extra data. Purged Times Series is explained [here][2]. There are many configuration variables below to allow you to experiment. Use either GPU or TPU. You can control which years are loaded, which neural networks are used, and whether to use feature engineering. You can experiment with different data preprocessing, model architecture, loss, optimizers, and learning rate schedules. The extra datasets contain the full history of the assets at the same format of the competition, so you can input that into your model too.
# 
# **NOTE:** this notebook lets you run a different experiment in each fold if you want to run lots of experiments. (Then it is like running multiple holdout validaiton experiments but in that case note that the overall CV score is meaningless because LB will be much different when the multiple experiments are ensembled to predict test). **If you want a proper CV with a reliable overall CV score you need to choose the same configuration for each fold.**
# 
# **Transformer**:
# 
# Transformer is very popular among deep learning competitions. This notebook provides the training and inference pipelines for transformer encoder implementation. Due to the long training time, it only runs for a small amount of epochs.  You may get better results by tuning the structure and hyperparameters.
# 
# <center><img src = "data:image/png;base64,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"></center>
# 
# Base Transformer structure from https://www.tensorflow.org/tutorials/text/transformer, modified [here](https://www.kaggle.com/gogo827jz/moa-lstm-pure-transformer-fast-and-not-bad) by Yirun Zhang with gelu activation function. No positional embedding is needed so it is removed and the embedding layer was also changed to a dense layer.
# 
# 
# [1]: TBD

# #### Code starts here ⬇

# In[ ]:


import os
import traceback
import tensorflow as tf
from scipy.stats import pearsonr
import pandas as pd, numpy as np
import jpx_tokyo_market_prediction
from sklearn.metrics import r2_score
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# # <span class="title-section w3-xxlarge" id="config">Configuration 🎚️</span>
# <hr >
# 
# In order to be a proper cross validation with a meaningful overall CV score, **you need to choose the same** `INC2021`, `INC2020`, `INC2019`, `INC2018`, `INC2017`, `INCCOMP`, `INCSUPP`, and `DEPTH_NETS`, `WIDTH_NETS` **for each fold**. If your goal is to just run lots of experiments, then you can choose to have a different experiment in each fold. Then each fold is like a holdout validation experiment. When you find a configuration you like, you can use that configuration for all folds.
# * DEVICE - is GPU or TPU
# * SEED - a different seed produces a different triple stratified kfold split.
# * FOLDS - number of folds. Best set to 3, 5, or 15 but can be any number between 2 and 15
# * INC2022 - This controls whether to include the extra historical prices during 2022.
# * INC2021 - This controls whether to include the extra historical prices during 2021.
# * INC2020 - This controls whether to include the extra historical prices during 2020.
# * INC2019 - This controls whether to include the extra historical prices during 2019.
# * INC2018 - This controls whether to include the extra historical prices during 2018.
# * INC2017 - This controls whether to include the extra historical prices during 2017.
# * INCCOMP - This controls whether to include the original data of the competition.
# * BATCH_SIZES - is a list of length FOLDS. These are batch sizes for each fold. For maximum speed, it is best to use the largest batch size your GPU or TPU allows.
# * EPOCHS - is a list of length FOLDS. These are maximum epochs. Note that each fold, the best epoch model is saved and used. So if epochs is too large, it won't matter.
# * DEPTH_NETS - is a list of length FOLDS. These are the Network Depths to use each fold. The number refers to the number of layers. So a number of `1` refers to 1 layer, and `2` refers to 2 layers, etc.
# * WIDTH_NETS - is a list of length FOLDS. These are the Network Widths to use each fold. The number refers to the number of units per layer. So a number of `32` refers to 32 per layer, and `643` refers to 64 layers, etc.

# In[ ]:


DEVICE = "TPU" #or "GPU"

SEED = 42

# CV PARAMS
FOLDS = 5
GROUP_GAP = 130
MAX_TEST_GROUP_SIZE = 180
MAX_TRAIN_GROUP_SIZE = 280

# WHICH YEARS TO INCLUDE? YES=1 NO=0
INC2022 = 1
INC2021 = 1
INC2020 = 1
INC2019 = 1
INC2018 = 1
INC2017 = 1
INCSUPP = 1

# BATCH SIZE AND EPOCHS
BATCH_SIZES = [2048] * FOLDS
EPOCHS = [1] * FOLDS

# WHICH NETWORK ARCHITECTURE TO USE?
DEPTH_NETS = [2, 2, 2, 2, 2] 
WIDTH_NETS = [64, 64, 64, 64, 64]


# In[ ]:


if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None
    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except: print("failed to initialize TPU")
    else: DEVICE = "GPU"

if DEVICE != "TPU": strategy = tf.distribute.get_strategy()
if DEVICE == "GPU": print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
AUTO     = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync


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


def upper_shadow(df): return df['High'] - np.maximum(df['Close'], df['Open'])
def lower_shadow(df): return np.minimum(df['Close'], df['Open']) - df['Low']

def get_features(df):
    df_feat = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_feat['upper_Shadow'] = upper_shadow(df_feat)
    df_feat['lower_Shadow'] = lower_shadow(df_feat)
    df_feat["high_div_low"] = df_feat["High"] / df_feat["Low"]
    df_feat["open_sub_close"] = df_feat["Open"] - df_feat["Close"]
    return df_feat


# # <span class="title-section w3-xxlarge" id="modelconf">Configure the model ⚙️</span>
# <hr>
# 
# This is a simple model with simple set of hyperparameters. Consider experimenting with different models, parameters, ensembles and so on.

# **The Model**

# In[ ]:


import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from random import choices
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GroupKFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b = True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:

        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):

        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm = [0, 2, 1, 3])

    def call(self, v, k, q, mask):

        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    refer : https://github.com/google-research/bert/blob/bee6030e31e42a9394ac567da170a89a98d2062f/modeling.py#L264
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def point_wise_feed_forward_network(d_model, dff):

    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation = gelu),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate = 0.1):

        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training = training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, rate = 0.1):

        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.embedding = tf.keras.layers.Dense(self.d_model)

        self.enc_layers = [EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate)
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(self.rate)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout': self.dropout,
        })
        return config

    def call(self, x, training, mask = None):

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)

        x = self.dropout(x, training = training)

        for i in range(self.num_layers):

            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


# In[ ]:


def build_model(fold, dim = 128, weight = 1.0):   
    num_layers, d_model, num_heads, dff, dropout_rate = 3, 128, 8, 256, 0.4
    num_layers = DEPTH_NETS[fold]
    d_model = WIDTH_NETS[fold]
    dropout_rates = 0.003
    
    inp = tf.keras.layers.Input(shape = (dim, ))
    x = tf.keras.layers.Reshape((1, dim))(inp)      
    
    x = TransformerEncoder(num_layers, d_model, num_heads, dff, dropout_rate)(x)[:, 0, :]      

    out = tf.keras.layers.Dense(1, activation = 'linear', name = 'action')(x)
    model = tf.keras.models.Model(inputs = inp, outputs = out)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = tf.keras.losses.MeanSquaredError())
    return model


# # <span class="title-section w3-xxlarge" id="sched">LR Scheduler ⏱️</span>
# <hr>
# 
# This is a common train schedule in kaggle competitions. The learning rate starts near zero, then increases to a maximum, then decays over time. Consider changing the schedule and/or learning rates. Note how the learning rate max is larger with larger batches sizes. This is a good practice to follow.

# In[ ]:


def get_lr_callback(batch_size = 8):
    lr_start   = 0.000005
    lr_max     = 0.00000125 * REPLICAS * batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
    def lrfn(epoch):
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        else: lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        return lr
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback


# # Time Series Cross Validation
# 
# > "There are many different ways one can do cross-validation, and **it is the most critical step when building a good machine learning model** which is generalizable when it comes to unseen data."
# -- **Approaching (Almost) Any Machine Learning Problem**, by Abhishek Thakur
# 
# CV is the **first** step, but very few notebooks are talking about this. Here we look at "purged rolling time series CV" and actually apply it in hyperparameter tuning for a basic estimator. This notebook owes a debt of gratitude to the notebook ["Found the Holy Grail GroupTimeSeriesSplit"](https://www.kaggle.com/jorijnsmit/found-the-holy-grail-grouptimeseriessplit). That notebook is excellent and this solution is an extention of the quoted pending sklearn estimator. I modify that estimator to make it more suitable for the task at hand in this competition. The changes are
# 
# - you can specify a **gap** between each train and validation split. This is important because even though the **group** aspect keeps whole days together, we suspect that the anonymized features have some kind of lag or window calculations in them (which would be standard for financial features). By introducing a gap, we mitigate the risk that we leak information from train into validation
# - we can specify the size of the train and validation splits in terms of **number of days**. The ability to specify a validation set size is new and the the ability to specify days, as opposed to samples, is new.
# 
# The code for `PurgedTimeSeriesSplit` is below. I've hidden it because it is really meant to act as an imported class. If you want to see the code and copy for your work, click on the "Code" box.

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
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

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


# # <span class="title-section w3-xxlarge" id="training">Training 🏋️</span>
# <hr>
# 
# Our model will be trained for the number of FOLDS and EPOCHS you chose in the configuration above. Each fold the model with lowest validation loss will be saved and used to predict OOF and test. Adjust the variable `VERBOSE`. The variable `VERBOSE=1 or 2` will display the training and validation loss for each epoch as text. 

# **Let's take a look at our CV**

# In[ ]:


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    cmap_cv = plt.cm.coolwarm
    jet = plt.cm.get_cmap('jet', 256)
    seq = np.linspace(0, 1, 256)
    _ = np.random.shuffle(seq)   # inplace
    cmap_data = ListedColormap(jet(seq))    
    for ii, (tr, tt) in enumerate(list(cv.split(X=X, y=y, groups=group))):
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0        
        ax.scatter(range(len(indices)), [ii + .5] * len(indices), c=indices, marker='_', lw=lw, cmap=cmap_cv, vmin=-.2, vmax=1.2)
    ax.scatter(range(len(X)), [ii + 1.5] * len(X), c=y, marker='_', lw=lw, cmap=plt.cm.Set3)
    ax.scatter(range(len(X)), [ii + 2.5] * len(X), c=group, marker='_', lw=lw, cmap=cmap_data)
    yticklabels = list(range(n_splits)) + ['target', 'day']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels, xlabel='Sample index', ylabel="CV iteration", ylim=[n_splits+2.2, -.2], xlim=[0, len(y)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax

asset_id = 1301
df = load_training_data(asset_id)
df_proc = get_features(df)
df_proc['date'] = df['date'].copy()
df_proc['y'] = df['Target']
df_proc = df_proc.dropna(how="any")
X = df_proc.drop("y", axis=1)
y = df_proc["y"]
groups = pd.factorize(X['date'].dt.day.astype(str) + '_' + X['date'].dt.month.astype(str) + '_' + X['date'].dt.year.astype(str))[0]
X = X.drop(columns = 'date')

fig, ax = plt.subplots(figsize = (12, 6))
cv = PurgedGroupTimeSeriesSplit(n_splits = FOLDS, group_gap = GROUP_GAP, max_train_group_size=MAX_TRAIN_GROUP_SIZE, max_test_group_size=MAX_TEST_GROUP_SIZE)
plot_cv_indices(cv, X, y, groups, ax, FOLDS, lw=20)


# **Main Training Function**

# In[ ]:


# USE VERBOSE=0 for silent, VERBOSE=1 for interactive, VERBOSE=2 for commit
VERBOSE = 1

def get_Xy_and_model():
    df = load_training_data()
    df_proc = get_features(df)
    df_proc['date'] = df['date'].copy()
    df_proc['y'] = df['Target']
    df_proc = df_proc.dropna(how="any")
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    groups = pd.factorize(X['date'].dt.day.astype(str) + '_' + X['date'].dt.month.astype(str) + '_' + X['date'].dt.year.astype(str))[0]
    X = X.drop(columns = 'date')
    oof_preds = np.zeros(len(X))
    scores, models = [], []
    
    for fold, (train_idx, val_idx) in enumerate(PurgedGroupTimeSeriesSplit(n_splits = FOLDS, group_gap = GROUP_GAP, max_train_group_size = MAX_TRAIN_GROUP_SIZE, max_test_group_size = MAX_TEST_GROUP_SIZE).split(X, y, groups)):
        # GET TRAINING, VALIDATION SET
        x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # DISPLAY FOLD INFO
        if DEVICE == 'TPU':
            if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)
        print('#'*25); print('#### FOLD',fold+1)        

        # BUILD MODEL
        K.clear_session()
        with strategy.scope(): model = build_model(fold, dim = x_train.shape[1])

        # SAVE BEST MODEL EACH FOLD
        sv = tf.keras.callbacks.ModelCheckpoint('fold-%i.h5' % fold, monitor = 'val_loss', verbose = 0, save_best_only = True, save_weights_only = True, mode = 'min', save_freq = 'epoch')

        # TRAIN
        history = model.fit( x_train, y_train, epochs = EPOCHS[fold], callbacks = [sv,get_lr_callback(BATCH_SIZES[fold])], validation_data = (x_val, y_val), verbose=VERBOSE)
        model.load_weights('fold-%i.h5' % fold)

        # PREDICT OOF
        pred = model.predict(x_val, verbose = VERBOSE)
        models.append(model)

        # REPORT RESULTS
        try: mse = mean_squared_error(np.nan_to_num(y_val), np.nan_to_num(pred))
        except: mse = 0.0
        scores.append(mse)
        oof_preds[val_idx] = pred.flatten()
        print('#### FOLD %i OOF MSE %.3f' % (fold + 1, mse))
    
    df_proc['SecuritiesCode'] = df['SecuritiesCode']
    df = df_proc
    df['oof_preds'] = np.nan_to_num(oof_preds)
    print('\n\n' + ('-' * 80) + '\n' + 'Finished trainings. Results:')
    print('Model: r2_score: %s | pearsonr: %s ' % (r2_score(df['y'], df['oof_preds']), pearsonr(df['y'], df['oof_preds'])[0]))
    print('Predictions std: %s | Target std: %s' % (df['oof_preds'].std(), df['y'].std()))

    try: plt.close()
    except: pass
    df2 = df.reset_index().set_index('date')
    df2 = df2.loc[df2['SecuritiesCode'] == 1301] # For demonstration purpose only.
    fig = plt.figure(figsize = (12, 6))
    # fig, ax_left = plt.subplots(figsize = (12, 6))
    ax_left = fig.add_subplot(111)
    ax_left.set_facecolor('azure')
    ax_right = ax_left.twinx()
    ax_left.plot(df2['y'].rolling(3 * 30 * 24 * 60).corr(df2['oof_preds']).iloc[::24 * 60], color = 'crimson', label = "Corr")
    ax_right.plot(df2['Close'].iloc[::24 * 60], color = 'darkgrey', label = "%s Close" % stock_name_dict[asset_id])
    plt.legend()
    plt.grid()
    plt.xlabel('Time')
    plt.title('3 month rolling pearsonr for %s' % (stock_name_dict[asset_id]))
    plt.show()
    
    return scores, oof_preds, models, y

print(f"Training model")
cur_scores, cur_oof_preds, cur_models, cur_targets = get_Xy_and_model()
scores, oof_preds, models, targets = cur_scores, cur_oof_preds, cur_models, cur_targets


# # <span class="title-section w3-xxlarge" id="codebook">Calculate OOF MSE</span>
# The OOF (out of fold) predictions are saved to disk. If you wish to ensemble multiple models, use the OOF to determine what are the best weights to blend your models with. Choose weights that maximize OOF CV score when used to blend OOF. Then use those same weights to blend your test predictions.

# In[ ]:


print('Overall MEAN OOF MSE %s' % np.mean(list(scores)))


# # <span class="title-section w3-xxlarge" id="submit">Submit To Kaggle 🇰</span>
# <hr>

# In[ ]:


env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for (df_test, options, financials, trades, secondary_prices, df_pred) in iter_test:
    x_test = get_features(df_test)
    y_pred = np.mean(np.concatenate([np.expand_dims(model.predict(x_test), axis = 0) for model in models], axis = 0), axis = 0)
    df_pred['Target'] = y_pred[:, 0]
    df_pred = df_pred.sort_values(by = "Target", ascending = False)
    df_pred['Rank'] = np.arange(0,2000)
    df_pred = df_pred.sort_values(by = "SecuritiesCode", ascending = True)
    df_pred.drop(["Target"], axis = 1)
    submission = df_pred[["Date", "SecuritiesCode", "Rank"]]
    env.predict(submission)

