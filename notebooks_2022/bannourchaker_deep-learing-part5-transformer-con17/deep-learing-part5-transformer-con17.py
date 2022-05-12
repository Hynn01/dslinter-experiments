#!/usr/bin/env python
# coding: utf-8

# # Introduction
# ## Overview
# In this competition, you'll classify 60-second sequences of sensor data, indicating whether a subject was in either of two activity states for the duration of the sequence.
# 
# ##  Files descriptions
# train.csv - the training set, comprising ~26,000 60-second recordings of thirteen biological sensors for almost one thousand experimental participants
# 
# * train_labels.csv - the class label for each sequence.
# * test.csv - the test set. For each of the ~12,000 sequences, you should predict a value for that sequence's state.
# * sample_submission.csv - a sample submission file in the correct format.
# 
# ## Fields
# 
# 1. sequence - a unique id for each sequence
# 1. subject - a unique id for the subject in the experiment
# 1. step - time step of the recording, in one second intervals
# 1. sensor_00 - sensor_12 - the value for each of the thirteen sensors at that time step
# 1. state - the state associated to each sequence. This is the target which you are trying to predict.
# 
# ## Plan : 
# 
# I Will solve this problem using  **Deep Learning** :
# 
# Deep learning is the application of artificial neural networks using modern hardware.
# 
# It allows the development, training, and use of neural networks that are much larger (more layers) than was previously thought possible.
# 
# There are thousands of types of specific neural networks proposed by researchers as modifications or tweaks to existing models. Sometimes wholly new approaches.
# 
# As a practitioner, I recommend waiting until a model emerges as generally applicable. It is hard to tease out the signal of what works well generally from the noise of the vast number of publications released daily or weekly.
# 
# There are three classes of artificial neural networks that I recommend that you focus on in general. They are:
# 
# 
# 
# 
# 
# *  Multilayer Perceptrons (MLPs)
# 
# *  Convolutional Neural Networks (CNNs)
# 
# *  Recurrent Neural Networks (RNNs)
# 
# [ref:](https://machinelearningmastery.com/when-to-use-mlp-cnn-and-rnn-neural-networks/) 
# 
# Follow my notebooks:
# 
# - **MLP:**
# 
# * https://www.kaggle.com/code/bannourchaker/deep-learing-part1-mlp2-deep-and-cross-con4
# 
# * https://www.kaggle.com/code/bannourchaker/deep-learing-part1-mlp2-cross-net-con3
# 
# * https://www.kaggle.com/code/bannourchaker/deep-learing-part1-mlp2-wide-and-deep-con2
# 
# * https://www.kaggle.com/code/bannourchaker/deep-learing-part1-mlp-con1
# 
# 
# 
# - **RNN:** 
# 
# * https://www.kaggle.com/bannourchaker/deep-learing-part2-stack-rnn-con5
# 
# * https://www.kaggle.com/bannourchaker/deep-learing-part2-bilstm-densenet-rnn-con6
# 
# * https://www.kaggle.com/bannourchaker/deep-learing-part2-deepbilstm-attention-rnn-con7
# 
# - **CNN:**
# 
# * https://www.kaggle.com/bannourchaker/deep-learing-part3-cnn-isallwhatyouneed-con8
# 
# * https://www.kaggle.com/bannourchaker/deep-learing-part3-cnn-denstnet-con9
# 
# * https://www.kaggle.com/bannourchaker/deep-learing-part3-cnn-denstnet-advanced-con10
# 
# * https://www.kaggle.com/bannourchaker/deep-learing-part3-cnn-inceptiontime-con11
# 
# * https://www.kaggle.com/bannourchaker/deep-learing-part3-cnn-googlenet-inception-con12
# 
# * https://www.kaggle.com/code/bannourchaker/deep-learing-part3-cnn-encoder-decoder-con13
# - **Hybrid CNN_LSTM:**
# * https://www.kaggle.com/bannourchaker/deep-learing-part4-hybrid-cnnlstm-sequentiel-con14
# 
# * https://www.kaggle.com/bannourchaker/deep-learing-part4-hybrid-cnn-lstm-parallel-con155
# 
# * https://www.kaggle.com/bannourchaker/deep-learing-part4-hybrid-cnn-lstm-parallel-con166
# 

#  
# <a id="top"></a>
# 
# <div class="list-group" id="list-tab" role="tablist">
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#1777C4; border:0' role="tab" aria-controls="home"><center>Get Started in Deep Learning</center></h3>
# 
# * [The 5-Step Model Life-Cycle](#1)
# * [Import](#2)
# * [Prepare Data/Impute Missing Value](#3)
# * [Sequential Model API](#4)   
# * [Modeling](#100)
# * [Submission](#100)
#     
# Predictive modeling with deep learning is a skill that modern developers need to know.
# 
# TensorFlow is the premier open-source deep learning framework developed and maintained by Google. Although using TensorFlow directly can be challenging, the modern tf.keras API beings the simplicity and ease of use of Keras to the TensorFlow project.
# 
# Using tf.keras allows you to design, fit, evaluate, and use deep learning models to make predictions in just a few lines of code. It makes common deep learning tasks, such as classification and regression predictive modeling, accessible to average developers looking to get things done.
# 
# In this Notebook, you will discover a step-by-step guide to developing deep learning models in TensorFlow using the tf.keras API.
#  Deep Learning Model Life-Cycle
# 
# In this section, you will discover the life-cycle for a deep learning model and the two tf.keras APIs that you can use to define models.
#     
# 
# **Standalone Keras.** The standalone open source project that supports TensorFlow, Theano and CNTK backends.
#     
# **tf.keras.**  The Keras API integrated into TensorFlow 2.
#     
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#1777C4; border:0' role="tab" aria-controls="home"><center>The 5-Step Model Life-Cycle</center></h3>
# 
# A model has a life-cycle, and this very simple knowledge provides the backbone for both modeling a dataset and understanding the tf.keras API.
# 
# The five steps in the life-cycle are as follows:
#     
# 
#                 Define the model.
# 
#                 Compile the model.
# 
#                 Fit the model.
# 
#                 Evaluate the model.
# 
#                 Make predictions.    
# 
#    
# 
#     
# 
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#1777C4; border:0' role="tab" aria-controls="home"><center> Basics of Neural Network</center></h3>
#    
#     
# ![image.png](attachment:9fb793a9-ed54-4165-a6a7-bae9e5d59a09.png)
#     
#     
#  **Activation function -** the activation function of a node defines the output of that node given an input or set of inputs.
# 
# **Loss Function its also called error function -**When you train Deep learning models, you feed data to the network, generate predictions, compare them with the actual values (the targets) and then compute what is known as a loss. This loss essentially tells you something about the performance of the network: the higher it is, the worse your network performs overall.
# 
# **Highly important -** activation function is for individual perceptron(basic unit of neural network) while loss function is calculated at final output layer..
# 
# **Optimization** the calculated loss, which tells us how poorly the model is performing at that current instant. Now we need to use this loss to train our network such that it performs better. Essentially what we need to do is to take the loss and try to minimize it, because a lower loss means our model is going to perform better. The process of minimizing (or maximizing) any mathematical expression is called optimization. Optimizers are algorithms or methods used to change the attributes of the neural network such as weights and learning rate to reduce the losses. Optimizers are used to solve optimization problems by minimizing the function.   
#     
#  ![image.png](attachment:359a1e81-1fc7-4e6a-a2ff-cf3826e5fdef.png)
#     
# **List of optimizer on tensorflow**
# 
# * Gradient Descent
# * Stochastic Gradient Descent (SGD)
# * Mini Batch Stochastic Gradient Descent (MB-SGD)
# * SGD with momentum
# * Nesterov Accelerated Gradient (NAG)
# * Adaptive Gradient (AdaGrad)
# * AdaDelta
# * RMSprop
# * Adam
#     
# **Gradient -**  in short its rate of change of error(loss) with respect of neural network parameter.. its like adjusting weights so that loss is minimum. Gradient Descent is an optimization algorithm for finding a local minimum of a differentiable function.
# 
# **learning rate -** the learning rate is a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function  
#     
#  ![image.png](attachment:264c7c54-559c-4aa6-b07c-425358c3e03c.png)
# 
#     
# [ ref course:](https://github.com/DeepSparkChaker/COURS/blob/main/Deep%20Learning%20Interview.pdf) 
#     
#     
#  <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#1777C4; border:0' role="tab" aria-controls="home"><center>Import</center></h3>   

# In[ ]:


get_ipython().system('pip install -U scikit-learn')


# In[ ]:


###############################################################################
#                       Load Library                                          #
###############################################################################

#Load the librarys
import pandas as pd #To work with dataset
import numpy as np #Math library
import matplotlib.gridspec as gridspec
import seaborn as sns #Graph library that use matplot in background
import matplotlib.pyplot as plt #to plot some parameters in seaborn
import warnings
# Preparation  
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer, StandardScaler,Normalizer,RobustScaler,MaxAbsScaler,MinMaxScaler,QuantileTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer
# Import StandardScaler from scikit-learn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer,IterativeImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer,ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline,FeatureUnion
from sklearn.manifold import TSNE
# Import train_test_split()
# Metrics
from sklearn.metrics import roc_auc_score, average_precision_score,recall_score
from sklearn.metrics import make_scorer,mean_absolute_error
from sklearn.metrics import mean_squared_error,classification_report,f1_score
from sklearn.metrics import roc_curve,confusion_matrix
from datetime import datetime, date
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.linear_model import LogisticRegression

import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.metrics import AUC
#import smogn
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone,ClassifierMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
# For training random forest model
import lightgbm as lgb
from scipy import sparse
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
# Model selection
from sklearn.model_selection import StratifiedKFold,GroupKFold, StratifiedGroupKFold  
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression,f_classif,chi2
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_classif,VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
import lightgbm as lgbm
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from xgboost import XGBClassifier,XGBRegressor
from sklearn import set_config
from itertools import combinations
# Cluster :
from sklearn.cluster import MiniBatchKMeans
#from yellowbrick.cluster import KElbowVisualizer
#import smong 
import category_encoders as ce
import warnings
#import optuna 
from joblib import Parallel, delayed
import joblib 
from sklearn import set_config
from typing import List, Optional, Union
import itertools
import shap

# Imbalanced data 
from imblearn.datasets import fetch_datasets
# to correctly set up the cross-validation
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import (
    RandomUnderSampler,
    CondensedNearestNeighbour,
    TomekLinks,
    OneSidedSelection,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    NeighbourhoodCleaningRule,
    NearMiss,
    InstanceHardnessThreshold
)
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
    SVMSMOTE,
)
from tqdm import tqdm 
import gc
import os
import shutil
import logging
import tensorflow_addons as tfa
set_config(display='diagram')
warnings.filterwarnings('ignore')


# <a id="top"></a>
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#1777C4; border:0' role="tab" aria-controls="home"><center>Load Data</center></h3>
# 
# ## Load the training data

# In[ ]:


get_ipython().run_cell_magic('time', '', "###############################################################################\n#                         Load data                                           #\n###############################################################################\nTRAIN_DATA_PATH = '../input/tabular-playground-series-apr-2022/train.csv'\nTRAIN_LABELS_PATH = '../input/tabular-playground-series-apr-2022/train_labels.csv'\nTEST_DATA_PATH = '../input/tabular-playground-series-apr-2022/test.csv'\nSAMPLE_SUBMISSION = '../input/tabular-playground-series-apr-2022/sample_submission.csv'\nSUBMISSION_FILE = 'submission.csv'\nINDEX = 'sequence'\nTARGET = 'state'\nSENSORS = ['sensor_{:02d}'.format(x) for x in range(0, 13)]")


# In[ ]:


get_ipython().run_cell_magic('time', '', '###############################################################################\n#                         Load data                                           #\n###############################################################################\ndef read_data():\n    """Reads the data sets\n    """\n    train = pd.read_csv(TRAIN_DATA_PATH)\n    test = pd.read_csv(TEST_DATA_PATH)\n    labels = pd.read_csv(TRAIN_LABELS_PATH)\n    submission = pd.read_csv(SAMPLE_SUBMISSION, index_col=INDEX)\n    \n    return train, test, labels, submission\ntrain, test, labels, submission = read_data()')


# In[ ]:


# adding labels to train data
train = pd.merge(train, labels,how='left', on="sequence")


# In[ ]:


y=train['state'].to_numpy().reshape(-1, 60)


# ## Extract groups and prevent data Leakage

# In[ ]:


groups = train["sequence"]
groups.nunique()


# In[ ]:


labels.head()


# In[ ]:


train.head()


# In[ ]:


train.subject.nunique()


# # Simple EDA

# In[ ]:


def plot_state_line(train, breath_ids=None, additional_cols=None):
    if breath_ids is None:
        breath_ids = train["sequence"].unique()[:4].tolist()
        
    fig, axes = plt.subplots(figsize=(25, 18), ncols=4, nrows=3)
    axes = np.ravel(axes)
    plot_cols = ["sensor_00", "sensor_01", "sensor_03"]
    
    if additional_cols:
        plot_cols += additional_cols
        
    for b, ax in zip(breath_ids, axes):
        _df = train[train["sequence"]==b].copy()
        (_df
         .set_index("step")[plot_cols]
         .plot(colormap='Paired',
               ax=ax,
               title=f"sequence={b}", 
               linewidth=2)
        )

    fig.subplots_adjust(hspace=0.3)
    return fig

def plot_state_line_px(input_df, breath_id, additional_cols=[]):
    import plotly.express as px
    print('hi')
    cols = ["sensor_00",  "sensor_01", "sensor_02"] + additional_cols
    breath_df = input_df[input_df["sequence"]==breath_id]
    plot_df = pd.DataFrame()
    for col in cols:
        _df = breath_df[cols].rename(columns={col:"y"})
        _df["step"] = breath_df["step"].values
        _df["color"] = col
        plot_df = pd.concat([plot_df, _df])
    fig = px.line(plot_df, x="step", y="y", color="color",title=f"Sequence={breath_id}, Subject={breath_df['subject'].unique()}")
    fig.update_traces(mode='markers+lines')
    return fig


# In[ ]:


breath_ids = list(train["sequence"].sample(12))
fig = plot_state_line(train, breath_ids, additional_cols=[])


# In[ ]:


temp = train.groupby('sequence').subject.min() # dataframe with one row per sequence
temp = labels.merge(temp, on='sequence') # add a column with the labels
temp = temp.groupby('subject').agg({'state': 'mean', 'sequence': 'count'}).rename(columns={'state': 'probability', 'sequence': 'sequence_count'})
temp1 = temp[temp.sequence_count >= 25].probability.rename('Probability of state==1')
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.hist(temp1, bins=20)
plt.ylabel('Subject count')
plt.xlabel('Probability for state==1')
plt.title('Histogram of state probabilities per subject')

plt.subplot(1, 2, 2)
plt.scatter(temp.sequence_count, temp.probability)
plt.xlabel('sequence count')
plt.ylabel('probability')
plt.title('Probability depends on sequence count')
plt.show()

print()
print(f"The standard deviation of {temp[temp.sequence_count >= 25].probability.std():.2f} is much higher than 0.1.")
print()
print('Subjects which are always in state 0:', (temp.probability == 0).sum())


# **Insight:**
# 
# * These views on the data confirms the earlier insight that generalization to previously unseen subjects is important and that the cross-validation strategy has to take this into account.
#  
# * The more often a subject occurs in the training data, the higher its probability for state=1. Although we must not use the subject as a feature, we should count how many sequences belong to every subject and use this count as a feature. (I tried it and this feature improved the predictions.)
# 
# * The lower left of the scattergram shows some dots which are grouped to hyperbolas. This is an artefact of low sequence counts and cannot be exploited for prediction. (If a subject has only one or two positive sequences, the probability is 1 / sequence count or 2 / sequence count, respectively, and these are the hyperbolas.
# 
# 
# [ref1](https://www.kaggle.com/code/rnepal2/tps-dnn-augmentation-bandpass-shift-etc)
# 
# [ref2](https://www.kaggle.com/code/abdulravoofshaik/early-eda-and-insights/notebook)

# # Config

# In[ ]:


# ========================================
# Utils
# ========================================

class Logger:
    """save log"""
    def __init__(self, path):
        self.general_logger = logging.getLogger(path)
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(os.path.join(path, 'Experiment.log'))
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)

    def info(self, message):
        # display time
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    @staticmethod
    def now_string():
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    
class Util:
    """save & load"""
    @classmethod
    def dump(cls, value, path):
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
    
    
class HorizontalDisplay:
    """display dataframe"""
    def __init__(self, *args):
        self.args = args

    def _repr_html_(self):
        template = '<div style="float: left; padding: 10px;">{0}</div>'
        return "\n".join(template.format(arg._repr_html_())
                         for arg in self.args)


# In[ ]:


class Config:
    name_v1 = "Exp-101-ResBiLSTM-v1"  
    editor = "chaka_abderrazak"
    lr = 1e-5
    weight_decay = 2e-5
    epochs =1000
    scheduler = "CosineDecayRestarts"  # : ReduceLROnPlateau: CosineDecayRestarts#Custum
    early_stop = True
    seq_len = 60
    masking = False
    steps_per_epochs = None
    train_batch_size = 64
    valid_batch_size = 32
    test_batch_size = 64
    n_fold = 12
    trn_fold = list(range(12))
    seed = 2022
    target_col = "state"
    debug = False

    # Kaggle Env
    kaggle_dataset_path = "../input/exp-101"

if Config.debug:
    Config.epochs = 2
    Config.trn_fold = [0]


# In[ ]:


print("This environment is Kaggle Kernel")
INPUT = "../input/tabular-playground-series-apr-2022"
EXP, OUTPUT, SUBMISSION = "./", "./", "./"
EXP_MODEL = os.path.join(EXP, "model")
EXP_FIG = os.path.join(EXP, "fig")
EXP_PREDS = os.path.join(EXP, "preds")

#if Config.kaggle_dataset_path is not None:
 #   KD_MODEL = os.path.join(Config.kaggle_dataset_path, "model")
  #  KD_EXP_PREDS = os.path.join(Config.kaggle_dataset_path, "preds")
   # shutil.copytree(KD_MODEL, EXP_MODEL)
   # shutil.copytree(KD_EXP_PREDS, EXP_PREDS)

# make dirs
for d in [EXP_MODEL, EXP_FIG, EXP_PREDS]:
    os.makedirs(d, exist_ok=True)

# utils
logger = Logger(EXP)

# utils
warnings.filterwarnings("ignore")
sns.set(style='whitegrid')


# # Utils

# In[ ]:


class colors: # You may need to change color settings
    RED = '\033[31m'
    ENDC = '\033[m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    
def combiner(x:pd.DataFrame)->List:
    #https://stackoverflow.com/questions/62071474/combine-two-columns-containing-lists-of-first-and-last-names-into-one-column-tha
    return [str(i[0])+str(i[1]) if i[1].isdigit() else str(i[0]) for i in zip(x["DISTANCE"], x["DEVICE"])]
#train1["DISTANCE"]=train1.apply(lambda x: combiner(x), axis=1)

def plot_roc(y,y_pred):
    from sklearn.metrics import auc
    fpr, tpr, thresholds = roc_curve(y_true=y, y_score=y_pred)
    auc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Model(area = {:.3f})'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
def plot_acc(f_scores):
    from matplotlib.ticker import MaxNLocator
    for fold in range(f_scores['folds'].nunique()):
        history_f = f_scores[f_scores['folds'] == fold]

        best_epoch = np.argmin(np.array(history_f['val_loss']))

        best_val_loss = history_f['val_loss'][best_epoch]
        print(f'The best val loss  epoch is :{best_epoch} and the val_loss:{best_val_loss}')
        fig, ax1 = plt.subplots(1, 2, tight_layout=True, figsize=(15,4))

        fig.suptitle('Fold : '+ str(fold+1) +
                     " Validation Loss: {:0.4f}".format(history_f['val_loss'].min()) +
                     " Validation Accuracy: {:0.4f}".format(history_f[ACC_VAL_METRICS].max()) +
                     " LR: {:0.8f}".format(history_f['lr'].min())
                     , fontsize=14)

        plt.subplot(1,2,1)
        plt.plot(history_f.loc[:, ['loss', 'val_loss']], label= ['loss', 'val_loss'])

        from_epoch = 0
        if best_epoch >= from_epoch:
            plt.scatter([best_epoch], [best_val_loss], c = 'r', label = f'Best val_loss = {best_val_loss:.5f}')
        if best_epoch > 0:
            almost_epoch = np.argmin(np.array(history_f['val_loss'])[:best_epoch])
            almost_val_loss = history_f['val_loss'][almost_epoch]
            if almost_epoch >= from_epoch:
                plt.scatter([almost_epoch], [almost_val_loss], c='orange', label = 'Second best val_loss')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper left')   

        ax2 = plt.gca().twinx()
        ax2.plot(history_f.loc[:, ['lr']], 'y:', label='lr' ) # default color is same as first ax
        ax2.set_ylabel('Learning rate')
        ax2.legend(loc = 'upper right')
        ax2.grid()

        best_epoch = np.argmax(np.array(history_f[ACC_VAL_METRICS]))
        best_val_acc = history_f[ACC_VAL_METRICS][best_epoch]
        print(f'The best val acc  epoch is :{best_epoch} and the val_acc:{best_val_acc}')
        plt.subplot(1,2,2)
        plt.plot(history_f.loc[:, [ACC_METRICS, ACC_VAL_METRICS]],label= [ACC_METRICS, ACC_VAL_METRICS])
        if best_epoch >= from_epoch:
            plt.scatter([best_epoch], [best_val_acc], c = 'r', label = f'Best val_acc = {best_val_acc:.5f}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc = 'lower left')
        plt.legend(fontsize = 15)
        plt.grid(b = True, linestyle = '-')
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Seed set to: {seed}")


# In[ ]:


###############################################################################
#                         Visualization                                       #
###############################################################################
'''
Extract info for each layer in a keras model.
'''
def utils_nn_config(model):
    lst_layers = []
    if "Sequential" in str(model): #-> Sequential doesn't show the input layer
        layer = model.layers[0]
        lst_layers.append({"name":"input", "in":int(layer.input.shape[-1]), "neurons":0, 
                           "out":int(layer.input.shape[-1]), "activation":None,
                           "params":0, "bias":0})
    for layer in model.layers:
        try:
            dic_layer = {"name":layer.name, "in":int(layer.input.shape[-1]), "neurons":layer.units, 
                         "out":int(layer.output.shape[-1]), "activation":layer.get_config()["activation"],
                         "params":layer.get_weights()[0], "bias":layer.get_weights()[1]}
        except:
            dic_layer = {"name":layer.name, "in":int(layer.input.shape[-1]), "neurons":0, 
                         "out":int(layer.output.shape[-1]), "activation":None,
                         "params":0, "bias":0}
        lst_layers.append(dic_layer)
    return lst_layers



'''
Plot the structure of a keras neural network.
'''
def visualize_nn(model, description=False, figsize=(10,8)):
    ## get layers info
    lst_layers = utils_nn_config(model)
    layer_sizes = [layer["out"] for layer in lst_layers]
    
    ## fig setup
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    ax.set(title=model.name)
    ax.axis('off')
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    x_space = (right-left) / float(len(layer_sizes)-1)
    y_space = (top-bottom) / float(max(layer_sizes))
    p = 0.025
    
    ## nodes
    for i,n in enumerate(layer_sizes):
        top_on_layer = y_space*(n-1)/2.0 + (top+bottom)/2.0
        layer = lst_layers[i]
        color = "green" if i in [0, len(layer_sizes)-1] else "blue"
        color = "red" if (layer['neurons'] == 0) and (i > 0) else color
        
        ### add description
        if (description is True):
            d = i if i == 0 else i-0.5
            if layer['activation'] is None:
                plt.text(x=left+d*x_space, y=top, fontsize=10, color=color, s=layer["name"].upper())
            else:
                plt.text(x=left+d*x_space, y=top, fontsize=10, color=color, s=layer["name"].upper())
                plt.text(x=left+d*x_space, y=top-p, fontsize=10, color=color, s=layer['activation']+" (")
                plt.text(x=left+d*x_space, y=top-2*p, fontsize=10, color=color, s="Σ"+str(layer['in'])+"[X*w]+b")
                out = " Y"  if i == len(layer_sizes)-1 else " out"
                plt.text(x=left+d*x_space, y=top-3*p, fontsize=10, color=color, s=") = "+str(layer['neurons'])+out)
        
        ### circles
        for m in range(n):
            color = "limegreen" if color == "green" else color
            circle = plt.Circle(xy=(left+i*x_space, top_on_layer-m*y_space-4*p), radius=y_space/4.0, color=color, ec='k', zorder=4)
            ax.add_artist(circle)
            
            ### add text
            if i == 0:
                plt.text(x=left-4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$X_{'+str(m+1)+'}$')
            elif i == len(layer_sizes)-1:
                plt.text(x=right+4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$y_{'+str(m+1)+'}$')
            else:
                plt.text(x=left+i*x_space+p, y=top_on_layer-m*y_space+(y_space/8.+0.01*y_space)-4*p, fontsize=10, s=r'$H_{'+str(m+1)+'}$')
    
    ## links
    for i, (n_a, n_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer = lst_layers[i+1]
        color = "green" if i == len(layer_sizes)-2 else "blue"
        color = "red" if layer['neurons'] == 0 else color
        layer_top_a = y_space*(n_a-1)/2. + (top+bottom)/2. -4*p
        layer_top_b = y_space*(n_b-1)/2. + (top+bottom)/2. -4*p
        for m in range(n_a):
            for o in range(n_b):
                line = plt.Line2D([i*x_space+left, (i+1)*x_space+left], 
                                  [layer_top_a-m*y_space, layer_top_b-o*y_space], 
                                  c=color, alpha=0.5)
                if layer['activation'] is None:
                    if o == m:
                        ax.add_artist(line)
                else:
                    ax.add_artist(line)
    plt.show()
##########################################
import matplotlib.pyplot as plt
plt.style.use('ggplot')
def plot_history(history):
    acc = history.history['auc']
    val_acc = history.history['val_auc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training auc')
    plt.plot(x, val_acc, 'r', label='Validation auc')
    plt.title('Training and validation AUC')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
#plot_history(history)


# <a id="top"></a>
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='color:white; background:#1777C4; border:0' role="tab" aria-controls="home"><center>Prepare Data/Impute Missing Value</center></h3>
# 
# 
# ## Prepare Data :

# In[ ]:


train.columns


# In[ ]:


get_ipython().run_cell_magic('time', '', '###############################################################################\n#                        Features engineer                                    #\n###############################################################################\nclass FeaturesEngineer(BaseEstimator, TransformerMixin):\n    #https://github.com/facebookresearch/Kats/blob/main/tutorials/kats_203_tsfeatures.ipynb\n    # https://github.com/predict-idlab/tsflex\n    #https://www.kaggle.com/code/ryanholbrook/tps-april-2022-benchmark\n    #https://www.kaggle.com/code/siukeitin/tps042022-fe-2500-features-with-tsfresh-catch22\n    #https://www.kaggle.com/competitions/tabular-playground-series-apr-2022/discussion/316757\n    # https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#tsfresh.feature_extraction.feature_calculators.ar_coefficient\n    def fit(self, X, y=None):\n        return self \n    # Author : https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n    def reduce_mem_usage(self,df:pd.DataFrame)->pd.DataFrame:\n        """ iterate through all the columns of a dataframe and modify the data type\n            to reduce memory usage.        \n        """\n        start_mem = df.memory_usage().sum() / 1024**2\n        print(\'Memory usage of dataframe is {:.2f} MB\'.format(start_mem))\n\n        for col in tqdm(df.columns):\n            col_type = df[col].dtype\n            name =df[col].dtype.name \n\n            if col_type != object and col_type.name != \'category\':\n            #if name != "category":    \n                c_min = df[col].min()\n                c_max = df[col].max()\n                if str(col_type)[:3] == \'int\':\n                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                        df[col] = df[col].astype(np.int8)\n                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                        df[col] = df[col].astype(np.int16)\n                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                        df[col] = df[col].astype(np.int32)\n                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                        df[col] = df[col].astype(np.int64)  \n                else:\n                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                        df[col] = df[col].astype(np.float16)\n                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                        df[col] = df[col].astype(np.float32)\n                    else:\n                        df[col] = df[col].astype(np.float64)\n            else:\n                df[col] = df[col].astype(\'category\')\n\n        end_mem = df.memory_usage().sum() / 1024**2\n        print(\'Memory usage after optimization is: {:.2f} MB\'.format(end_mem))\n        print(\'Decreased by {:.1f}%\'.format(100 * (start_mem - end_mem) / start_mem))\n        return df\n    \n    def transform(self, X, y=None):\n        train= pd.DataFrame(X).copy()\n        train=self.reduce_mem_usage(train)\n        if \'state\' in train.columns:\n            train = train.drop([\'state\'], axis=1)\n        features = train.columns.tolist()[3:]\n        for feature in features:\n            #train[feature + \'_lag1\'] = train.groupby(\'sequence\')[feature].shift(1)\n            #train[feature + \'_lag3\'] = train.groupby(\'sequence\')[feature].shift(3)\n            #train[feature + \'_back1\'] = train.groupby(\'sequence\')[feature].shift(-1)\n            #train[feature + \'_max\'] = train.groupby(\'sequence\')[feature].transform(\'max\')\n            #train[feature + \'_min\'] = train.groupby(\'sequence\')[feature].transform(\'min\')\n            #train[feature + \'_std\'] = train.groupby(\'sequence\')[feature].transform(\'std\')\n            #train[feature + \'mean\'] = train.groupby(\'sequence\')[feature].transform(\'mean\')\n            #train[feature+\'_diffmean\'] = train[feature + \'mean\'] - train[feature]\n            #train[feature + \'_cumsum\'] = train.groupby([\'sequence\'])[feature].cumsum()\n            for window in [3,6,12,24]:\n                train[feature+\'_lead_diff\'+str(window)] = train[feature] - train.groupby(\'sequence\')[feature].shift(window).fillna(0)\n                train[feature+\'_lag_diff\'+str(window)] = train[feature] - train.groupby(\'sequence\')[feature].shift(-1*window).fillna(0)\n            for window in [3,6,12,24]:\n                train[feature+\'_roll_\'+str(window)+\'_mean\'] = train.groupby(\'sequence\')[feature]\\\n                                                         .rolling(window=window, min_periods=1)\\\n                                                         .mean().reset_index(level=0,drop=True)\n                train[feature+\'_roll_\'+str(window)+\'_std\'] = train.groupby(\'sequence\')[feature]\\\n                                                        .rolling(window=window, min_periods=1)\\\n                                                        .std().reset_index(level=0,drop=True)\n                train[feature+\'_roll_\'+str(window)+\'_sum\'] = train.groupby(\'sequence\')[feature]\\\n                                                        .rolling(window=window, min_periods=1)\\\n                                                        .sum().reset_index(level=0,drop=True)\n            #train[\'one\'] = 1\n            #train[\'count\'] = (train[\'one\']).groupby(train[\'sequence\']).cumsum()\n            #train[feature +\'_ummean\'] =train[feature + \'_cumsum\'] /train[\'count\']\n            train[feature +\'_ewm_mean\'] = (train\\\n                          .groupby(\'sequence\')[feature]\\\n                          .ewm(halflife=10)\\\n                       .mean()\\\n                        .reset_index(level=0,drop=True))\n            #train[feature +\'_rolling_mean\'] = train.groupby(\'sequence\')[feature].rolling(window=10, min_periods=1).mean().reset_index(level=0,drop=True)\n            #train[feature +\'_expand_mean\'] = train.groupby(\'sequence\')[feature].expanding(10).mean().reset_index(level=0,drop=True)\n            train.fillna(0, inplace=True)\n            #train[\'sensor_02_up\'] = (train.sensor_02.diff(axis=1) > 0).sum(axis=1)\n            #train[\'sensor_02_down\'] = (train.sensor_02.diff(axis=1) < 0).sum(axis=1)\n              \n\n        print("Step-1...Completed")\n        #train[\'sensor_04\' + \'_kurt\'] = train.groupby(\'sequence\')[\'sensor_04\'].transform(pd.DataFrame.kurt)\n        train[\'sequence_count_of_subject\']=train.groupby(\'subject\')[\'sequence\'].transform(\'count\')\n        count_in_sequences = 60\n        train[\'cum_count\']=train.groupby(\'sequence\').cumcount()+1\n        train[\'month_sin\'] = np.sin(2*np.pi*train[\'cum_count\']/count_in_sequences)\n        train[\'month_cos\'] = np.cos(2*np.pi*train[\'cum_count\']/count_in_sequences) \n        train.fillna(0, inplace=True)\n        train = train.drop(["sequence", "subject", "step",\'cum_count\'], axis=1)\n        print("Step-2...Completed")\n        #train = train.drop(["sequence", "subject", "step",\'one\'], axis=1)\n        # Convert Dtypes   \n        train=self.reduce_mem_usage(train)\n        return train ')


# In[ ]:


get_ipython().run_cell_magic('time', '', '###############################################################################\n#                        Features Engineer                                    # \n###############################################################################\n#FeaturesEngineer().fit(train)\ntrain =FeaturesEngineer().fit_transform(train)\ntest =FeaturesEngineer().fit_transform(test)')


# In[ ]:


train.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "###############################################################################\n#                        Defien the features                                  #\n###############################################################################\nFEATURES = train.columns \nnum_features = len(FEATURES)\nprint(f'Number of features: {num_features}')")


# In[ ]:


get_ipython().run_cell_magic('time', '', '###############################################################################\n#                         Target Distrubution                                 #\n###############################################################################\n\ncolors = [\'#7A5197\', \'#BB5098\', \'#5344A9\', \'#F5C63C\', \'#F47F6B\']\nplt.subplots(figsize=(25, 10), facecolor=\'#f6f5f5\')\nplt.pie(labels[\'state\'].value_counts(), startangle=90, wedgeprops={\'width\':0.3}, colors=[\'#F5C63C\', \'#7A5197\'] )\nplt.title(\'Target Balance Pie Chart\', loc=\'center\', fontsize=24, color=\'#7A5197\', fontweight=\'bold\');\nplt.text(0, 0, f"{labels[\'state\'].value_counts()[0] / labels[\'state\'].count() * 100:.2f}%", ha=\'center\', va=\'center\', fontweight=\'bold\', fontsize=42, color=\'#7A5197\');\nplt.legend(labels[\'state\'].value_counts().index, ncol=2, facecolor=\'#f6f5f5\', edgecolor=\'#f6f5f5\', loc=\'lower center\', fontsize=16);\nplt.show();')


# In[ ]:


###############################################################################
#                        Cat_columns                                          #
###############################################################################
cat_columns = train.select_dtypes(include=['category','object','bool']).columns
cat_columns


# In[ ]:


###############################################################################
#                        Num_columns                                          #
###############################################################################
num_columns =train.select_dtypes(exclude=['category','object','bool']).columns
num_columns


# # Preprocess the data 

# In[ ]:


###############################################################################
#                         Define differrent Transformer                       #
###############################################################################

#Define cat pipeline
 
categorical_transformer_ohe = Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='constant',
                                                  fill_value='MISSING',
                                                #  add_indicator=True
                                                 )
                         ),
                        ('encoder', OneHotEncoder()),#(Numerical Input, Categorical Output)
                    ]
                    )
#Define num pipeline
numeric_transformer_power = Pipeline(
                            steps=[
                            ('imputer', SimpleImputer(strategy='median'
                                                     # ,add_indicator=True
                                                     )),
                            ('scaler', PowerTransformer()),#(Numerical Input, Numerical Output)                 
                                
                            ]
                            )

numeric_transformer_ss = Pipeline(
                            steps=[
                            ('imputer', SimpleImputer(strategy='median'
                                                     # ,add_indicator=True
                                                     )),
                            ('scaler', StandardScaler()),#(Numerical Input, Numerical Output)
                            ]
                            )
numeric_transformer_RS = Pipeline(
                            steps=[
                            ('imputer', SimpleImputer(strategy='median'
                                                     # ,add_indicator=True
                                                     )),
                            ('scaler', RobustScaler()),#(Numerical Input, Numerical Output)
                            ]
                            )
numeric_transformer_QuantileTransformer = Pipeline(
                            steps=[
                            ('imputer', SimpleImputer(strategy='median'
                                                     # ,add_indicator=True
                                                     )),
                            ('scaler', QuantileTransformer(n_quantiles=2000, output_distribution='normal', 
                            random_state=42))
                            ]
                            )
#num_columns=FEATURES
# Features union cat + num 
preprocessor_basic = ColumnTransformer(
            transformers=[
                ('numerical',  numeric_transformer_power, num_columns),
                ('categorical', categorical_transformer_ohe, cat_columns)
            ]
            )

preprocessor_basic


# In[ ]:


get_ipython().run_cell_magic('time', '', '###############################################################################\n#                       Preprocess Train and Test DATA                         # \n###############################################################################\nnumeric_transformer_QuantileTransformer.fit(train[FEATURES])\ntrain[FEATURES]=numeric_transformer_QuantileTransformer.transform(train[FEATURES])\ntest[FEATURES]=numeric_transformer_QuantileTransformer.transform(test[FEATURES])')


# In[ ]:


###############################################################################
#                        Extract features and Target                          #
###############################################################################
num_features = len(FEATURES)
train_data = train[FEATURES].values.reshape(int(len(train) / 60), 60, num_features)
test_data = test[FEATURES].values.reshape(int(len(test) / 60), 60, num_features)
display(train_data.shape)
display(test_data.shape)
display(labels.shape)


# In[ ]:


###############################################################################
#                        Extract features and Target                          #
###############################################################################
target= "state"
X = train_data.copy()# axis=1
#y = labels[target]
del train_data


# In[ ]:


###############################################################################
#        Split the dataset and labels into training and test sets             #
###############################################################################
class colors: # You may need to change color settings
    RED = '\033[31m'
    ENDC = '\033[m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=0,stratify=y )
print(f" {colors.BLUE}{X_test.shape[0]} {colors.ENDC} rows in test set vs. {colors.BLUE}{ X_train.shape[0]} {colors.ENDC}in training set. {colors.BLUE}{X_test.shape[1]}{colors.ENDC} Features.")


# In[ ]:


###############################################################################
#                         Check that we handle all columns                    #
###############################################################################
all_columns = (num_columns.append(cat_columns))
print(set(train.columns.tolist()).difference(all_columns))
assert set(train.columns.tolist())==set(all_columns)
assert len(train.columns.tolist())==len(all_columns)


# ## Define model/more deep 
# 
# Defining the model requires that you first select the type of model that you need and then choose the architecture or network topology.
# 
# From an API perspective, this involves defining the layers of the model, configuring each layer with a number of nodes and activation function, and connecting the layers together into a cohesive model.
# 
# Models can be defined either with the Sequential API or the Functional API, and we will take a look at this in the next section.
# 
# ### Wich Activation Funtions to use 
# * Sigmoid functions and their combinations generally work better in the case of classification problems.
# * Sigmoid and tanh functions are sometimes avoided due to the vanishing gradient problem.
# * Tanh is avoided most of the time due to dead neuron problem.
# * ReLU activation function is widely used and is default choice as it yields better results.
# * If we encounter a case of dead neurons in our networks the leaky ReLU function is the best choice.
# * ReLU function should only be used in the hidden layers.
# * An output layer can be linear activation function in case of regression problems.
# * [see list of available Activation functions in tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/activations)
# * swish is outperforming Relu.. so a better to start from
# 
# ## Compile the Model
# 
# Compiling the model requires that you first select a loss function that you want to optimize, such as mean squared error or cross-entropy.
# 
# It also requires that you select an algorithm to perform the optimization procedure, typically stochastic gradient descent, or a modern variation, such as Adam. It may also require that you select any performance metrics to keep track of during the model training process.
# 
# From an API perspective, this involves calling a function to compile the model with the chosen configuration, which will prepare the appropriate data structures required for the efficient use of the model you have defined.
# 
# The optimizer can be specified as a string for a known optimizer class, e.g. ‘sgd‘ for stochastic gradient descent, or you can configure an instance of an optimizer class and use that.
# 
# For a list of supported optimizers, see this:[optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
# 
# 
#         #compile the model
#         opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
#         model.compile(optimizer=opt, loss='binary_crossentropy')
# 
# The three most common loss functions are:
# 
#     ‘binary_crossentropy‘ for binary classification.
#     ‘sparse_categorical_crossentropy‘ for multi-class classification.
#     ‘mse‘ (mean squared error) for regression.
# [Sparse_categorical_crossentropy vs categorical_crossentropy](https://stats.stackexchange.com/questions/326065/cross-entropy-vs-sparse-cross-entropy-when-to-use-one-over-the-other)    
# For a list of supported loss functions, see:[ tf.keras Loss Functions](https://www.tensorflow.org/api_docs/python/tf/keras/losses)
# 
# Metrics are defined as a list of strings for known metric functions or a list of functions to call to evaluate predictions.
# 
# For a list of supported metrics, see: [tf.keras Metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
# 
#     # compile the model
#     model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
#     
# ##  Fit the Model
# 
# Fitting the model requires that you first select the training configuration, such as the number of epochs (loops through the training dataset) and the batch size (number of samples in an epoch used to estimate model error).
# 
# Training applies the chosen optimization algorithm to minimize the chosen loss function and updates the model using the backpropagation of error algorithm.
# 
# Fitting the model is the slow part of the whole process and can take seconds to hours to days, depending on the complexity of the model, the hardware you’re using, and the size of the training dataset.
# 
# From an API perspective, this involves calling a function to perform the training process. This function will block (not return) until the training process has finished. 
# 
#         # fit the model
# 
#         # model.fit(X, y, epochs=100, batch_size=32)
# While fitting the model, a progress bar will summarize the status of each epoch and the overall training process. This can be simplified to a simple report of model performance each epoch by setting the “verbose” argument to 2. All output can be turned off during training by setting “verbose” to 0.
#      
#         # fit the model
#         # model.fit(X, y, epochs=100, batch_size=32, verbose=0)
#         
#         
# ## Evaluate the Model
# 
# Evaluating the model requires that you first choose a holdout dataset used to evaluate the model. This should be data not used in the training process so that we can get an unbiased estimate of the performance of the model when making predictions on new data.
# 
# The speed of model evaluation is proportional to the amount of data you want to use for the evaluation, although it is much faster than training as the model is not changed.
# 
# From an API perspective, this involves calling a function with the holdout dataset and getting a loss and perhaps other metrics that can be reported.
# 
#     ...
#     # evaluate the model
#     # loss = model.evaluate(X, y, verbose=0)
#     
# ## Make a Prediction
# 
# 
# Making a prediction is the final step in the life-cycle. It is why we wanted the model in the first place.
# 
# It requires you have new data for which a prediction is required, e.g. where you do not have the target values.
# 
# From an API perspective, you simply call a function to make a prediction of a class label, probability, or numerical value: whatever you designed your model to predict.
# 
# You may want to save the model and later load it to make predictions. You may also choose to fit a model on all of the available data before you start using it.
# 
# Now that we are familiar with the model life-cycle, let’s take a look at the two main ways to use the tf.keras API to build models: sequential and functional.
# 
#         # make a prediction
#         # yhat = model.predict(X)
# 
# ##  Get Better Model Performance
# 
# **More deep and avoid overfitting:**
# 
# * Regularization in Deep Learning:  L1, L2,
# * BatchNormalization,
# * Dropout
# 
# **Reduce Overfitting With Dropout:**
# 
# Dropout is a clever regularization method that reduces overfitting of the training dataset and makes the model more robust.
# 
# This is achieved during training, where some number of layer outputs are randomly ignored or “dropped out.” This has the effect of making the layer look like – and be treated like – a layer with a different number of nodes and connectivity to the prior layer.
# 
# Dropout has the effect of making the training process noisy, forcing nodes within a layer to probabilistically take on more or less responsibility for the inputs.
# 
# You can add dropout to your models as a new layer prior to the layer that you want to have input connections dropped-out.
# 
# This involves adding a layer called Dropout() that takes an argument that specifies the probability that each output from the previous to drop. E.g. 0.4 means 40% percent of inputs will be dropped each update to the model.
# 
# **How to Accelerate Training With Batch Normalization**
# 
# The scale and distribution of inputs to a layer can greatly impact how easy or quickly that layer can be trained.
# 
# This is generally why it is a good idea to scale input data prior to modeling it with a neural network model.
# 
# Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks.
# 
# You can use batch normalization in your network by adding a batch normalization layer prior to the layer that you wish to have standardized inputs. You can use batch normalization with MLP, CNN, and RNN models.
# 
# **How to Halt Training at the Right Time With Early Stopping**
# 
# Neural networks are challenging to train. Too little training and the model is underfit; too much training and the model overfits the training dataset. Both cases result in a model that is less effective than it could be.
# 
# One approach to solving this problem is to use **early stopping**. This involves monitoring the loss on the training dataset and a validation dataset (a subset of the training set not used to fit the model). As soon as loss for the validation set starts to show signs of overfitting, the training process can be stopped.
# 
# Early stopping can be used with your model by first ensuring that you have a validation dataset. You can define the validation dataset manually via the validation_data argument to the fit() function, or you can use the validation_split and specify the amount of the training dataset to hold back for validation.
# 
# You can then define an EarlyStopping and instruct it on which performance measure to monitor, such as ‘val_loss‘ for loss on the validation dataset, and the number of epochs to observed overfitting before taking action, e.g. 
# 
# This configured EarlyStopping callback can then be provided to the fit() function via the “callbacks” argument that takes a list of callbacks.
# 
# This allows you to set the number of epochs to a large number and be confident that training will end as soon as the model starts overfitting. You might also like to create a learning curve to discover more insights into the learning dynamics of the run and when training was halted.
# 
# # Hardware config

# In[ ]:


###############################################################################
#                         Hardware configurations                             #
###############################################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)


# In[ ]:


from tensorflow.python.client import device_lib

device_lib.list_local_devices()


# In[ ]:


tf.test.is_gpu_available()


# In[ ]:


def get_hardware_strategy():
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        tf.config.optimizer.set_jit(True)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    return tpu, strategy

tpu, strategy = get_hardware_strategy()


# In[ ]:


# Detect hardware, return appropriate distribution strategy
print(tf.version.VERSION)
print("REPLICAS: ", strategy.num_replicas_in_sync)


# ## Define/compile the model 

# In[ ]:


HIDDEN_LAYERS = [800, 600,400, 200, 100]
DROPOUT = 0.05 
ACTIVATION = 'swish' # swish mish relu selu ;swish overfit more cause of narrow global minimun
KERNEL_INIT = "glorot_normal" # Minimal impact, but give your init the right foot forward glorot_uniform lecun_normal


# # Transformer :
# 
# All you need to know about the state of the art Transformer Neural Network Architecture, adapted to Time Series Tasks. Keras code included.
# ## Introduction
# Attention Is All You Need they said. Is it a more robust convolution? Is it just a hack to squeeze more learning capacity out of fewer parameters? Is it supposed to be sparse? How did the original authors come up with this architecture?
# 
# - It’s better than RNNs because it’s not recurrent and can use previous time step features without a loss in detail
# 
# - It’s the top performer architecture on plethera of tasks, including but not limited to: NLP, Vision, Regression (it scales)
# 
# - It is pretty easy to switch from an existing RNN model to the Attention architecture. Inputs are of the same shape!
# 
# ## Preprocessing
# Using Transformers for Time Series Tasks is different than using them for NLP or Computer Vision. We neither tokenize data, nor cut them into 16x16 image chunks. Instead, we follow a more classic / old school way of preparing data for training.
# 
# One thing that is definitely true is that we have to feed data in the same value range as input, to eliminate bias. This is typically on the [0, 1] or [-1, 1] range. In general, it is recommended to apply the same kind of preprocessing pipeline on all of your input features to eliminate this bias. Individual use cases may be exempt from this, different models and data are unique! Think about the origin of your data for a moment.
# 
# Popular time series preprocessing techniques include:
# 
# * Just scaling to [0, 1] or [-1, 1]
# * Standard Scaling (removing mean, dividing by standard deviation)
# * Power Transforming (using a power function to push the data to a more normal distribution, typically used on skewed data / where outliers are present)
# * Outlier Removal
# * Pairwise Diffing or Calculating Percentage Differences
# * Seasonal Decomposition (trying to make the time series stationary)
# * Engineering More Features (automated feature extractors, bucketing to percentiles, etc)
# * Resampling in the time dimension
# * Resampling in a feature dimension (instead of using the time interval, use a predicate on a feature to re-arrange your time steps — for example when recorded quantity exceeds N units)
# * Rolling Values
# * Aggregations
# * Combinations of these techniques
# 
# Again, preprocessing decisions are tightly coupled to the problem and data at hand, but this is a nice list to get your started.
# If your time series can become stationary by doing preprocessing such as seasonal decomposition, you could get good quality predictions by using smaller models (that also get trained way faster and require less code and effort), such as NeuralProphet or Tensorflow Probability.
# 
# Deep Neural Networks can learn linear and periodic components on their own, during training (we will use Time 2 Vec later). That said, I would advise against seasonal decomposition as a preprocessing step.
# Other decisions such as calculating aggregates and pairwise differences, depend on the nature of your data, and what you want to predict.
# 
# Treating sequence length as a hyper parameter, this leads us to an input tensor shape that is similar to RNNs: (batch size, sequence length, features) .
# ## Architecture
# We are going to use Multi-Head Self-Attention (setting Q, K and V to depend on the input through different dense layers/matrices). The next part is optional and depends on the scale of your model and data, but we are also going to ditch the decoder part completely. This means, that we are only going to use one or more attention block layers.
# In the last part, we are going to use a few (one or more) Dense layers to predict whatever we want to predict.
# ![image.png](attachment:ef489c86-e37e-410f-97b0-0d39a1f06218.png)
# 
# Each Attention Block consists of Self Attention, Layer Normalizations and a Feed — Forward Block. The input dimensions of each block are equal to it’s output dimensions.
# 
# ## Bag Of Tricks
# Things to consider when using Transformers and Attention, to get the most out of your model.
# * Start Small: 
# Don’t go crazy with hyperparameters. Start with a single, humble attention layer, a couple of heads and a low dimension. Observe results and adjust hyper parameters accordingly — don’t overfit! Scale your model along with your data. Nevertheless, nothing is stopping you from scheduling a huge hyperparameter search job :).
# * Learning Rate Warmup
# 
# A crucial part of the attention mechanism that leads to greater stability is learning-rate warmup. Start with a small learnign rate and gradually increase it till you reach the base one, then decrease again. You can go crazy with exponential — decaying schedules and sophisticated formulas
# * Use Adam (or variants)
# Non-accelerated gradient descent optimization methods do not work well with Transformers. Adam is a good initial optimizer choice to train with. Keep an eye out for newer (and possibly better) optimization techniques like AdamW or NovoGrad!
# 
# [ref](http://jalammar.github.io/illustrated-transformer/) 
# ##  Define the Model
# ### Utils

# In[ ]:


#feat_dim = X.shape[-1] + 32
feat_dim = X.shape[-1]
embed_dim = X.shape[-1]  # Embedding size for attention
num_heads = 8  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer
dropout_rate = 0.1
num_blocks = 12


# In[ ]:


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="gelu"), tf.keras.layers.Dense(feat_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# In[ ]:


def Transformer(x_pre):
    with strategy.scope():
        inputs = tf.keras.layers.Input(shape=(x_pre.shape[-2:]))
        
        # "EMBEDDING LAYER"
        x = tf.keras.layers.Dense(feat_dim)(inputs)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # TRANSFORMER BLOCKS
        for k in range(num_blocks):
            x_old = x
            transformer_block = TransformerBlock(embed_dim, feat_dim, num_heads, ff_dim, dropout_rate)
            x = transformer_block(x)
            x = 0.7*x + 0.3*x_old # SKIP CONNECTION

        # Classification HEAD
        out1=x
        #out1 = tf.keras.layers.BatchNormalization()(out1)
        for _ in range(5):

            out = tf.keras.layers.Conv1D(filters=64, kernel_size=6, strides=1, padding='same')(out1)
            out = tf.keras.layers.BatchNormalization()(out)
            out = tf.keras.layers.Activation("relu")(out)
            out = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same')(out)
            out = tf.keras.layers.BatchNormalization()(out)
            out = tf.keras.layers.Concatenate()([out, out1])
            out = tf.keras.layers.Activation("relu")(out)
            out1 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2,padding='same')(out)

        out = tf.keras.layers.Flatten()(out1)
        out = tf.keras.layers.Dense(128)(out)
        out = tf.keras.layers.Activation("selu")(out)
        outputs = layers.Dense(60, activation="sigmoid")(out)
           
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='adam', loss='binary_crossentropy', 
                                #metrics=['accuracy',F1])
                                 metrics=[AUC(name='auc')])
    return model 


# In[ ]:


###############################################################################
#                         Define and compile model                            #
###############################################################################
#model_reg=generate_baseline2()
simple_basic_dl =Transformer(X)
#model_reg.summary()
simple_basic_dl.summary()


# In[ ]:


###############################################################################
#                         Visualization                                       #
###############################################################################
#tf.keras.utils.plot_model(model=simple_basic_dl, show_shapes=True, dpi=76, )


# ## Fitting concept

# In[ ]:


###############################################################################
#                        Learning Rate Scheduler                              #
###############################################################################
def get_scheduler(monitor='val_loss'):
    if Config.scheduler == "custom-v1":
        def custom_scheduler(epoch):
            x = Config.lr
            if epoch >= 125: x = 0.0007
            if epoch >= 185: x = 0.0004
            if epoch >= 250: x = 0.0003
            if epoch >= 275: x = 0.0002
            if epoch >= 290: x = 0.00015
            if epoch >= 305: x = 0.0001
            if epoch >= 320: x = 0.000075
            if epoch >= 325: x = 0.00006
            if epoch >= 330: x = 0.00004
            if epoch >= 330: x = 0.00003
            if epoch >= 340: x = 0.00002
            if epoch >= 345: x = 0.00001
            return x
        
        # plot steps
        plt.plot([custom_scheduler(i) for i in range(Config.epochs)])
        plt.show()
        scheduler = tf.keras.callbacks.LearningRateScheduler(custom_scheduler, verbose=1)
    elif Config.scheduler == "custom-v2":
        def get_lr_callback(epoch,lr):
            lr_start   = 0.00001
            lr_max     = 0.01#0.00000125 * 1 * batch_size
            lr_min     = 0.0001
            lr_ramp_ep = 2
            lr_sus_ep  = 2
            lr_decay   = 0.8

            def lrfn(epoch):
                if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
                elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
                else:
                    lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
                return lr
            return lrfn(epoch)

        # plot steps
        scheduler = tf.keras.callbacks.LearningRateScheduler(get_lr_callback, verbose=True)
        x = [x for x in range(1000)]
        plt.plot(x,[get_lr_callback(x,.1) for x in x])
        plt.show()
    elif Config.scheduler == "ReduceLROnPlateau":
        scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, 
            factor=0.7,
            patience=4,
            min_lr=1e-7, 
            verbose=1
                     )
    elif Config.scheduler == "custom-v3":
        # set learning rate scheduler
        # we can chage learning rate during learning
        def lr_schedul(epoch):
            x = 0.0001
            if epoch >= 20:
                x= 0.00001
            if epoch >= 40:
                x = 0.00001
            if epoch >= 60:
                x = 0.000001
            if epoch >= 80:
                x = 0.0000001
            if epoch >= 100:
                x = 0.00000001
            if epoch >= 120:
                x = 0.000000001        
            return x

            def lrfn(epoch):
                if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
                elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
                else:
                    lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
                return lr
            return lrfn(epoch)

        # plot steps
        scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedul, verbose=True)
           
    elif Config.scheduler == "CosineDecayRestarts":
        cisine_decay_r = tf.keras.experimental.CosineDecayRestarts(
            Config.lr,
            first_decay_steps=Config.epochs // 100,
            t_mul=1,
            m_mul=1,
            alpha=0.01
            )
        
        # plot steps
        plt.plot([cisine_decay_r(i) for i in range(Config.epochs)])
        plt.show()
        scheduler = tf.keras.callbacks.LearningRateScheduler(cisine_decay_r, verbose=1)
    else:
        raise NotImplementedError
    
    return scheduler


# In[ ]:


###############################################################################
#                       configure early stopping                              #
###############################################################################
es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=10,
                   min_delta=0.000000001,
                   restore_best_weights=True)
###############################################################################
#                       Save Models                                          #
###############################################################################
#filepath = 'my_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
# define the checkpoint
#filepath = "model.hdf5"
#checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,  monitor='val_auc', verbose=1,  save_best_only=True, save_weights_only=True,mode='max')


# # StratifiedGroupKFold  Evaluation 

# In[ ]:


class colors:
    # You may need to change color settings
    RED = '\033[31m'
    ENDC = '\033[m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'


# In[ ]:


get_ipython().run_cell_magic('time', '', '###############################################################################\n#                         CV                                                 #\n###############################################################################\n# flatten input\n#n_input = X.shape[1] * X.shape[2]\n#X = X.reshape((X.shape[0], n_input))\n# flatten test data\n#n_input = test_data.shape[1] * test_data.shape[2]\n#test_data = test_data.reshape((test_data.shape[0], n_input))')


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


len(groups.unique())


# In[ ]:


get_ipython().run_cell_magic('time', '', '###############################################################################\n#                         CV                                                 #\n###############################################################################\nfrom sklearn  import metrics\n# evaluate each strategy on the dataset\nresults = list()\n# Setting a 10-fold stratified cross-validation (note: shuffle=True)\nseed = 2022\nset_seed(seed)\nACC_VAL_METRICS = \'val_auc\'\nACC_METRICS=\'auc\'\nSEED = 2022\nFOLDS =12\nEPOCHS =1000\nskf = GroupKFold(\n        n_splits=Config.n_fold, \n        #shuffle=True,\n        #random_state=Config.seed\n)\n# CV interations\n# Create arrays for the features and the response variable\nroc_auc = list()\naverage_precision = list()\noof = np.empty((X.shape[0],))\npredictions=[]\nf_scores = []\nmean_auc = 0\nmean_ap=0\nF1 = list()\nRecall=list()\nbest_iteration = list()\nfor fold, (train_idx, test_idx) in enumerate(skf.split(X, y,groups.unique())):    \n    X_train, y_train =X[list(train_idx),:], y[list(train_idx)]\n    X_test, y_test = X[list(test_idx),:],y[list(test_idx)]\n    # fit model using our gpu\n    filepath = f\'my_best_model{fold}.hdf5\'\n    # define the checkpoint\n    #filepath = "model.hdf5"\n    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, \n                             monitor=\'val_auc\',\n                             verbose=1, \n                             save_best_only=True, \n                             save_weights_only=True,\n                             mode=\'max\')\n \n    history = simple_basic_dl.fit(X_train,y_train,\n                                           #y_train,\n                                           validation_data=(X_test, y_test),\n                                           batch_size=64,epochs=EPOCHS, \n                                           verbose=1 ,\n                                           shuffle=True,\n                                           #callbacks=[lr_callback ,es,checkpoint])\n                                           #callbacks=[lr_decay ,es,checkpoint])\n                                           callbacks=[get_scheduler(\'val_loss\') ,es,checkpoint])\n                                           # callbacks=[lr_decay ,es])\n                                           #callbacks=[plateau,es,checkpoint])\n\n    #  ----------------------------------------------------------\n    #  Saving scores to plot the end\n    scores = pd.DataFrame(history.history)\n    scores[\'folds\'] = fold\n    if fold == 0:\n        f_scores = scores\n        #model.summary()\n #         keras.utils.plot_model(model, show_shapes=True, rankdir="LR")\n    else:\n        f_scores = pd.concat([f_scores, scores], axis=0)\n\n    #  ----------------------------------------------------------\n    #plot_history(history)\n    preds = simple_basic_dl.predict(X_test)\n    auc_score= roc_auc_score(y_true=y_test, y_score=preds.squeeze())\n    roc_auc.append(auc_score)\n    ap=average_precision_score(y_true=y_test, y_score=preds.squeeze())\n    print(\'-\' * 80) \n    plot_roc(y_test.squeeze().reshape(-1, 1).squeeze(),preds.squeeze().reshape(-1, 1).squeeze())\n    mean_auc += auc_score / FOLDS\n    mean_ap+=ap/FOLDS\n    predictions.append(simple_basic_dl.predict(test_data)) \n    print(f"Fold {fold} | AUC: {colors.GREEN}{auc_score}{colors.ENDC}")\n    print(f"Fold {fold} | average_precision_score:{colors.GREEN} {ap}{colors.ENDC}")\n    roc_auc.append(auc_score)\n    gc.collect()\nprint(\'-\' * 80)  \nprint(f"\\nOverall mean AUC score : {colors.RED}{mean_auc}{colors.ENDC}")\nprint(f"\\nOverall mean average_precision_score :{colors.RED} {mean_ap}{colors.ENDC}")')


# In[ ]:


plot_acc(f_scores)


# In[ ]:


predictions = np.mean(np.column_stack(predictions), axis=1)
#predictions = model.predict(x_test_pre)
print(len(predictions))
predictions.shape


# In[ ]:


predictions


# In[ ]:


submission.shape


# In[ ]:


#submission[TARGET] = np.mean(y_preds, axis=0)
submission[TARGET] = predictions
submission.to_csv(SUBMISSION_FILE, index=True)
submission


# # Summuray 
# 
# We tried  differents tecthniques in order to have better results 
# As is always the case with neural networks, there are a huge number of possible architectures and parameters to play around with. Possible interesting avenues include:
# 
#     adding / removing / widening / shortening layers in the network
#     adjusting the regularization layers / parameters (dropout / l2)
#     adding new regularization parameters
#     changing activation functions
#     changing the gradient descent optimizer
#     adjusting the learning rate and no of epochs
#     Other things to try:
#     Add lr scheduler.✔
#     Add early stoppping.✔
#     Add evaluation plots and confusion matrices.
#     Remove batch norms.✔
#     Try different number of folds.✔
#     Increase number of epochs.✔
#     Try different sizes of hidden layers.✔
#     Try drop out and add regularization. ✔
#     Add voting classifier.✔
#     Features engineering.✔
#     Embeddings.✔
#     etc, etc.....
# 
# With some tinkering, you'll find that the networks performance can quickly approach that of the aforementioned popular models. Neural networks like this also offer a huge amount of flexibility which can offer some really promising avenues for improving your scores.
# 
# **Note: you can dramatically improve the training time of the model by increasing the batch size and utilizing a GPU session!**
# 
# reference : 
# 
# https://www.kaggle.com/code/faressayah/tensorflow-2-tutorial-get-started-in-deep-learning/notebook
# 
# to try : 
# 
# advanced arichtecture ; 
# 
# https://www.kaggle.com/code/teckmengwong/dcnv2-softmaxclassification
# 
# https://www.kaggle.com/code/teckmengwong/dcnv2-softmaxclassification/comments
# 
# https://www.kaggle.com/code/mlanhenke/tps-12-deep-cross-nn-keras/notebook
# 
# https://keras.io/examples/structured_data/wide_deep_cross_networks/
# 
# https://www.tensorflow.org/recommenders/examples/dcn
# 
# https://www.kaggle.com/code/mlanhenke/tps-12-g-res-variable-selection-nn-keras
# 
# https://www.kaggle.com/code/pourchot/tps-12-simple-nn-with-skip-connection
# 
# https://www.kaggle.com/code/mlanhenke/tps-12-deep-cross-nn-keras/notebook
# 
# deep + kfold : 
# 
# https://www.kaggle.com/code/gulshanmishra/tps-dec-21-tensorflow-nn-feature-engineering
# 
# 
# https://www.kaggle.com/lucamassaron/deep-learning-for-tabular-data
# 
# 
# 
# https://mmuratarat.github.io/2019-06-12/embeddings-with-numeric-variables-Keras
# 
# https://www.slideshare.net/MeetupDataScienceRoma/deep-learning-for-tabular-data-luca-massaron
# 
# https://medium.com/analytics-vidhya/tensorflow-2-tutorial-on-categorical-features-embedding-93dd81027ea9
# 
# https://www.kaggle.com/mtinti/keras-starter-with-bagging-1111-84364
# 
# https://www.kaggle.com/faressayah/tensorflow-2-tutorial-get-started-in-deep-learning
# 
# https://www.kaggle.com/colinmorris/embedding-layers
# 
# https://www.kaggle.com/dustyturner/dense-nn-with-categorical-embeddings
# 
# wrap keras regressor : 
# 
# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# 
# 
# 
# 
# 
# cat embedding : 
# 
# https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/
# 
# 
# https://www.kaggle.com/lucamassaron/deep-learning-for-tabular-data
# 
# https://www.kaggle.com/lukaszborecki/tps-09-nn
# 
# https://www.kaggle.com/siavrez/kerasembeddings
# 
# https://www.kaggle.com/code/mst8823/19th-place-best-single-model-resbilstm
# 
# incep:
# https://towardsdatascience.com/deep-learning-for-time-series-classification-inceptiontime-245703f422db
# 
# Things to try : 
# 
# https://www.kaggle.com/datafan07/top-1-approach-eda-new-models-and-stacking
# 
# 
# Deep  : 
# https://www.kaggle.com/shivansh002/tame-your-neural-network-once-for-all
# 
# https://www.kaggle.com/lukaszborecki/tps-09-nn/
# 
# https://www.kaggle.com/bannourchaker/10-deeplearning-embedding-rnn-tf-keras
# 
# https://www.kaggle.com/bannourchaker/single-nn/edit
# 
# https://www.kaggle.com/bannourchaker/deep-learning-starter-gpu
# 
# learing rate adaptative vs scheulde : 
# 
# https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
# 
# Best mine : 
# 
# 
# https://www.kaggle.com/deepchaka/deep-learning-ventilator-time-series-regress/edit/run/78653879
# 
# https://www.kaggle.com/deepchaka/deeplearning2/edit/run/78470709
# 
# 
# https://www.kaggle.com/deepchaka/deeplearning2/edit/run/78470709https://www.kaggle.com/deepchaka/deeplearning2/edit/run/78470709
# 
# https://www.kaggle.com/deepchaka/deep-lstm-test-kaggle-quantile-cat-freduced-tpu/edit/run/78738587
# 
# https://www.kaggle.com/deepchaka/deep-lstm-test-kaggle-quantile-cat-freduced-tpu/edit/run/78738587
# 
# https://www.kaggle.com/bannourchaker/deep-learning-starter-gpu
#     
# https://www.kaggle.com/bannourchaker/10-deeplearning-embedding-rnn-tf-keras/edit/run/74137751    
# 
# 
# 
# 
