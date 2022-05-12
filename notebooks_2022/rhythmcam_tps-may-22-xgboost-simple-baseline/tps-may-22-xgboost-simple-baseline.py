#!/usr/bin/env python
# coding: utf-8

# # Define Data

# In[ ]:


# imports 
import numpy as np
import pandas as pd 
import os,random,gc
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler,MaxAbsScaler,RobustScaler

from xgboost import XGBClassifier

# variables
TRAIN_PATH = "../input/tabular-playground-series-may-2022/train.csv"
TEST_PATH = "../input/tabular-playground-series-may-2022/test.csv"
SAMPLE_SUBMISSION_PATH = "../input/tabular-playground-series-may-2022/sample_submission.csv"
SUBMISSION_PATH = "submission.csv"

ID = "id"
TARGET = "target"

SEED = 2022
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything()


# # Preprocess Data

# In[ ]:


def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train = reduce_memory_usage(train)
# train = train.sample(10000)
gc.collect()
test = reduce_memory_usage(test)
gc.collect()

print(train.describe(include="O"))

train_len = len(train)

train_test = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

str_col_list = train.describe(include="O").columns.tolist()
for col in str_col_list:
    def uniqueLength(row):
        return len( list( set( row[col] ) ) )

    train_test[col + "_num"] = train_test.apply(uniqueLength, axis=1)

for i in range(10):
    train_test[f"f_27_ch_ord{i}"] = train_test["f_27"].str.get(i).apply(ord) - ord('A')
    train_test[f"f_27_ch{i}"] = train_test["f_27"].str.get(i)


# In[ ]:


train_test.head()


# In[ ]:


train_test.groupby("f_27_ch0")[TARGET].mean()


# In[ ]:


for i in range(10):
    meanData = train_test.groupby(f"f_27_ch{i}")[TARGET].mean()
    train_test[f"f_27_ch{i}Mean"] = train_test[f"f_27_ch{i}"].map(meanData)
    train_test = train_test.drop(f"f_27_ch{i}",axis=1)
    

train = train_test[:train_len]
test = train_test[train_len:]
test.drop(labels=[TARGET],axis = 1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.columns


# # Build Model

# In[ ]:


y = train[TARGET]
X = train.drop([ID,TARGET,"f_27"],axis=1)
X_test = test.drop([ID,"f_27"],axis=1)

MODEL_TREE_METHOD = 'gpu_hist'
MODEL_EVAL_METRIC = "auc"

model = XGBClassifier(tree_method=MODEL_TREE_METHOD,
                      eval_metric=MODEL_EVAL_METRIC,
                      learning_rate=0.09399,
                      max_depth=16,
                      
                     ) 
model.fit(X, y)


# # Evaluate Model

# In[ ]:


from sklearn.metrics import roc_auc_score
pred_y = model.predict_proba(X)[:, 1]
roc_auc_score(y, pred_y)


# # Predict Data

# In[ ]:


pred_test = model.predict_proba(X_test)[:, 1]
print(pred_test[:5])

sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
sub[TARGET] = pred_test
sub.to_csv(SUBMISSION_PATH, index=False)
sub.head()

