#!/usr/bin/env python
# coding: utf-8

# # To Do 
# 
# #### Feature engineering: 
# * Relative features (float cols) ----Completed outside kaggle [Notebook showing process](https://www.kaggle.com/code/slythe/feature-creation-selection-feature-engine/edit)
# * Grouped Mathematical features (float cols)
# 
# * Categorical columns (Int columns)
# 
# * Text Features - Done 
# 
# * Hyper parameter tuning

# # 📩 Import Libraries 📩 

# In[ ]:


# Data and visualization
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import Counter

# hyperparameter tuning 
import optuna 

#modelling
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from sklearn.calibration import calibration_curve, CalibratedClassifierCV


# In[ ]:


# parameters 
sns.set_theme()

CALIBRATION = False
EPOCHS = 5000

OPTUNA = False


# # 💾 Load Data 💾

# In[ ]:


train_original = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv",index_col = 0)
test_original = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv",index_col = 0)

# train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv",index_col = 0)
# test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv",index_col = 0)
sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv",index_col = 0)


# # 🌟 Feature Engineering 🌟

# * Unicode (ord) code adapted from [cabaxiom](https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model#Feature-Engineering)
# 
# cabaxiom already identified f_29 and f_30 as potential categorical columns. 
# Lets try improve on this

# In[ ]:


int_cols = train_original.dtypes[(train_original.dtypes =="int64") & (train_original.dtypes.index != "target") ].index
float_cols = train_original.dtypes[train_original.dtypes =="float64" ].index


# In[ ]:


all_letters = ['A', 'B', 'D', 'E', 'P', 'C', 'S', 'G', 'F', 'Q', 'H', 'N', 'K', 'R', 'M', 'T', 'O', 'J', 'I', 'L']

def feature_engineering(df):
    #letter count 
#     for letter in all_letters: 
#         #Count letter
#         df[letter] = df["f_27"].str.count(letter)
#         #Contains letter
#         df[f"contains_{letter}"]  = df["f_27"].str.contains(letter).astype("category")
    
    #Unicoding
    for i in range(10):
        df["f_27_"+str(i)] = df["f_27"].str[i].apply(lambda x: ord(x) - ord("A"))
    
    # Get Unique letters
    df["unique_text_str"] = df["f_27"].apply(lambda x :  ''.join([str(n) for n in list(set(x))]) )
    df["unique_text_str"] = df["unique_text_str"].astype("category")

    df["unique_text_len"] = df.f_27.apply(lambda s: len(set(s)))
    
#     #Merge categorical columns 
#     df["f29_f30"] = df[["f_29","f_30"]].apply(lambda x: str( x["f_29"] ) + str(x["f_30"]), axis =1) 
#     df["f29_f30"] = df["f29_f30"].astype("category")
     
#     # get max and min letter (use 'Counter' to get count of letters and then get max/min from this dictionary )
#     df["max_letter"] = df["f_27"].apply(lambda x : Counter(x)).apply(lambda x : max(x, key=x.get))
#     df["max_letter"] = df["max_letter"].astype("category")
#     df["min_letter"] = df["f_27"].apply(lambda x : Counter(x)).apply(lambda x : min(x, key=x.get))
#     df["min_letter"] = df["min_letter"].astype("category")
    
    return df

train = feature_engineering(train_original)
test = feature_engineering(test_original)


# ## Mathematical Features 
# * We will do this with certain columns i.e. the float columns (but certain groupings)
# * We need a list of functions and/or function names, e.g. [np.sum, ‘mean’] 

# In[ ]:


train_original[float_cols].describe()


# #### Group Float columns 
# * We can see from the above that certain columns have similar std/ min/ max, we will group them
# * f_00 to f_06 => Group1
# * f_19 to f_26 => Group2
# * f28 looks to be seperate from both groups

# In[ ]:


group1_float =['f_00','f_01','f_02','f_03','f_04','f_05','f_06']
group2_float = ['f_19','f_20','f_21','f_22','f_23','f_24','f_25','f_26']

def mathematical_feats(df,cols, suffix):
    df[f"sum_{suffix}"] = df[cols].sum(axis = 1)
    df[f"mean_{suffix}"] = df[cols].mean(axis = 1)
    df[f"std_{suffix}"] = df[cols].std(axis = 1)
    df[f"min_{suffix}"] = df[cols].min(axis = 1)
    df[f"max_{suffix}"] = df[cols].max(axis = 1)
    df[f"median_{suffix}"] = df[cols].median(axis = 1)
    df[f"mad_{suffix}"] = df[cols].mad(axis = 1)

    #potentially change periods OR changes axis OR fillna with actuals
    #df[f"diff_{suffix}"] = df[cols].diff(periods=1, axis = 1)
    
    df[f"max-min_{suffix}"] = df[cols].max(axis = 1) - df[cols].min(axis = 1)
    df[f"q01_{suffix}"] = df[cols].quantile(q= 0.1, axis =1)
    df[f"q25_{suffix}"] = df[cols].quantile(q= 0.25, axis =1) 
    #df[f"q50_{suffix}"] = df[cols].quantile(q= 0.5, axis =1) 
    df[f"q75_{suffix}"] = df[cols].quantile(q= 0.75, axis =1) 
    df[f"q95_{suffix}"] = df[cols].quantile(q= 0.95, axis =1) 
    df[f"q99_{suffix}"] = df[cols].quantile(q= 0.99, axis =1)
    df[f"kurt_{suffix}"] = df[cols].kurt(axis =1) 
    df[f"skew_{suffix}"] = df[cols].skew( axis =1)
    
    return df

mathematical_feats(train, float_cols, "group2_float")
mathematical_feats(test, float_cols, "group2_float")
# mathematical_feats(train, float_cols, "group1_float")
# mathematical_feats(test, float_cols, "group1_float")


# ## Drop unimportant features 
# From previous runs 

# In[ ]:


feats = ['q50_group1_float', 'mean_group1_float']

def drop_feats(df, feats):
    df.drop(feats ,axis = 1 ,inplace = True )
    return df 

# drop_feats(train, feats)
# drop_feats(test, feats)


# # 🚀 Base Model 🚀

# In[ ]:


categorical_features = ["unique_text_str"
                        #, "f29_f30"
                        #,"min_letter"
                        #,"max_letter"
                        #,"f_29","f_30"
                       ]

categorical_features.extend( [col for col in train.columns if "contains" in col] ) 


# In[ ]:


# drop the text column as we already have features created earlier
X = train.drop(["target","f_27"],axis =1)
y= train["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)


# # Optuna - Hyperparameter tuning 

# In[ ]:


def objective(trial):
    # 2. Suggest values of the hyperparameters using a trial object.
    lgb_params = {
        'objective': 'binary',
        'metric': "auc",
        'verbosity': -1,
        'num_iterations': EPOCHS,
        "num_threads": -1,
        #"force_col_wise": True,
        "learning_rate": trial.suggest_float('learning_rate',0.01,0.2),
        'boosting_type': trial.suggest_categorical('boosting',["gbdt"]),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 256),
        #'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1000, 10000),
        'max_depth': trial.suggest_int('max_depth', -1,15)
    }
        
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial,metric = "auc")  
    
    train_set = lgb.Dataset(X_train, y_train)
    valid_set = lgb.Dataset(X_test, y_test)

    model = lgb.train(params=lgb_params,
                          train_set= train_set, 
                          valid_sets= [valid_set], 
                          num_boost_round= EPOCHS,
                          callbacks=[lgb.early_stopping(30),pruning_callback] ,categorical_feature = categorical_features ) 

    val_preds = model.predict(X_test)
    auc = roc_auc_score(y_test, val_preds)
    print("Val AUC:", auc)
    
    return auc


# In[ ]:


if OPTUNA:
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)

    trial = study.best_trial
    best_params = study.best_params

    #Print our results
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    print(" AUC Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# ## Base model

# In[ ]:


model = lgb.LGBMClassifier(
    objective= 'binary',
    metric= "auc",
    num_iterations = EPOCHS,
    num_threads= -1,
    learning_rate= 0.18319492258552644,
    boosting= 'gbdt',
    lambda_l1= 0.00028648667113792726,
    lambda_l2= 0.00026863027834978876,
    num_leaves= 229,
    max_depth= 0,
    min_child_samples=80,
    max_bins=511, 
    random_state=42 
)

model.fit(X_train,y_train, eval_set=[(X_test,y_test)], callbacks = [lgb.early_stopping(30)],eval_metric="auc" , 
          categorical_feature = categorical_features
         )

val_preds = model.predict_proba(X_test)
y_preds = model.predict_proba(X_train)

print("Intrinsic AUC:", roc_auc_score(y_train, y_preds[:,1]))
print("Validation AUC:", roc_auc_score(y_test, val_preds[:, 1] ))


# In[ ]:


feat_importance = pd.DataFrame(data = model.feature_importances_, index= train.drop(["target","f_27"],axis =1).columns).sort_values(ascending = False, by= [0] )

plt.figure(figsize= (25,10))
sns.barplot(y= feat_importance[0], x= feat_importance.index)
plt.xticks(rotation = 90) 
plt.show()


# In[ ]:


feat_importance[feat_importance[0] <50].index


# ## Calibration 
# Taken from last months kernel [TPS April ](https://www.kaggle.com/code/slythe/calibrated-xgboost-human-activity-recognition)

# In[ ]:


prob_true, prob_pred = calibration_curve(y_test, val_preds[:,1], n_bins=10)


# In[ ]:


calibrator = CalibratedClassifierCV(model, method = "isotonic", cv='prefit')
calibrator.fit(X_test, y_test)
cal_preds = calibrator.predict_proba(X_test)

print("Validation AUC:" , roc_auc_score(y_test, val_preds[:, 1] ))
print("Calibrated AUC:" , roc_auc_score(y_test, cal_preds[:, 1] ))


# In[ ]:


fig, ax = plt.subplots(figsize=(20,10))
plt.plot(prob_pred,prob_true, marker='o', linewidth=1, label='xgb model probabilities')

# reference line
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
#plt.axvline(x=0.2, color = "r")
fig.suptitle('Calibration plot')
ax.set_xlabel('Predicted probability (mean)')
ax.set_ylabel('Fraction of positives (%True  in each bin)')
plt.legend()
plt.show()


# # ❎ Cross validation ❎

# In[ ]:


cv = KFold(n_splits = 5, shuffle = True,random_state=42)


# In[ ]:


preds = []
auc_cv = []
for fold, (idx_train, idx_val) in enumerate(cv.split(X,y)):
    print("\n")
    print("#"*10, f"Fold: {fold}","#"*10)
    X_train , X_test = X.iloc[idx_train] , X.iloc[idx_val]
    y_train , y_test = y[idx_train] , y[idx_val]
    
    model = lgb.LGBMClassifier(
    objective= 'binary',
    metric= "auc",
    num_iterations = EPOCHS,
    num_threads= -1,
    learning_rate= 0.18319492258552644,
    boosting= 'gbdt',
    lambda_l1= 0.00028648667113792726,
    lambda_l2= 0.00026863027834978876,
    num_leaves= 229,
    max_depth= 0,
    min_child_samples=80,
    max_bins=511, 
    random_state=42 )
    
    model.fit(X_train,y_train, eval_set=[(X_test,y_test)], callbacks = [lgb.early_stopping(30)],eval_metric="auc")
    
    if CALIBRATION:
        calibrator = CalibratedClassifierCV(model, method = "isotonic", cv='prefit')
        calibrator.fit(X_test, y_test)
        auc = roc_auc_score(y_test, calibrator.predict_proba(X_test)[:, 1])
        print("\n Calibration AUC:" , auc)
        preds.append(calibrator.predict_proba(test.drop("f_27",axis =1))[:, 1])
    else:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print("\n Validation AUC:" , auc)
        preds.append(model.predict_proba(test.drop("f_27",axis =1))[:, 1])
        
    auc_cv.append(auc)
    
print("FINAL AUC: ", np.mean(auc_cv))


# # 📡 Submission 📡

# In[ ]:


sub["target"] = np.array(preds).mean(axis =0)
sub.to_csv("submission.csv")
sub


# In[ ]:


sub.plot(kind= "hist",figsize= (25,8))
plt.show()


# In[ ]:




