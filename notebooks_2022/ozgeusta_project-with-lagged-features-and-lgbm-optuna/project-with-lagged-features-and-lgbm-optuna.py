#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import optuna
from functools import partial


# In[ ]:


df=pd.read_csv('../input/enerjisa-uretim-hackathon/features.csv')
df['Timestamp'] = pd.to_datetime(df["Timestamp"])
df.head()


# In all columns, the max value is 99999. We thought that these values are entered incorrectly. So, we replaced them with Nan.

# In[ ]:


df.describe()


# In[ ]:


for i in list(set(df.columns) - set("Timestamp")):
    df.loc[df[i]==99999,i] = np.nan


# In[ ]:


df_power = pd.read_csv('../input/enerjisa-uretim-hackathon/power.csv')
df_power['Timestamp'] = pd.to_datetime(df_power["Timestamp"])


# **Distribution of the target value:**

# In[ ]:


plt.hist(df_power["Power(kW)"], bins=30)


# **Determination of categorical and numerical columns:**

# In[ ]:


categorical_column = ["Particle Counter"]
numerical_columns = list(set(df.columns) - set(["Timestamp"] + categorical_column))


# Even though the quantiles look much better after removing the 99999 values, there are still some outliers at both the upper and lower bounds. So, we capped the data from 0.01 and 0.99 quantiles.

# In[ ]:


df.describe()


# In[ ]:


for i in numerical_columns:
    df.loc[df[i]<df[i].quantile(0.01),i] = df[i].quantile(0.01)
    df.loc[df[i]>df[i].quantile(0.99),i] = df[i].quantile(0.99)


# After that, we added some lagged data such as before 10 and 20 minutes, average of 30 minutes, percentage changes and increases.

# In[ ]:


df_power1 = df_power.merge(df, on= "Timestamp", how = "left")

for i in numerical_columns:
    df_power1[i+"_10_min_lagged"] = df_power1[i].rolling(1).mean()
    df_power1[i+"_20_min_lagged"] = df_power1[i].shift(1).rolling(1).mean()
    df_power1[i+"_30_min_avg"] = df_power1[i].rolling(3).mean()

for i in numerical_columns :
    df_power1[i+"_10_min_lagged_percentage"] = (df_power1[i+"_10_min_lagged"]-df_power1[i])/df_power1[i+"_10_min_lagged"]*100
    df_power1[i+"_10_min_lagged_increase"] = df_power1[i] - df_power1[i+"_10_min_lagged"]
    df_power1[i+"_20_min_lagged_increase"] = df_power1[i] - df_power1[i+"_20_min_lagged"]
    df_power1[i+"_30_min_avg_percentage"] = (df_power1[i+"_30_min_avg"]-df_power1[i])/df_power1[i+"_30_min_avg"]*100
    df_power1[i+"_30_min_avg_increase"] = df_power1[i] - df_power1[i+"_30_min_avg"]


# ### Model

# In[ ]:


X = df_power1.drop(["Timestamp","Power(kW)"],axis = 1)
y = df_power1[["Power(kW)"]]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[ ]:


lgbm = LGBMRegressor()
lgbm.fit(X_train,y_train,eval_metric="rmse")
y_pred = lgbm.predict(X_test)
print(mean_squared_error(y_train, lgbm.predict(X_train), squared = False))
print(mean_squared_error(y_test, y_pred, squared = False))


# In[ ]:


important_features = pd.Series(data=lgbm.feature_importances_, index=X_train.columns)
important_features.sort_values(ascending=False,inplace=True)
important_features[:30]


# In[ ]:


for i in [30,35,40,45,50,55,60]:  
    lgbm = LGBMRegressor()
    lgbm.fit(X_train[list(important_features[:i].index)],y_train,eval_metric="rmse")
    y_pred = lgbm.predict(X_test[list(important_features[:i].index)])
    print(i)
    print("train:",mean_squared_error(y_train, lgbm.predict(X_train[list(important_features[:i].index)]), squared = False))
    print("test:",mean_squared_error(y_test, y_pred, squared = False))


# In[ ]:


lgbm = LGBMRegressor()
lgbm.fit(X_train[list(important_features[:50].index)],y_train,eval_metric="rmse")
y_pred = lgbm.predict(X_test[list(important_features[:50].index)])
print(mean_squared_error(y_train, lgbm.predict(X_train[list(important_features[:50].index)]), squared = False))
print(mean_squared_error(y_test, y_pred, squared = False))


# In[ ]:


xgb1 = XGBRegressor()
xgb1.fit(X_train[list(important_features[:50].index)],y_train,eval_metric="rmse")
y_pred_1 = xgb1.predict(X_test[list(important_features[:50].index)])
print(mean_squared_error(y_train, xgb1.predict(X_train[list(important_features[:50].index)]), squared = False))
print(mean_squared_error(y_test, y_pred_1, squared = False))


# In[ ]:


df_power2 = df_power1[list(set(df_power1.columns)-set(categorical_column))].fillna(df_power1[list(set(df_power1.columns)-set(categorical_column))].median())
df_power2[categorical_column] = df_power1[categorical_column].fillna(0)
df_power2["Power(kW)"] = df_power1["Power(kW)"]

X_1 = df_power2.drop(["Power(kW)"],axis = 1)
y_1 = df_power2[["Power(kW)"]]

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1,y_1, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators = 100, n_jobs =-1, max_depth = 20) 
rf.fit(X_train_1[list(important_features[:50].index)],y_train_1)
y_pred_2 = rf.predict(X_test_1[list(important_features[:50].index)])


# In[ ]:


def objective(trial):
    
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [100,200]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 100, 400, step=20),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100, step = 10),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6,0.7,0.8,0.9,1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.7,0.8,0.9,1])
    }
    
    model = LGBMRegressor(**params)
        
    model.fit(X_train[list(important_features[:50].index)], y_train, eval_set=[(X_test[list(important_features[:50].index)], y_test)], 
              eval_metric=['rmse'], early_stopping_rounds=100, verbose=0)  
    
    preds = model.predict(X_test[list(important_features[:50].index)])
    test_rmse = mean_squared_error(y_test,preds,squared= False)
    train_rmse = mean_squared_error(y_train,model.predict(X_train[list(important_features[:50].index)]),squared= False)
    
    print("Train score:",train_rmse)
    print("Test score:",test_rmse)
     
    return test_rmse


# In[ ]:


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)


# In[ ]:


print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[ ]:


params_lgbm=study.best_params   
params_lgbm['n_estimators'] = 2000
params_lgbm['metric'] = 'rmse'


# In[ ]:


lgbm = LGBMRegressor(**params_lgbm)
lgbm.fit(X_train[list(important_features[:50].index)],y_train,eval_metric="rmse")
y_pred = lgbm.predict(X_test[list(important_features[:50].index)])
print(mean_squared_error(y_train, lgbm.predict(X_train[list(important_features[:50].index)]), squared = False))
print(mean_squared_error(y_test, y_pred, squared = False))


# In[ ]:


df_models = pd.DataFrame()
 
df_models["XGB"] = y_pred_1
df_models["LGBM"] = y_pred
df_models["RF"] = y_pred_2

df_models["actual"] = y_test[["Power(kW)"]].to_numpy()

df_models.head()


# In[ ]:


X_2 = df_models[['XGB', 'LGBM', 'RF']]
y_2 = df_models[["actual"]]

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2,y_2, test_size=0.2, random_state=42)


# In[ ]:


reg = LinearRegression().fit(X_train_2, y_train_2)
pred = reg.predict(X_test_2)
print(mean_squared_error(y_test_2, pred, squared = False))


# In[ ]:


df_test=pd.read_csv('../input/enerjisa-uretim-hackathon/sample_submission.csv')
df_test['Timestamp'] = pd.to_datetime(df_test["Timestamp"])


# In[ ]:


df_1 = df.copy()

for i in numerical_columns:
    df_1[i+"_10_min_lagged"] = df_1[i].rolling(1).mean()
    df_1[i+"_20_min_lagged"] = df_1[i].shift(1).rolling(1).mean()
    df_1[i+"_30_min_avg"] = df_1[i].rolling(3).mean()
    
for i in numerical_columns :
    df_1[i+"_10_min_lagged_percentage"] = (df_1[i+"_10_min_lagged"]-df_1[i])/df_1[i+"_10_min_lagged"]*100
    df_1[i+"_10_min_lagged_increase"] = df_1[i] - df_1[i+"_10_min_lagged"]
    df_1[i+"_20_min_lagged_increase"] = df_1[i] - df_1[i+"_20_min_lagged"]
    df_1[i+"_30_min_avg_percentage"] = (df_1[i+"_30_min_avg"]-df_1[i])/df_1[i+"_30_min_avg"]*100
    df_1[i+"_30_min_avg_increase"] = df_1[i] - df_1[i+"_30_min_avg"]


# In[ ]:


df_test = df_test.drop(["Power(kW)"], axis = 1).merge(df_1[list(important_features[:50].index)+["Timestamp"]],on="Timestamp",how="left")


# In[ ]:


pred_1 = xgb1.predict(df_test.drop(["Timestamp"], axis =1))
pred_2 = lgbm.predict(df_test.drop(["Timestamp"], axis =1))

df_test = df_test.fillna(df_power1[df_test.columns].median())

pred_3 = rf.predict(df_test.drop(["Timestamp"], axis =1))

df_models = pd.DataFrame()
 
df_models["XGB"] = pred_1
df_models["LGBM"] = pred_2
df_models["RF"] = pred_3

pred_final = reg.predict(df_models)
df_test["Power(kW)"] = pred_final


# In[ ]:


df_test[["Timestamp","Power(kW)"]]

