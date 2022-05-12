#!/usr/bin/env python
# coding: utf-8

# <p style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:10px 10px;font-weight:bold;border:2px solid #673AB7;">Car Rental Demand Forcasting 🚗</p>
# 

# <center><img src= "https://raw.githubusercontent.com/ashwinshetgaonkar/kaggle-kernel-images/main/car-rental-images.jpg" alt ="rental-cars" style='width:600px;'></center><br>

# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Importing Libraries</span>
# 

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook',font_scale=1.25)
from IPython.core.display import HTML,display
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.visualization import plot_param_importances


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Importing Data</span>

# In[ ]:


df=pd.read_csv('../input/analytics-vidhya-hackathon-april-2022/train_E1GspfA.csv')
test_df=pd.read_csv('../input/analytics-vidhya-hackathon-april-2022/test_6QvDdzb.csv')
df.head()


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 210px">Basic EDA</span>

# In[ ]:


# checking the number of rows and columns for training dataset
rows,columns=df.shape[0],df.shape[1]


# In[ ]:


display(HTML(f"<h3>  <b style='color:red;font-size:22px;'>Inference </b>:<ul><li>There are {rows} rows and {columns} columns in the training dataset.</li></ul></h3>"))


# In[ ]:


# checking the number of rows and columns for test dataset
rows,columns=test_df.shape[0],test_df.shape[1]


# In[ ]:


display(HTML(f"<h3>  <b style='color:red;font-size:22px;'>Inference </b>:<ul><li>There are {rows} rows and {columns} columns in the test dataset.</li></ul></h3>"))


# In[ ]:


#checking the data types of the training datset columns
df.dtypes


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Parsing Date Column</span>
# 

# In[ ]:


df['date']=pd.to_datetime(df['date'])
test_df['date']=pd.to_datetime(test_df['date'])


# In[ ]:


#checking the data types of the training datset columns
df.dtypes


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 210px">Visualizing the Training Data</span>
# 

# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Entire Training Datset</span>
# 

# In[ ]:


plt.figure(figsize=(20,7))
sns.lineplot(data=df,x='date',y='demand',lw=2,color='red');


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">One year Training Data</span>
# 

# In[ ]:


one_year_df=df.loc[(df.date >=pd.to_datetime('2019-01-01')) & (df.date<pd.to_datetime('2020-01-01'))]
plt.figure(figsize=(20,7))
sns.lineplot(data=one_year_df,x='date',y='demand',lw=2,color='red');


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">One month Training Data</span>
# 

# In[ ]:


one_month_df=df.loc[(df.date >=pd.to_datetime('2019-01-01')) & (df.date<pd.to_datetime('2019-02-01'))]
plt.figure(figsize=(20,7))
sns.lineplot(data=one_month_df,x='date',y='demand',lw=2,color='red');


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Two month Training Data</span>

# In[ ]:


two_month_df=df.loc[(df.date >=pd.to_datetime('2019-01-01')) & (df.date<pd.to_datetime('2019-03-01'))]
plt.figure(figsize=(20,7))
sns.lineplot(data=two_month_df,x='date',y='demand',lw=2,color='red');


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">One day Training Data</span>

# In[ ]:


one_day_df=df.loc[(df.date >=pd.to_datetime('2019-01-01')) & (df.date<pd.to_datetime('2019-01-02'))]
plt.figure(figsize=(20,7))
sns.lineplot(data=one_day_df,x='hour',y='demand',lw=2,color='red');


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 210px">Feature Engineering</span>

# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Decomposing Date Features</span>

# In[ ]:


# decompose date 
df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
df['day'] = df['date'].dt.day
df['week_of_year']=(df['date'].dt.isocalendar().week).astype(int)
df['day_of_week']=df['date'].dt.weekday
df['quartor']=df['date'].dt.quarter

# for test data
test_df['year']=test_df['date'].dt.year
test_df['month']=test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['week_of_year']=(test_df['date'].dt.isocalendar().week).astype(int)
test_df['day_of_week']=test_df['date'].dt.weekday
test_df['quartor']=test_df['date'].dt.quarter


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Grouping Demand by months</span>

# In[ ]:


plt.figure(figsize=(20,7))
temp1=df.groupby('month')['demand'].mean().reset_index()
temp2=df.groupby('month')['demand'].median().reset_index()
plt.subplot(1,2,1)
sns.barplot(data=temp1,x='month',y='demand',palette='Set1')
plt.title("Mean")
plt.subplot(1,2,2)
sns.barplot(data=temp2,x='month',y='demand',palette='Set1');
plt.title("Median")
plt.tight_layout()


# <h3>  <b style='color:red;font-size:22px;'>Inference </b>:<ul><li>The month of November has higher demand as compared to other months of the year.</li></ul></h3>

# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Adding is_november feature</span>

# In[ ]:


df['is_november']=(df['month']==11).astype(int)
test_df['is_november']=(test_df['month']==11).astype(int)


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Grouping Demand by hour</span>

# In[ ]:


plt.figure(figsize=(20,7))
temp1=df.groupby('hour')['demand'].mean().reset_index()
temp2=df.groupby('hour')['demand'].median().reset_index()
plt.subplot(1,2,1)
sns.barplot(data=temp1,x='hour',y='demand',palette='Set1')
plt.title("Mean")
plt.subplot(1,2,2)
sns.barplot(data=temp2,x='hour',y='demand',palette='Set1');
plt.title("Median")
plt.tight_layout()


# <h3>  <b style='color:red;font-size:22px;'>Inference </b>:<ul><li>We can observe that hours from 7 to 22 have higher demand as compared to the other hours of the day.</li></ul></h3>
# 

# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Adding peak_hours feature</span>

# In[ ]:


df['peak_hours'] = ((df['hour']>=7 ) &  (df['hour']<=22)).astype(int)
test_df['peak_hours'] = ((test_df['hour']>=7) &  (test_df['hour']<=22)).astype(int)


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Grouping Demand by year</span>

# In[ ]:


plt.figure(figsize=(20,7))
temp1=df.groupby('year')['demand'].mean().reset_index()
temp2=df.groupby('year')['demand'].median().reset_index()
plt.subplot(1,2,1)
sns.barplot(data=temp1,x='year',y='demand',palette='Set1')
plt.title("Mean")
plt.subplot(1,2,2)
sns.barplot(data=temp2,x='year',y='demand',palette='Set1');
plt.title("Median")
plt.tight_layout()


# <h3> <b style='color:red;font-size:22px;'>Inference </b>:<ul>
#     <li>We can observe that as demand goes on increasing every year.</li><br>
#     <li> I will define a feature year_code to measure the number of years from beginning.</li></ul></h3>
# 

# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Adding year_code featrue</span>

# In[ ]:


df['year_code']=df.year-2017
test_df['year_code']=test_df.year-2017


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Grouping Demand by quartor</span>

# In[ ]:


plt.figure(figsize=(20,7))
temp1=df.groupby('quartor')['demand'].mean().reset_index()
temp2=df.groupby('quartor')['demand'].median().reset_index()
plt.subplot(1,2,1)
sns.barplot(data=temp1,x='quartor',y='demand',palette='Set1')
plt.title("Mean")
plt.subplot(1,2,2)
sns.barplot(data=temp2,x='quartor',y='demand',palette='Set1');
plt.title("Median")
plt.tight_layout()


# <h3>  <b style='color:red;font-size:22px;'>Inference </b>:<ul>
#     <li>We can observe that quartor 3 has lower demand as compared to other quartors of the year.</li>
#     </ul>
# </h3>

# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Adding is_quartor_three feature</span>

# In[ ]:


df['is_quartor_three'] = (df['quartor']==3).astype(int)
test_df['is_quartor_three'] = (test_df['quartor']==3).astype(int)


# 
# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Grouping Demand by week of the year</span>
# 

# In[ ]:


plt.figure(figsize=(24,7))
temp1=df.groupby('week_of_year')['demand'].mean().reset_index()
temp2=df.groupby('week_of_year')['demand'].median().reset_index()
plt.subplot(1,2,1)
sns.barplot(data=temp1,x='week_of_year',y='demand',palette='Set1')
plt.title("Mean")
plt.subplot(1,2,2)
sns.barplot(data=temp2,x='week_of_year',y='demand',palette='Set1');
plt.title("Median")
plt.tight_layout()


# <h3><b style='color:red;font-size:22px;'>Inference </b>:<ul>
#     <li>We can observe that there is one major spike for two weeks(45,46).</li><br>
#     <li>But I have already considered this when I added the is_november feature.</li>
#     </ul></h3>

# 
# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Grouping Demand by day of the week</span>

# In[ ]:


plt.figure(figsize=(20,7))
temp1=df.groupby('day_of_week')['demand'].mean().reset_index()
temp2=df.groupby('day_of_week')['demand'].median().reset_index()
plt.subplot(1,2,1)
sns.barplot(data=temp1,x='day_of_week',y='demand',palette='Set1')
plt.title("Mean")
plt.subplot(1,2,2)
sns.barplot(data=temp2,x='day_of_week',y='demand',palette='Set1');
plt.title("Median")
plt.tight_layout()


# <h3> <b style='color:red;font-size:22px;'>Inference </b>:<ul>
#     <li>We can observe that the weekend days (Friday,Saturday,Sunday) have high demand.</li><br>
#     <li>Since there are only 7 categories, there is not need to additionally add any feature as the model will figure this out pretty easily.
#     </ul></h3>
# 

# 
# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Grouping Demand by day of the month</span>
# 

# In[ ]:


plt.figure(figsize=(24,7))
temp1=df.groupby('day')['demand'].mean().reset_index()
temp2=df.groupby('day')['demand'].median().reset_index()
plt.subplot(1,2,1)
sns.barplot(data=temp1,x='day',y='demand',palette='Set1')
plt.title("Mean")
plt.subplot(1,2,2)
sns.barplot(data=temp2,x='day',y='demand',palette='Set1');
plt.title("Median")
plt.tight_layout()


# <h3>  <b style='color:red;font-size:22px;'>Inference </b>:<ul><li>No specific trend observed.</li></ul></h3>

# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 210px">Splitting the Data into train and val</span>

# In[ ]:


# compute 20% number of samples
df.shape[0]*0.2


# In[ ]:


train=df.iloc[:-3600,:]
val=df.iloc[-3600:,:]


# In[ ]:


# checking the number of rows and columns for train dataset
rows,columns=train.shape[0],train.shape[1]


# In[ ]:


display(HTML(f"<h3>  <b style='color:red;font-size:22px;'>Inference </b>:<ul><li>There are {rows} rows and {columns} columns in the train dataset.</li></ul></h3>"))


# In[ ]:


# checking the number of rows and columns for val dataset
rows,columns=val.shape[0],val.shape[1]


# In[ ]:


display(HTML(f"<h3>  <b style='color:red;font-size:22px;'>Inference </b>:<ul><li>There are {rows} rows and {columns} columns in the val dataset.</li></ul></h3>"))


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Dropping unnecessary features</span>

# In[ ]:


xtrain=train.drop(columns=['demand','year','date','quartor']) #quartor feature is dropped as is_quartor_three feature will carry the info.
ytrain=train['demand']

xval=val.drop(columns=['year','demand','date','quartor']) #quartor feature is dropped as is_quartor_three feature will carry the info.
yval=val['demand']

test_df.drop(columns=['date','year','quartor'],inplace=True)


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 210px">Modelling</span>

# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">LightGBM model with my defeault parameters</span>

# In[ ]:


model=lgb.LGBMRegressor(n_estimators=3000,learning_rate=0.01)
model.fit(
               xtrain.values,ytrain,
               eval_set=[(xval.values,yval)],
               eval_metric='rmse',
               callbacks=[lgb.early_stopping(100)]
        );


# In[ ]:


display(HTML(f"<h3> <b style='color:#673AB7;font-size:22px;'>This model gave rmse: <b style='color:red;'>{model.best_score_['valid_0']['rmse']:0.4F}</b></h3>"))


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">LightGBM model with Hyperparameter tunning using Optuna</span>

# In[ ]:


def objective(trial):
    
    params = {
    
    'n_estimators':4000,
    'num_leaves':trial.suggest_int('num_leaves',35,80),
    'subsample':1,
    'min_child_samples':trial.suggest_int("min_child_samples",30,100),
    'learning_rate':trial.suggest_categorical("learning_rate",[0.001,0.01,0.03,0.05,0.07]),
     'max_depth':trial.suggest_int("max_depth",4,12),
    'reg_alpha':trial.suggest_float('reg_alpha',0.0,50),
    'reg_lambda':trial.suggest_float('reg_lambda',0.0,50),
    "min_split_gain": trial.suggest_float("min_split_gain", 0.0,20),
    'subsample_freq' : trial.suggest_categorical("subsample_freq", [1])
        
            }
    
    model=lgb.LGBMRegressor(**params)
    model.fit(
               xtrain.values,ytrain,
               eval_set=[(xval.values,yval)],
               eval_metric='rmse',
               callbacks=[lgb.early_stopping(100,verbose=0)]
        )
    pred=model.predict(xval)
    
    score=mean_squared_error(yval,pred,squared=False)
    
    return score


# In[ ]:


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=300,show_progress_bar=True,n_jobs=1)


# In[ ]:


trial = study.best_trial
best_params_lgbm=trial.params
study.best_value
display(HTML(f"<h3 style='color:#673AB7'>Best Params :<br><br><pre>{best_params_lgbm:}</h3>"))


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Training the model with Best Parameters</span>

# In[ ]:


model=lgb.LGBMRegressor(**best_params_lgbm,n_estimators=4000)
model.fit(
               xtrain.values,ytrain,
               eval_set=[(xval.values,yval)],
               eval_metric='rmse',
               callbacks=[lgb.early_stopping(100)]
        )


# In[ ]:


display(HTML(f"<h3> <b style='color:#673AB7;font-size:22px;'>This model gave rmse: <b style='color:red;'>{model.best_score_['valid_0']['rmse']:0.4F}</b></h3>"))


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">LightGBM Parameter Importances</span>

# In[ ]:


plot_param_importances(study)


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Making Predictions on the Test Dataset and Saving it</span>

# In[ ]:


preds=model.predict(test_df.values)
sub=pd.read_csv('../input/analytics-vidhya-hackathon-april-2022/sample_4E0BhPN.csv')
sub['demand']=preds
sub.to_csv('submission.csv',index=False)
sub.head()


# <span style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #673AB7;padding:0px 20px">Feature Importance</span>

# In[ ]:


pd.DataFrame(model.feature_importances_.tolist(),columns=['Importance'],index=xtrain.columns.to_list()).sort_values(by='Importance',ascending=False).plot(kind='bar',figsize=(14,7),color=['#673AB7']);


# <h2 style='text-align:center;color:#673AB7;font-weight:bold'>Using this Feature Engineering I got rank of 29 in the recently concluded Jobathon by Analytics Vidhya.</h2>

# <h2 style='text-align:center;color:#673AB7;font-weight:bold'> Do share your feedback in the comments section,I hope you found this to be helpful.🙌</h2>

# <p style="background-color:#673AB7;color:white;font-size:22px;text-align:center;border-radius:10px 10px;font-weight:bold;border:2px solid #673AB7;">Thank you 😊 !!!!!!!!!!!</p>
