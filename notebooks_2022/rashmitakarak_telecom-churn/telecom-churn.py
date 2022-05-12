#!/usr/bin/env python
# coding: utf-8

# <h1><strong><em><span style="color:MediumSlateBlue"><span style="font-family:Helvetica;"> Telecom Churn Case Study (Machine Learning - II)  </span></span></em></strong></h1>

# ### Business problem overview
# In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.For many incumbent operators, retaining high profitable customers is the number one business goal.
# - To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.
# 
# - In this project, you will analyse customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn and identify the main indicators of churn.
# 
# ***Understanding and defining churn***
# There are two main models of payment in the telecom industry - postpaid (customers pay a monthly/annual bill after using the services) and prepaid (customers pay/recharge with a certain amount in advance and then use the services).
# 
# - In the postpaid model, when customers want to switch to another operator, they usually inform the existing operator to terminate the services, and you directly know that this is an instance of churn.
# 
# However, in the prepaid model, customers who want to switch to another network can simply stop using the services without any notice, and it is hard to know whether someone has actually churned or is simply not using the services temporarily (e.g. someone may be on a trip abroad for a month or two and then intend to resume using the services again).
# 
# Thus, churn prediction is usually more critical (and non-trivial) for prepaid customers, and the term ‘churn’ should be defined carefully.  Also, prepaid is the most common model in India and Southeast Asia, while postpaid is more common in Europe in North America.
# 
# ***This project is based on the Indian and Southeast Asian market.***
# 
# 
# ***Definitions of churn***
# There are various ways to define churn, such as:
# 
# ***Revenue-based churn:*** Customers who have not utilised any revenue-generating facilities such as mobile internet, outgoing calls, SMS etc. over a given period of time. One could also use aggregate metrics such as ‘customers who have generated less than INR 4 per month in total/average/median revenue’.
# 
# The main shortcoming of this definition is that there are customers who only receive calls/SMSes from their wage-earning counterparts, i.e. they don’t generate revenue but use the services. For example, many users in rural areas only receive calls from their wage-earning siblings in urban areas.
# 
# ***Usage-based churn:*** Customers who have not done any usage, either incoming or outgoing - in terms of calls, internet etc. over a period of time.
# 
# A potential shortcoming of this definition is that when the customer has stopped using the services for a while, it may be too late to take any corrective actions to retain them. For e.g., if you define churn based on a ‘two-months zero usage’ period, predicting churn could be useless since by that time the customer would have already switched to another operator.
# 
# - In this project, you will use the usage-based definition to define churn.
# 
# ***High-value churn***
# In the Indian and the Southeast Asian market, approximately 80% of revenue comes from the top 20% customers (called high-value customers). Thus, if we can reduce churn of the high-value customers, we will be able to reduce significant revenue leakage.
# 
# In this project, you will define high-value customers based on a certain metric (mentioned later below) and predict churn only on high-value customers.
# 
# ***Understanding the business objective and the data***
# The dataset contains customer-level information for a span of four consecutive months - June, July, August and September. The months are encoded as 6, 7, 8 and 9, respectively. 
# 
# The business objective is to predict the churn in the last (i.e. the ninth) month using the data (features) from the first three months. To do this task well, understanding the typical customer behaviour during churn will be helpful.
# 
#  
# ***Understanding customer behaviour during churn***
# Customers usually do not decide to switch to another competitor instantly, but rather over a period of time (this is especially applicable to high-value customers). In churn prediction, we assume that there are three phases of customer lifecycle :
# 
# The ‘good’ phase: In this phase, the customer is happy with the service and behaves as usual.
# 
# The ‘action’ phase: The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a  competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than the ‘good’ months. Also, it is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)
# 
# The ‘churn’ phase: In this phase, the customer is said to have churned. You define churn based on this phase. Also, it is important to note that at the time of prediction (i.e. the action months), this data is not available to you for prediction. Thus, after tagging churn as 1/0 based on this phase, you discard all data corresponding to this phase.
# 
#  
# 
# In this case, since you are working over a four-month window, the first two months are the ‘good’ phase, the third month is the ‘action’ phase, while the fourth month is the ‘churn’ phase.

# **P.S. Kindly let me know how you like the notebook and do suggest!! :)**

# ### Importing the necessary packages.

# In[ ]:


# import required libraries
import pandas as pd, numpy as np
import seaborn as sns, matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
#from imblearn.metrics import sensitivity_specificity_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import statsmodels.api as sm

from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix,precision_score, f1_score,classification_report
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# ### Reading the dataset.

# In[ ]:


data= pd.read_csv("../input/telecom-churn/telecom_churn_data.csv",None)
pd.set_option('max_columns', None)
data.head(10)


# ### Reading the features and information of the Dataset.

# In[ ]:


data.shape


# In[ ]:


data.info(null_counts=True, verbose=True)


# In[ ]:


data.describe()


# In[ ]:


# Removing columns with 0 values in all rows, as shown in the table above:
data.drop(['mobile_number','loc_og_t2o_mou','std_og_t2o_mou','loc_ic_t2o_mou','std_ic_t2o_mou_6','std_ic_t2o_mou_7',
           'std_ic_t2o_mou_8','std_og_t2c_mou_8','std_ic_t2o_mou_9'], axis=1, inplace=True)


# In[ ]:


data.circle_id.value_counts(normalize=True )


# In[ ]:


#there is only 1 circle id value and since its not related to churn rate, we can remove this column too
data.drop(['circle_id'], axis=1, inplace=True)


# ## Imputing Missing values

# In[ ]:


date_cols = ['last_date_of_month_6',
             'last_date_of_month_7',
             'last_date_of_month_8',
             'last_date_of_month_9',
             'date_of_last_rech_6',
             'date_of_last_rech_7',
             'date_of_last_rech_8',
             'date_of_last_rech_9',
             'date_of_last_rech_data_6',
             'date_of_last_rech_data_7',
             'date_of_last_rech_data_8',
             'date_of_last_rech_data_9'
            ]

fb_night_cols =  ['night_pck_user_6',
             'night_pck_user_7',
             'night_pck_user_8',
             'night_pck_user_9',
             'fb_user_6',
             'fb_user_7',
             'fb_user_8',
             'fb_user_9'
            ]

num_cols = [column for column in data.columns if column not in date_cols + fb_night_cols]

print("Date cols:%d\nNumeric cols:%d\nfb_night cols:%d" % (len(date_cols), len(num_cols), len(fb_night_cols)))


# In[ ]:


(data.isnull().sum()/len(data)*100).sort_values(ascending = False)


# ## Imputing the missing recharge values with '0'

# In[ ]:


# some recharge columns have minimum value of 1 while some don't
recharge_cols = ['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8', 'total_rech_data_9',
                 'max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8', 'max_rech_data_9',
                 'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8', 'av_rech_amt_data_9',
                 ]
data[recharge_cols] = data[recharge_cols].apply(lambda x: x.fillna(0))


# ### Checking after imputation

# In[ ]:


data[recharge_cols].isnull().sum()


# ### Replaceing the NAN values in fb_night_cols columns with '0'

# In[ ]:


data[fb_night_cols] = data[fb_night_cols].apply(lambda x: x.fillna(0))


# In[ ]:


data[fb_night_cols].isnull().sum()


# In[ ]:


(data.isnull().sum()/len(data)*100).sort_values(ascending = False)


# In[ ]:


dele_col =['arpu_2g_6','date_of_last_rech_data_6','count_rech_2g_6','arpu_3g_6','count_rech_3g_6','count_rech_3g_7',
           'date_of_last_rech_data_7','arpu_3g_7','count_rech_2g_7','arpu_2g_7','arpu_2g_9','date_of_last_rech_data_9',
           'arpu_3g_9','count_rech_3g_9','count_rech_2g_9','count_rech_3g_8','date_of_last_rech_data_8','arpu_3g_8',
           'count_rech_2g_8','arpu_2g_8']
data.drop(data[dele_col],axis=1, inplace=True)


# In[ ]:


imp_val = (data.isnull().sum()/len(data)*100)


# In[ ]:


percent_missing = data.isnull().sum() * 100 / len(data)
percent_missing = percent_missing[percent_missing>=1]
to_impute = percent_missing.index
sample = pd.DataFrame(percent_missing)
sample.index


# In[ ]:


data.fillna(data[to_impute].mean(), axis=0,inplace=True)


# In[ ]:


data[to_impute].head()


# ## Imputing the rest of the missing values using KNNImputer

# In[ ]:


data.drop(['last_date_of_month_6',
             'last_date_of_month_7',
             'last_date_of_month_8',
             'last_date_of_month_9',
             'date_of_last_rech_6',
             'date_of_last_rech_7',
             'date_of_last_rech_8',
             'date_of_last_rech_9'], axis=1,inplace=True)


# In[ ]:


data.shape


# In[ ]:


imp_val = (data.isnull().sum()/len(data)*100)
imp_val


# ## Filtering High Value Customers

# ### Creating average recharge amount for June and July

# In[ ]:


data.shape


# In[ ]:


# calculate the total data recharge amount for June and July --> number of recharges * average recharge amount
data['total_data_rech_6'] = data['total_rech_data_6'] * data['av_rech_amt_data_6']
data['total_data_rech_7'] = data['total_rech_data_7'] * data['av_rech_amt_data_7']


# In[ ]:


# add total data recharge and total recharge to get total combined recharge amount for a month
# calculate total recharge amount for June and July --> call recharge amount + data recharge amount
data['amt_data_6'] = data[['total_rech_amt_6','total_data_rech_6']].sum(axis=1)
data['amt_data_7'] = data[['total_rech_amt_7','total_data_rech_7']].sum(axis=1)


# In[ ]:


# calculate average recharge done by customer in June and July
data["avg_rech_amt_6_7"] = (data["amt_data_6"] + data["amt_data_7"])/2
data["avg_rech_amt_6_7"].head()


# ### Finding the 70th percentile

# In[ ]:


# look at the 70th percentile recharge amount
print("Recharge amount at 70th percentile: {0}".format(data.avg_rech_amt_6_7.quantile(0.7)))


# In[ ]:


# retain only those customers who have recharged their mobiles with more than or equal to 70th percentile amount
filtered_data = data.loc[data.avg_rech_amt_6_7 >= data.avg_rech_amt_6_7.quantile(0.7), :]
filtered_data = filtered_data.reset_index(drop=True)
filtered_data.shape


# **As mentioned in the problem statement, the rows above 70th percentile are almost at range equal to 29.9K**

# ## Derive Churn

#  Here, we are using only 9th month data to calculate the target variable (churn_rate)

# In[ ]:


# calculate total incoming and outgoing minutes of usage
filtered_data['total_calls_mou_9'] = filtered_data.total_ic_mou_9 + filtered_data.total_og_mou_9


# In[ ]:


# calculate 2g and 3g data consumption
filtered_data['total_internet_mb_9'] =  filtered_data.vol_2g_mb_9 + filtered_data.vol_3g_mb_9


# In[ ]:


# create churn variable: those who have not used either calls or internet in the month of September are customers who have churned
# 0 - not churn, 1 - churn
filtered_data['churn_rate'] = filtered_data.apply(lambda row: 1 if (row.total_calls_mou_9 == 0 and row.total_internet_mb_9 == 0) else 0, axis=1)


# In[ ]:


# filtered_data['total_calls_mou_9'] = filtered_data.total_ic_mou_9 + filtered_data.total_og_mou_9
# filtered_data['total_internet_mb_9'] =  filtered_data.vol_2g_mb_9 + filtered_data.vol_3g_mb_9
# filtered_data['churn_rate'] = filtered_data.apply(lambda row: 1 if (row.total_calls_mou_9 == 0 and row.total_internet_mb_9 == 0) else 0, axis=1)
round(filtered_data['churn_rate'].value_counts(normalize=True)*100,2)


# ### Visualizing the dependent variable

# In[ ]:


## Visualizing the churn rate in the form of a plot
plt.figure(figsize=(10,5))
ax=filtered_data['churn_rate'].value_counts()
plt.pie(ax,autopct='%1.1f%%',labels=ax.index,colors=['red','black'])
plt.ylabel('Count')
plt.xlabel('Churn Status')
plt.title('Churn Status Distribution',fontsize=14)
plt.show()


# In[ ]:


## Visualizing the churn rate in the form of a plot
plt.figure(figsize=(10,5))
ax=filtered_data['churn_rate'].value_counts().plot(kind = 'bar')
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.xticks(rotation = 45)
plt.ylabel('Count')
plt.xlabel('Churn Status')
plt.title('Churn Status Distribution',fontsize=14)
plt.show()


# In[ ]:


# delete derived variables
filtered_data = filtered_data.drop(['total_calls_mou_9', 'total_internet_mb_9'], axis=1)


# # Calculate difference between 8th and previous months

# In[ ]:


filtered_data['arpu_diff'] = filtered_data.arpu_8 - ((filtered_data.arpu_6 + filtered_data.arpu_7)/2)

filtered_data['onnet_mou_diff'] = filtered_data.onnet_mou_8 - ((filtered_data.onnet_mou_6 + filtered_data.onnet_mou_7)/2)

filtered_data['offnet_mou_diff'] = filtered_data.offnet_mou_8 - ((filtered_data.offnet_mou_6 + filtered_data.offnet_mou_7)/2)

filtered_data['roam_ic_mou_diff'] = filtered_data.roam_ic_mou_8 - ((filtered_data.roam_ic_mou_6 + filtered_data.roam_ic_mou_7)/2)

filtered_data['roam_og_mou_diff'] = filtered_data.roam_og_mou_8 - ((filtered_data.roam_og_mou_6 + filtered_data.roam_og_mou_7)/2)

filtered_data['loc_og_mou_diff'] = filtered_data.loc_og_mou_8 - ((filtered_data.loc_og_mou_6 + filtered_data.loc_og_mou_7)/2)

filtered_data['std_og_mou_diff'] = filtered_data.std_og_mou_8 - ((filtered_data.std_og_mou_6 + filtered_data.std_og_mou_7)/2)

filtered_data['isd_og_mou_diff'] = filtered_data.isd_og_mou_8 - ((filtered_data.isd_og_mou_6 + filtered_data.isd_og_mou_7)/2)

filtered_data['spl_og_mou_diff'] = filtered_data.spl_og_mou_8 - ((filtered_data.spl_og_mou_6 + filtered_data.spl_og_mou_7)/2)

filtered_data['total_og_mou_diff'] = filtered_data.total_og_mou_8 - ((filtered_data.total_og_mou_6 + filtered_data.total_og_mou_7)/2)

filtered_data['loc_ic_mou_diff'] = filtered_data.loc_ic_mou_8 - ((filtered_data.loc_ic_mou_6 + filtered_data.loc_ic_mou_7)/2)

filtered_data['std_ic_mou_diff'] = filtered_data.std_ic_mou_8 - ((filtered_data.std_ic_mou_6 + filtered_data.std_ic_mou_7)/2)

filtered_data['isd_ic_mou_diff'] = filtered_data.isd_ic_mou_8 - ((filtered_data.isd_ic_mou_6 + filtered_data.isd_ic_mou_7)/2)

filtered_data['spl_ic_mou_diff'] = filtered_data.spl_ic_mou_8 - ((filtered_data.spl_ic_mou_6 + filtered_data.spl_ic_mou_7)/2)

filtered_data['total_ic_mou_diff'] = filtered_data.total_ic_mou_8 - ((filtered_data.total_ic_mou_6 + filtered_data.total_ic_mou_7)/2)

filtered_data['total_rech_num_diff'] = filtered_data.total_rech_num_8 - ((filtered_data.total_rech_num_6 + filtered_data.total_rech_num_7)/2)

filtered_data['total_rech_amt_diff'] = filtered_data.total_rech_amt_8 - ((filtered_data.total_rech_amt_6 + filtered_data.total_rech_amt_7)/2)

filtered_data['max_rech_amt_diff'] = filtered_data.max_rech_amt_8 - ((filtered_data.max_rech_amt_6 + filtered_data.max_rech_amt_7)/2)

filtered_data['total_rech_data_diff'] = filtered_data.total_rech_data_8 - ((filtered_data.total_rech_data_6 + filtered_data.total_rech_data_7)/2)

filtered_data['max_rech_data_diff'] = filtered_data.max_rech_data_8 - ((filtered_data.max_rech_data_6 + filtered_data.max_rech_data_7)/2)

filtered_data['av_rech_amt_data_diff'] = filtered_data.av_rech_amt_data_8 - ((filtered_data.av_rech_amt_data_6 + filtered_data.av_rech_amt_data_7)/2)

filtered_data['vol_2g_mb_diff'] = filtered_data.vol_2g_mb_8 - ((filtered_data.vol_2g_mb_6 + filtered_data.vol_2g_mb_7)/2)

filtered_data['vol_3g_mb_diff'] = filtered_data.vol_3g_mb_8 - ((filtered_data.vol_3g_mb_6 + filtered_data.vol_3g_mb_7)/2)


# In[ ]:


#filtered_data['']


# In[ ]:


filtered_data.describe()


# # Deleting the 9th Month variable from the dataset

# In[ ]:


filtered_data = filtered_data.filter(regex="[^9]$",axis=1)
filtered_data.shape


# In[ ]:


# extract all names that end with 9
col_9_names = data.filter(regex='9$', axis=1).columns

# update num_cols and cat_cols column name list
fb_night_cols = [col for col in fb_night_cols if col not in col_9_names]
fb_night_cols.append('churn')
num_cols = [col for col in filtered_data.columns if col not in fb_night_cols]


# In[ ]:


filtered_data.shape


# In[ ]:


filtered_data_pca = filtered_data.copy()


# In[ ]:


filtered_data_pca.shape


# # Visualizing the Data 

# In[ ]:


#fig, ax = plt.subplots(figsize =(10, 7))
sns.histplot(filtered_data['total_rech_amt_8'], bins = [0, 25, 50, 75, 100])
plt.show()


# ### From the plot above we can conclude that Higher the recharge amount , lower the people interested to invest in the plan.

# In[ ]:


# visualizing box plots for  6th, 7th and 8th month
def box_plot_func(param):
    plt.figure(figsize=(20,16))
    df = filtered_data
    plt.title("Churn Rate For ", )
    plt.subplot(2,3,1)
    sns.boxplot(data=df, y=param+"_6",x="churn_rate",hue="churn_rate",
                showfliers=False,palette=['g','r'],linewidth=2.5)
    plt.subplot(2,3,2)
    sns.boxplot(data=df, y=param+"_7",x="churn_rate",hue="churn_rate",
                showfliers=False,palette=("Set2"),linewidth=2.5)
    plt.subplot(2,3,3)
    sns.boxplot(data=df, y=param+"_8",x="churn_rate",hue="churn_rate",
                showfliers=False,palette=("Paired"),linewidth=2.5)
    plt.show()


# ### Average Revenue Per User for Good and Action Phases

# In[ ]:


box_plot_func("arpu")


# *** There is a slight dip in the churn rate in the month of August i.e. Action Phase ***

# ### All kind of calls within the same operator network.

# In[ ]:


box_plot_func('onnet_mou')


# ### There is a dip in the data for Churning customers whereas for Non-churning customers it has increased.

# ### Local Outgoing minutes of call of Operator T to fixed lines of T
# 

# In[ ]:


box_plot_func('loc_og_t2f_mou')


# So we observe that Churn Rate has risen in month 8 ,for this feature

# ### STD outgoing call minutes for Operator T to other operator mobile

# In[ ]:


box_plot_func('std_og_t2m_mou')


# *** Churn Rate has dropped significantly in the 8th month with this feature. ***

# In[ ]:


sns.histplot(filtered_data['std_og_mou_8'], bins = [0, 25, 50, 75, 100])
plt.show()


# ## Plotting Total Recharge Amount 

# In[ ]:


box_plot_func('total_rech_amt')


# *** Conclusion: We can see there is significant dip in the total recharge Amount in the 8th month for churn customers. ***

# ### Plotting Last Day Recharge Amount Vs Churn Rate

# In[ ]:


box_plot_func('last_day_rch_amt')


# *** Analysis: There is a huge drop in the recharge amount for churned customers in the 8th month. ***

# ### Plotting Maximum Recharge against Churn Rate

# In[ ]:


box_plot_func('max_rech_amt')


# *** Analysis: There is a drop in the recharge amount for churned customers in the 8th month. ***

# In[ ]:


box_plot_func('loc_og_mou') # local outgoing calls in months


# In[ ]:


box_plot_func('max_rech_data') #max mobile data recharge


# In[ ]:


box_plot_func('night_pck_user')


# In[ ]:


# fig, axes = plt.subplots(round(len(num_cols) / 6), 6, figsize=(25, 65))

# for i, ax in enumerate(fig.axes):
#     if i < len(num_cols):
#         ax.hist(filtered_data.loc[filtered_data['churn_rate']==0, num_cols[i]],color='black')
#         ax.hist(filtered_data.loc[filtered_data['churn_rate']==1, num_cols[i]],color='red')
#         # adjusting font size of X-Labels and Y-Labels
#         ax.set_xlabel(num_cols[i],fontsize=12)
#         ax.legend(['Not Churn','Churn'],loc = 'best')        
# plt.show()


# ### Dropping 6th and 7th month columns , because we have derived new features from them.

# In[ ]:


to_drop = filtered_data.filter(regex='_6|_7').columns
len(to_drop)


# In[ ]:


# converting Age on Number from days to months and hence dropping the previous column
filtered_data['aon_in_months'] = filtered_data['aon']/30
filtered_data['aon_in_months'].head()


# In[ ]:


# Dropping the aon columns since we are having aon_in_months
filtered_data.drop('aon',axis=1, inplace=True)


# In[ ]:


# Dropping the columns in 'to_drop' from 'filtered_data'
filtered_data.drop(to_drop, axis=1, inplace=True)


# In[ ]:


filtered_data.shape


# In[ ]:


# we are initializing the independent features into X and target variable into y before splitting
X = filtered_data.drop("churn_rate", axis=1)
y = filtered_data["churn_rate"]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, train_size=0.7, random_state =42)
print("X train Shape: ", X_train.shape, "\nX test shape: ",X_test.shape, "\ny_train Shape: ",y_train.shape, "\ny_test shape: ",y_test.shape)


# In[ ]:


X = filtered_data_pca.drop("churn_rate", axis=1)
y = filtered_data_pca["churn_rate"]
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X,y, test_size = 0.3, train_size=0.7, random_state =42)
print("X train Shape: ", X_train.shape, "\nX test shape: ",X_test.shape, "\ny_train Shape: ",y_train.shape, "\ny_test shape: ",y_test.shape)


# In[ ]:


X_train_cols = X_train.columns


# In[ ]:


# box_plot_func('offnet_mou') ///////////////////////////////////delete


# In[ ]:


X_train.head()


# In[ ]:



scalar = StandardScaler()
X_train= scalar.fit_transform(X_train)
X_train


# In[ ]:



scalar1 = StandardScaler()
X_test= scalar1.fit_transform(X_test)
X_test


# In[ ]:


print(type(X_train))


# In[ ]:


# Converting the numpy array into Dataframe
X_train = pd.DataFrame(X_train, columns=X_train_cols)
X_test = pd.DataFrame(X_test, columns=X_train_cols)


# In[ ]:


# X_train['offnet_mou_8'].sort_values(ascending=False)


# In[ ]:


# plot feature correlation
import seaborn as sns
plt.rcParams["figure.figsize"] =(10,10)
htmap = sns.diverging_palette(199, 359, s=99, center="light", as_cmap=True)
sns.heatmap(data=X_train.corr(), center=0.0,cmap=htmap)
plt.show()


# In[ ]:


plt.figure(figsize = (40,20))        # Size of the figure
sns.heatmap(data = X_train.corr(),annot = True,annot_kws={'size':6})
plt.show()


# In[ ]:


## Removing highly correlated values in train
X_train.drop(['arpu_8','sachet_2g_8','av_rech_amt_data_8','loc_ic_mou_8',
              'total_rech_num_8','jun_vbc_3g', 'arpu_diff', 'loc_ic_t2m_mou_8',
              'total_ic_mou_diff', 'std_og_mou_diff','onnet_mou_8','offnet_mou_8', 
              'offnet_mou_diff','total_og_mou_diff','fb_user_8','std_og_mou_8',
              'loc_og_mou_8','std_ic_mou_8'],axis=1, inplace=True)


# In[ ]:


## Removing highly correlated values in test
X_test.drop(['arpu_8','sachet_2g_8','av_rech_amt_data_8','loc_ic_mou_8',
              'total_rech_num_8','jun_vbc_3g', 'arpu_diff', 'loc_ic_t2m_mou_8',
              'total_ic_mou_diff', 'std_og_mou_diff','onnet_mou_8','offnet_mou_8', 
              'offnet_mou_diff','total_og_mou_diff','fb_user_8','std_og_mou_8',
              'loc_og_mou_8','std_ic_mou_8'],axis=1, inplace=True)


# In[ ]:


X_train.shape


# In[ ]:


def evaluation_table(y_act, y_pred):
    accuracy  = round(metrics.accuracy_score(y_act, y_pred),2)
    precision = round(precision_score(y_act, y_pred),2)
    recall = round(recall_score(y_act, y_pred),2)
    f1 = round(f1_score(y_act, y_pred),2)
    tn, fp, fn, tp = confusion_matrix(y_act, y_pred).ravel() # confusion matrix
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
       
    return (accuracy,precision,recall,f1,specificity,sensitivity)


# ## Logistic regression Without PCA

# ### With class imbalance

# In[ ]:



lr = LogisticRegression()


# In[ ]:


lr_model = lr.fit(X_train,y_train)


# In[ ]:


y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)


# In[ ]:


table = pd.DataFrame()


# In[ ]:


accuracy,precision,recall,f1,specificity,sensitivity = evaluation_table(y_test,y_test_pred)
tempTable = pd.DataFrame({'Model':['Logistic regression Without PCA With class imbalance'], 
                            'Accuracy': [accuracy],
                            'Precision': [precision],
                            'Recall': [recall], 
                            'Sensitivity': [sensitivity],
                            'Specificity': [specificity],
                            'F1 score': [f1] 
                             })

table = pd.concat([table, tempTable])
table = table[['Model', 'Accuracy', 'Precision', 'Recall', 'Sensitivity','Specificity', 'F1 score' ]]
table


# In[ ]:


print("Train Evaluation with imbalanced data:\n\n ",classification_report(y_train,y_train_pred ))
print("Train accuracy value:",metrics.accuracy_score(y_train, y_train_pred))
print("Test accuracy value:",metrics.accuracy_score(y_test, y_test_pred))
print("Test Evaluation with imbalanced data:\n\n ",classification_report(y_test,y_test_pred ))


# In[ ]:


y_train = pd.DataFrame(y_train)
y_train.value_counts()/21000


# *** Here we can see that the churn_rate is imbalanced.
# To balance the data, we are using **SMOTE** ***

# ## Logistic Regression with balanced data using SMOTE

# In[ ]:


smote = SMOTE(random_state=0)
X_smote, y_smote = smote.fit_resample(X_train,y_train)


# In[ ]:


lr1 = LogisticRegression()
lr_model1 = lr1.fit(X_smote, y_smote)
print(classification_report(y_smote, lr_model1.predict(X_smote)))


# In[ ]:


y_smote.value_counts()


# #### Hence we can see that data is balanced from churn rate values above.

# In[ ]:


smote_test = SMOTE(random_state=0)
X_smote_test, y_smote_test = smote_test.fit_resample(X_test,y_test)
lr2 = LogisticRegression()
lr_model2 = lr2.fit(X_smote_test, y_smote_test)
print(classification_report(y_smote_test, lr_model2.predict(X_smote_test)))


# ### From the values shown above we observe 86% accuracy for Test Data and 85% accuracy for Train Data. 
# ### They are almost equal to each other.

# In[ ]:


# Important co-eff from the interpretable models
#total_og_mou_8,std_og_t2t_mou_8, std_og_t2m_mou_8, loc_og_t2t_mou_8, loc_og_t2m_mou_8


# In[ ]:


y_train = pd.DataFrame(y_train,columns=['churn_rate'])


# In[ ]:


y_train.value_counts()/21000


# In[ ]:


y_train.shape,X_train.shape


# In[ ]:


# predict churn on test data
y_pred_load = lr.predict(X_smote_test)

# create onfusion matrix
cm_load = confusion_matrix(y_smote_test, y_pred_load)
print(cm_load)
TP = cm_load[1,1] # true positive 

TN = cm_load[0,0] # true negatives
FP = cm_load[0,1] # false positives
FN = cm_load[1,0] # false negatives
#Sensitivity
LR_Smote_Sensitivity= TP / float(TP+FN)
LR_Smote_Specificity = TN / float(TN+FP)
LR_Smote_Accuracy = metrics.accuracy_score(y_smote_test, y_pred_load)
print('Sensitivity:' ,TP / float(TP+FN))
print('Specificity:' ,TN / float(TN+FP))
print("Accuracy :",metrics.accuracy_score(y_smote_test, y_pred_load))


# Here we can see that the churn_rate is imbalanced.
# To balance the data, we are using **SMOTE**

# In[ ]:


## Creating new row for this model in table.
y_pred_load = lr.predict(X_smote_test)
accuracy2,precision2,recall2,f1_2,specificity2,sensitivity2 = evaluation_table(y_smote_test, y_pred_load)
tempTable2 = pd.DataFrame({'Model':['Logistic Regression with balanced data using SMOTE'], 
                            'Accuracy': [accuracy2],
                            'Precision': [precision2],
                            'Recall': [recall2], 
                            'Sensitivity': [sensitivity2],
                            'Specificity': [specificity2],
                            'F1 score': [f1_2] })

table = pd.concat([table, tempTable2])
table = table[['Model', 'Accuracy', 'Precision', 'Recall', 'Sensitivity','Specificity', 'F1 score' ]]
table


# ## High Performance Model

# ### Logistic regression using PCA

# In[ ]:


filtered_data_pca.shape


# In[ ]:


pca = Pipeline([('scaler', StandardScaler()), ('pca', PCA())])
pca.fit(filtered_data_pca)
churn_pca = pca.fit_transform(filtered_data_pca)


# In[ ]:


# extract pca model from pipeline
pca = pca.named_steps['pca']

# look at explained variance of PCA components
print(pd.Series(np.round(pca.explained_variance_ratio_.cumsum(), 4)*100))


# In[ ]:


# plot feature variance
features = range(pca.n_components_)
cumulative_variance = np.round(np.cumsum(pca.explained_variance_ratio_)*100, decimals=4)
plt.figure(figsize=(175/20,100/20)) # 100 elements on y-axis; 175 elements on x-axis; 20 is normalising factor
plt.title("Explained Variance")
plt.xlabel("No. of components")
plt.ylabel("Cumulative variance")
plt.plot(cumulative_variance)
plt.show()


# ## PCA with Logistic regression

# In[ ]:


# creating pipeline
PCA_VARS = 95
steps = [('scaler', StandardScaler()),
         ("pca", PCA(n_components=PCA_VARS)),
         ("logistic", LogisticRegression(class_weight='balanced'))
        ]
pipeline = Pipeline(steps)


# In[ ]:


# fit model
pipeline.fit(X_train_pca, y_train_pca)

# check score on train data
pipeline.score(X_train_pca, y_train_pca)


# ### Evaluation on test

# In[ ]:


# predict churn on test data

y_pred = pipeline.predict(X_test_pca)

# create onfusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_pca, y_pred)
print(cm)

# check sensitivity and specificity
from sklearn.metrics import roc_auc_score
TP = cm[1,1] # true positive 
TN = cm[0,0] # true negatives
FP = cm[0,1] # false positives
FN = cm[1,0] # false negatives
#Sensitivity
print('Sensitivity:' ,TP / float(TP+FN))
print('Specificity:' ,TN / float(TN+FP))
print("Accuracy :",metrics.accuracy_score(y_test_pca, y_pred))

# check area under curve
y_pred_prob = pipeline.predict_proba(X_test_pca)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_test_pca, y_pred),2))


# In[ ]:


accuracy3,precision3,recall3,f1_3,specificity3,sensitivity3 = evaluation_table(y_test_pca, y_pred)
tempTable3 = pd.DataFrame({'Model':['Logistic regression using PCA'], 
                            'Accuracy': [accuracy3],
                            'Precision': [precision3],
                            'Recall': [recall3], 
                            'Sensitivity': [sensitivity3],
                            'Specificity': [specificity3],
                            'F1 score': [f1_3] })

table = pd.concat([table, tempTable3])
table = table[['Model', 'Accuracy', 'Precision', 'Recall', 'Sensitivity','Specificity', 'F1 score' ]]
table


# ## Hyperparameter tuning - PCA and Logistic Regression
# 

# In[ ]:


y_train.value_counts()/y_train.shape


# In[ ]:


# PCA
pca = PCA()

# logistic regression - the class weight is used to handle class imbalance - it adjusts the cost function
logistic = LogisticRegression(class_weight={0:0.1, 1: 0.9})

# create pipeline
steps = [("scaler", StandardScaler()), 
         ("pca", pca),
         ("logistic", logistic)
        ]

# compile pipeline
pca_logistic = Pipeline(steps)

# hyperparameter space
params = {'pca__n_components': [60, 80], 'logistic__C': [0.1, 0.5, 1, 2, 3, 4, 5, 10], 'logistic__penalty': ['l1', 'l2']}

# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)

# create gridsearch object
model = GridSearchCV(estimator=pca_logistic, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)


# In[ ]:


# fit model
model.fit(X_train_pca, y_train_pca)


# In[ ]:


# cross validation results
pd.DataFrame(model.cv_results_)


# In[ ]:


# print best hyperparameters
print("Best AUC: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)


# In[ ]:


X_test_pca.shape


# In[ ]:


# predict churn on test data
y_pred = model.predict(X_test_pca)

# create onfusion matrix
cm = confusion_matrix(y_test_pca, y_pred)
print(cm)


# check area under curve
y_pred_prob = model.predict_proba(X_test_pca)[:, 1]
print("AUC:    \t", round(roc_auc_score(y_test_pca, y_pred_prob),2))


TP = cm[1,1] # true positive 
TN = cm[0,0] # true negatives
FP = cm[0,1] # false positives
FN = cm[1,0] # false negatives
#Sensitivity
print('Sensitivity:' ,TP / float(TP+FN))
print('Specificity:' ,TN / float(TN+FP))
print("Accuracy :",metrics.accuracy_score(y_test_pca, y_pred))


# In[ ]:


accuracy4,precision4,recall4,f1_4,specificity4,sensitivity4 = evaluation_table(y_test_pca, y_pred)
tempTable4 = pd.DataFrame({'Model':['Hyperparameter tuning - PCA and Logistic Regression'], 
                            'Accuracy': [accuracy4],
                            'Precision': [precision4],
                            'Recall': [recall4], 
                            'Sensitivity': [sensitivity4],
                            'Specificity': [specificity4],
                            'F1 score': [f1_4] })

table = pd.concat([table, tempTable4])
table = table[['Model', 'Accuracy', 'Precision', 'Recall', 'Sensitivity','Specificity', 'F1 score' ]]
table


# ## Random Forest with Hyperparameter tuning and Balanced class

# In[ ]:


# random forest - the class weight is used to handle class imbalance - it adjusts the cost function
forest = RandomForestClassifier(class_weight={0:0.1, 1: 0.9}, n_jobs = -1)

# hyperparameter space
params = {"criterion": ['gini', 'entropy'], "max_features": ['auto', 0.4]}

# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)

# create gridsearch object
model = GridSearchCV(estimator=forest, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)


# In[ ]:


# fit model
model.fit(X_train_pca, y_train_pca)


# In[ ]:


# print best hyperparameters
print("Best AUC: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)


# In[ ]:


# predict churn on test data
y_pred = model.predict(X_test_pca)

# create onfusion matrix
cm = confusion_matrix(y_test_pca, y_pred)
print(cm)
TP = cm[1,1] # true positive 
TN = cm[0,0] # true negatives
FP = cm[0,1] # false positives
FN = cm[1,0] # false negatives
#Sensitivity
print('Sensitivity:' ,TP / float(TP+FN))
print('Specificity:' ,TN / float(TN+FP))
print("Accuracy :",metrics.accuracy_score(y_test_pca, y_pred))


# ## Random Forest without Hyperparameters

# In[ ]:


# run a random forest model on train data
max_features = int(round(np.sqrt(X_train_pca.shape[1])))    # number of variables to consider to split each node
print(max_features)

rf_model = RandomForestClassifier(n_estimators=100, max_features=max_features, class_weight={0:0.1, 1: 0.9}, oob_score=True, random_state=4, verbose=1)


# In[ ]:


# fit model
rf_model.fit(X_train_pca, y_train_pca)


# In[ ]:


# OOB score
rf_model.oob_score_


# In[ ]:


len(X_train_pca.columns)


# In[ ]:


# predict churn on test data
y_pred = rf_model.predict(X_test_pca)

# create onfusion matrix
cm = confusion_matrix(y_test_pca, y_pred)
print(cm)

TP = cm[1,1] # true positive 
TN = cm[0,0] # true negatives
FP = cm[0,1] # false positives
FN = cm[1,0] # false negatives
#Sensitivity
print('Sensitivity:' ,TP / float(TP+FN))
print('Specificity:' ,TN / float(TN+FP))
print("Accuracy :",metrics.accuracy_score(y_test_pca, y_pred))


# In[ ]:


## 
accuracy5,precision5,recall5,f1_5,specificity5,sensitivity5 = evaluation_table(y_test_pca, y_pred)
tempTable5 = pd.DataFrame({'Model':['Random Forest with Hyperparameter tuning and Balanced class'], 
                            'Accuracy': [accuracy5],
                            'Precision': [precision5],
                            'Recall': [recall5], 
                            'Sensitivity': [sensitivity5],
                            'Specificity': [specificity5],
                            'F1 score': [f1_5] })

table = pd.concat([table, tempTable5])
table = table[['Model', 'Accuracy', 'Precision', 'Recall', 'Sensitivity','Specificity', 'F1 score' ]]
table


# ## XGBoost Classification

# In[ ]:


import xgboost as xgb

from xgboost import XGBClassifier
from sklearn import metrics

xgclf = xgb.XGBClassifier()

xgclf.fit(X_smote, y_smote)


# In[ ]:


print('AUC on train data by XGBoost =', metrics.roc_auc_score(y_true=y_smote,
                                                              y_score=xgclf.predict_proba(X_smote)[:, 1]))


# In[ ]:


print('AUC on test data by XGBoost =', metrics.roc_auc_score(y_true=y_smote_test,
                                                             y_score=xgclf.predict_proba(X_smote_test)[:, 1]))


# In[ ]:


from sklearn import model_selection
xgb_model = xgb.XGBClassifier()

# Default-Run of default-hyperparameters
parameters = {'learning_rate': [0.3],
              'max_depth': [6],
              'min_child_weight': [1],
              'n_estimators': [100]}

scorer = metrics.make_scorer(metrics.roc_auc_score,
                             greater_is_better=True,
                             needs_proba=True,
                             needs_threshold=False)

clf_xgb = model_selection.GridSearchCV(estimator=xgb_model,
                                       param_grid=parameters,
                                       n_jobs=-1,
                                       cv=3,
                                       scoring=scorer,
                                       refit=True)

clf_xgb.fit(X_train, y_train)


# In[ ]:


print(clf_xgb.best_score_)


# In[ ]:



y_train_xgb = xgclf.predict(X_smote)
print("Train accuracy value:",metrics.accuracy_score(y_smote, y_train_xgb))


# In[ ]:


y_test_xgb = xgclf.predict(X_smote_test)
print("Test accuracy value:",metrics.accuracy_score(y_smote_test, y_test_xgb))


# In[ ]:


## 
accuracy6,precision6,recall6,f1_6,specificity6,sensitivity6 = evaluation_table(y_smote_test, y_test_xgb)
tempTable6 = pd.DataFrame({'Model':['XGBoost Classification'], 
                            'Accuracy': [accuracy6],
                            'Precision': [precision6],
                            'Recall': [recall6], 
                            'Sensitivity': [sensitivity6],
                            'Specificity': [specificity6],
                            'F1 score': [f1_6] })

table = pd.concat([table, tempTable6])
table = table[['Model', 'Accuracy', 'Precision', 'Recall', 'Sensitivity','Specificity', 'F1 score' ]]
table


# ## Feature Importance

# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(rf_model.feature_importances_,X_train.columns)), columns=['Value','Feature'])

plt.figure(figsize=(15,20))
# sns.color_palette("pastel")
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False),palette='gist_rainbow',linewidth=2.5)
plt.title('Random Forest with Hyperparameter tuning and Balanced class')
plt.tight_layout()
plt.show()


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


feature_imp=feature_imp.sort_values("Value",ascending = False)
feature_imp.head(20)


# <h1>
# <center><span style="color:DarkCyan"><span style="font-family:Helvetica;">-: Business Recommendation :-</span></span> </center>
# </h1>
# 

# 
# - ***We have seen from the above models that the accuracy values of the imbalanced data are high.***
# - ***The scores for the balanced data are reduced.***
# - ***The top 5 features which we recommend are  as follows :-***
# 
#     - <strong><em><span style="color:MediumVioletRed">Outgoing Calls</span></strong></em> - As analysed by the model(s), the rate of Churn of customers depend highly on their outgoing calls and it is among the top 10 features. Higher the frequency of outgoing calls, lesser would be the Churn Rate. In order to retain the high value customers,the company should provide offers on talktime recharges when purchased by customers.
#     - <strong><em><span style="color:MediumVioletRed">Incoming Calls</span></strong></em> - Churn Rate also depends vastly on incoming calls of the customers. Hence the telecom company should launch various kinds of discounts on recharges so that these valued customers do not churn.
#     - <strong><em><span style="color:MediumVioletRed">ISD Incoming Calls</span></strong></em> - This feature ranks fourth in the top 10 features and affects the churn rate significantly. Higher  ISD Incoming Calls suggest higher usage of the operator for the users. Hence our recommendation would be to reduce the rates of incoming ISD calls and also launch schemes for various countries so that customers are able to avail ISD calls whenever required.
#     - <strong><em><span style="color:MediumVioletRed">Roaming Calls</span></strong></em> - The rate of Roaming calls affect the use of the telecom service and therefore also affect the Churn rate of customers. It is advisable that roaming packs according to countries for a week or various time spans should be launched, so as to ease the use of calls.
#     - <strong><em><span style="color:MediumVioletRed">Night Packs</span></strong></em> - Night packs are being used by many customers and hence it is within the top 5 most important features. Monthly night packages or power recharges are crucial for retention of the customers and if not taken care of might lead to loss of these customers.
# 
# ***Apart from the points above , high value existing customers must be informed about new schemes , thus preventing them to churn.***
