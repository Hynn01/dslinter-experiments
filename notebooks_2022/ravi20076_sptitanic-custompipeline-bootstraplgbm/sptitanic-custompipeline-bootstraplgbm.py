#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# General imports:-
from math import floor;
import numpy as np;
import pandas as pd;
from scipy.stats import iqr, mode;
from termcolor import colored;
from warnings import filterwarnings;
from gc import collect;

import seaborn as sns;
import matplotlib.pyplot as plt;
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Sklearn and model imports:-
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import FunctionTransformer;
from sklearn_pandas import DataFrameMapper, gen_features;

from sklearn.preprocessing import LabelEncoder, StandardScaler;
from lightgbm import LGBMClassifier;
from xgboost import XGBClassifier;
from sklearn.metrics import auc, roc_auc_score;


# # Spaceship Titanic Classification:-

# In[ ]:


# Importing relevant data:-
xytrain = pd.read_csv('../input/spaceship-titanic/train.csv', encoding= 'utf8');
xtest = pd.read_csv('../input/spaceship-titanic/test.csv', encoding= 'utf8');
sub_fl = pd.read_csv('../input/spaceship-titanic/sample_submission.csv', encoding= 'utf8');

# Visualizing the data:-
print(colored(f"\nTrain data:-\n", color = 'blue', attrs= ['bold', 'dark']));
display(xytrain.head(5));
print(colored(f"\nTest data:-\n", color = 'blue', attrs= ['bold', 'dark']));
display(xtest.head(5));
print(colored(f"\nSample Submission:-\n", color = 'blue', attrs= ['bold', 'dark']));
display(sub_fl.head(5));

Ftre_Lst = list(xytrain.drop('Transported', axis=1).columns);
Target = 'Transported';

print(colored(f"\nModel Features:-\n", color = 'blue', attrs= ['bold', 'dark']));
print(colored(f"{Ftre_Lst}", color = 'blue'));


# # 1. Data visualization and pre-processing:-
# 
# In this section, we develop data visualizations and a strategy to fill nulls across the columns in the train-test sets

# In[ ]:


# Train-test information and description:-
print(colored(f"\nTrain set info\n", color=  'blue', attrs= ['bold', 'dark']));
display(xytrain.info());

print(colored(f"\nTest set info\n", color=  'blue', attrs= ['bold', 'dark']));
display(xtest.info());

print(colored(f"\nTrain set description\n", color=  'blue', attrs= ['bold', 'dark']));
display(xytrain.describe().transpose().style.format('{:,.2f}'));

print(colored(f"\nTest set description\n", color=  'blue', attrs= ['bold', 'dark']));
display(xtest.describe().transpose().style.format('{:,.2f}'));


# In[ ]:


# Target column balance plot:-
fig, ax = plt.subplots(1,1, figsize= (3,5));
xytrain.Transported.value_counts(normalize= True).plot.bar(ax= ax, color = 'tab:blue');
ax.set_title(f"\nTarget column analysis\n", color = 'tab:blue', fontsize= 12);
ax.grid(visible= True, linestyle= '--', which = 'both', color = 'lightgrey');
plt.show();


# In[ ]:


# Analyzing numerical feature distributions:-
Num_Ftre_Lst = list(xytrain.select_dtypes(include= np.number).columns);

fig, ax = plt.subplots(3,2, figsize=(20,14));
for i, col in enumerate(Num_Ftre_Lst):
    sns.distplot(x=xytrain[col], color = 'tab:blue', ax=ax[floor(i/2), i%2]);
    ax[floor(i/2), i%2].grid(visible= True, color= 'lightgrey', linestyle= '--', which= 'both');
plt.suptitle(f"Numerical feature distributions", color= 'tab:blue', fontsize=12);
plt.show();


# In[ ]:


# Analyzing object feature distributions:-
fig, ax = plt.subplots(2,2, figsize=(10,10));
for i, col in enumerate(['HomePlanet', 'CryoSleep', 'Destination', 'VIP']):
    sns.countplot(x=xytrain[col], color = 'tab:blue', ax=ax[floor(i/2), i%2]);
    ax[floor(i/2), i%2].grid(visible= True, color= 'lightgrey', linestyle= '--', which= 'both');
plt.suptitle(f"Object feature distributions", color= 'tab:blue', fontsize=12);
plt.show();


# In[ ]:


# Null check:-
_ = pd.concat((xytrain[Ftre_Lst].isna().sum(axis=0)/ len(xytrain), 
               xtest.isna().sum(axis=0)/ len(xtest)), axis=1).\
rename({0:'Train', 1:'Test'}, axis=1);

fig, ax = plt.subplots(1,1, figsize= (14,7));
_.plot.bar(ax=ax);
ax.grid(visible= True, linestyle= '--', which = 'both', color = 'lightgrey');
ax.set_title("\nNull records in the train-test data\n", color = 'tab:blue', fontsize= 12);
plt.show();

print(colored(f"\nNull records in the train-test data\n", 
              color='blue', attrs= ['bold', 'dark']));
display(_.drop('PassengerId').style.format('{:.2%}').highlight_max(color= 'lightblue', axis=0).       highlight_min(color = 'lightyellow', axis=0));


# In[ ]:


# Analyzing categorical feature levels and associated target states:-
for col in xytrain.drop(['Name', 'PassengerId', 'Cabin'], axis=1).select_dtypes(exclude= np.number).columns:
    print(colored(f"\nAnalysis for {col}\n", color = 'blue', attrs= ['dark', 'bold']));
    display(xytrain.groupby(col,dropna=False).    agg(Nb_Records=pd.NamedAgg('Transported', np.size),
        Nb_Transported = pd.NamedAgg('Transported', np.sum)));


# In[ ]:


# Analyzing the cabin column further:-
_cabin_prf = xytrain.Cabin.str.split('/', expand= True).add_prefix('Cabin').join(xytrain.Transported);
_cabin_prf['CabinCtg'] = _cabin_prf['Cabin0']+_cabin_prf['Cabin2'];

fig, ax = plt.subplots(1,1, figsize= (14,7));
_cabin_prf.groupby('CabinCtg').agg(Nb_Passengers= pd.NamedAgg('Transported',np.size), 
                                   Nb_Transported= pd.NamedAgg('Transported',np.sum)
                                  ).plot.bar(ax= ax);
ax.set_title(f"\nCabin category verus transported passengers\n", color= 'tab:blue', fontsize= 12);
ax.grid(visible= True, linestyle= '--', which = 'both', color = 'lightgrey');
ax.set_yticks(range(0,1600,100), fontsize= 8);
ax.set_xlabel("\nCabin Category");
plt.show();


# In[ ]:


# Generating cross-tab between cabin category and destination:-
_ = _cabin_prf.join(xytrain.Destination).groupby(['Destination', 'CabinCtg']).agg({'Transported': [np.size, np.sum]}).reset_index().pivot(index= 'CabinCtg', columns= 'Destination');

_.columns = [j+'-'+k for i, j,k in _.columns.to_flat_index()];

print(colored(f"\nCross-tab between cabin category and destination\n", 
              color = 'blue', attrs= ['bold', 'dark']));
display(_.style.format(precision = 0));

del _;


# In[ ]:


# Generating cross-tab between cabin category and embarkation:-
_ = _cabin_prf.join(xytrain.HomePlanet).groupby(['HomePlanet', 'CabinCtg']).agg({'Transported': [np.size, np.sum]}).reset_index().pivot(index= 'CabinCtg', columns= 'HomePlanet');

_.columns = [j+'-'+k for i, j,k in _.columns.to_flat_index()];

print(colored(f"\nCross-tab between cabin category and HomePlanet\n", 
              color = 'blue', attrs= ['bold', 'dark']));
display(_.style.format(precision = 0));

del _;


# In[ ]:


# Generating cross-tab between cabin category and VIP status:-
_ = _cabin_prf.join(xytrain.VIP).groupby(['VIP', 'CabinCtg']).agg({'Transported': [np.size, np.sum]}).reset_index().pivot(index= 'CabinCtg', columns= 'VIP');
_.columns = [j+'-'+str(k) for i, j,k in _.columns.to_flat_index()];

print(colored(f"\nCross-tab between cabin category and VIP-status\n", 
              color = 'blue', attrs= ['bold', 'dark']));
display(_.style.highlight_max(axis=0, color = 'lightblue').        highlight_min(axis=0, color = 'lightyellow').        format('{:,.0f}'));

del _;


# In[ ]:


# Analyzing the correlation between the training features and the target:-
fig, ax = plt.subplots(1,1,figsize= (14,8));
sns.heatmap(data=xytrain.corr(), cmap = 'Spectral_r', ax=ax,linecolor= 'black', center=True,
            linewidth = 1.0, annot= True, fmt= '.2%');
ax.set_title(f"Correlation heatmap for train data\n", color = 'black', fontsize= 12);
plt.yticks(rotation= 45, fontsize= 8);
plt.xticks(rotation= 45, fontsize= 8);
plt.show();


# In[ ]:


# Analyzing the HomePlanet and destination interaction feature:-

_ = pd.concat((xytrain['HomePlanet']+ ' - ' + xytrain['Destination'], xytrain.Transported), axis=1).rename({0:'Journey_Lbl'}, axis=1).groupby('Journey_Lbl').agg({'Transported': [np.size, np.sum]});
_.columns = ['Nb_Passengers', 'Nb_Transported'];

fig, ax = plt.subplots(1,1, figsize= (14,9));
_.plot.bar(ax =ax);
ax.set_title("Interaction- journey details\n", fontsize= 12, color = 'black');
ax.grid(visible= True, linestyle= '--', which = 'both', color = 'lightgrey');
ax.set_xlabel('');
ax.set_yticks(range(0,3300,100));
plt.xticks(rotation= 45, fontsize= 7);
plt.show();

del fig, ax;

print(colored(f"\nInteraction- journey details\n", color=  'blue', attrs= ['bold', 'dark']));
display(_.style.highlight_max(axis=0, color = 'lightblue').
        highlight_min(axis=0, color = 'lightyellow').
        format('{:,.0f}'));
del _;


# In[ ]:


# Analyzing customer spending by VIP-status and cryosleep:-
filterwarnings('ignore');
_ = xytrain[['CryoSleep','VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']];
_['Total_Spend'] = _.select_dtypes(include= np.number).sum(axis=1);

print(colored(f"Cross-tab:- Total Spend versus Cryosleep and VIP status\n",
      color = 'blue', attrs= ['dark', 'bold']));
display(_.groupby(['CryoSleep', 'VIP'])['Total_Spend'].describe().        style.highlight_max(axis=0, color= 'lightblue').        format('{:,.2f}'));

# Analyzing transported passengers versus VIP-status and cryosleep:-
_xtab = _[['VIP','CryoSleep']].join(xytrain.Transported).groupby(['VIP','CryoSleep']).agg({'Transported': [np.size, np.sum]});
_xtab.columns = ['Nb_Passengers', 'Nb_Transported'];
_xtab['Rt_Transported'] = _xtab['Nb_Transported']/ _xtab['Nb_Passengers'];

print(colored(f"\nCross-tab:- Total transported passengers versus Cryosleep and VIP status\n",
      color = 'blue', attrs= ['dark', 'bold']));
display(_xtab.style.format({'Nb_Passengers': '{:,.0f}','Nb_Transported': '{:,.0f}',
                           'Rt_Transported': '{:.2%}'}));

del _xtab;


# In[ ]:


# Analyzing total spending and target distribution by age:-
_ = _[['Total_Spend', 'VIP', 'CryoSleep']].join(xytrain[['Age', 'Transported']]);
_['Lifestage'] = np.select([_.Age < 13.0, _.Age<18.0], ['1.Child', '2.Teen'], '3.Adult');

print(colored(f"Cross-tab:- Total transported passengers versus Cryosleep,VIP status and Lifestage\n",
      color = 'blue', attrs= ['dark', 'bold']));
display(
_.groupby(['Lifestage','VIP','CryoSleep']).\
agg({'Transported': [np.size, np.sum],'Total_Spend': [np.mean, np.median]}).\
style.format('{:,.0f}').highlight_max(axis=1, color= 'lightblue').\
highlight_min(axis=1, color= 'lightyellow')
);


# In[ ]:


# Analyzing Name and Passenger ID to elicit filial relations:-
_ = pd.concat((xytrain['PassengerId'].str.split('_', expand= True).add_prefix('ID'),
               xytrain['Name'].str.split(' ', expand= True).\
               rename({0:'FirstName', 1:'LastName'}, axis=1),
              _[['Total_Spend']],
              xytrain[['CryoSleep', 'Transported', 'VIP']]
              ),
              axis=1);
_['CryoSleep'] = np.where(_.CryoSleep == True, 1,0);

# Pooling spending details and passengers per family name:-
_family_dtl = _.groupby(['VIP','LastName']).agg(Total_Spend = pd.NamedAgg('Total_Spend',np.sum),
    Nb_Passengers = pd.NamedAgg('Transported', np.size),
    Nb_Transported = pd.NamedAgg('Transported', np.sum),
    Nb_CryoSleep = pd.NamedAgg('CryoSleep', np.sum)
   ).\
reset_index().sort_values('Total_Spend', ascending= False);

_family_dtl['Rt_Transport'] = _family_dtl['Nb_Transported']/ _family_dtl['Nb_Passengers'];

# Plotting the top 20 spender families:-
fig, ax= plt.subplots(1,1, figsize= (20,8));
sns.barplot(data= _family_dtl.head(20),y= 'Total_Spend', x= 'LastName', 
            palette= 'Blues', ax=ax);
ax.set_title("Top 20 spender families\n", color= 'tab:blue', fontsize=12);
ax.grid(visible= True, which= 'both', linestyle= '--', color= 'lightgrey');
ax.set_yticks(range(0,53000,2000));
ax.set_xlabel(f"\nFamily Name\n", fontsize=12);
ax.set_ylabel(f"Spending\n", fontsize=12);
plt.xticks(rotation=45);
plt.show();


# ## Key notes and inferences:-
# 
# 1. People in cryosleep did not spend any money throughout the trip
# 2. Transportation rate for cryosleep > others
# 3. VIP passengers also have higher transportation rate than others
# 4. Children have a higher propensity for cryosleep and transportation
# 5. Children under 12 years of age did not spend any money at all (could be a policy)
# 

# # 2. Data Transformation Pipeline:-
# 
# In this section, we develop a pipeline that elicits the data transformation process and returns the associated model train-test sets. Custom classes and functions are used to develop the pipeline.
# 
# This is divided into 4 functions as below-
# 1. Add features- create new columns for name, cabinID components and treat nulls in float columns for amenities
# 2. Treat nulls in age and cryosleep columns based on family name and spending details
# 3. Create journey column based on null treated home planet and destination, based on family name
# 4. Treat nulls in VIP and cabin columns based on family names
# 
# The pipeline is then developed using FunctionTransformer as all of these functions are stateless

# In[ ]:


def AddFeatures(X: pd.DataFrame):
    "This function adds the name split, total spending features and treats nulls in amenities";
    
    df = pd.concat((X[Ftre_Lst].drop(['Name', 'Cabin'],axis=1),
                X.Name.str.split(' ', expand= True).add_prefix('Name'),
                X.Cabin.str.split('/', expand= True).add_prefix('Cabin')), axis=1);

    df[['RoomService', 'FoodCourt','ShoppingMall','Spa','VRDeck']]=    df[['RoomService', 'FoodCourt','ShoppingMall','Spa','VRDeck']].fillna(0.0);
    df['TotalSpend'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] +     df['Spa'] + df['VRDeck'];
    
    df['VIP'] = df['VIP']*1.0;
    df['VIP'] = df['VIP'].astype(np.float16);
    
    return df;


# In[ ]:


def TrtNullAgeCrSlp(df:pd.DataFrame):
    """
    This function fills nulls in age and cryosleep as below-
    Cryosleep:-
    1. For non-spenders, cryosleep=1 as cryosleep customers don't spend
    2. If age <=12 and cryosleep is null, then cryosleep= 1
    3. For spenders, cryosleep= 0
    
    Age:-
    1. For spenders/ non-cryosleep, median age > 12 for family is considered (child cannot spend)
    2. For all remaining nulls, overall median age is used
    
    Flag for Is_Child (Age <=12) is also created
    """;
    
    # 1. Filling nulls in cryosleep based on spending and age details:-
    df['CryoSleep'] = np.float16(df['CryoSleep']*1.0);
    df.loc[(df.CryoSleep.isna()==True) & (df.TotalSpend == 0.0), ['CryoSleep']] = 1.0;
    # Assuming child (age <=12) and null cryosleep = cryosleep
    df.loc[(df.CryoSleep.isna()==True) & (df.Age <=12), ['CryoSleep']] = 1.0;
    # Assuming no cryosleep for spenders:-
    df.loc[(df.CryoSleep.isna()==True) & (df['TotalSpend'] > 0.0), ['CryoSleep']] = 0.0;
    df['CryoSleep'] = df['CryoSleep'].astype(np.int8);
    
    # 2. Assuming average family age for spenders:-
    df = df.merge(df.loc[df.Age >12,['Name1', 'Age']].dropna().groupby('Name1').                  agg(_Age= pd.NamedAgg('Age', np.median)),
                  how= 'left', left_on= 'Name1', right_on='Name1', suffixes= ('',''));
    df.loc[(df.Age.isna()==True) & ((df.TotalSpend > 0.0) | (df.CryoSleep==0)), ['Age']] = df._Age;
    # Filling median age for remaining nulls:-
    df['Age'] = df['Age'].fillna(df.Age.median());
    
    # 3. Creating flag for child:-
    df['Is_Child'] = np.where(df.Age <= 12, 1,0);
    df['Is_Child'] = df['Is_Child'].astype(np.int8);
    
    df = df.drop(['_Age'], axis=1);  
    df['Age'] = df['Age'].astype(np.int8);
    return df;


# In[ ]:


def CreateJourney(df: pd.DataFrame):
    """
    This function treats nulls in HomePlanet and Destination and combines them to form Journey
    1. Home Planet:-
    a. Based on family name, home planet nulls are filled (all family members have same home planet)
    b. Remaining nulls are filled using the overall mode
    
    2. Destination:-
    a. Based on family name, mode of destination is created and filled up for nulls
    b. For all remaining nulls, overall mode is used
    
    3. Journey = HomePlanet - Destination is the interaction feature
    """;
    
    # 1. Fostering null treatment for HomePlanet based on last name and overall mode:-
    df = df.merge(df[['Name1', 'HomePlanet']].drop_duplicates().dropna(), 
                 how= 'left',left_on= 'Name1', right_on= 'Name1', suffixes= ('', '_'));
    df['HomePlanet'] = df['HomePlanet'].fillna(df.HomePlanet_);
    df['HomePlanet'] = df['HomePlanet'].fillna(df[['HomePlanet']].                                               apply(lambda x: x.mode()).values[0][0]);

    # 2. Fostering null treatment for destination based on last name and overall mode:-
    _ = df[['Name1', 'Destination']].groupby('Name1')['Destination'].    value_counts(ascending= False);
    _.name = 'Nb_Destination';
    _ = _.reset_index().groupby(['Name1']).head(1);

    df = df.merge(_, how = 'left', left_on= 'Name1', right_on= 'Name1', suffixes= ('', '_'));
    df['Destination'] = df['Destination'].fillna(df['Destination_']);
    df['Destination'] = df['Destination'].fillna(df[['Destination']].                                                 apply(lambda x: x.mode()).values[0][0]);
    del _;

    # 3. Developing interaction column for journey:-
    df['Journey'] = df['HomePlanet'] + ' - ' + df['Destination'];
    df = df.drop(['HomePlanet','Destination','HomePlanet_','Destination_','Nb_Destination'], 
                 axis=1,errors = 'ignore');

    return df;


# In[ ]:


def TrtNullVIPCabin(df: pd.DataFrame):
    """
    This function treats nulls in VIP and cabin columns using the last name.
    We assume that members of the same family have the same cabin and VIP IDs
    As an addition, it downcasts the float64 columns to conserve memory.
    """;

    # Assuming that members of the same family have the same VIP ID:-
    df = df.merge(df[['VIP', 'Name1']].groupby('Name1')['VIP'].max(), 
                how = 'left', left_on= 'Name1', right_on= 'Name1', suffixes= ('','_'));
    df['VIP'] = df['VIP'].fillna(df.VIP_);
    df['VIP'] = df['VIP'].fillna(0.0);
    df['VIP'] = df['VIP'].astype(np.int8);

    # Assuming that members of the same family have the same cabin0/ cabin2 ID:-
    _ = df[['Cabin0', 'Name1']].groupby('Name1')['Cabin0'].value_counts();
    _.name = 'Nb_Records';
    df = df.merge(_.reset_index().groupby('Name1').head(1).drop('Nb_Records', axis=1), 
                how= 'left', left_on= 'Name1', right_on= 'Name1', suffixes= ('','_'));
    df['Cabin0'] = df['Cabin0'].fillna(df.Cabin0_);
    del _;

    _ = df[['Cabin2', 'Name1']].groupby('Name1')['Cabin2'].value_counts();
    _.name = 'Nb_Records';
    df = df.merge(_.reset_index().groupby('Name1').head(1).drop('Nb_Records', axis=1), 
                how= 'left', left_on= 'Name1', right_on= 'Name1', suffixes= ('','_'));
    df['Cabin2'] = df['Cabin2'].fillna(df.Cabin2_);

    df = df.drop(['Cabin0_', 'Cabin2_'], axis=1, errors= 'ignore');
    del _;

    # Considering remaining nulls with overall cabin mode based on VIP status:-
    df = df.merge(df.groupby(['VIP']).agg({'Cabin0': lambda df: df.mode(), 'Cabin2': lambda y: y.mode()}),
                how= 'left', left_on= 'VIP', right_index= True, suffixes= ('', '_'));
    df['Cabin0'] = df['Cabin0'].fillna(df.Cabin0_);
    df['Cabin2'] = df['Cabin2'].fillna(df.Cabin2_);
    df = df.drop(['Cabin0_', 'Cabin2_', 'VIP_'], axis=1, errors= 'ignore');
 
    # Downcasting columns to conserve memory:-    
    df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','TotalSpend']] =     df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','TotalSpend']].astype(np.float16);  
    
    # Dropping extra columns after usage:-
    df = df.drop(['Cabin1', 'Name0', 'Name1'], axis=1, errors= 'ignore');
    return df;


# In[ ]:


# Organizing the target columns:-
ytrain = np.where(xytrain[['Transported']] == True, 1,0).ravel();

# Collating train data for the feature transformation:-
Xtrain = xytrain.drop('Transported', axis=1);

# Developing the data transformer pipeline:-
Data_Xformer=Pipeline(steps= 
         [('AddFeatures', FunctionTransformer(AddFeatures)),
          ('TrtNullAgeCrSlp', FunctionTransformer(TrtNullAgeCrSlp)),
          ('CreateJourney', FunctionTransformer(CreateJourney)),
          ('TrtNullVIPCabin', FunctionTransformer(TrtNullVIPCabin)),
          ('LblEncode', DataFrameMapper(input_df= True, df_out= True, drop_cols= ['PassengerId'],default=None,
                                       features=gen_features(columns= [['Journey'], ['Cabin0'], ['Cabin2']],
                                                             classes= [LabelEncoder])
                                       ))
         ], verbose= True);

# Implementing the pipeline on the training set and test set:-
Xtrain = Data_Xformer.fit_transform(Xtrain, ytrain);
Xtest = Data_Xformer.transform(xtest);

print(colored(f"\nTrain-Test pipeline implementation results", color= 'blue', attrs= ['bold', 'dark']));
print(colored(f"{len(Xtrain), len(Xtest)}", color = 'blue'));

print(colored(f"\nTrain-set pipeline output columns", color= 'blue', attrs= ['bold', 'dark']));
print(colored(f"{list(Xtrain.columns)}", color = 'blue'));

print(colored(f"\nTest-set pipeline output columns", color= 'blue', attrs= ['bold', 'dark']));
print(colored(f"{list(Xtest.columns)}", color = 'blue'));

collect();


# In[ ]:


# Plotting correlation plot after the pipeline:-
fig, ax= plt.subplots(1,1, figsize= (16,10));
sns.heatmap(Xtrain.corr(), cmap = 'icefire', annot= True, fmt= '.1%', 
            linewidth= 1.0, linecolor= 'black', ax=ax, center= True);
ax.set_title("Correlation heatmap after the pipeline implementation\n", fontsize= 12, color= 'tab:blue');
plt.yticks(rotation= 45, fontsize= 8);
plt.xticks(rotation= 45, fontsize= 8);
plt.show();


# # 3. Model Training and Development:-
# 
# In this section, we train ML models (tree based models) and develop the test set predictions using the pipeline output. We follow the below routine:-
# 
# 1. Sample the train data using a higher fraction (say- 90%) with different random states. Develop the model train, development sets thereby
# 2. Use an ensemble approach on the sampled model train set and use the validation set for the scoring metric calculation
# 3. Make a test set prediction using the model trained and so fitted. Predicting probabilities seems to be a better option than predicting labels
# 4. Store the test set prediction in an output dataframe 
# 5. Once all models are executed, consider the probability prediction average across the n-model runs and use it as a final prediction
# 6. Consider calibrating the model if needed. This is a last decision after the model development. Mostly, calibration is not necessary

# In[ ]:


# Creating the output dataframe for test predictions:-
Mdl_Pred_Prf = pd.DataFrame(data= None, index= xtest['PassengerId'], columns= None);
sample_frac = 0.95;
n_mdl_runs= 250;

for i in range(0,n_mdl_runs,1):
    print(colored(f"\nIteration{i}", color= 'blue', attrs= ['bold', 'dark']));
    xtr = Xtrain.sample(random_state = i, frac= sample_frac);
    xdev = Xtrain.loc[~Xtrain.index.isin(xtr.index)];
    ytr = ytrain[xtr.index];
    ydev = ytrain[xdev.index];
    
    print(colored(f"LGBM:-", color = 'blue'));
    mdl = LGBMClassifier(n_estimators= 500, objective= 'binary', learning_rate = 0.08);
    mdl.fit(xtr, ytr, eval_set=[(xtr, ytr), (xdev, ydev)], eval_metric= ['auc','binary_logloss'], 
            early_stopping_rounds= 50,verbose= 50);
    Mdl_Pred_Prf[f"LBGM{i}"] = mdl.predict_proba(Xtest)[:,1];
    
    print(colored(f"XGBoost:-", color = 'blue'));
    mdl = XGBClassifier(n_estimators = 500);
    mdl.fit(xtr, ytr, eval_metric= 'auc', eval_set=[(xtr, ytr), (xdev, ydev)], 
            early_stopping_rounds= 50,verbose= 50);
    Mdl_Pred_Prf[f"XGBoost{i}"] = mdl.predict_proba(Xtest)[:,1];
    
    del xtr, xdev, ytr, ydev;
    collect();


# In[ ]:


# Preparing the submission file:-
Mdl_Pred_Prf['Transported'] = Mdl_Pred_Prf.median(axis=1);
Mdl_Pred_Prf['Transported'] = np.where(Mdl_Pred_Prf['Transported'] >=0.50, True,False)

Mdl_Pred_Prf.reset_index()[['PassengerId', 'Transported']].to_csv("submission.csv", index= False);

