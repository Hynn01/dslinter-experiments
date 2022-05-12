#!/usr/bin/env python
# coding: utf-8

# # Various models to predict Autism spectrum Disorders (ADS)
# _**Autism Spectrum Disorder prediction with supervised models for binary classification**_
# 
# ---
# 
# ## Contents
# 
# 1. [Summary](#Summary)
# 1. [Preparation](#Preparation)
#     1. [Data Loading](#Data_Loading)
# 1. [Preprocessing](#Preprocessing)
#     1. [Exploratory Data Analysis](#Exploratory_Data_Analysis)
#     1. [Feature enginering & selection](#Feature_enginering_&_selection)
# 1. [Training](#Training)
#     1. [Random Forest](#Random_Forest)
#     1. [Logistic Regression](#Logistic_Regression)
#     1. [XGBoost](#XGBoost)
# 1. [Testing](#Testing)
# 1. [Conclusion](#Conclusion)
# 
# ---
# 
# # Summary
# [Back to Contents](#Contents)
# 
# Autism Spectrum Disorders (ASD) are an ensemble of neurodevelopmental psychiatric disorders that cause diverse social and cognitive impairments. These impairments are associated with changes in the brain's early maturation of cortical circuits, where genetic and socio-cultural aspects are most prominent. However, neuropsychological evaluation at early stages is complex, and standard psychological examination is absent in young patients. 
# 
# This notebook aims to predict the likelihood of a patient with ASD using survey and demographic variables and standard psychological tests. We used several supervised models based and monitored their efficacy in [Weights and Biases](https://wandb.ai) that you can visit [here](https://wandb.ai/neurohost/ASD/https://wandb.ai/neurohost/ASD/)
# 
# ---
# 
# # Acknoledgements
# 
# We thank [Tensor Girl](https://www.kaggle.com/usharengaraju) for hosting the competition, [Satoshi Datamoto](https://www.kaggle.com/satoshidatamoto) for suggesting the feature selection, and [Mahsa Zamanifard](https://www.kaggle.com/mahsazamanifard) for resampling to increase the scoring of previous versions of the notebook. To [Bala Baskar](https://www.kaggle.com/balabaskar) for the idea of label encoding based on the frequency of the categorical variable. Also, to [Andrada Olteanu](https://www.kaggle.com/andradaolteanu) for providing the [country-mapping dataset](https://www.kaggle.com/datasets/andradaolteanu/country-mapping-iso-continent-region)
# 
# # Preparation
# [Back to Contents](#Contents)
# 
# This notebook uses scikit-learn Transformers for data enginering and trains common supervised models for a binary classification task. 
# 
# It requires familiarity with:
# * Standard scientific modules for data handling (pandas),
# * Modules for scientific analysis (Scipy, NumPy and machine learning Scikit-learn)
# * The scientific library for data visualization (matplotlib).
# 
# It also assumes that basic machine learning methods, like test/train split and hyperparameter searching. It is otherwise possible to follow the notebook with a minimal background.
# 
# 

# In[ ]:


# data handling
import pandas as pd

# numerical analysis
import numpy as np

# OS-independent path
import pathlib

# data visualization (See Preprocessing -> Exploratory Data Analysis)
import matplotlib.pyplot as plt
plt.style.use('https://raw.githubusercontent.com/JoseGuzman/minibrain/master/minibrain/paper.mplstyle') # minibrain plotting

# Display Pipelines and models
from sklearn import set_config
set_config(display='diagram')

# other (e.g. for python >=3.6 type definitions like age: int = 1)
from typing import List, Tuple

# A progress bar
from tqdm import tqdm

# my utility script
from reducing import PandaReducer


# # Data_Loading
# [Back to Contents](#Contents)
# 
# We first load test and train datasets into a Pandas DataFrame object and the target variable
# as a Pandas DataSeries object.

# In[ ]:


# Data Loading
# Define file paths, test and train files
mypath = pathlib.Path('../input/autismdiagnosis/Autism_Prediction/')
train_file = mypath / 'train.csv'
test_file = mypath / 'test.csv'
type(test_file)


# In[ ]:


# Data Loading
def data_loader(file:pathlib.PosixPath, target:str=None, verbose:bool = False, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads csv file and return a tuple with a pandas dataset 
    containing all features, and a pandas Series with the 
    target variable.
    """
    
    data = pd.read_csv(file, **kwargs)
    df = PandaReducer().reduce(data) # see reduce.py in Utility Script 
    
    if target is not None:
        #target = (df[target] == 'yes').astype(int)
        target = (df[target]).astype(int)
    
    if verbose:
        print('The dataset contains {0} entries and {1} features'.format(*df.shape))
    
    return df, target


# # 

# In[ ]:


# Data Loading
train, train_target = data_loader(file = train_file, target = 'Class/ASD', verbose=True, index_col='ID')
test, _ = data_loader(file = test_file, target=None, verbose=True, index_col='ID') # note target is None


# In[ ]:


train.head(n=5)


# # Preprocessing
# [Back to Contents](#Contents)
# 
# We first need a brief exploration of variable types and target variable. 

# ## Exploratory_Data_Analysis
# [Back to Contents](#Contents)
# 
# The visualization of types and distributions of the -independent- variables, also called features. 

# In[ ]:


train.info()


# In[ ]:


# Exploration: there are no missing values in both train and test datasets
test.isnull().values.any(), train.isnull().values.any() # 


# In[ ]:


# Exploration: evaluate if data is uniformly distributed
train['Class/ASD'].value_counts() # data is umbalanced!


# In[ ]:


# drop target variable, we have it in train_target
train.drop(['Class/ASD'],axis=1, inplace=True) 


# In[ ]:


# Exploration: visualization of target variable

autistic = train_target.value_counts() 

fig, ax = plt.subplots(1,2, figsize =(6,3))
fig.tight_layout(pad = 1, h_pad = 2, w_pad = 4)

mylabels = ['ASD (No)', 'ASD (Yes)']
mycolors = ['tab:blue', 'tab:orange']

ax[0].bar(x=mylabels, height = autistic, color = mycolors, width = 0.75, alpha = .6)

for tick in ax[0].get_xticklabels():
    tick.set_rotation(90)
ax[0].set_ylabel('counts')#, ax[0].set_yticks(np.arange(0,4000,500))

ax[1].pie(autistic.values, labels = mylabels, colors = mycolors, autopct='%2.2f%%',shadow=True, startangle=90);


# In[ ]:


train.describe() # numeric variables, 


# It seems like **A_*Score** are the evaluation tests, and vary between 0 and 1. We need to provide normalization to **age** and **result**

# In[ ]:


test.describe() # numeric variables, 


# We see the content of the different variables in test and train datasets. It is important, because the evaluation is made on the test sets, and we want all the variables and contents to be trained before (in train dataset).

# In[ ]:


# train.select_dtypes(['category']) # category variables

for col in train.select_dtypes(['category']).columns :
    myval = train[col].unique().tolist()
    if len(myval) == 2:
        print(f'BIVARIATE   : {col} -> {train[col].unique().tolist()}')
    elif len(myval) == 1:
        print(f'UNIVARIATE   : {col} -> {train[col].unique().tolist()}')
    else:
        print(f'MULTIVARIATE: {col} -> {train[col].unique().tolist()}')


# In[ ]:


for col in test.select_dtypes(['category']).columns :
    myval = test[col].unique().tolist()
    if len(myval) == 2:
        print(f'BIVARIATE   : {col} -> {test[col].unique().tolist()}')
    elif len(myval) == 1:
        print(f'UNIVARIATE   : {col} -> {test[col].unique().tolist()}')
    else:
        print(f'MULTIVARIATE: {col} -> {test[col].unique().tolist()}')


# In[ ]:


train['age_desc'].unique(), test['age_desc'].unique() # this variable is not informative, we will remove it


# In[ ]:


# let me check if both datasets relations are idential
print(f'{"test":25s} --   {"train":10s}')
print(f'{"="*45}')
for i,j in zip(np.sort(test.relation.unique()), np.sort(train.relation.unique())):
    print(f'{i:25s} --   {j:15s}')
    
len(test.relation.unique()), len(train.relation.unique())    


# In[ ]:


# let me check if both datasets are representatives
print(f'{"test":15s} --   {"train":10s}')
print(f'{"="*35}')
for i,j in zip(np.sort(test.ethnicity.unique()), np.sort(train.ethnicity.unique())):
    print(f'{i:15s} --   {j:15s}')
len(test.ethnicity.unique()), len(train.ethnicity.unique())


# In[ ]:



for i,j in enumerate(np.sort(test.ethnicity.unique())):
    print(f'{i:2d} --   {j:15s}')


# In[ ]:


for i,j in enumerate(np.sort(train.ethnicity.unique())):
    print(f'{i:2d} --   {j:15s}')


# In[ ]:


train['ethnicity'].value_counts()


# In[ ]:


# To homogeneize both datasets, I will remove three records with 'others'
#train = train[train.ethnicity != 'others']
#df.drop(df.loc[df['line_race']==0].index, inplace=True)

#df.drop(train.loc[train['ethnicity']==0].index, inplace=True)
del_idx = train.loc[train['ethnicity']=='others'].index
train.drop(del_idx, inplace=True)
#train = train[~train.ethnicity.str.contains("others")]
train_target.drop(del_idx, inplace=True)


# In[ ]:


train['ethnicity'].value_counts()


# In[ ]:


# If different ethnicit this will affect one-hot-encoding
len(test.ethnicity.unique()), len(train.ethnicity.unique())


# In[ ]:


# Different ethnicities, this will affect one-hot-encoding!!!
len(train['contry_of_res'].unique()), len(test['contry_of_res'].unique())


# In[ ]:


#Different countries, this will affect one-hot-encoding
np.array_equiv(train['contry_of_res'].unique(), test['contry_of_res'].unique())


# Countries are different, we will collect all of them into continents using [this dataset](https://www.kaggle.com/datasets/andradaolteanu/country-mapping-iso-continent-region) to train with continents.

# In[ ]:


# We will map country

# Load the dataset of country/region
data = pd.read_csv('../input/country-mapping-iso-continent-region/continents2.csv') # small dataset, no need to reduce size


print(data.region.unique())
continent = pd.Series(data.region.values, index = data.name).to_dict() # create a dictionary


train['region'] = train['contry_of_res'].map(continent) # could use .replace here
test['region'] = test['contry_of_res'].map(continent)

#assert(train.region.unique() == test.region.unique())

train[['contry_of_res','region']].head(n=10)


# In addition, we rank the countries by frequency of appareance in the train dataset, as suggested in [this notebook](https://www.kaggle.com/code/balabaskar/autism-prediction-eda-with-0-827-score)

# In[ ]:



# create a dataframe with train and test to account for all possible countries
df = pd.DataFrame(train['contry_of_res'].append(test['contry_of_res'], ignore_index = True))
ncountry = df['contry_of_res'].nunique()
print(f'Number of unique countries = {ncountry}')

ranking = range(1,ncountry+1)[::-1]
country = df['contry_of_res'].value_counts().index.to_list()
rank_country = dict( zip(country, ranking))
print(f'Length ranking dictionary  = {len(rank_country)}')


# ## Feature_enginering_&_selection
# [Back to Contents](#Contents)
# 
# We first start some some custom definitions for custom Transformers and PipeLines.

# A ColumnTransformer will apply the transformation to a single feature or list of features. An alternative methods is to use common Pipelines where the column transformation is defined at the initialization of the
# transformer. I tend to combine both methods
# 
# 
# _Look [here](https://machinelearningmastery.com/columntransformer-for-numerical-and-categorical-data/) to learn how to use ColumnTransformer_

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin


# In[ ]:


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Custom feature for Label encoding based on frequency 
    of categorical variables.
    To use it:
    >>> myrank = {'united states': 1, 'spain':10}
    >>> mydropper = FrequencyEncoder(features = ['country_of_res'], rank = myrank)
    >>> df = mydropper.fit_transform(X = train)
    """

    def __init__(self, col_name:str, rank:dict)-> None:
        """
        Remove the list of features from a pandas 
        Dataframe object.
        
        Parameter
        ---------
        col_name:  the variable to remove
        rank: (dict) containg the variable and frequency to 
        be substitued (eg. myrank = {'united states': 1, 'spain':10}
        
        """

        self.col_name = col_name
        self.rank = rank
        self.df = None
   
       

    def fit(self, X:pd.DataFrame, y = None):
        """
        Remove the column lists and update dataset
        """
        df = X.copy()
                
        df[self.col_name] = df[self.col_name].map(self.rank)
        
        self.df = df
        return self
   
    def transform(self, X:pd.DataFrame = None) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with removed features.
        
        Parameter
        ---------
        dataframe:  Pandas DataFrame object
        """
        df = X.copy()
        
        # Drop features
        df[self.col_name] = df[self.col_name].map(self.rank)
        
        return self.df


# In[ ]:


fencoder = FrequencyEncoder(col_name = 'contry_of_res', rank = rank_country)
df = fencoder.fit_transform(X= train)
df['contry_of_res']


# In[ ]:


class RegionTransformer(BaseEstimator, TransformerMixin):
    """
    Custom feature transform a Country in one of the 
    five continents of the world, 'Asia' 'Europe' 
    'Africa' 'Oceania' 'Americas', or 'nan'.
    To use it:
    >>> myregion = RegionTransformer(continent = data)
    >>> df = myregion.fit_transform(X = train)
    """

    def __init__(self, continent:dict) -> None:
        """
        Remove the list of features from a pandas 
        Dataframe object.
        
        Parameter
        ---------
        continent:  (dict) of countries/continent pairs.
        """

        self.continent = continent
        self.df = None
    
    
    def get_feature_names_out(self) -> List[str]:
        """
        Get column names (necessary for Pipelines)
        """
        
        if self.df is None:
            mycols = ['None']
        else:
            mycols =  self.df.columns.tolist()
            
        return mycols
        

    def fit(self, X:pd.DataFrame, y = None):
        """
        Remove the column lists and update dataset
        """
        
        df = X.copy()
        df['region'] = df['contry_of_res'].map(self.continent)
        self.df = df
        
        return self
   
    def transform(self, X:pd.DataFrame = None) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with removed features.
        
        Parameter
        ---------
        dataframe:  Pandas DataFrame object
        """
        
        df = X.copy()
        
        # Add region to dataset
        df['region'] = df['contry_of_res'].map(self.continent)
        return self.df


# In[ ]:


myregion = RegionTransformer(continent = continent)
df = myregion.fit_transform(X = test)
myregion.get_feature_names_out()


# In[ ]:


class DropperTransformer(BaseEstimator, TransformerMixin):
    """
    Custom feature dropper to add to custom Pipelines.
    To use it:
    >>> mydropper = DropperTransformer(features = ['age'])
    >>> df = mydropper.fit_transform(X = train)
    """

    def __init__(self, features:List[str])-> None:
        """
        Remove the list of features from a pandas 
        Dataframe object.
        
        Parameter
        ---------
        features:  (list) of variables to remove
        """

        self.features = features
        self.df = None
    
    
    def get_feature_names_out(self)-> List[str]:
        """
        Get column names (necessary for Pipelines)
        """
        if self.df is None:
            mycols = ['None']
        else:
            mycols =  self.df.columns.tolist()
            
        return mycols
        

    def fit(self, X:pd.DataFrame, y = None):
        """
        Remove the column lists and update dataset
        """
        df = X.copy()
        self.df = df.drop(self.features, axis = 1)
        
        return self
   
    def transform(self, X:pd.DataFrame = None) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with removed features.
        
        Parameter
        ---------
        dataframe:  Pandas DataFrame object
        """
        df = X.copy()
        
        # Drop features
        self.df = df.drop(self.features, axis = 1)
        return self.df
        


# In[ ]:


mydropper = DropperTransformer(features = ['age_desc'])
mydropper.fit(train)
#df = mydropper.fit_transform(train)
for i, col in enumerate(mydropper.get_feature_names_out()):
    print(f'{i:2d} - {col}')


# ## Feature selection with Pearson's Chi-square Test for Independence
# * $\chi^2$ statistic for testing binary categorical variables relationship to categorical. Chi-square is applied to categorical variables and is especially useful when those variables are nominal (where order doesn't matter, like marital status or gender).
# 
# The null hypothesis (H0) of the Chi-Square test is that no relationship exists on the categorical variables in the population; they are independent. If the probability that the null hypothesis (H0) is higher than 5%, then the null hypothesis is valid (variables are thus independent). Note that I tend to use the test in the opposite direction (p<0.05 to test the dependency between variables)

# In[ ]:


from sklearn.feature_selection import chi2


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


binary_features = ['gender', 'jaundice', 'austim', 'used_app_before']
binary_encoder = OneHotEncoder(sparse=False, drop= 'if_binary')
#foo = binary_encoder.fit_transform(train[binary_features])
#cols = binary_encoder.get_feature_names_out()

chi_score, p = chi2(X = binary_encoder.fit_transform(train[binary_features])  ,y = train_target)
star = p <= 0.05 # if they are dependent, probability of Ho (independency) must be lower than 5%
pd.DataFrame(zip(binary_features, chi_score, p, star), columns=['category', 'Chi-Square', 'p-value', 'P<0.05'])


# The null hypothesis (H0) of the Chi-Square test is that no relationship exists on the categorical variables in the population (i.e. the variable is indepedent). If we discard categorical variables with probability 5% or more, we discard **gender** and **used_app_before**.

# In[ ]:


# Note that A*_Score are also binary variables, we will test if they are related to the independent variable
mylist = [f'A{i}_Score' for i in range(1,10)]

chi_score, p = chi2(X = binary_encoder.fit_transform(train[mylist])  ,y = train_target)
star = p <= 0.05
pd.DataFrame(zip(mylist, chi_score, p, star), columns=['category', 'Chi-Square', 'p-value', 'P<0.05'])


# All A*_Scores are related (with probability of 95% or more) to the depedent variable.

# ## Designing preprocessing Pipeline

# In[ ]:


# Transformation: create processing pipeline
#from sklearn.pipeline import make_pipeline # to concatenate estimators and transformers (like Pipeline)
#from sklearn.compose import make_column_transformer # to apply transformers to categories (ColumnTransformer)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# We test first the indices of the variables resulting after our custom transformers. These indices are important because ColumnTransformed used indices in the Pipeline.

# In[ ]:


fencoder = ('country_encoder', FrequencyEncoder(col_name = 'contry_of_res', rank = rank_country))
country = ('country', RegionTransformer(continent = continent))
dropper = ('dropper', DropperTransformer(features = ['age_desc', 'gender', 'used_app_before']))

in_process = (country, dropper)
pipeline = Pipeline(steps = in_process)
df = pipeline.fit_transform(train)
#pipeline[1].get_feature_names_out()
for i,col in enumerate(pipeline[1].get_feature_names_out()):
    print(f'[{i:2d}] -> {col}')


# In[ ]:


# =====================================================================
# Tuples for Pipeline contain only the 'key' & Transformers
# =====================================================================
fencoder = ('country_encoder', FrequencyEncoder(col_name = 'contry_of_res', rank = rank_country))
country = ('country', RegionTransformer(continent = continent))
dropper = ('dropper', DropperTransformer(features = ['age_desc', 'gender', 'used_app_before']))

# =====================================================================
# Tuples for ColumnTransformer contain only the 'key',Transformers,col
# =====================================================================
z_scoring = ('z_scoring', StandardScaler(), [10, 14] ) # age, result
binarize = ('binarize', OneHotEncoder(sparse=False, drop= 'if_binary'), [12,13] ) # jaundice, austim
one_hot =  ('one_hot',  OneHotEncoder(sparse=False, handle_unknown='ignore'), [11, 16, 17] ) # ethnicity, relation, region

col_transformer = ColumnTransformer(transformers = (z_scoring, binarize, one_hot), remainder= 'passthrough')

col_preprocess = ('col_transformer', col_transformer)

preprocess = Pipeline( steps = (fencoder, country, dropper, col_preprocess))#, RandomForestClassifier(random_state = 42))
preprocess
#make_pipeline(mypipeline, RandomForestClassifier(random_state=42))


# In[ ]:


# Check the number of resulting variables are the same
preprocess.fit_transform(X=train).shape, preprocess.fit_transform(X=test).shape


# In[ ]:


# =====================================================================
# Tuples for make_pipeline contain only the Transformers
# =====================================================================
#country = (RegionTransformer(continent = continent))
#dropper = (DropperTransformer(features = ['age_desc', 'contry_of_res']))

# =====================================================================
# Tuples for ColumnTransformer contain only the 'key',Transformers,col
# =====================================================================
#z_scoring = ('z_scoring', StandardScaler(), [10, 17] ) # age, result
#binarize = ('binarize', OneHotEncoder(sparse=False, drop= 'if_binary'), [11, 13, 14, 16] ) # gender, jaundice, austim, used_app_before
#one_hot =  ('one_hot',  OneHotEncoder(sparse=False, handle_unknown='ignore'), [12, 15, 18] ) # ethnicity, country_of_res, relation

#col_preprocess = ColumnTransformer(transformers = (z_scoring, binarize, one_hot), remainder= 'passthrough')

#preprocess = make_pipeline(country, dropper, col_preprocess)# note we don't need key, value tuples
#preprocess
#make_pipeline(mypipeline, RandomForestClassifier(random_state=42))


# # Training
# [Back to Contents](#Contents)  
# 
# We will test the accuracy of our models when training the dataset with the most common classification methods.

# In[ ]:


# We first apply the preprocessing pipeline
Xtrain = preprocess.fit_transform(X = train)
Xtest = preprocess.fit_transform(X = test)

# check resulting variables are the same after preprocessing
assert(Xtrain.shape[1] == Xtest.shape[1])

# check the same number of independent variables
assert(Xtrain.shape[0] == train_target.shape[0])

#Xtrain.shape, Xtest.shape, train_target.shape


# In[ ]:


# minimal reporting here
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score


# In[ ]:


# minimal train/test split
from sklearn.model_selection import train_test_split

# we train with one-third of the dataset
X_train, X_test, y_train, y_test = train_test_split(Xtrain, train_target, test_size=1/5., random_state=42)


# ## Random_Forest
# [Back to Contents](#Contents)
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


RF_clf = RandomForestClassifier(random_state = 42) # instance of model with default methods


# In[ ]:



RF_clf.fit(X = X_train, y = y_train) # training with 4/5 of the data

prediction = RF_clf.predict(X = X_test) # predict the rest 1/5 

print(classification_report(y_test, prediction)) # accuracy 0.82


# In[ ]:


def plot_metrics(model, X:np.array, y_target:np.array) -> plt.figure:
    """
    Plots confusion matrix and Receiver Operating Characteristic
    (ROC) curve of the classifier
    
    Arguments:
    ----------
    predictor
    X (array): input matrix
    y_target (array) : target vector
    """
    prediction = model.predict( X )
    
    # Compute donfusion matrix
    cm = confusion_matrix(y_target, prediction, labels = model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)
    
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    fig.tight_layout(pad = 3, h_pad = 2, w_pad = 4)
    fig.suptitle(type(model).__name__)
    ax[0] = disp.plot(ax = ax[0])

    
    # ROC curve
    y_pred_prob = model.predict_proba(X)[::,1]
    test_FP, test_TP, thresholds = roc_curve(y_target ,y_pred_prob)
    auc = roc_auc_score(y_target, y_pred_prob)
    
    ax[1].plot(test_FP, test_TP, color='C0', label = f'AUC = {auc:2.2f}')
    ax[1].plot([0, 1], [0, 1],'r--', lw=1)

    ax[1].legend(loc =4, fontsize=10);

    ax[1].set_ylabel('True Positive (TP)', fontsize=10);
    ax[1].set_xlabel('False Positive (FP)', fontsize=10);

    ax[1].set_title('Receiver Operating Characteristic (ROC) curve', fontsize=10);
    
    #return fig


# In[ ]:


# RandomForest performance on test dataset (1/5 of the training dataset)
plot_metrics(model = RF_clf,  X = X_test, y_target = y_test)


# In[ ]:


def get_predictionfile(prediction: np.array, filename:str):
    """
    Return submission file
    
    Arguments:
    prediction: (array)
        the estimator result of predict() method  
    filename : (str)
        filename to be saved
    """
    # test input size
    check_shape = prediction.shape == (200,)
    assert check_shape, f'prediction shape expected (200,), got: {prediction.shape}'
    
    df = pd.DataFrame({'ID': test.index, 'Class/ASD': prediction})
    print(f'{filename} with {df.shape[0]} predictions created')
    return df.to_csv(filename, index=False)
    


# In[ ]:


# we test our trained model with the Xtest restulting from the competition 
get_predictionfile(RF_clf.predict_proba(Xtest)[:,1], 'RF_submission.csv') # Score: 


# ## Logistic_Regression
# [Back to Contents](#Contents)

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


LR_clf = LogisticRegression(random_state = 42,solver='liblinear')#, max_iter = 1500)


# In[ ]:



LR_clf.fit(X = X_train, y = y_train) # training with 4/5 of the data

prediction = LR_clf.predict(X = X_test) # predict the rest 1/5 

print(classification_report(y_test, prediction)) # accuracy 0.84


# In[ ]:


# Logistic Regression performance on test dataset (1/5 of the training dataset)
plot_metrics(model = LR_clf,  X = X_test, y_target = y_test)


# In[ ]:


# we test our trained model with the Xtest restulting from the competition 
get_predictionfile(LR_clf.predict_proba(Xtest)[:,1], 'LR_submission.csv') # Score: 0.69984


# ## Decission_Tree
# [Back to Contents](#Contents)

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


DT_clf = DecisionTreeClassifier(random_state=42)


# In[ ]:



DT_clf.fit(X = X_train, y = y_train) # training with 4/5 of the data

prediction = DT_clf.predict(X = X_test) # predict the rest 1/5 

print(classification_report(y_test, prediction)) # accuracy 0.81


# In[ ]:


# Logistic Regression performance on test dataset (1/5 of the training dataset)
plot_metrics(model = DT_clf,  X = X_test, y_target = y_test)


# In[ ]:


# we test our trained model with the Xtest restulting from the competition 
get_predictionfile(prediction = DT_clf.predict_proba(Xtest)[:,1], filename = 'DT_submission.csv') # Score: 0.65117


# ## Test models
# [Back to Contents](#Contents)
# 
# We will test a list of models and perform an hyperparameter search hyperparameter tunning and evaluate accuracy and some other metrics.

# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

myKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # balanced split categories


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import BernoulliNB





# 1. Instances of all models 
RF_clf = RandomForestClassifier(random_state = 42)
AD_clf = AdaBoostClassifier(random_state = 42)
LR_clf = LogisticRegression(random_state = 42, solver='liblinear', max_iter = 1500)
DT_clf = DecisionTreeClassifier(random_state=42)

KN_clf = KNeighborsClassifier( )
SVC_clf = SVC(degree=10, probability = True, random_state = 42)

NB_clf = BernoulliNB( )
GB_clf = GradientBoostingClassifier(random_state=42)


# Prepare hyperparameter dictionary of each estimator each having a key as ‘classifier’ and value as estimator object. The hyperparameter keys should start with the word of the classifier separated by ‘__’ (double underscore). Check [this link](https://towardsdatascience.com/how-to-tune-multiple-ml-models-with-gridsearchcv-at-once-9fcebfcc6c23)

# In[ ]:



# ================================================
# Random Forest 
# ================================================
param_RF = {}
param_RF['classifier__n_estimators'] = [10, 50, 100, 250]
param_RF['classifier__max_features'] = ['auto', 'sqrt', 'log2']
param_RF['classifier__max_depth'] = [5, 10, 20]
param_RF['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param_RF['classifier'] = [RF_clf]

# ================================================
# Adaboost 
# ================================================
param_AD = {}
param_AD['classifier__n_estimators'] =  [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
param_AD['classifier__learning_rate'] =  [(0.97 + x / 100) for x in range(0, 8)],
param_AD['classifier__algorithm'] =  ['SAMME', 'SAMME.R']
param_AD['classifier'] = [AD_clf]

# ================================================
# Logistic Regression
# ================================================
param_LR = {}
param_LR['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
param_LR['classifier__penalty'] = ['l1', 'l2']
param_LR['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param_LR['classifier'] = [LR_clf]

# ================================================
# Decission Tree
# ================================================
param_DT = {}
param_DT['classifier__max_depth'] = [5,10,25,None]
param_DT['classifier__min_samples_leaf'] = [2,5,10]
param_DT['classifier__criterion'] = ["gini", "entropy"]
param_DT['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param_DT['classifier'] = [DT_clf]

# ================================================
# k-Nearest Neighbours
# ================================================
param_KN = {}
param_KN['classifier__n_neighbors'] = [5,7,9,11,13,15],
#param_KN['classifier__weights'] = ['uniform','distance'],
#param_KN['classifier__metric'] = ['minkowski','euclidean','manhattan']
param_KN['classifier'] = [KN_clf]

# ================================================
# Support Vector Classifier
# ================================================
param_SVC = {}
param_SVC['classifier__C'] =  [0.1, 1, 10, 100], 
param_SVC['classifier__gamma'] = [1,0.1,0.01,0.001],
param_SVC['classifier__kernel'] = ['rbf', 'poly', 'sigmoid']
param_SVC['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
param_SVC['classifier'] = [SVC_clf]

# ================================================
# Naive Bayes
# ================================================
param_NB = {}
param_NB['classifier__alpha'] = np.logspace(0,-9, num=100)
param_NB['classifier'] = [NB_clf]

# ================================================
# Gradient Boosting
# ================================================
param_GB = {}
param_GB['classifier__n_estimators'] = [10, 50, 100, 250]
param_GB['classifier__max_depth'] = [5, 10, 20]
param_GB['classifier'] = [GB_clf]


# In[ ]:


# IList the hyperparameter dictionary and prepare a pipeline of the 1st classifier
pipeline = Pipeline([('classifier', RF_clf)])
myparams = [param_RF, param_AD, param_LR, param_DT, param_KN, param_SVC, param_NB, param_GB]


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Train a RandomizedSearchCV model with the pipeline and parameter dictionary list.\n\nmymodel = RandomizedSearchCV(pipeline, myparams, cv=myKFold, n_jobs=-1, scoring='roc_auc').fit(X = Xtrain, y = train_target)")


# In[ ]:


mymodel.best_params_


# In[ ]:


# ROC-AUC score for the best model
mymodel.best_score_


# # Submission

# In[ ]:


mymodel.best_estimator_


# In[ ]:


predictions = pd.DataFrame(mymodel.best_estimator_.predict_proba(Xtest))
predictions


# In[ ]:


# check https://www.kaggle.com/competitions/autismdiagnosis/discussion/323303
get_predictionfile(mymodel.best_estimator_.predict_proba(Xtest)[:,1], 'submission.csv') # score: 0.84344


# In[ ]:




