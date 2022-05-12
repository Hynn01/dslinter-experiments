#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing general packages:-
import numpy as np;
import pandas as pd;
from scipy.stats import iqr, skew, kurtosis, mode;
from scipy.fft import rfft;
from warnings import filterwarnings;
from termcolor import colored;
import gc;

import seaborn as sns;
import matplotlib.pyplot as plt;
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(10);
pd.options.display.float_format = '{:.2f}'.format;


# In[ ]:


# Important sklearn and ensemble specific packages:-
from sklearn.base import BaseEstimator, TransformerMixin;
from sklearn.pipeline import Pipeline;
from sklearn_pandas import DataFrameMapper, gen_features;
from sklearn.preprocessing import FunctionTransformer,RobustScaler, StandardScaler;
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score;
from sklearn.decomposition import PCA;
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score;

from sklearn.tree import DecisionTreeClassifier as DTree;
from xgboost import XGBClassifier;
from lightgbm import LGBMClassifier;
from catboost import CatBoostClassifier;


# # Tabular Playground Series- April 2022:-

# # 1. Data Loading and pre-model checks
# 
# We load the train-test data into the kernel, elicit sample data to assess the structures and engender basic checks for further steps in this section

# In[ ]:


# Loading the relevant data-sets with basic checks:-
xtrain = pd.read_csv('../input/tabular-playground-series-apr-2022/train.csv', encoding = 'utf8');
ytrain = pd.read_csv('../input/tabular-playground-series-apr-2022/train_labels.csv', encoding = 'utf8');
xtest = pd.read_csv('../input/tabular-playground-series-apr-2022/test.csv', encoding = 'utf8');
sub_fl = pd.read_csv('../input/tabular-playground-series-apr-2022/sample_submission.csv', encoding = 'utf8');

print(colored(f"Train, train-label,test set lengths = {len(xtrain), len(ytrain), len(xtest)}\n", 
              color = 'blue', attrs= ['bold']));

print(colored(f"\nTrain-test data samples\n", color = 'blue', attrs= ['bold', 'dark']));
display(xtrain.head(5));
print('\n');
display(xtest.head(5));

print(colored(f"\nSample submission\n", color = 'blue', attrs= ['bold', 'dark']));
display(sub_fl.head(5));

print(colored(f"\nTrain-set columns\n{list(xtrain.columns)}\n", color = 'blue'));
print(colored(f"\nTest-set columns\n{list(xtest.columns)}\n", color = 'blue'));

print(colored(f"\nTrain labels-set columns\n{list(ytrain.columns)}\n", color = 'blue'));
print(colored(f"\nTrain labels sample\n", color = 'blue', attrs = ['bold', 'dark']));
display(ytrain.head(5));


# In[ ]:


# Checking if the target class is balanced/ imbalanced:-
fig, ax= plt.subplots(1,1, figsize= (4,6));
sns.barplot(y= ytrain.state.value_counts(normalize= False), x= ytrain.state.unique(), palette= 'Blues',
            saturation= 0.90, ax=ax);
ax.set_title(f"Target column distribution for (im)balanced classes\n", color= 'tab:blue', fontsize= 12);
ax.set_yticks(range(0,14001,1000));
ax.set_xlabel('Target Class');
plt.show();


# ### Model Development Plan:-
# 
# These tables are structured as below-
# 1. xtrain:- This encapsulates the training features only. Each sequence number is associated with 60 time steps with numerical readings for that sequence number from 13 sensors on a unique subject number for the sequence.
# 2. ytrain:- This is the target table, with sequence numbers and class labels for the sequence. This data is balanced, as seen in the previous cell target distribution, hence, no over-sampling is needed
# 3. xtest:- This continues the sequences from the train-set but we are unaware of the classification state
# 
# Model development requires feature engineering with new features encapsulating the sensor readings' descriptive statistics over the sequence number. 
# New columns including the mean, std, skewness, kurtosis, IQR, median, etc. can be created across all 13 sensors and their efficacy in the model development may be assessed for inclusion. 
# Classifier models like an LSTM/ ML models could be used after eliciting relevant features and pre-processing

# # 2. Feature processing
# 
# In this section, we analyse the train-set features, plot graphs as deemed necessary and elicit model development specific inferences from the raw data. We prepare an interim table xytrain with all features and the target state to facilitate lower computational burden during the analysis herewith.

# In[ ]:


# Displaying train-set information and description:-
print(colored(f"\nTrain-set information\n", color = 'blue', attrs = ['bold', 'dark']));
display(xtrain.info());

print(colored(f"\nTrain-set description\n", color = 'blue', attrs = ['bold', 'dark']));
display(xtrain.describe().transpose().style.format('{:,.1f}'));

print(colored(f"\nTrain-test set dtypes\n", color = 'blue', attrs = ['bold', 'dark']));
display(pd.concat((xtrain.dtypes, xtest.dtypes), axis= 1).rename({0: 'Train_Dtypes', 1: 'Test_Dtypes'}, axis=1));


# In[ ]:


# Plotting correlation heatmap for the train data sensor readings:-
fig, ax= plt.subplots(1,1, figsize= (18,10));
sns.heatmap(data= xtrain.iloc[:, -13:].corr(), vmin= 0.0, vmax=1.00, annot= True, fmt= '.1%',
            cmap= sns.color_palette('Spectral_r'), linecolor='black', linewidth= 1.00,
            ax=ax);
ax.set_title('Correlation heatmap for the train-set sensor data\n', fontsize=12, color= 'black');
plt.yticks(rotation= 0, fontsize= 9);
plt.xticks(rotation= 90, fontsize= 9);
plt.show();


# In[ ]:


# Preparing an interim dataframe with all the training features and their state for analysis:-
xytrain = xtrain.merge(ytrain, how= 'left', on='sequence');


# In[ ]:


# Plotting sensor readings with the target to elicit mutual information and importance:-
fig, ax= plt.subplots(1,1, figsize= (12,6));
xytrain.drop(['sequence', 'subject', 'step'], axis=1).corr()[['state']].drop('state').plot.bar(ax= ax);
ax.set_title("Correlation analysis for all sensor columns\n", color= 'tab:blue', fontsize= 12);
ax.grid(visible= True, which= 'both', color= 'grey', linestyle= '--', linewidth= 0.50);
ax.set_xlabel('\nColumns', color= 'black');
ax.set_ylabel('Correlation', color= 'black');
plt.show();


# In[ ]:


# Plotting boxplots to study the column distributions:-

fig, ax= plt.subplots(1,1, figsize= (18,10));
sns.boxplot(data= xtrain.iloc[:, -13:], ax=ax);
ax.set_title(f"Distribution analysis for sensor data in train-set\n", color= 'tab:blue', fontsize= 12);
ax.set_yticks(range(-600,700,100));
ax.grid(visible=True, which='both', color= 'lightgrey', linestyle= '--');
ax.set_xlabel('\nSensor Columns\n', fontsize= 12, color= 'tab:blue');
ax.set_ylabel(f'Sensor Readings\n', fontsize= 12, color= 'tab:blue');

plt.xticks(rotation= 90);
plt.show()


# ### Subject analysis:- 
# 
# This sub-section elicits key insights derived from the train-test set subjects as below-
# 1. We plan to study the common subject characteristics for state
# 2. We also plan to develop descriptive statistics of sensor readings based on subjects 
# 3. We will check if data leakage exists between the train-test subjects for any manual adjustments over the model results at the end of the assignment

# In[ ]:


# Analyzing subject characteristics:-
sub_prf_train= xytrain[['subject', 'sequence', 'state']].drop_duplicates().set_index('sequence').pivot_table(index= 'subject', values= 'state', aggfunc= [np.size, np.sum]);
sub_prf_train.columns= ['Nb_Min', 'Nb_S1'];
sub_prf_train['Nb_S0'] = sub_prf_train['Nb_Min'] - sub_prf_train['Nb_S1'];
sub_prf_train['S1_Rate'] = sub_prf_train['Nb_S1']/ sub_prf_train['Nb_Min'];

sub_prf_train.sort_values(['S1_Rate'], ascending= False);

print(colored(f"\nTrain set subject inferences:-", color= 'red', attrs= ['bold', 'dark']));
print(colored(f"Number of train-set subjects = {len(sub_prf_train)}", color = 'blue'));
print(colored(f"Number of train-set subjects never going to state 1 = {len(sub_prf_train.query('S1_Rate == 0.0'))}", 
              color = 'blue'));
print(colored(f"Number of train-set subjects never going to state 0 = {len(sub_prf_train.query('S1_Rate == 1.0'))}", 
              color = 'blue'));

print(colored(f"\nDescriptive summary statistics for the training subjects\n", color = 'red', attrs= ['bold']));
display(sub_prf_train.iloc[:,:-1].describe().transpose().style.format('{:.1f}'));

print(colored(f"\nDescriptive summary statistics for the test-set subjects\n", color = 'red', attrs= ['bold']));
display(xtest[['sequence', 'subject']].drop_duplicates().groupby(['subject']).agg(Nb_Min = pd.NamedAgg('sequence', np.size)).describe().transpose().style.format('{:.1f}'));

print(colored(f"\nDescriptive summary statistics for the training subjects never in state 1\n", 
              color = 'red', attrs= ['bold']));
display(sub_prf_train.loc[sub_prf_train.S1_Rate== 0.0].describe().transpose().style.format('{:.1f}'));

print(colored(f"\nSensor summary statistics for the training subjects never in state 1\n", 
              color = 'red', attrs= ['bold']));
display(xtrain.loc[xtrain.subject.isin(sub_prf_train.loc[sub_prf_train.S1_Rate== 0.0].index), 
           xtrain.columns.str.startswith('sensor')].describe().transpose().style.format('{:,.1f}'));

print(colored(f"\nSensor summary statistics for the training subjects in state 1 and 0\n", 
              color = 'red', attrs= ['bold']));
display(xtrain.loc[xtrain.subject.isin(sub_prf_train.loc[sub_prf_train.S1_Rate > 0.0].index), 
           xtrain.columns.str.startswith('sensor')].describe().transpose().style.format('{:,.1f}'));

print(colored(f"\nSensor summary statistics for all training subjects\n", color = 'red', attrs= ['bold']));
display(xtrain.loc[:,xtrain.columns.str.startswith('sensor')].describe().transpose().style.format('{:,.1f}'));


# In[ ]:


# Plotting unique sequences per subject:-

_ = xytrain[['sequence', 'subject', 'state']].drop_duplicates().             groupby(['subject','state'])['sequence'].nunique().reset_index().             pivot_table(index= 'subject', columns= 'state', values= 'sequence', aggfunc= [np.sum]);
_.columns = ['Nb_Unq_Seq0', 'Nb_Unq_Seq1'];

fig, ax= plt.subplots(2,1, figsize= (18,15));

sns.lineplot(data= _ , palette= 'rainbow', ax= ax[0], linestyle= '-');
ax[0].set_title(f"Number of unique sequences per subject in the training set\n", color= 'black', fontsize= 12);
ax[0].legend(loc= 'upper right', fontsize= 8);
ax[0].set_xlabel('Subjects\n', color= 'black', fontsize= 10);
ax[0].set_ylabel('Sequences', color= 'black', fontsize= 10);
ax[0].grid(visible= True, which= 'both', linestyle= '-', color= 'lightgrey');
ax[0].set_xticks(range(0, 680, 25));
ax[0].set_yticks(range(0, 181, 15));

sns.lineplot(data=_.loc[sub_prf_train.loc[sub_prf_train.S1_Rate== 0.0].index][['Nb_Unq_Seq0']].values, 
             palette= 'Dark2',ax= ax[1]);
ax[1].set_title(f"\nNumber of unique sequences per subject in the training set never in state1\n",
                color= 'black', fontsize= 12);
ax[1].grid(visible= True, which= 'both', linestyle= '-', color= 'lightgrey');
ax[1].set_xticks(range(0, 65, 5));
plt.show();

del _;


# In[ ]:


# Analyzing the distributions of the sensor readings across state:-
_ = xytrain.iloc[0:2,:].columns[xytrain.iloc[0:2,:].columns.str.startswith('sensor_')];
for col in _:
    fig, ax= plt.subplots(1,1, figsize = (12,3.5));
    sns.kdeplot(data=xytrain[[col, 'state']], x= col, hue="state", multiple="stack", palette = 'rainbow', ax= ax);
    ax.grid(visible= True, which= 'both', color= 'lightgrey');
    ax.set_xlabel('');
    ax.set_title(f'\n{col}\n');
    plt.show();

del _;


# In[ ]:


# Analyzing the univariate characteristics for all sensors across the states:-

print(colored(f"\nState 0 descriptions\n", color= 'blue', attrs= ['bold', 'dark'])); 
display(xytrain.loc[xytrain.state == 0, xytrain.columns.str.startswith('sensor')].describe().transpose()        .style.format('{:,.2f}'));
print();

print(colored(f"\nState 1 descriptions\n", color= 'blue', attrs= ['bold', 'dark'])); 
display(xytrain.loc[xytrain.state == 1, xytrain.columns.str.startswith('sensor')].describe().transpose().        style.format('{:,.2f}'));


# In[ ]:


# Deleting the combined analysis table after usage:-
del xytrain;
gc.collect();


# ### Pipeline adjutant functions and classes:-
# 
# The below functions and classes are used to develop the pipeline for data transformation and processing

# In[ ]:


# Reducing memory usage by reassigning new datatypes to the relevant tables:-
def ReduceMemory(df:pd.DataFrame):
    """
    This function assigns new dtypes to the relevant dataset attributes and reduces memory usage.
    The relevant data-type is determined from the description seen earlier in the kernel.
    
    Input:- df (dataframe):- Analysis dataframe
    Returns:- df (dataframe):- Modified dataframe
    """; 
    
    df[['subject']] = df[['subject']].astype(np.int16);
    df[['sequence']] = df[['sequence']].astype(np.int32);
    df[['step']] = df[['step']].astype(np.int8);
    
    #  selecting all sensor float columns and reassigning data-types:-   
    _ = df.iloc[0, -13:].index;
    df[_] = df[_].astype(np.float32);
    del _;
    
    return df; 


# In[ ]:


class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    This class does the below tasks:-
    1. Generates a model dataframe object as return object for all transforms and all sensors
    2. Creates descriptive statistics based summaries 
    3. Appends each set of transforms to the master dataframe (mdl_df)
    4. Separately calculates the MAD and appends it to the model master dataframe
    5. Creates FFT based features if requested by the user and appends to the model dataframe
    6. Creates a global feature list to be used elsewhere in the code
    """;
    
    def __init__(self, FFT_req:str):self.FFT_req = FFT_req;    
    def fit(self, X, y= None, **fit_params): 
        if self.FFT_req == 'Y': self.nb_FFT = np.int8(len(X[['step']].drop_duplicates())/2+1);
        else: self.nb_FFT = 0;
            
        self.sensor_col = X.iloc[0,-13:].index; 
        return self;   
    
    # Creating FFT facilitator function:-
    def DoFFT(self, X):
        """This function generates real valued Fast Fourier Transforms for the sensor readings in the dataframe""" ;        
        FFT_df =         pd.concat([pd.Series(np.abs(rfft(X[col].values)), 
                             index=[f'freq{i}_{col}' for i in range(self.nb_FFT)]) 
               for col in self.sensor_col]);
        return FFT_df; 
        
    def transform(self, X, y= None, **transform_params):
        "This function provides the transformed features for the dataframe";
        
        # Creating output master dataframe with sequence and subject:- 
        mdl_df = X[['sequence', 'subject']].drop_duplicates().set_index('sequence'); 
  
        # Creating summary statistics based features:-
        for col in self.sensor_col:
            _xform = X.groupby('sequence').agg({col: [np.mean, np.amin, np.median, np.amax, iqr, skew, kurtosis]})
            _xform.columns= [j+'_'+i for i, j in _xform.columns.to_flat_index()];
            mdl_df = mdl_df.join(_xform);
            del _xform; 
            
         # Calculating MAD and appending with the master dataframe:-   
        mdl_df = mdl_df.join(X.loc[:, self.sensor_col].groupby(X.sequence).mad().add_prefix('mad_'));
        
          # Creating the occurances of a given subject in the data-set as a feature:-   
        _nb_subj_rcrd = X[['subject', 'sequence']].groupby('subject')[['sequence']].nunique().rank(method= 'max');
        _nb_subj_rcrd['pctl_rcrd_subj'] = (_nb_subj_rcrd.sequence - 1)*100/len(_nb_subj_rcrd);
        mdl_df = mdl_df.merge(_nb_subj_rcrd[['pctl_rcrd_subj']], how= 'left', 
                              left_on= 'subject', right_index= True);
        del _nb_subj_rcrd;
        
        # Creating FFT if requested by user:-
        if self.FFT_req == 'Y': 
            _fft = X.groupby(['sequence', 'subject']).apply(self.DoFFT).droplevel(level=1, axis='index');
            mdl_df = pd.concat((mdl_df, _fft), axis=1);
            del _fft;
        
        # Creating the list of features engineered from the give dataset:-        
        global Ftre_Engr_Lst;
        Ftre_Engr_Lst = list(mdl_df.columns);    
        print(colored(f"\nTotal features generated ={len(Ftre_Engr_Lst):,.0f}\n", color= 'blue')); 
        
        return mdl_df; 


# In[ ]:


# Creating an output table structure for all transforms across all features without FFT:- 
def MakeFeatures(df:pd.DataFrame):
    """
    This class does the below tasks:-
    1. Generates a model dataframe object as return object for all transforms and all sensors
    2. Creates descriptive statistics based summaries 
    3. Appends each set of transforms to the master dataframe (mdl_df)
    4. Separately calculates the MAD and appends it to the model master dataframe
    5. Creates a global feature list to be used elsewhere in the code
    """;
    
    # Creating output master dataframe with sequence and subject:- 
    mdl_df = df[['sequence', 'subject']].drop_duplicates().set_index('sequence');
    
    # Aggregating sensor columns:-    
    sensor_col_lst = df.iloc[0,-13:].index; 
    
    for col in sensor_col_lst:
        _xform = df.groupby('sequence').agg({col: [np.mean, np.amin, np.median, np.amax, iqr, skew, kurtosis]})
        _xform.columns= [j+'_'+i for i, j in _xform.columns.to_flat_index()];
        mdl_df = mdl_df.join(_xform);
        del _xform;
    
    # Calculating MAD and appending with the master dataframe:-   
    mdl_df = mdl_df.join(df.loc[:, sensor_col_lst].groupby(df.sequence).mad().add_prefix('mad_'));
    
    # Creating the occurances of a given subject in the data-set as a feature:-   
    _nb_subj_rcrd = df[['subject', 'sequence']].groupby('subject')[['sequence']].nunique().rank(method= 'max');
    _nb_subj_rcrd['pctl_rcrd_subj'] = (_nb_subj_rcrd.sequence - 1)*100/len(_nb_subj_rcrd);
    mdl_df = mdl_df.merge(_nb_subj_rcrd[['pctl_rcrd_subj']], how= 'left', 
                          left_on= 'subject', right_index= True);
        
    del sensor_col_lst, _nb_subj_rcrd;       
    global Ftre_Engr_Lst;
    Ftre_Engr_Lst = list(mdl_df.columns);    
    print(colored(f"\nTotal features generated are {len(Ftre_Engr_Lst):,.0f}\n", color= 'blue')); 
    return mdl_df;


# In[ ]:


# Removing outliers from the feature engineered dataset:-
class OutlierRemover(BaseEstimator, TransformerMixin):
    "This class removes outliers based on IQR multiplier (usually 1.5*IQR)";
    
    def __init__(self, iqr_mult:float = 1.50):
        "This function initializes the IQR multiplier for the outlier removal";
        self.iqr_mult_ = iqr_mult;
        
    def fit(self, X, y=None, **fit_params):
        "This function calculates the cutoff for outlier removal on the train-data";
        X_iqr = iqr(X, axis=0);
        self.OtlrLB_ = np.percentile(X, 25, axis=0) - self.iqr_mult_* X_iqr;
        self.OtlrUB_ = np.percentile(X, 75, axis=0) + self.iqr_mult_* X_iqr;
        del X_iqr;
        return self;
    
    def transform(self, X, y= None, **transform_params):
        "This function clips the outliers off the data-set";
        return pd.DataFrame(data= np.clip(X, a_min= self.OtlrLB_, a_max= self.OtlrUB_), 
                            index= X.index, columns= X.columns);


# In[ ]:


# Performing univariate analysis on the relevant columns and selecting useful model features:-
class FeatureSelector(BaseEstimator, TransformerMixin):
    "This class calculates the correlation and mutual information scores for the features to help in model selection";
    
    def __init__(self, abs_corr_cutoff:np.float32= 0.0,std_cutoff:np.float32 = 0.0): 
        "This method initializes the cutoffs for the correlation and mutual information metrics";
        self.corr_cutoff_ = abs_corr_cutoff;
        self.std_cutoff_ = std_cutoff;
    
    def fit(self, X, y, **fit_params):
        """
        This method calculates the correlation on the training data 
        It also shortlists selected columns for the transform step
        """;
        global Ftre_Engr_Lst, Sel_Ftre_Lst_V1, Unv_Prf;
        
        _ = pd.concat((X, y), axis=1);       
        self.Unv_Prf_ =         pd.concat((_.corr('pearson')[['state']].drop(['state'], axis=0).rename({'state': 'PRSN_COR_VAL'}, axis=1),
                  pd.DataFrame(np.std(_, axis=0), index= Ftre_Engr_Lst, columns= ['STD_VAL'])                  
                  ), axis=1).drop(['subject'], axis=0);

        self.Unv_Sel_Ftre_ =         self.Unv_Prf_.loc[(abs(self.Unv_Prf_['PRSN_COR_VAL']) >= self.corr_cutoff_) &
                          (self.Unv_Prf_['STD_VAL'] >= self.std_cutoff_)].index;
        
        Unv_Prf = self.Unv_Prf_;
        
        Sel_Ftre_Lst_V1 = list(self.Unv_Sel_Ftre_);
        print(colored(f"\nSelected features after feature engineering", color= 'blue', attrs= ['bold']));
        print(colored(f"{list(Sel_Ftre_Lst_V1)}\n", color = 'blue'));
        return self;
    
    def transform(self, X, y=None, **transform_param):
        """
        This function returns the correlation and mutual information results.
        It also returns the data-set with reduced columns as per feature selection strategy
        """;
        X1= X.copy();
        sel_cols = Sel_Ftre_Lst_V1 + ['subject'];
        return X1[sel_cols];                      


# In[ ]:


# Further shortlisting features based on purging highly correlated features to avoid collinearity:-
class CorrFeatureDropper(BaseEstimator, TransformerMixin):
    """
    This class shortlists features to drop after preliminary feature selection based on correlation among variables
    """;
    
    def __init__(self, corr_cutoff:np.float32 = 0.80): 
        self.corr_cutoff = corr_cutoff;
        
    def fit(self, X, y= None,  **fit_params):
        """This function determines the columns to be dropped based on feature-correlation""";
        
        global Unv_Prf;
        
        # Generating the root column for further group-by object:-
        Unv_Prf['ROOT_FTRE_NM'] = np.where(Unv_Prf.index.str.contains('sensor'), 
                                           Unv_Prf.index.str[-9:], Unv_Prf.index);
        Unv_Prf['ABS_CORR_VAL'] = abs(Unv_Prf.PRSN_COR_VAL);

        # Creating a lower triangular matrix of correlations from the shortlisted features and dropping 'subject':-
        _ftre_corr_prf =         pd.DataFrame(np.tril(X.corr()),index= X.columns, columns= X.columns).        drop('subject', axis=0).drop('subject', axis=1);

        # Collating columns to be dropped based on high correlations among features:-
        drop_ftre_ =         pd.concat((Unv_Prf, 
                   _ftre_corr_prf[(abs(_ftre_corr_prf) > self.corr_cutoff) & 
                                  (abs(_ftre_corr_prf) < 1.0)].any(axis=0)), axis=1).\
        rename({0: 'CORR_FTRE_FL'}, axis=1).query("CORR_FTRE_FL == True").\
        groupby('ROOT_FTRE_NM')[['ABS_CORR_VAL','STD_VAL']].rank(method= 'dense',ascending= False).\
        query("ABS_CORR_VAL != 1.0 and STD_VAL !=1.0").index;
        
        self.drop_ftre_lst_ = list(drop_ftre_);
        
        print(colored(F"\nColumns to be dropped based on collinearity check are", 
                      color= 'blue', attrs= ['bold', 'dark']));
        print(colored(f"{list(drop_ftre_)}\n", color = 'blue'));       
        return self;
    
    def transform(self, X, y= None, **transform_params):
        "This function drops the shortlisted columns based on the fit method results";
        X1 = X.drop(self.drop_ftre_lst_, axis=1);
        Sel_Ftre_Lst_V2 = list(X1.columns);
        return X1;


# ### Pipeline development:-
# 
# This pipeline does the below-
# 1. Reduce memory usage by reassigning data-types
# 2. Develop new features from the sensor readings using descriptive statistics aggregators and if needed, FFT
# 3. Remove outliers using the column IQR
# 4. Shortlist important features using Pearson Correlation and standard deviation
# 5. Drop correlated features to avoid collinearity issues
# 6. Standardize the data using appropriate scaling and centering strategy
# 
# **The FFT based step is adapted with thanks from the notebook titled 'LGBM with Fourier transform' originally written by PAVEL SALIKOV**

# In[ ]:


# Data processing specific globals:-
# 1. Creating empty lists for the data-processor pipeline for feature shortlisting:-
Sel_Ftre_Lst_V1 = [];
Sel_Ftre_Lst_V2 = [];

# 2. Other cutoffs for the feature-selection:-
abs_corr_cutoff = 0.05;
highcorr_cutoff = 0.68;
std_cutoff = 0.30;

# 2. Standardization class label (StandardScaler/ RobustScaler):
Std_Class_Lbl = RobustScaler;

# 3. FFT requirement flag:-
FFT_req= 'Y';


# In[ ]:


# Designing the data processor pipeline:-
Data_Processor = Pipeline(verbose= True,  
         steps= \
         [('ReduceMemory', FunctionTransformer(ReduceMemory)),
          ('MakeFeatures', FeatureCreator(FFT_req='Y')),
          ('RemoveOutliers',OutlierRemover(iqr_mult=1.5)),
          ('SelectFeatures', FeatureSelector(abs_corr_cutoff= abs_corr_cutoff, std_cutoff = std_cutoff)),
          ('DropCorrFeatures', CorrFeatureDropper(highcorr_cutoff)),
          ('StdFeatures', DataFrameMapper(drop_cols= None, input_df= True, df_out= True, default= None,
                                          features= gen_features(
                                              columns=[col.split(' ') for col in Sel_Ftre_Lst_V2 if col != 'subject'],
                                              classes= [Std_Class_Lbl])
                                         ))
         ]
        );

# Implementing the pipeline on the train-test sets:-
print(colored(f"\nImplementing the data processor pipeline on the training data\n",
              color= 'blue', attrs= ['dark', 'bold']));
Ftre_Prf_Train = Data_Processor.fit_transform(xtrain, ytrain.state);
print(colored(f"\nTraining data shape = {Ftre_Prf_Train.shape}\n", color= 'blue'));

print(colored(f"\nImplementing the data processor pipeline on the test data\n",
              color= 'blue', attrs= ['dark', 'bold']));
Ftre_Prf_Test = Data_Processor.transform(xtest);
print(colored(f"\nTest data shape = {Ftre_Prf_Test.shape}\n", color= 'blue'));


# # 3. Model Training
# 
# We shall now train appropriate ML models on the features to elicit appropriate test set predictions
# 
# We shall use the Group KFold cross validation strategy with training and evaluation metrics displayed through the model training cycle.We will also use 'early stopping' to prevent overfitting
# 
# We will then store the predictions from each model component in a dataframe to be used later for the submission file

# In[ ]:


# Model training specific globals:-
# 1. Verbose indicator:-
verbose_nb = 150;
# 2. Early cut-off indicator:-
early_cutoff_itr_nb = 300;
# 3. CV splits with Group Kfold:-
n_splits_cv = 5;
# 4. Estimators for ensemble:-
nb_trees = 2500;


# In[ ]:


# Creating output dataframe to store the diagnostics:-
mdl_diag_prf = pd.DataFrame(data=None, index= None, 
                            columns= ['Mdl_Lbl', 'Fold_Nb', 'Precision', 'Recall', 'F1', 'Accuracy', 'GINI']);

# Creating output dataframe to store the test-set predictions:-
mdl_pred_prf = pd.DataFrame(data= None, index= sub_fl.sequence, columns= None);

# Developing model scoring function:-
def Score_Model(y, ypred, mdl_lbl, fold_nb):
    "This functions creates the precision, recall, F1, accuracy and GINI metrics for the given model instance";
    global mdl_diag_prf;
    return pd.concat((mdl_diag_prf, 
                      pd.DataFrame(data= ([mdl_lbl, fold_nb, 
                                           precision_score(y, ypred), recall_score(y, ypred), 
                                           f1_score(y, ypred),accuracy_score(y, ypred), 
                                           roc_auc_score(y, ypred)]), 
                                   index = mdl_diag_prf.columns).T), axis=0,ignore_index= True);

# Defining the cross-validation strategy using Group-KFold:-
fold_ = GroupKFold(n_splits=n_splits_cv);

# Designing the model training function:-
def Train_Model(mdl, mdl_nm, fold_, verbose_nb=verbose_nb, early_cutoff_itr_nb = early_cutoff_itr_nb):
    """
    This function trains the model based on the provided k-fold structure and provides the test-set diagnostics and predictions
    Inputs- 
    mdl (model instance)
    mdl_nm (string)- name of the model
    fold_ (CV structure)
    globals (early_cutoff_itr_nb, verbose_nb)- global variables for verbose and early-cutoff            
    """;
    
    global mdl_diag_prf;    
    print(colored(f"\nCurrent model is {mdl_nm}\n", color= 'red', attrs= ['dark', 'bold']));
    
    for fold_nb, (train_idx, dev_idx) in enumerate(list(
        fold_.split(X= Ftre_Prf_Train,y= None,groups= Ftre_Prf_Train.subject))):
        
        xtr, xdev = Ftre_Prf_Train.loc[train_idx], Ftre_Prf_Train.loc[dev_idx];
        ytr, ydev = ytrain[['state']].loc[train_idx], ytrain[['state']].loc[dev_idx];

        mdl = mdl;
        
        if mdl_nm.lower() != 'catboost':
            mdl.fit(xtr, ytr, 
                    early_stopping_rounds= early_cutoff_itr_nb,
                    eval_set=[(xtr, ytr), (xdev, ydev)],verbose= verbose_nb, eval_metric=['auc']
                   );
        elif mdl_nm.lower() == 'catboost':
            mdl.fit(xtr, ytr, early_stopping_rounds= early_cutoff_itr_nb,
                    eval_set=[(xtr, ytr), (xdev, ydev)],verbose= verbose_nb);
            
        ytr_pred= mdl.predict(xtr);
        ydev_pred= mdl.predict(xdev);
        
        mdl_diag_prf = Score_Model(ydev, ydev_pred, mdl_lbl= mdl_nm, fold_nb=fold_nb);
        mdl_pred_prf[mdl_nm + str(fold_nb)] = mdl.predict_proba(Ftre_Prf_Test)[:,1];


# In[ ]:


# Implementing the model training functions:-
filterwarnings('ignore')

# 1.LGBM classifier:-
Train_Model(mdl= LGBMClassifier(random_state= 10, metric= 'auc', n_estimators= nb_trees, 
                                boosting_type= 'gbdt',max_depth = 7,learning_rate= 0.10,
                                subsample= 0.80, colsample_bytree= 0.75, objective= 'binary'), 
            mdl_nm= 'LGBM', fold_= fold_);


# In[ ]:


# Implementing the model training functions:-
filterwarnings('ignore')

# 1.LGBM classifier:-
Train_Model(mdl= LGBMClassifier(random_state= 10, metric= 'auc', n_estimators= nb_trees, 
                                boosting_type= 'gbdt',max_depth = 7,learning_rate= 0.10,
                                subsample= 0.80, colsample_bytree= 0.75, objective= 'binary'), 
            mdl_nm= 'LGBM', fold_= fold_);

# 2.XgBoost classifier:-
Train_Model(mdl= XGBClassifier(random_state= 10, n_estimators= nb_trees, learning_rate= 0.10, eval_metric= 'logloss',
                              use_label_encoder=False),
           mdl_nm= 'XGBoost', fold_ = fold_);

# 3.CatBoost classifier:-
Train_Model(mdl = CatBoostClassifier(verbose= False, eval_metric='AUC'), mdl_nm= 'CatBoost', fold_= fold_);

# Printing all model diagnostics:-
print(colored("\nModel dev-set diagnostics\n", color= 'blue', attrs= ['bold', 'dark']));
display(mdl_diag_prf.style.highlight_max(subset= ['Precision', 'Recall', 'F1', 'Accuracy', 'GINI'], 
                                     color= 'lightblue').format(precision= 4));


# # 4. Submission File
# 
# We shall prepare the submission file based on the mean probability of the classifiers built to enrapture all candidate models' inputs

# In[ ]:


mdl_pred_prf['AllModels'] = mdl_pred_prf.mean(axis=1);
mdl_pred_prf[['AllModels']].reset_index().rename({'AllModels': 'state'}, axis=1).to_csv("Submission.csv", index= False);

