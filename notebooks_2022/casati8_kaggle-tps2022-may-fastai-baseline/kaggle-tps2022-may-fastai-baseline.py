#!/usr/bin/env python
# coding: utf-8

# This notebook is a quick demonstration, who to use the Fastai v2 library for a Kaggle tabular competition. Fastai v2 is based on pytorch and allows you, to build a decent machine learning application. For more information please visit the Fastai documentation: https://docs.fast.ai/. I will link to "Chapter 9, Tabular Modelling Deep Dive" and the notebook "09_tabular.ipynb".
# 
# 
# This monthly competition is a binary classification problem: find the correct state bases on 32 differend features. In this notebook i will use a neural network approach and i will train this network with the offered traing data set.
# 
# Let's start and import the needed stuff ..

# In[ ]:


from fastai.tabular.all import * 
from fastai.test_utils import show_install
from IPython.display import display, clear_output
import holidays
import seaborn as sns

show_install()


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[ ]:


def set_seed_value(seed=718):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed_value()


# In[ ]:


path = Path('../input/tabular-playground-series-may-2022/')
Path.BASE_PATH = path
path.ls()


# In[ ]:


train_df = pd.read_csv(os.path.join(path, 'train.csv')).set_index('id')
test_df = pd.read_csv(os.path.join(path, 'test.csv')).set_index('id')
sample_submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))

string_var = 'f_27'
dep_var = 'target'


# I use Pandas to import them and to verify, where null values are there or some values are missing. The result shows, that the data set is complete, so that no additional data completion is needed. That's a goog result!

# In[ ]:


train_df.isna().sum().sum(), test_df.isna().sum().sum(), train_df.isnull().sum().sum(), test_df.isnull().sum().sum()


# 
# 
# Let's have look on the training data set.
# 

# In[ ]:


train_df.head()


# Let's see how the values for the depended variable, the taget, are distributed:

# In[ ]:


train_df.hist(column=dep_var)


# Bothe values are equal distributed, no skewness in their distrubtion can be detected; fine!
# 
# 
# First i will look at the correlation matrix to verify how important a feature is.

# In[ ]:


#train_df = train_df[string_var].astype('category')
#test_df = test_df[string_var].astype('category')


# In[ ]:


corr = train_df.corr()

fig, axes = plt.subplots(figsize=(30, 15))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask=mask, linewidths=.5, annot=True, cmap='rainbow')

plt.show()


# In[ ]:


train_df.info()


# Was we can ses there are 9 clumns with float values, 21 with integer values and one column with string values ('f_27'). I should verify how the integer and the string values are distributed. Why? The integer and the string values are transformed into categorized variables and for these variables the number of the unique values are important.

# In[ ]:


cat_columns = train_df.columns[(train_df.dtypes.values != np.dtype('float64'))]
cat_columns


# In[ ]:


for col in cat_columns:
    print('column ', col, ' number of unique values ', train_df[col].nunique())


# As we can see besides the column 'f_27' we have a small number of unique values for each of these featues. We can transform them into categorized variables without getting to wide structures in the hidden embedded layer later on. 
# 
# For the column 'f_27', the string feature, we have 741354 differend values by 900000 values in total. Building a categorized variable for this features makes on sense at this point.
# 
# As i can see the values in column 'f_27' have the equal width of 10 characters.

# In[ ]:


train_df[string_var].str.len().min(), train_df[string_var].str.len().max(), test_df[string_var].str.len().min(), test_df[string_var].str.len().max()


# This insight for the values in column 'f_27' brings 2 different solution for a baseline approach:
# * drop the column 'f_27' from the training and test data set
# * split the 10 character value like 'ABABDADBAB' into 10 new columns like 'A','B','A','B','D','A','D','B','A','B' and create catgeorized variable for each of these new columns.
# 
# 
# The following function will implement this feature conversion.

# In[ ]:


def convert_feature_27(df, do_convert = True):
    if do_convert:
        for i in range(10):
            df[f'f_27_{i}'] = df[string_var].str.get(i)
    df.drop([string_var], axis=1, inplace=True)
    return df


# In[ ]:


train_df = convert_feature_27(train_df, do_convert=True)
test_df = convert_feature_27(test_df, do_convert=True)


# In[ ]:


train_df.shape, test_df.shape


# In[ ]:


train_df.head()


# I need a list of the column names, which are candidates for category variables and which are no candidates, also called continous variables. The Fastai library offers the function 'cont_cat_split' to do this for us. Our training data set contains only floating values for the independed variables, therefore we expect that no category variables are available.

# In[ ]:


cont_vars, cat_vars = cont_cat_split(train_df, dep_var=dep_var, max_card=25)
len(cont_vars), len(cat_vars),cont_vars,cat_vars


# 

# The next step is to create a data loader. The Fastai library offers a powerful helper called 'TabularPandas'. It needs the data frame, list of the category and continous variables, the depened variable and a splitter. The splitter divides the data set into two parts: one for the training and one for the validation and for internal optimization step in each epoch. The batch size is set to 1024, because we have a large data set. We can use a random split because the rows in the data set are independed.

# In[ ]:


def getData(df, batchSize=1024):
    
    to_train = TabularPandas(df, 
                           [Normalize, Categorify, FillMissing],
                           cat_names=cat_vars,
                           cont_names=cont_vars, 
                           splits=RandomSplitter(valid_pct=0.2)(df),  
                           device = device,
                           y_block=CategoryBlock(),
                           y_names=dep_var) 

    return to_train.dataloaders(bs=batchSize)


# In[ ]:


dls = getData(train_df, batchSize=2048)
len(dls.train), len(dls.valid)


# Show me the transformed data, which will be used in the network later.

# In[ ]:


dls.show_batch()


# At least i create a learner pasing the dataloader into it. The default settings are two hidden layers with 200 and 100 elements. 
# Increasing the number of parameters in the neural network will improve the accuarcy and score, hopefully.

# In[ ]:


my_config = tabular_config(y_range=(0,1) )
learn = tabular_learner(dls,
                       config = my_config,
                       metrics=[accuracy])

learn.summary()


# In[ ]:


learn.lr_find()


# I will use a maximum learning rate of 3e-3. Starting the learning process is quite easy, i will run for 50 epochs. I will save the model with the best, with the lowest validation lost value. The Fastai library offers the SaveModelCallback callback. You must specify the file name only. The option with_opt=True stores the values of the optimizer also. You will find the new file in the subdirectory  'models'.

# In[ ]:


learn.fit_one_cycle(100, 3e-3, wd=0.01, cbs=SaveModelCallback(fname='kaggle_tps2022_may', with_opt=True))


# In[ ]:


learn.show_results()


# The confusion matrix below shows us the quality of test data predictions.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(normalize=True, norm_dec=3)


# Now it's time to calculate the predictions for the test data set.

# In[ ]:


learn.load('kaggle_tps2022_may')


# I got the 'one hot encoded' prediction values, which are probabilities for the different target values. np.argmax returns the index with the maximum probability value, like 0 or 1.

# In[ ]:


dlt = learn.dls.test_dl(test_df, bs=1024) 
nn_preds,_ ,preds = learn.get_preds(dl=dlt , with_decoded=True) 

nn_preds, preds


# In[ ]:


sample_submission[dep_var] = np.argmax(nn_preds, axis=-1)
sample_submission.to_csv("submission.csv", index=False)
sample_submission.head(10)


# Here is the distribution of the predicted target values

# In[ ]:


sample_submission.hist(column=dep_var)


# In[ ]:


get_ipython().system('ls -al')


# As we can see we achieve an accuracy value of 0.94 with the default fastai settings and with a minimal features engineering for the special column 'f_27'. That is a greate result and is the baseline to investigate in more feature engineering and/or modeling to get a better final result. At this point you can start your own experience. Fell free and use my notebokk if you like.
