#!/usr/bin/env python
# coding: utf-8

# Quick and dirty EDA
# ==
# 
# I'm going to do a fairly minimal EDA before I do any modelling. First, let's read in the data and verify that pandas infers the data types for the train and test sets to be the same:

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np

sns.set(
    style='darkgrid', context='notebook', rc={'figure.figsize': (12, 8), 'figure.frameon': False, 'legend.frameon': False}
)
def compress_mem(df):
    floats = df.columns[df.dtypes == np.float64]
    ints = df.columns[df.dtypes == np.int64]
    return df.astype(
        {col: np.float32 for col in floats}
    ).astype({col: np.int32 for col in ints})

train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv').pipe(compress_mem)
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv').pipe(compress_mem)

(train.dtypes.drop('target') == test.dtypes).all()


# Are the train test and test set similar?
# ==
# 
# Okay, with that out of the way, I want to start by finding out whether the train and test sets have big differences. I'll take a look at everything that's non-float first, and check whether we have some feature values that are present only in test, or vice-versa. Let's start by checking the number of possible values for the non-numerical columns:

# In[ ]:


df = pd.concat([train.drop(columns=['target']).assign(ds='train'), test.assign(ds='test')]).set_index('id')

summary = df.select_dtypes(exclude=np.float32).groupby('ds').nunique().reset_index().melt(id_vars=['ds'])

sns.catplot(
    data=summary, y='variable', x='value', col='ds', kind='bar', orient='horizontal'
);


# `f_27` needs special treatment
# --
# 
# Okay, we'll need to set aside some time for `f_27` later, that's clearly ready to be used as a feature yet. With this many possible values, it's highly likely that some values only occur in test, but maybe we can interpret the data to make it more valuable. Let's check out the rest:

# In[ ]:


sns.catplot(
    data=summary.loc[summary.variable != 'f_27'], y='variable', x='value', hue='ds', kind='bar', orient='horizontal'
);


# Some feature values occur only in test
# --
# 
# There are columns where test has more values than train, but it's not the common case. Let's try to check which features that have values that only occur in the test set or only in the train set.

# In[ ]:


possible = df.select_dtypes(exclude=np.float32).drop(columns=['f_27']).melt(
    id_vars=['ds']
).drop_duplicates()

in_train = possible.loc[possible.ds == 'train'].drop(columns=['ds'])
in_test = possible.loc[possible.ds == 'test'].drop(columns=['ds'])
joined = in_train.merge(in_test, how='outer', indicator=True)
joined = joined.loc[joined._merge != 'both']
joined = joined.replace({'left_only': 'train', 'right_only': 'test'}).rename(columns={'_merge': 'unique_in'})
joined


# Maybe just discrete numbers, not categoricals?
# --
# 
# Okay, there are actually a few cases here. Let's try to find out whether we can treat any of these as numerical columns.
# 

# In[ ]:


sns.catplot(
    data=df.loc[:, joined.variable.to_list() + ['ds']].melt(id_vars=['ds']),
    col='variable', x='value', hue='ds', kind='count', col_wrap=3
);


# Looks plausible that these are just discrete numerical data, and we might get away with just scaling them, instead of treating them as categoricals. This data set is supposed look like a manufacturing problem of some sort, so it might be that some sensors output discrete measurements or settings of some type. At a glance, there are no huge differences between train and test for these columns.
# 
# Looking into `f_27`
# ==
# 
# This column is very different from anything we've seen so far. Let's check same possible values:
# 

# In[ ]:


df.f_27.sample(n=10)


# Okay, my first thought is that this is "more than 1" variable. My second thought is that maybe this is a single variable because the order matters, e.g. this is some sort of instruction sequence. Let's break this column apart to start with:
# 

# In[ ]:


f_27 = df.f_27.str.split('', expand=True).drop(columns=[0, 11]) # Empty string in first and last column
f_27.nunique().plot.bar(title='Unique values by string position').set_xlabel('position');


# This looks like it could plausibly be used as categorical variables, one way or the other. We should probably also check if some of these occur only in the train or test set.
# 
# Really hard to say anything meaningful about what this represents. We'll need to check the distributions of the various positions, these could easily represent numbers from some sensor or equipment. We're on a mission to find out if train and test have glaring differences, so let's check if we're in trouble here:
# 

# In[ ]:


f_27 = f_27.join(df.ds).melt(id_vars=['ds'], var_name=['position'])
sns.catplot(
    data=f_27,
    col='position', x='value', kind='count', hue='ds', col_wrap=3
);


# Seemingly no big differences between train and test here. It's probably still a good idea to check if any of these value/position pairs occur only in one of the data sets, so let's work on that:
# 

# In[ ]:


unique = f_27.drop_duplicates()
unique_train = unique.loc[unique.ds == 'train'].drop(columns='ds')
unique_test = unique.loc[unique.ds == 'test'].drop(columns='ds')
joined = unique_train.merge(unique_test, how='outer', indicator=True)
joined = joined.loc[joined._merge != 'both'].replace(
    {'left_only': 'train_only', 'right_only': 'test_only'}
).rename(columns={'_merge': 'in'})
joined


# Okay, so a possible pitfall if we were to try using these as categoricals is that not all position/value combinations occur in both data sets. From the distributions we looked at above, it seems likely that position 2, 5 and 9 might be numerical values anyway.
# 
# Let's quickly check if it is also the case that the floating point values follow roughly the same distributions between train/test, then we'll stop looking at the test set and start looking at the target instead.
# 

# In[ ]:


sns.displot(
    data=df.select_dtypes(np.float32).join(df.ds).melt(id_vars=['ds']),
    col='variable', hue='ds', x='value', kind='hist', col_wrap=3,
    facet_kws={'sharex': False}, bins=50, common_bins=False,
);


# Concluding the differences between train and test
# ==
# 
# All of these look very similar. To sum up:
# 
# - `f_27` needs to be handled separately, either by splitting it into constituent parts, or possibly by treating it as a sequence (RNN?)
# - When split, `f_27` has 3 positions where not all values occur in both train and test. We can probably treat positions 2, 5 and 9 as numbers, though.
# - The discrete variables we've got have values that occur in only train, or only test, which could create problems if we introduce categorical encoding for them.
# - The distributions we've looked at appear to be very similar between train and test. We've no reason (this time...) to expect local CV and LB score to be wildly different.
# 
# Investigating how features relate to target
# ==
# 
# Now, let's stop snooping in the test set and start looking only at the train data. We already know that we'll need to split apart the `f_27` feature, so we'll just get that out of the way right away, then translate the character values in those values to numbers.
# 

# In[ ]:


del df
del f_27
df = train.set_index('id').drop(columns=['f_27'])
f_27 = train.set_index('id').f_27.str.split('', expand=True).drop(columns=[0, 11]).rename(columns='f_27_{}'.format)
f_27 = f_27.applymap(ord) - ord('A')
df = df.join(f_27)
test = test.set_index('id')
test = test.drop(columns=['f_27']).join(
    test.f_27.str.split('', expand=True).drop(columns=[0, 11]).rename(columns='f_27_{}'.format).pipe(
        lambda df: df.applymap(ord) - ord('A')
    )
)
df.head().T


# Let's first check the size of train vs test, and the target balance:

# In[ ]:


print('Train samples:', len(train), 'test samples:', len(test), 'mean(target) =', train.target.mean())


# Okay, train is about the same size as test, and the training data is nearly balanced.
# 
# Let's check if a correlation map suggests anything at all:

# In[ ]:


sns.heatmap(df.select_dtypes(np.float32).join(df.target).corr(), cmap='coolwarm', annot=True, fmt='.2f');


# There are some correlated features, but the correlations aren't very strong. There are mostly weak correlations with the target:
# 

# In[ ]:


df.corr()['target'].drop('target').plot.bar(title='Correlation with target');


# Candidates for categorical interpretation
# --
# 
# Can any of our low-ordinal columns predict the target well?

# In[ ]:


cats = df.columns[df.nunique() <= 30]
sns.catplot(
    data=df[cats].melt(id_vars=['target']),
    col='variable', x='value', kind='count', hue='target',
    col_wrap=3, sharex=False, sharey=False
);


# Nothing that is super-strong, but the binary features definitely seem useful, as do several of the `f_27` string positions. Looking at it right now, I think we're probably OK with encoding many of the string positions simply as numbers.
# 
# Distribution of numerical features
# --
# 
# Let's check if there are any distribution differences for the float features that seem promising:

# In[ ]:


sns.displot(
    data=df.select_dtypes(np.float32).join(df.target).melt(id_vars='target'),
    x='value', col='variable', hue='target', col_wrap=3, kind='ecdf', facet_kws={'sharex': False}
);


# It is easy to see that a model should be able to exploit some of these. They're all fairly normally distributed and should work fine with any kind of scaler with no big adjustment needed. Because we know that some integer features have values that occur only in test, I am tempted to just send them through a scaler as well. I'm thinking we'll focus on tree models and NN here, and we can start playing with trees without doing any transformations at all.
# 
# Let's keep the bools, turn everything else into floats and get the ball rolling, that way we won't have to deal with values that don't exist in train:
# 

# In[ ]:


train = df.astype({col: np.float32 for col in df.columns[(df.dtypes == np.int64) | (df.dtypes == np.int32)]})
train = train.astype({col: np.bool_ for col in df.columns[df.nunique() == 2]})
test = test.astype(train.dtypes.drop('target'))
train.to_parquet('train.pq')
test.to_parquet('test.pq')

train.info()


# Check sklearn models
# ==
# 
# At this point I feel ready to do a simple grid search to check if any particular kind of model seems more promising than others. Let's do some imports and set up seeds, CV splits.

# In[ ]:


import os
import random
from sklearn import metrics, model_selection, ensemble, pipeline, linear_model, neural_network, preprocessing

random.seed(42)
np.random.seed(42)
cv = model_selection.KFold(shuffle=True, random_state=42)

n_jobs = os.cpu_count()
if n_jobs > 4:
    n_jobs = n_jobs // 2 # hyper-threading

pipe = pipeline.Pipeline([
    ('scaler', preprocessing.StandardScaler()),
    ('model', linear_model.LogisticRegression())
])
scalers = [[preprocessing.StandardScaler()], [preprocessing.RobustScaler()]]

grid = model_selection.ParameterGrid([
    {'scaler': scalers, 'model': [[linear_model.LogisticRegression()], [linear_model.SGDClassifier()]]},
    {'scaler': scalers, 'model': [[neural_network.MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True)]]},
    {'scaler': [[None]], 'model': [
        [ensemble.RandomForestClassifier()],
        [ensemble.HistGradientBoostingClassifier()],
    ]}
])
search = model_selection.GridSearchCV(pipe, grid, scoring='roc_auc', n_jobs=n_jobs)
search.fit(train.drop(columns=['target']), train.target)
pd.DataFrame(search.cv_results_)


# NN looks promising
# ==
# 
# At first glance, it looks like we should be focusing on NN, and linear models are possibly a waste of time. It's probably worth looking at gradient boosters + additional feature engineering on `f_27` too.
# 
# There's very little spread in score across the CVs, for hyper-parameter tuning we could probably get away with using less than the whole data set or simply not using all the splits.
# 
# Let's invest a little bit of time training an MLP, and submit before we round off. In another notebook, we're going to tune an NN with pytorch and optuna instead of guesstimating parameters.

# In[ ]:


pipe.set_params(model=neural_network.MLPClassifier(
    hidden_layer_sizes=(64, 64, 32), batch_size=4096, alpha=1e-3, max_iter=50 * 200, # roughly 50 epochs
    early_stopping=True, learning_rate_init=5e-3, n_iter_no_change=25
))
pipe.fit(train.drop(columns=['target']), train.target)
proba = pipe.predict_proba(test)[:, 1]


# In[ ]:


test.assign(target=proba)[['target']].to_csv('submission.csv')

