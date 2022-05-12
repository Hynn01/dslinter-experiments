#!/usr/bin/env python
# coding: utf-8

# # Train vs. Test dataset distributions
# Before getting started on this competition I quickly wanted to check the distributions of the test dataset against that of the training dataset, and if possible see how different from each other they are.

# In[ ]:


import gc
import itertools
from copy import deepcopy

import numpy as np
import pandas as pd

from tqdm import tqdm

from scipy.stats import ks_2samp

from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

from sklearn import manifold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. t-SNE Distribution Overview
# To start out I'll take out an equal amount of samples from the train and test dataset (4459 samples from both, i.e. entire training set and sample of test set), and perform a t-SNE on the combined data. I'm scaling all the data with mean-variance, but for columns where we have outliers (> 3x standard deviation) I also do a log-transform prior to scaling.
# 
# ## 1.0. Data Pre-Processing
# Current pre-processing procedure:
# * Get 4459 rows from training set and test set and concatenate them
# * Columns with standard deviation of 0 in training set removed
# * Columns which are duplicate in training set removed
# * Log-transform all columns which have significant outliers (> 3x standard deviation)
# * Create datasets with: 
#     * Mean-variance scale all columns including 0-values!
#     * Mean-variance scale all columns **excluding** 0-values!
#     

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# How many samples to take from both train and test\nSAMPLE_SIZE = 4459\n\n# Read train and test files\ntrain_df = pd.read_csv(\'../input/train.csv\').sample(SAMPLE_SIZE)\ntest_df = pd.read_csv(\'../input/test.csv\').sample(SAMPLE_SIZE)\n\n# Get the combined data\ntotal_df = pd.concat([train_df.drop(\'target\', axis=1), test_df], axis=0).drop(\'ID\', axis=1)\n\n# Columns to drop because there is no variation in training set\nzero_std_cols = train_df.drop("ID", axis=1).columns[train_df.std() == 0]\ntotal_df.drop(zero_std_cols, axis=1, inplace=True)\nprint(f">> Removed {len(zero_std_cols)} constant columns")\n\n# Removing duplicate columns\n# Taken from: https://www.kaggle.com/scirpus/santander-poor-mans-tsne\ncolsToRemove = []\ncolsScaned = []\ndupList = {}\ncolumns = total_df.columns\nfor i in range(len(columns)-1):\n    v = train_df[columns[i]].values\n    dupCols = []\n    for j in range(i+1,len(columns)):\n        if np.array_equal(v, train_df[columns[j]].values):\n            colsToRemove.append(columns[j])\n            if columns[j] not in colsScaned:\n                dupCols.append(columns[j]) \n                colsScaned.append(columns[j])\n                dupList[columns[i]] = dupCols\ncolsToRemove = list(set(colsToRemove))\ntotal_df.drop(colsToRemove, axis=1, inplace=True)\nprint(f">> Dropped {len(colsToRemove)} duplicate columns")\n\n# Go through the columns one at a time (can\'t do it all at once for this dataset)\ntotal_df_all = deepcopy(total_df)              \nfor col in total_df.columns:\n    \n    # Detect outliers in this column\n    data = total_df[col].values\n    data_mean, data_std = np.mean(data), np.std(data)\n    cut_off = data_std * 3\n    lower, upper = data_mean - cut_off, data_mean + cut_off\n    outliers = [x for x in data if x < lower or x > upper]\n    \n    # If there are crazy high values, do a log-transform\n    if len(outliers) > 0:\n        non_zero_idx = data != 0\n        total_df.loc[non_zero_idx, col] = np.log(data[non_zero_idx])\n    \n    # Scale non-zero column values\n    nonzero_rows = total_df[col] != 0\n    total_df.loc[nonzero_rows, col] = scale(total_df.loc[nonzero_rows, col])\n    \n    # Scale all column values\n    total_df_all[col] = scale(total_df_all[col])\n    gc.collect()\n    \n# Train and test\ntrain_idx = range(0, len(train_df))\ntest_idx = range(len(train_df), len(total_df))')


# With that I end up with two dataframe, pre-processed slightly differently in terms of either scaling with sparse entries or without.

# ## 1.1. Performing PCA
# Since we have so many features, I thought it'd be a good idea to perform PCA prior to the t-SNE to reduce the dimensionality a bit. Arbitrarily I chose to include 1000 PCA components, which includes about 80% of the variation in the dataset, which I think it allright for saying something about the distributions, but also speeding up t-SNE a bit. In the following I show just the visualize only the plots from PCA on the dataset where scaling was performed excluding zeroes.

# In[ ]:


def test_pca(data, create_plots=True):
    """Run PCA analysis, return embedding"""
    
    # Create a PCA object, specifying how many components we wish to keep
    pca = PCA(n_components=1000)

    # Run PCA on scaled numeric dataframe, and retrieve the projected data
    pca_trafo = pca.fit_transform(data)    

    # The transformed data is in a numpy matrix. This may be inconvenient if we want to further
    # process the data, and have a more visual impression of what each column is etc. We therefore
    # put transformed/projected data into new dataframe, where we specify column names and index
    pca_df = pd.DataFrame(
        pca_trafo,
        index=total_df.index,
        columns=["PC" + str(i + 1) for i in range(pca_trafo.shape[1])]
    )

    # Only construct plots if requested
    if create_plots:
        
        # Create two plots next to each other
        _, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = list(itertools.chain.from_iterable(axes))

        # Plot the explained variance# Plot t 
        axes[0].plot(
            pca.explained_variance_ratio_, "--o", linewidth=2,
            label="Explained variance ratio"
        )

        # Plot the cumulative explained variance
        axes[0].plot(
            pca.explained_variance_ratio_.cumsum(), "--o", linewidth=2,
            label="Cumulative explained variance ratio"
        )

        # Show legend
        axes[0].legend(loc="best", frameon=True)

        # Show biplots
        for i in range(1, 4):

            # Components to be plottet
            x, y = "PC"+str(i), "PC"+str(i+1)

            # Plot biplots
            settings = {'kind': 'scatter', 'ax': axes[i], 'alpha': 0.2, 'x': x, 'y': y}
            pca_df.iloc[train_idx].plot(label='Train', c='#ff7f0e', **settings)
            pca_df.iloc[test_idx].plot(label='Test',  c='#1f77b4', **settings)    

        # Show the plot
        plt.show()
    
    return pca_df

# Run the PCA and get the embedded dimension
pca_df = test_pca(total_df)
pca_df_all = test_pca(total_df_all, create_plots=False)


# I included to plot the biplots just for fun, even though only very few percent of the variation are described by those components. Looks fun, and also hints at the training data being more spread out in those components than the test data, which seems more tightly clustered around the center.
# 
# ## 1.2. Running t-SNE
# Having reduced the dimensionality a bit it's now possible to run the t-SNE in about 5min or so, and subsequently plot both training and test data in the embedded 2D space. In the following I do that for both the dataset cases I have prepared to see any differences.

# In[ ]:


def test_tsne(data, ax=None, title='t-SNE'):
    """Run t-SNE and return embedding"""

    # Run t-SNE
    tsne = TSNE(n_components=2, init='pca')
    Y = tsne.fit_transform(data)

    # Create plot
    for name, idx in zip(["Train", "Test"], [train_idx, test_idx]):
        ax.scatter(Y[idx, 0], Y[idx, 1], label=name, alpha=0.2)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
    ax.legend()        
    return Y

# Run t-SNE on PCA embedding
_, axes = plt.subplots(1, 2, figsize=(20, 8))

tsne_df = test_tsne(
    pca_df, axes[0],
    title='t-SNE: Scaling on non-zeros'
)
tsne_df_unique = test_tsne(
    pca_df_all, axes[1],
    title='t-SNE: Scaling on all entries'
)

plt.axis('tight')
plt.show()  


# From this is seems like if scaling is performed only on non-zero entries, then the training and test set look more similar. If scaling is performed on all entries it seems like the two datasets are more separated from each other. In a previous notebook I didn't remove duplicate columns or columns with zero standard deviation - in that case even more significant differences were observed. Of course, it's still always important to be careful with t-SNE intepretation in my experience, and it might be worth looking into in more detail; both in terms of t-SNE parameters, pre-processing, etc.
# 
# ### 1.2.1. t-SNE colored by row-index or zero-count
# @avloss commented on this kernel about the fact that the data is time separated, so I thought it'd be interesting to look a bit more into why the t-SNE looks as it does. The two most obvious measures to investigate, that I could come up with off the top of my head, were the index of the rows (as a measure of 'time', assuming data is not shuffled), and the number of zeros for the given rows.

# In[ ]:


gc.collect()
# Get our color map
cm = plt.cm.get_cmap('RdYlBu')

# Create plot
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
sc = axes[0].scatter(tsne_df[:, 0], tsne_df[:, 1], alpha=0.2, c=range(len(tsne_df)), cmap=cm)
cbar = fig.colorbar(sc, ax=axes[0])
cbar.set_label('Entry index')
axes[0].set_title("t-SNE colored by index")
axes[0].xaxis.set_major_formatter(NullFormatter())
axes[0].yaxis.set_major_formatter(NullFormatter())

zero_count = (total_df == 0).sum(axis=1).values
sc = axes[1].scatter(tsne_df[:, 0], tsne_df[:, 1], alpha=0.2, c=zero_count, cmap=cm)
cbar = fig.colorbar(sc, ax=axes[1])
cbar.set_label('#sparse entries')
axes[1].set_title("t-SNE colored by number of zeros")
axes[1].xaxis.set_major_formatter(NullFormatter())
axes[1].yaxis.set_major_formatter(NullFormatter())
 


# Looks pretty interesting - seems like the higher-index rows are located more at the center of the plot. Also, we see a small cluster of rows with few zero-entries, as well as a few more clusters in the right-hand figure.
# 
# ### 1.2.2. t-SNE with different perplexities
# t-SNE can give some pretty tricky to intepret results depending on the perplexity parameter used. So just to be sure in the following I check for a few different values of the perplexity parameter.

# In[ ]:


_, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, perplexity in enumerate([5, 30, 50, 100]):
    
    # Create projection
    Y = TSNE(init='pca', perplexity=perplexity).fit_transform(pca_df)
    
    # Plot t-SNE
    for name, idx in zip(["Train", "Test"], [train_idx, test_idx]):
        axes[i].scatter(Y[idx, 0], Y[idx, 1], label=name, alpha=0.2)
    axes[i].set_title("Perplexity=%d" % perplexity)
    axes[i].xaxis.set_major_formatter(NullFormatter())
    axes[i].yaxis.set_major_formatter(NullFormatter())
    axes[i].legend() 

plt.show()


# Overall these all look pretty similar and show the same trend, so no need to worry about the perplexity parameter it seems.
# 
# ### 1.2.3. t-SNE colored by target
# For the training set it may be interesting to see how the different target values are separated on the embedded two dimensions.

# In[ ]:


# Create plot
fig, axes = plt.subplots(1, 1, figsize=(10, 8))
sc = axes.scatter(tsne_df[train_idx, 0], tsne_df[train_idx, 1], alpha=0.2, c=np.log1p(train_df.target), cmap=cm)
cbar = fig.colorbar(sc, ax=axes)
cbar.set_label('Log1p(target)')
axes.set_title("t-SNE colored by target")
axes.xaxis.set_major_formatter(NullFormatter())
axes.yaxis.set_major_formatter(NullFormatter())


# Clearly the different train target values are located at different locations in the t-SNE plot.

# # 2. Classification of Test vs. Train
# Another good check is to see how well we can classify whether a given entry belongs to test or training dataset - if it is possible to do this reasonably well, that is an indication of differences between the two dataset distributions. I'll just run a simple shuffled 10-fold cross-validation with a basic random forest model to see how well it performs this task. First let's try that classification on the case where scaling is performed on all entries:

# In[ ]:


def test_prediction(data):
    """Try to classify train/test samples from total dataframe"""

    # Create a target which is 1 for training rows, 0 for test rows
    y = np.zeros(len(data))
    y[train_idx] = 1

    # Perform shuffled CV predictions of train/test label
    predictions = cross_val_predict(
        ExtraTreesClassifier(n_estimators=100, n_jobs=4),
        data, y,
        cv=StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=42
        )
    )

    # Show the classification report
    print(classification_report(y, predictions))
    
# Run classification on total raw data
test_prediction(total_df_all)


# On the current notebook this gives about a 0.71 f1 score, which means we can do this prediction quite well, indicating some significant differences between the datasets. Let us try on the dataset where we only scaled non-zero values:

# In[ ]:


test_prediction(total_df)


# This reduced the f1 score a little bit down to 0.68; corresponding to what we observed in the t-SNE analysis, but still it's apparently quite easy for the model to decently well distinguish between train and test - considering the very simple classifcation model used here.

# # 3. Feature-by-feature distribution similarity
# Next let us try to look at the problem feature-by-feature, and perform Kolomogorov-Smirnov tests to see if the distribution in test and training set is similar. I'll use the function [scipy.stats.ks_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html#scipy.stats.ks_2samp) from scipy to run the tests. For all those features where the distributions are highly distinguishable, we may benefit from ignoring those columns, so as to avoid overfitting on training data. In the following I just identify those columns, and plot the distributions as a sanity check for some of the features

# In[ ]:


def get_diff_columns(train_df, test_df, show_plots=True, show_all=False, threshold=0.1):
    """Use KS to estimate columns where distributions differ a lot from each other"""

    # Find the columns where the distributions are very different
    diff_data = []
    for col in tqdm(train_df.columns):
        statistic, pvalue = ks_2samp(
            train_df[col].values, 
            test_df[col].values
        )
        if pvalue <= 0.05 and np.abs(statistic) > threshold:
            diff_data.append({'feature': col, 'p': np.round(pvalue, 5), 'statistic': np.round(np.abs(statistic), 2)})

    # Put the differences into a dataframe
    diff_df = pd.DataFrame(diff_data).sort_values(by='statistic', ascending=False)

    if show_plots:
        # Let us see the distributions of these columns to confirm they are indeed different
        n_cols = 7
        if show_all:
            n_rows = int(len(diff_df) / 7)
        else:
            n_rows = 2
        _, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows))
        axes = [x for l in axes for x in l]

        # Create plots
        for i, (_, row) in enumerate(diff_df.iterrows()):
            if i >= len(axes):
                break
            extreme = np.max(np.abs(train_df[row.feature].tolist() + test_df[row.feature].tolist()))
            train_df.loc[:, row.feature].apply(np.log1p).hist(
                ax=axes[i], alpha=0.5, label='Train', density=True,
                bins=np.arange(-extreme, extreme, 0.25)
            )
            test_df.loc[:, row.feature].apply(np.log1p).hist(
                ax=axes[i], alpha=0.5, label='Test', density=True,
                bins=np.arange(-extreme, extreme, 0.25)
            )
            axes[i].set_title(f"Statistic = {row.statistic}, p = {row.p}")
            axes[i].set_xlabel(f'Log({row.feature})')
            axes[i].legend()

        plt.tight_layout()
        plt.show()
        
    return diff_df

# Get the columns which differ a lot between test and train
diff_df = get_diff_columns(total_df.iloc[train_idx], total_df.iloc[test_idx])


# On my run it dropped about 150 features. Let's try a classification report again to see if we can distinguish test from train.

# In[ ]:


# Run classification on total raw data
print(f">> Dropping {len(diff_df)} features based on KS tests")
test_prediction(
    total_df.drop(diff_df.feature.values, axis=1)
)


# Here we actually see lower score, down from 68% to 62%, meaning train and test are more similar. I've not tested these things with any regressors yet, but I'd think it might be interesting to drop some if not all of these features which may enable the model to overfit on training data. I'm not sure Kolmogorov–Smirnov is neccesarily the absolute best statistical test for comparing these kinda-discrete distributions - I've tried only running it on non-zero entries, but in that case we end up removing many more features, while still allowing the model to easily distinguish between train and test based on the zeroes. Suggestions on how to approach this more thoroughly would be appreciated.
# 
# # 4. Decomposition Feature
# So far I've only looked at PCA components, but most kernels look at several decomposition methods, so it may be interesting to look at t-SNE of these 10-50 components of each method instead of 1000 PCA components. Furthermore, it's interesting to see how well we can classify test/train based on this reduced feature space.
# 
# 

# In[ ]:


COMPONENTS = 20

# List of decomposition methods to use
methods = [
    TruncatedSVD(n_components=COMPONENTS),
    PCA(n_components=COMPONENTS),
    FastICA(n_components=COMPONENTS),
    GaussianRandomProjection(n_components=COMPONENTS, eps=0.1),
    SparseRandomProjection(n_components=COMPONENTS, dense_output=True)    
]

# Run all the methods
embeddings = []
for method in methods:
    name = method.__class__.__name__    
    embeddings.append(
        pd.DataFrame(method.fit_transform(total_df), columns=[f"{name}_{i}" for i in range(COMPONENTS)])
    )
    print(f">> Ran {name}")
    
# Put all components into one dataframe
components_df = pd.concat(embeddings, axis=1)

# Prepare plot
_, axes = plt.subplots(1, 3, figsize=(20, 5))

# Run t-SNE on components
tsne_df = test_tsne(
    components_df, axes[0],
    title='t-SNE: with decomposition features'
)

# Color by index
sc = axes[1].scatter(tsne_df[:, 0], tsne_df[:, 1], alpha=0.2, c=range(len(tsne_df)), cmap=cm)
cbar = fig.colorbar(sc, ax=axes[1])
cbar.set_label('Entry index')
axes[1].set_title("t-SNE colored by index")
axes[1].xaxis.set_major_formatter(NullFormatter())
axes[1].yaxis.set_major_formatter(NullFormatter())

# Color by target
sc = axes[2].scatter(tsne_df[train_idx, 0], tsne_df[train_idx, 1], alpha=0.2, c=np.log1p(train_df.target), cmap=cm)
cbar = fig.colorbar(sc, ax=axes[2])
cbar.set_label('Log1p(target)')
axes[2].set_title("t-SNE colored by target")
axes[2].xaxis.set_major_formatter(NullFormatter())
axes[2].yaxis.set_major_formatter(NullFormatter())

plt.axis('tight')
plt.show()  


# Let us check how well we can classify train from test with these feature:

# In[ ]:


test_prediction(components_df)


# So here we get a classification f1 score of about 0.83, which is pretty bad I would say. Clearly the test and training are very different from each other looking at these components. Let us try to use the KS tests again to eliminate columns that are significantly different from each other.

# In[ ]:


# Get the columns which differ a lot between test and train
diff_df = get_diff_columns(
    components_df.iloc[train_idx], components_df.iloc[test_idx],
    threshold=0.1
)

# Run classification on total raw data
print(f">> Dropping {len(diff_df)} features based on KS tests")
test_prediction(
    components_df.drop(diff_df.feature.values, axis=1)
)


# So by dropping 78 features we're down to an f1 score of 0.6. I've not tried testing any of this against either local CV score or public LB score, and probably all these features should not be dropped, but I imagine some of them could be leading to overfitting on the training set.
# 
# To be updated.
