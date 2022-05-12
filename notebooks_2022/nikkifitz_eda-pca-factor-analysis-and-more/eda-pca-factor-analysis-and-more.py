#!/usr/bin/env python
# coding: utf-8

# ## **Preliminaries**

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install Boruta\n!pip install fastcluster\n!pip install factor-analyzer')


# In[ ]:


# import libraries
import fastcluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from boruta import BorutaPy
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo, FactorAnalyzer # the sklearn FactorAnalysis function is inadequate
from scipy.cluster import hierarchy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import FeatureAgglomeration 
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif, RFECV
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import NuSVC, SVC


# In[ ]:


# import the data
train = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
sample_sub = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv')


# In[ ]:


# take a look at the last couple of lines from the training dataset
train.tail(10)


# In[ ]:


# take a look at the last couple of lines from the testing dataset
test.tail(10)


# In[ ]:


train.describe()


# In[ ]:


train.info()


# So at first glance, it looks like we have a mix of continuous/interval, binary and ordinal/categorical variables in the mix. Let's visualise these features to confirm.

# In[ ]:


# but first we need to encode "f_27"
oenc = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = 999999)

train['f_27_enc'] = oenc.fit_transform(train['f_27'].values.reshape(-1,1))


# In[ ]:


fig, axs = plt.subplots(6, 6, figsize=(40, 20), sharex=False)
for i, ax in enumerate(axs.flatten()):
    if i < len(train.columns):
        if train.columns[i] != "f_27":
            sns.histplot(data = train, x = train.columns[i], ax = ax)
            sns.despine();
    else:
        ax.set_axis_off()


# In[ ]:


# let's drop the id column and f_27
train2 = train.drop(['id', 'f_27'], axis = 1)

# separate the target column out
target = train2.pop("target")
# and temporarily drop the new f_27
f_27 = train2.pop("f_27_enc")

# ... before reinserting it in it's proper place
train2.insert(loc = 27, column = "f_27", value = f_27)


# In[ ]:


# plot a cluster map - can't be visualised with the entire dataset/takes a loong time to do so
lut = dict(zip(target.unique(), "rbg"))
row_colors = target.map(lut)

sns.clustermap(train2.iloc[:1000], metric = "euclidean", method = "ward", row_colors = row_colors, cmap="mako", z_score = 1);


# In[ ]:


# reinsert the target variable
train2_incl_target = train2.copy()
train2_incl_target.insert(loc = 31, column = "target", value = target)


# In[ ]:


# calculate the correlations
corr = train2_incl_target.corr(method = "pearson")

plt.figure(figsize=(40, 10))
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n = 200), square=True, linewidth = 0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right');


# There's not a lot to work with here, but let's proceed with performing a number of different decomposition techniques to see what comes out.

# ## **PCA**

# In[ ]:


# scale the data
sc = StandardScaler()
train2_sc = sc.fit_transform(train2)

# perform the PCA
pca = PCA(n_components = 5)
train_pc = pca.fit_transform(train2_sc)

pc_df = pd.DataFrame(data = train_pc, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
pc_df['target'] = target

# observed the amount of explained variation by each component
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# extract the feature weights for each component
weights = pca.components_
weights_df = pd.DataFrame(data = weights.reshape(31,5), columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index = train2.columns)
#weights_df = weights_df.sort_values(by = 'PC1', axis = 0, ascending = False)
#weights_df = weights_df.sort_values(by = 'PC2', axis = 0, ascending = False)
#weights_df.head()


# In[ ]:


# loadings plot
plt.figure(figsize = (10,10))
p1 = sns.scatterplot(x = "PC1", y = "PC2", data = weights_df)
plt.axvline(x = 0, color = 'red', alpha = 0.5, ls = '--')
plt.axhline(y = 0, color = 'red', alpha = 0.5, ls = '--')
sns.despine();

# add text annotations
for line in range(0, weights_df.shape[0]):
     p1.text(weights_df['PC1'][line]+0.01, weights_df['PC2'][line], 
             weights_df.index[line], horizontalalignment='left', 
             size = 'medium', color='black')


# In[ ]:


# observations plot
plt.figure(figsize = (10,10))
sns.scatterplot(x = "PC1", y = "PC2", hue = "target", data = pc_df)
sns.despine();


# Clearly doing a horrible job at distinguishing between the two classes.

# ## **Factor Analysis**

# In[ ]:


# first test if the dataset is a good candidate for factor analysis - which it isn't but we'll proceed anyway
chi_square_value, p_value = calculate_bartlett_sphericity(train2)
kmo_all, kmo_model = calculate_kmo(train2)
print('Bartlett-sphericity chi-square: {}'.format(chi_square_value))
print('Bartlett-sphericity p-value: {}'.format(p_value))
print('KMO score: {}'.format(kmo_model))


# In[ ]:


train2_sc = sc.fit_transform(train2)

# perform the Factor Analysis
base_fa = FactorAnalyzer(rotation = None)
train_fa = base_fa.fit_transform(train2_sc)

ev, v = base_fa.get_eigenvalues()
ev_df = pd.DataFrame(ev, columns = ['eigenvalue'])
ev_df['factor'] = range(1,32)

# scree plot
plt.figure(figsize = (15,5))
sns.lineplot(x = "factor", y = "eigenvalue", data = ev_df)
plt.axhline(y = 1.0, color = 'red', alpha = 0.5, ls = '--')
sns.despine();


# 13 factors have an eigenvalue greater than one, so this is the number of unobserved variables.

# In[ ]:


# repeat but with promax rotation and a pre-specified number of factors
fa = FactorAnalyzer(n_factors = 13, rotation = 'promax', method = "ml")
train_fa = fa.fit_transform(train2_sc)

fvariance = fa.get_factor_variance()
variance_df = pd.DataFrame(fvariance, columns = ['Factor {}'.format(i) for i in range(1, 13+1)], index = ['SS loadings', 'Proportion Var', 'Cumulative Var'])
variance_df


# In[ ]:


# get loadings
loadings = fa.loadings_
loadings_df = pd.DataFrame(loadings, columns = ['Factor {}'.format(i) for i in range(1, 13+1)], index = train2.columns)
loadings_df['highest loading'] = loadings_df.idxmax(axis = 1)
loadings_df


# So, now this is interesting:
# - Factor 1 is loaded almost purely on f_25
# - Factor 2 is loaded almost purely on f_30
# - Factor 3 is loaded almost purely on f_03
# - Factor 4 is loaded almost purely on f_20
# - Factor 5 is loaded almost purely on f_05
# - Factor 6 - f_28
# - Factor 7 - f_19
# - Factor 8 - a high loading on f_21
# - Factor 9 to 13 - nothing high, suggesting we can drop those factors
# 
# There is also agreement amongst the various rotation methods that actually 6/7 factors is sufficient.
# 
# The final column identifies the factor with the largest loading for each variable, which could provide some direction as to what groupings of variables to investigate for interactions.

# In[ ]:


# repeat but with promax rotation and 8 factors
fa_8 = FactorAnalyzer(n_factors = 8, rotation = 'promax', method = "ml")
train_fa_8 = fa_8.fit_transform(train2_sc)

fvariance = fa_8.get_factor_variance()
variance_df = pd.DataFrame(fvariance, columns = ['Factor {}'.format(i) for i in range(1, 8+1)], index = ['SS loadings', 'Proportion Var', 'Cumulative Var'])
variance_df


# In[ ]:


# get new loadings
loadings = fa_8.loadings_
loadings_df = pd.DataFrame(loadings, columns = ['Factor {}'.format(i) for i in range(1, 8+1)], index = train2.columns)
loadings_df['highest loading'] = loadings_df.idxmax(axis = 1)
loadings_df = loadings_df.sort_values(by = ['highest loading'])
loadings_df


# The point of all this was to attempt to find interlinked associations. i.e. to reduce the observed variables into a few latent/unobserved variables or identify groups of interlinked variables to reveal hidden relationships.
# 
# As per the previous comment, the last column could provide some direction on what variables to investigate for interactions. For example, the loadings table indicates that f_26, f_24 and f_20 could be describing the same unobserved feature.

# In[ ]:


# plot the heatmap for the loadings
plt.figure(figsize = (80, 5))

ax = sns.heatmap(loadings_df.drop('highest loading', axis=1).T, 
                 vmin = -1, vmax = 1, center = 0,
                 cmap = sns.diverging_palette(220, 20, n = 200),
                 square = True, annot = True, fmt = '.2f', annot_kws = {"fontsize":8})
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment='right');


# ## **Hierachical Clustering of Variables**

# In[ ]:


# lifted verbatim from the sklearn documentation
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    hierarchy.dendrogram(linkage_matrix, **kwargs)


# In[ ]:


# fit the two models to the scaled data
euc_fc = FeatureAgglomeration(distance_threshold = 0, n_clusters = None, affinity = "euclidean", linkage = "ward")
euc_fit = euc_fc.fit(train2_sc)

cos_fc = FeatureAgglomeration(distance_threshold = 0, n_clusters = None, affinity = "cosine", linkage = "complete")
cos_fit = cos_fc.fit(train2_sc)

# and plot their respective dendograms
plt.figure(figsize = (20, 8))
plt.title("Hierarchical Clustering Dendrogram - euclidean distances and ward linkage")
hierarchy.set_link_color_palette(['grey', 'maroon', 'burlywood', 'darkgreen', 'deepskyblue', 'mediumslateblue'])
plot_dendrogram(euc_fit, leaf_rotation = 0, color_threshold = 1400, labels = train2.columns)

plt.figure(figsize = (20, 8))
plt.title("Hierarchical Clustering Dendrogram - cosine distances and complete linkage")
plot_dendrogram(cos_fit, leaf_rotation = 0, color_threshold = 1.05, labels = train2.columns)

# reset the colour palette
hierarchy.set_link_color_palette(None)


# The cosine distance clustering is possibly slightly preferable as it gives fairly even clusters, whereas the euclidean clustering has one cluster much bigger than the rest. That being said, if you ignore the colours of the clusters (they mean absolutely nothing) and concentrate just on which variables have been placed together, you'll find remarkable consistency between the two results. 

# ## **Univariate Feature Selection**

# In[ ]:


mi_func = mutual_info_classif(train2, target)
f_func = f_classif(train2, target)

ufc_df = pd.DataFrame(data = {'F stat': f_func[0], 
                              'p-value': f_func[1],
                              'mutual info score': mi_func}, index = train2.columns)

fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex = False)

plot_data = ufc_df.sort_values(by = 'mutual info score', ascending = False)
p = sns.barplot(x = plot_data.index, y = "mutual info score", data = plot_data, ax = axs[0])
p.set_title('Feature importance based on mutual information scores')

plot_data = ufc_df.sort_values(by = 'F stat', ascending = False)
p = sns.barplot(x = plot_data.index, y = 'F stat', data = plot_data, ax = axs[1])
p.set_title('Feature importance based on ANOVA F-statistics')
sns.despine();


# So, according to both these methods, f_27 carries the most predictive power with respect to predicting the target class followed by f_30 or f_21 depending on your metric of choice. However, I wonder if that's purely because the encoded-values of f_27 are so much larger than any other feature. Let's find out...

# In[ ]:


mi_func = mutual_info_classif(train2_sc, target)
f_func = f_classif(train2_sc, target)

ufc_df = pd.DataFrame(data = {'F stat': f_func[0], 
                              'p-value': f_func[1],
                              'mutual info score': mi_func}, index = train2.columns)

fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex = False)

plot_data = ufc_df.sort_values(by = 'mutual info score', ascending = False)
p = sns.barplot(x = plot_data.index, y = "mutual info score", data = plot_data, ax = axs[0])
p.set_title('Feature importance based on mutual information scores and the scaled data')

plot_data = ufc_df.sort_values(by = 'F stat', ascending = False)
p = sns.barplot(x = plot_data.index, y = 'F stat', data = plot_data, ax = axs[1])
p.set_title('Feature importance based on ANOVA F-statistics and the scaled data')
sns.despine();


# So it's not the case and f_27 is actually the most important predictor. However, whilst the ANOVA metric maintains that f_21 is next-most important, the mutual information metric changes its mind and puts f_29 in second place (up from way down the list) and f_30 in third.

# ## **Boruta**
# 
# Boruta is an interesting package. It claims that it's an all-relevant feature selection method rather than just trying to minimise the error, which means that it's attempting to find *all* features carrying information useful for prediction.

# In[ ]:


# define the random forest classifier
rf = RandomForestClassifier(n_jobs = -1, max_depth = 3)  ## recommended max_depth 3 to 7

# define the Boruta feature selection method
# perc = 100 - threshold for comparison between real and shadow features, i.e. the maximum
# verbose = 2, print all output
feat_selector = BorutaPy(rf, n_estimators = 'auto', verbose = 2, max_iter = 100, random_state = 22)

# find all relevent features
feat_selector.fit(train2.values, target.ravel())  ## BorutaPy only accepts numpy arrays


# In[ ]:


# check the number of selected features
print('The number of selected features {}.'.format(feat_selector.n_features_))

# check which features were selected/retained
#print('Selected features {}'.format(feat_selector.support_))

# check feature rankings
#print('Feature rankings {}'.format(feat_selector.ranking_))


# ## **Scatter Plot Matrix (SPLOM)**

# In[ ]:


# apologies for this being so small
g = sns.PairGrid(train2_incl_target.iloc[:1000], hue = "target")
g.map_diag(sns.kdeplot)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)


# ## **Pairwise Interaction Modelling - Perceptron**

# A Perceptron classifier is a linear classifier suitable for large-scale learning. That is, it doesn't require a learning rate, isn't regularised and updates its model only on mistakes.

# In[ ]:


# taking an idea from another competitor - encoding each letter in f_27 as its own value
split_f27 = train.f_27.str.split('', n = 0, expand = True)

encoded_f27 = train.copy()
for col in np.arange(1, 10+1):
    encoded_f27['f_27_' + str(col)] = split_f27[col]
encoded_f27.drop(encoded_f27.columns[0:34], axis = 1, inplace = True)

# then encoding them
oenc2 = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = 99)

for col in encoded_f27.columns:
    encoded_f27[col] = oenc2.fit_transform(encoded_f27[col].values.reshape(-1,1))


# In[ ]:


# create all the pairwise interactions b/w f_27 (and variations thereof) and the other variables
# this results in a performance warning regarding a fragmented data frame, but it can be safely ignored (I think)
train_int = train2.copy()
train_int = pd.concat([train_int, encoded_f27], axis = 1)
    
size_col = train_int.columns.size
for i in range(0, size_col):
    for j in range (i + 1, size_col):
        col1 = str(train_int.columns[i])
        col2 = str(train_int.columns[j])
        col_name = col1 + "_" + col2
        train_int[col_name] = pd.Series(train_int[col1] * train_int[col2], name = col_name)


# In[ ]:


# split train data into training and validation sets (or development and evaluation as sklearn calls them)
# running into memory/RAM issues so only training on a subset of the data
X = train_int[:30000]
y = target[:30000]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 22)


# In[ ]:


# train an elastic net classifier model, using 5-fold cross-validation to identify the best combination of lambda and 
# alpha
params = {'l1_ratio': [0.2, 0.4, 0.6, 0.8, 1],
          'alpha': [0.0001, 0.001, 0.01, 0.1]}

pmod = Perceptron(penalty = "elasticnet", random_state = 22, n_jobs = -1, verbose = 0, early_stopping = True)
p_clf = GridSearchCV(estimator = pmod, 
                     param_grid = params, 
                     scoring = 'roc_auc',
                     n_jobs = -1)
p_clf.fit(X_train, y_train)


# In[ ]:


print(p_clf.best_estimator_)
print(p_clf.best_params_)
print(p_clf.best_score_)


# Repeated runs have indicated that an `alpha` of 0.01 and `l1 ratio` of 1 produces the highest score. This means that the model is basically a Lasso regression model, which means that it can perform feature selection by setting some coefficients in the model to zero.

# In[ ]:


p_coef_df = pd.DataFrame(data = {'feature': train_int.columns,
                                 'coefficient': p_clf.best_estimator_.coef_[0],
                                 'abs_coefficient': abs(p_clf.best_estimator_.coef_[0])})
p_coef_df = p_coef_df.sort_values(by = 'abs_coefficient', ascending = False)
p_coef_df = p_coef_df.loc[p_coef_df.coefficient > 0]

plt.figure(figsize = (20, 5))
p1 = sns.stripplot(data = p_coef_df[:30], x = "feature", y = "abs_coefficient", size = 10, jitter = False, 
                   palette = "mako", linewidth = 1, edgecolor = "w")
p1.set_xticklabels(p1.get_xticklabels(), rotation = 30)
sns.despine();


# There's a common factor here. Even if the model did no better than chance, it still thought that the interactions between f_27 and many of the other features were very important in predicting the target class. It's also interesting to note that the top 30 features in the model are all pairwise interactions...

# ... and that's all for this one folks. Thanks to everyone that has upvoted my notebook. It means a lot that my work is appreciated. There's a couple of other things I wanted to include, but they just didn't seem to want to work or were taking a crazy amount of time to run, even on a reduced dataset.
