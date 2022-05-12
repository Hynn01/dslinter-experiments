#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Hey, thanks for viewing my Kernel!
# 
# If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! :)

# In[ ]:


import numpy as np
import pandas as pd
import warnings

warnings.simplefilter("ignore")
train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
display(train.head())
display(test.head())
display(sub.head())


# In[ ]:


print("Train shape:",train.shape)
print("Test shape:",test.shape)


# In[ ]:


display(train.isna().sum().sum())
display(test.isna().sum().sum())


# In[ ]:


display(train.duplicated().sum())
display(test.duplicated().sum())


# In[ ]:


display(train['target'].value_counts(normalize=True))


# In[ ]:


int_features = list(test.select_dtypes(include='int').columns)
int_features.remove('id')
float_features = list(test.select_dtypes(include='float').columns)
object_features = list(test.select_dtypes(include='object').columns)
print("int featres:", int_features)
print("float featres:", float_features)
print("object featres:", object_features)


# In[ ]:


display(train[int_features].nunique())


# In[ ]:


display(train[object_features].nunique())


# In[ ]:


from IPython.core.display import HTML
def value_counts_all(df, columns):
    pd.set_option('display.max_rows', 50)
    table_list = []
    for col in columns:
        table_list.append(pd.DataFrame(df[col].value_counts()))
    return HTML(
        f"<table><tr> {''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list])} </tr></table>")


# In[ ]:


value_counts_all(train, int_features)


# In[ ]:


value_counts_all(test, int_features)


# ### Insights 1
# 
# - f_07, f_09, f_10, f_11, f_13 has one more different value in test set
# - f_08, f_14, f_15, f_16 has more value in train set than test set
# - f_12 has 16 in train but has 15 in test. f_17 has 14 in train but has 13 in test. Just one value.
# - f_29, f_30 can be object types

# # Distributions

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def plot_distributions(data, features, hue='target', ncols=3, method='hist'):
    nrows = round(len(features) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, round(nrows*16/ncols)))
    col_i, row_i = 0, 0
    for index, feature in enumerate(features):
        if method == 'hist':
            sns.kdeplot(data=data, x=feature, hue=hue, ax=axes[row_i][col_i])
        elif method == 'count':
            temp = data.sort_values(feature)
            sns.countplot(data=temp, x=feature, hue=hue, ax=axes[row_i][col_i])
        elif method == 'bar':
            temp = data.copy()
            temp['counts'] = 1
            temp = temp.groupby([hue, feature], as_index=False).agg({'counts':'sum'})
            sns.barplot(data=temp, x=feature, y='counts', hue=hue, ax=axes[row_i][col_i])
        col_i += 1
        if col_i == ncols:
            col_i = 0
            row_i += 1
    plt.show()
    


# In[ ]:


def histogram_correlation_plot(data, features, target, ncols=3, rolling_num=1000):
    nrows = round(len(features) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, round(nrows*16/ncols)))
    col_i, row_i = 0, 0
    for index, feature in enumerate(features):
        temp = data.sort_values(feature)
        temp.reset_index(inplace=True)
        axes[row_i][col_i].scatter(temp.index, temp[target].rolling(rolling_num).mean(), s=1, alpha=0.5)
        axes[row_i][col_i].set_title(feature)
        axes[row_i][col_i].set_xticks(np.arange(0, 900000, step=10000))
        col_i += 1
        if col_i == ncols:
            col_i = 0
            row_i += 1
    plt.show()


# In[ ]:


plot_distributions(train, float_features, hue='target', ncols=4, method='hist')


# In[ ]:


histogram_correlation_plot(train, float_features, 'target', ncols=4, rolling_num=10000)


# ### Insights 2
# 
# - f_00, f_01, f_02, f_05 has same pattern
# - f_21, f_22, has same pattern
# - f_20, f_25, has same pattern
# - f_23, f_28, has same pattern
# - f_28 is opposite to the f_25 and f_23 is opposite to the f_20
# - f_19 and f_24 are highly correlated with target
# - f_03, f_04, f_06 have a low relationship with the target

# In[ ]:


plot_distributions(train, int_features, hue='target', ncols=3, method='bar')


# In[ ]:


histogram_correlation_plot(train, int_features, 'target', ncols=3, rolling_num=10000)


# # Correlations

# In[ ]:


def display_p_values(df, columns, target, th=0.05, cut=False):
    from scipy.stats import pearsonr
    p_values_list = []
    for c in columns:
        p = round(pearsonr(train.loc[:,target], train.loc[:,c])[1], 4)
        p_values_list.append(p)

    p_values_df = pd.DataFrame(p_values_list, columns=[target], index=columns)
    def p_value_warning_background(cell_value):
        highlight = 'background-color: lightcoral;'
        default = ''
        if cell_value > th:
                return highlight
        return default
    
    if cut:
        p_values_df_high = p_values_df[p_values_df[target] > th]
    else:
        p_values_df_high = p_values_df.copy()
    display(p_values_df_high.style.applymap(p_value_warning_background))


# In[ ]:


display_p_values(train, float_features, 'target', th=0.05)


# In[ ]:


display_p_values(train, int_features, 'target', th=0.05)


# ### Insights 3
# 
# - f_03, f_04, f_06 have a low relationship with the target too in this part
# - f_12, f_17 have a low relationship with the target but not low as f_03, f_04

# # Feature Engineering

# In[ ]:


def create_features(data):
    object_data_cols = [f"f_27_{i+1}" for i in range(10)]
    object_data = pd.DataFrame(data['f_27'].apply(list).tolist(), columns=object_data_cols)
    for feature in object_data_cols:
        object_data[feature] = object_data[feature].apply(ord) - ord('A')
    
    data = pd.concat([data, object_data], 1)
    data["unique_characters"] = data.f_27.apply(lambda s: len(set(s)))
    return data


# In[ ]:


train_fe = create_features(train.copy())
test_fe = create_features(test.copy())
train_fe.head()


# In[ ]:


fe_object_features = [f"f_27_{i+1}" for i in range(10)]
fe_object_features.append("unique_characters")
plot_distributions(train_fe, fe_object_features, hue='target', ncols=3, method='count')


# # Dimension Reduction Analysis

# ## PCA Analysis

# In[ ]:


def plot_pca_target(X, y, features, title, figsize=(16, 8)):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    pca = PCA(n_components=2)
    X_std = StandardScaler().fit_transform(X[features])
    tr_p = pca.fit_transform(X_std)
    
    df_pca = pd.DataFrame()
    df_pca['x'] = tr_p[:, 0]
    df_pca['y'] = tr_p[:, 1]
    df_pca['target'] = y
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df_pca, x='x', y='y', hue='target', alpha=0.5, s=2)
    ax.set_title(title)
    plt.show()
    
def plot_pca_traintest(X_train, X_test, features, title, figsize=(16, 8)):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train[features])
    X_test_std = scaler.transform(X_test[features])
    tr_p = pca.fit_transform(X_train_std)
    te_p = pca.transform(X_test_std)
    
    plt.figure(figsize=figsize)
    plt.gca()
    plt.scatter(tr_p[:,0], tr_p[:,1], s=1, c='blue', label='Train') # train: blue
    plt.scatter(te_p[:,0], te_p[:,1], s=1, c='red', label='Test') # test: yellow
    plt.legend()
    plt.title(title)
    plt.show()
    


# In[ ]:


features = [col for col in train_fe.columns if col != "id" and col != "target" and col != "f_27"]
plot_pca_target(train_fe, train_fe[['target']], features, "PCA Target")


# In[ ]:


plot_pca_traintest(train_fe, test_fe, features, "PCA Train-Test")


# ## UMAP

# In[ ]:





# # Baseline

# Subset so things run in reasonable time.

# In[ ]:


from sklearn.model_selection import train_test_split
df_sample = train_fe.sample(n=100000)
X_train, X_test, y_train, y_test = train_test_split(
    df_sample.drop(['id', 'target', 'f_27'], 1), df_sample[['target']], test_size=0.33, random_state=42)


# In[ ]:


from catboost import CatBoostClassifier
model = CatBoostClassifier(cat_features=fe_object_features, 
                       grow_policy = "Lossguide")
model.fit(X_train, y_train, verbose=0)


# In[ ]:


prediction = model.predict(X_test)
accuracy = np.mean(prediction == y_test.to_numpy().flatten())
print("Accuracy: "+str(accuracy))


# # Feature Importance

# In[ ]:


def plot_feature_importance(importance,names,model_type, figsize=(10, 8)):
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=figsize)
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


# In[ ]:


plot_feature_importance(model.feature_importances_, X_train.columns, 'CatBoost', figsize=(8, 16))


# # Insights
# 
# ### Insights 1
# 
# - f_07, f_09, f_10, f_11, f_13 has one more different value in test set
# - f_08, f_14, f_15, f_16 has more value in train set than test set
# - f_12 has 16 in train but has 15 in test. f_17 has 14 in train but has 13 in test. Just one value.
# - f_29, f_30 can be object types
# 
# ### Insights 2
# 
# - f_00, f_01, f_02, f_05 has same pattern
# - f_21, f_22, has same pattern
# - f_20, f_25, has same pattern
# - f_23, f_28, has same pattern
# - f_28 is opposite to the f_25 and f_23 is opposite to the f_20
# - f_19 and f_24 are highly correlated with target
# - f_03, f_04, f_06 have a low relationship with the target
# 
# ### Insights 3
# 
# - f_03, f_04, f_06 have a low relationship with the target too in this part
# - f_12, f_17 have a low relationship with the target but not low as f_03, f_04

# # Subset for Testing

# # Bayesian Optimization

# In[ ]:


X1 = X_train
Y1 = y_train


# In[ ]:


# From: https://ai.plainenglish.io/catboost-cross-validated-bayesian-hyperparameter-tuning-91f1804b71dd

from catboost import Pool, cv, CatBoostClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import * 
from sklearn.metrics import *


def CB_opt(n_estimators, depth, learning_rate, max_bin,
             subsample, num_leaves, l2_leaf_reg, model_size_reg): 
  scores = []
  skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1944)
  for train_index, test_index in skf.split(X1, Y1):
    
    trainx, valx = X1.iloc[train_index], X1.iloc[test_index]
    trainy, valy = Y1.iloc[train_index], Y1.iloc[test_index]
 
    reg = CatBoostClassifier(verbose = 0,
                            n_estimators = int(n_estimators),
                            learning_rate = learning_rate,
                            subsample = subsample, 
                            l2_leaf_reg = l2_leaf_reg,
                            max_depth = int(depth),
                            num_leaves = int(num_leaves),
                            random_state = 88,
                            grow_policy = "Lossguide",
                            max_bin = int(max_bin),  
                            use_best_model = True, 
                            model_size_reg = model_size_reg,
                            cat_features = fe_object_features,
                           
                            )
    
    reg.fit(trainx, trainy, eval_set = (valx, valy))
    scores.append(accuracy_score(valy, reg.predict(valx)))
  return np.mean(scores)


# In[ ]:


pbounds = {"n_estimators": (150,500),
           "depth": (2,25),
           "learning_rate": (.01, 0.3),
           "subsample":(0.6, 1.),
           "num_leaves": (16,60),
           "max_bin":(150,350),
           "l2_leaf_reg":(0,10),
           "model_size_reg": (0,10)
}
optimizer = BayesianOptimization(
    f = CB_opt,
    pbounds = pbounds,
    verbose = 2,
    random_state = 888,
)

optimizer.maximize(init_points = 2, n_iter = 50)

print(optimizer.max)


# In[ ]:


optimizer.max["params"]


# In[ ]:


clf = CatBoostClassifier(verbose=0,
                       n_estimators=np.int(optimizer.max["params"]["n_estimators"]),
                       depth=np.int(optimizer.max["params"]["depth"]),
                       learning_rate=optimizer.max["params"]["learning_rate"],
                       subsample=optimizer.max["params"]["subsample"],
                       num_leaves=np.int(optimizer.max["params"]["num_leaves"]),
                       max_bin=np.int(optimizer.max["params"]["max_bin"]),
                       l2_leaf_reg=optimizer.max["params"]["l2_leaf_reg"],
                       model_size_reg=optimizer.max["params"]["model_size_reg"],
                       grow_policy = "Lossguide",
                       cat_features = fe_object_features,
                        )


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


prediction_BayesOpt = clf.predict(X_test)
accuracy_BayesOpt = np.mean(prediction_BayesOpt == y_test.to_numpy().flatten())
print("Accuracy: "+str(accuracy_BayesOpt))


# In[ ]:


#! pip install --upgrade scikit-learn
#! pip install lazypredict
#from lazypredict.Supervised import LazyClassifier


# In[ ]:


## fit all models
#clf = LazyClassifier(predictions=True)
#models, predictions = clf.fit(X_train, X_test, y_train, y_test)


# ## !!! Work in Progress !!!
