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


train['f_26_diff'] = train['f_26'].diff()
test['f_26_diff'] = test['f_26'].diff()
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)


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
    
def plot_pca_traintest(X_train, X_test, features, title, figsize=(16, 8), max_dis=0.5, find_hard_rows=True, th=0.75):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import RadiusNeighborsClassifier
    
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train[features])
    X_test_std = scaler.transform(X_test[features])
    tr_p = pca.fit_transform(X_train_std)
    te_p = pca.transform(X_test_std)
    
    fig, ax = plt.subplots(figsize=figsize)
            
    #ax.scatter(tr_p[:,0], tr_p[:,1], s=1, c='blue', label='Train') # train: blue
    #ax.scatter(te_p[:,0], te_p[:,1], s=1, c='orange', label='Test') # test: yellow
    sns.scatterplot(x=tr_p[:,0], y=tr_p[:,1], s=2, label='Train', ax=ax)
    sns.scatterplot(x=te_p[:,0], y=te_p[:,1], s=2, label='Test', ax=ax)
    ax.legend()
    ax.set_title(title)
    
    if find_hard_rows:
        neigh = RadiusNeighborsClassifier(radius=max_dis, algorithm='kd_tree', leaf_size=40)
        x_q75, x_q25 = np.percentile(tr_p[:, 0], [th, 100-th])
        y_q75, y_q25 = np.percentile(tr_p[:, 1], [th, 100-th])
        condition_tr = (((tr_p[:, 0]>x_q75)|(tr_p[:, 0]<x_q25))&((tr_p[:, 1]>y_q75)|(tr_p[:, 1]<y_q25)))
        s_tr_p = tr_p[condition_tr, :]
        condition_te = (((te_p[:, 0]>x_q75)|(te_p[:, 0]<x_q25))&((te_p[:, 1]>y_q75)|(te_p[:, 1]<y_q25)))
        indexs = np.arange(0, len(X_test))
        s_te_p = te_p[condition_te, :]
        s_indexs = indexs[condition_te]
        print("th x values:", x_q75, x_q25, "- th y values:", y_q75, y_q25)
        print("train shape:", s_tr_p.shape, "- test shape:", s_te_p.shape)
        
        neigh.fit(s_tr_p, np.ones(s_tr_p.shape[0]))
        neigh_radius = neigh.radius_neighbors(s_te_p)
        # neigh_radius[0][0]: distances, neigh_radius[1][0]: macthes
        hard_rows_list = []
        for i in range(s_te_p.shape[0]):
            if len(neigh_radius[0][i]) == 0:
                hard_rows_list.append(s_indexs[i])
        
        for index in hard_rows_list:
            circle = plt.Circle((te_p[index, 0], te_p[index, 1]), max_dis, color='red', fill=False)
            ax.add_patch(circle)
        print("hard_rows_list len:", len(hard_rows_list))
        
        return hard_rows_list
    


# In[ ]:


features = [col for col in train_fe.columns if col != "id" and col != "target" and col != "f_27"]
plot_pca_target(train_fe, train_fe[['target']], features, "PCA Target")


# In[ ]:


import gc

del train
del test
gc.collect()


# In[ ]:


hard_rows_list = plot_pca_traintest(train_fe, test_fe, features, "PCA Train-Test", 
                                    figsize=(16, 16), max_dis=0.5, th=95)


# In[ ]:


hard_rows_list


# ### Insights 4
# 
# - There is no training data near the [233210, 241822, 246986, 501132, 508003] test rows.

# ## UMAP

# In[ ]:


def plot_umap(embedding, df, col, ax=None):
    colors = pd.factorize(df.loc[:, col])
    colors_dict = {}
    for index, label in enumerate(df[col].unique()):
        colors_dict[index] = label
    color_list = sns.color_palette(None, len(df[col].unique()))
    
    if ax == None:
        fig, ax = plt.subplots(figsize=(12,12))
        for color_key in colors_dict.keys():
            indexs = colors[0] == color_key
            temp_embedding = embedding[indexs, :]
            ax.scatter(temp_embedding[:, 0], temp_embedding[:, 1], 
                        c=color_list[color_key], 
                        edgecolor='none', 
                        alpha=0.80,
                        label=colors_dict[color_key],
                        s=10)
        ax.legend(bbox_to_anchor=(1, 1), fontsize="x-large", markerscale=2.)
        ax.set_title('UMAP - ' + col, fontsize=18);
    else:
        for color_key in colors_dict.keys():
            indexs = colors[0] == color_key
            temp_embedding = embedding[indexs, :]
            ax.scatter(temp_embedding[:, 0], temp_embedding[:, 1], 
                        c=color_list[color_key], 
                        edgecolor='none', 
                        alpha=0.80,
                        label=colors_dict[color_key],
                        s=10)
        ax.legend(bbox_to_anchor=(1, 1), fontsize="x-large", markerscale=2.)
        ax.set_title('UMAP - ' + col, fontsize=18);


# In[ ]:


import umap

#embedding = umap.UMAP(n_neighbors=10, min_dist=0.3, metric='correlation').fit_transform(train_fe[features])


# In[ ]:


#fig, ax = plt.subplots(figsize=(16, 16))
#plot_umap(embedding, train_fe[features], "target", ax=ax)


# # Baseline

# In[ ]:


from catboost import CatBoostClassifier

X_train = train_fe.drop(['id', 'target', 'f_27'], 1)
y_train = train_fe[['target']]
model = CatBoostClassifier(cat_features=fe_object_features)
model.fit(X_train, y_train, verbose=0)


# In[ ]:


X_test = test_fe.drop(['id', 'f_27'], 1)
sub['target'] = model.predict_proba(X_test)[:, 1]
sub.to_csv('submission.csv', index=False)


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
# 
# ### Insights 4
# 
# - There is no training data near the [233210, 241822, 246986, 501132, 508003] test rows.
