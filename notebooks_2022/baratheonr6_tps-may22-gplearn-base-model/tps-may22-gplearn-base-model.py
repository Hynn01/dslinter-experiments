#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().system('pip install statannotations')
get_ipython().system('pip install matplotlib-venn')
import matplotlib.pyplot as plt
import matplotlib.style as style
from statannotations.Annotator import Annotator
import seaborn as sns
from matplotlib_venn import venn2, venn3_circles
style.use('fivethirtyeight')


# <div style="background:#2b6684   ;font-family:'Times';font-size:35px;color:  #F0CB8E" >&ensp;TPS - MAY2022</div>

# <div style="background:#2b6684   ;font-family:'Times';font-size:35px;color:  #F0CB8E" >&ensp;Table of Contents</div>
# <div class="alert alert-warning" role="alert">
# <ul style="font-family:cursive;font-size:18px" >
#     
# <li><a href = "#1"> 1. Loading Data 📲 </a></li>     
# <li ><a href = "#2"> 2. features interaction with target 📈 </a></li> 
# <li><a href = "#3"> 3. Analysis of f_27 </a></li>
# <ul>
# <li ><a  href = "#300"> 3.1 1st and 4th character of test set's f_27 feature has more characters than train's f_27 </a></li>
# <li ><a  href = "#301"> 3.2 Intersection among characters using venn diagrams </a></li>
# </ul> 
# <li><a href = "#4"> 4. Feature Engineering Using gplearn  🔧</a></li>
# <li ><a  href = "#5"> 5. Model Fitting  🔧 </a></li>
# </ul>
# </div>

# <a id="1"></a>
# <div class="alert alert-warning" role="alert">
# <ul style="font-family:cursive;font-size:25px; color:#0E198A">Loading Data 📲 
# </ul>
#     

# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')

print('Train')
display(train.head())

print('Test')
display(test.head())

print('Dataframe shapes:', train.shape, test.shape)


# <div class="alert alert-warning" role="alert">
# <ul style="font-family:cursive;font-size:18px; color:#A20404">Observations: 
# <li>Info shows us that we have 1 object column f_27.</li>
# </ul>
# </div>

# In[ ]:


train.info()


# <div class="alert alert-warning" role="alert">
# <ul style="font-family:cursive;font-size:18px; color:#A20404">Observations: 
# <li>Distribution of our target variable suggests that the target is binary and almost balanced.</li>
# </ul>
# </div>

# In[ ]:


train.target.value_counts(normalize = True)


# <a id="2"></a>
# <div class="alert alert-warning" role="alert">
# <ul style="font-family:cursive;font-size:25px; color:#0E198A">features interaction with target 📈
# </ul>

# <div class="alert alert-warning" role="alert">
# <ul style="font-family:cursive;font-size:18px; color:#A20404"><b style="font-size:25px"></b>    
# <li>There is no clear distinction of medians for all the features as analyzed with boxplots. As suggested in the competition description it would be useful to look into feature interaction instead </li>
# </ul>
# </div>

# In[ ]:


train['target_str'] = train['target'].astype(int).astype(str)
x = "target_str"
y = "f_28"
order = ['0','1']
ax = sns.boxplot(data=train, x=x, y=y, order=order)
annot = Annotator(ax, [('0','1')], data=train, x=x, y=y, order=order)
annot.configure(test='Mann-Whitney', text_format='star', loc='outside', verbose=2, comparisons_correction=None, line_height=0.05, text_offset=15)
annot.apply_test()
ax, test_results = annot.annotate()


# <div class="alert alert-warning" role="alert">
# <ul style="font-family:cursive;font-size:18px; color:#A20404">
# <a href="https://www.kaggle.com/code/ambrosm/tpsmay22-eda-which-makes-sense#The-16-float-features"> Scatterplot referenced from @AMBROSM  </a>
# </ul>
# </div>

# In[ ]:


float_features = [f for f in train.columns if train[f].dtype == 'float64']
fig, axs = plt.subplots(4, 4, figsize=(16, 16))
for f, ax in zip(float_features, axs.ravel()):
    temp = pd.DataFrame({f: train[f].values,
                         'state': train.target.values})
    temp = temp.sort_values(f)
    temp.reset_index(inplace=True)
    ax.scatter(temp[f], temp.state.rolling(25000, center=True).mean(), s=2)
    ax.set_xlabel(f'{f}')
plt.suptitle('Float Features Scatterplot')
plt.show()


# <a id="3"></a>
# <div class="alert alert-warning" role="alert">
# <ul style="font-family:cursive;font-size:25px; color:#0E198A">Analysis of f_27
# </ul>

# In[ ]:


cols = ['0_27','1_27','2_27','3_27','4_27','5_27','6_27','7_27','8_27','9_27']
num_cols = [x+'_num' for x in cols]


# In[ ]:


import string
mapper = dict(zip(string.ascii_uppercase,np.arange(0,len(string.ascii_lowercase))))
for i in range(10):
    train[str(i) +'_27']=train.f_27.str[i]
    train[str(i) +'_27_num']=train[str(i) +'_27'].map(mapper)
    test[str(i) +'_27']=test.f_27.str[i]
    test[str(i) +'_27_num']=test[str(i) +'_27'].map(mapper)

print(train[cols].head())


# <a id="300"></a>
# <div class="alert alert-warning" role="alert">
# <ul style="font-family:cursive;font-size:18px; color:#A20404"><b style="font-size:25px"></b>    
# 1st and 4th character of test set's f_27 feature has more characters than train's f_27 
# </ul>
# </div>

# In[ ]:


for f in cols:
    print( f,'   ',set(test[f].unique()).difference(set(train[f].unique())) )


# <a id="301"></a>
# <div class="alert alert-warning" role="alert">
# <ul style="font-family:cursive;font-size:18px; color:#A20404"><b style="font-size:25px"></b>    
# Intersection among characters using venn diagrams
# </ul>
# </div>

# In[ ]:


from itertools import combinations
comb = list(combinations(range(0,10), 2))
comb = [x for x in comb if x[0]!=x[1]]


# In[ ]:


fig, axs = plt.subplots(9,5,figsize=(60, 60))
for co, ax in zip(comb, axs.ravel()):  
    A = set(train.f_27.str[co[0]].unique())
    B = set(train.f_27.str[co[1]].unique())
 
    diagram = venn2([A, B], ("Set "+str(co[0]), "Set "+str(co[1])), ax= ax)

    diagram.get_label_by_id("10").set_text("\n".join(A - B))
    diagram.get_label_by_id("11").set_text("\n".join(A & B))
    diagram.get_label_by_id("01").set_text("\n".join(B - A))


# <a id="4"></a>
# <div class="alert alert-warning" role="alert">
# <ul style="font-family:cursive;font-size:25px; color:#0E198A">Feature Engineering Using gplearn  🔧
# <ul style="font-family:cursive;font-size:18px; color:#A20404"><b style="font-size:25px"></b>    
# <li>gplearn implements Genetic Programming in Python, with a scikit-learn inspired and compatible API. </li>
# <li><a href="https://gplearn.readthedocs.io/en/stable/"> Documentation of gplearn</a></li>
# </ul>  
# </ul>

# In[ ]:


from sklearn.model_selection import train_test_split
feature_list = [f for f in train.columns if train[f].dtype in ['float64','int64'] and f not in ['target','id']]
X_train, X_test, y_train, y_test = train_test_split( train[feature_list], train['target'], test_size=0.2, random_state=42, stratify=train['target'])


# In[ ]:


from gplearn.genetic import SymbolicTransformer

symbolic_transformer = SymbolicTransformer(
    population_size= 4000,
    init_depth=(1,5),
    feature_names=feature_list,
    verbose=1,
    random_state=12).fit(X_train, y_train)


# In[ ]:


df = pd.DataFrame(columns=["name", "description", "fitness"])
for i, fx in enumerate(symbolic_transformer):
    if str(fx) not in feature_list:  # Drop unchanged features
        df.loc[i] = ["", str(fx), fx.fitness_]
df = df.drop_duplicates()
df = df.nlargest(5, columns="fitness")
df


# In[ ]:


X_train[
    ["feature_" + str(int(x)) for x in range(len(feature_list), len(feature_list) + len(df))]
] = symbolic_transformer.transform(X_train[feature_list])[:, df.index]
X_test[
    ["feature_" + str(int(x)) for x in range(len(feature_list), len(feature_list) + len(df))]
] = symbolic_transformer.transform(X_test[feature_list])[:, df.index]
test[
    ["feature_" + str(int(x)) for x in range(len(feature_list), len(feature_list) + len(df))]
] = symbolic_transformer.transform(test[feature_list])[:, df.index]


# <a id="5"></a>
# <div class="alert alert-warning" role="alert">
# <ul style="font-family:cursive;font-size:25px; color:#0E198A">Model Fitting  🔧
# <ul style="font-family:cursive;font-size:18px; color:#A20404"><b style="font-size:25px"></b>    
# </ul>  
# </ul>

# In[ ]:


feature_list = [f for f in X_train.columns if X_train[f].dtype in ['float64','int64'] and f not in ['target','id']]
import lightgbm as lgbm
params = {'n_estimators': 10000,
          'lambda_l2': 0.0015, 
          'alpha': 9.82, 
          'learning_rate': 0.02, 
          'max_depth': 11, 
          'min_child_weight': 135}
model = lgbm.LGBMClassifier(**params)
model.fit(X_train[feature_list], y_train)
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test[feature_list])))
print(classification_report(y_train, model.predict(X_train[feature_list])))


# In[ ]:


def ret(x):
    if x=='f_01':
        return 1
    else:
        return 0

model = lgbm.LGBMClassifier(**params,monotone_constraints=[ret(index) for index in feature_list] )
model.fit(X_train[feature_list], y_train)
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test[feature_list])))
print(classification_report(y_train, model.predict(X_train[feature_list])))


# In[ ]:


f, ax = plt.subplots(figsize=(10, 10))
lgbm.plot_importance(model,ignore_zero=False,ax=ax)


# In[ ]:


sub = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv')
sub['target'] = model.predict_proba(test[feature_list])[:,1]
sub.to_csv('submission.csv', index=False)
sub


# <div style="background:#2b6684   ;font-family:'Times';font-size:35px;color:  #F0CB8E" >&ensp;Thanks for visiting my notebook ! if you have a few minutes to spare do visit my open souce work</div>
# <a id="5"></a>
# <div class="alert alert-warning" role="alert"> 
# <ul style="font-family:cursive;font-size:18px; color:#A20404"><b style="font-size:25px"></b>
# <a href="https://github.com/jaswinder9051998/zoofs">  Zoofs ( Zoo Feature Selection )  </a>
#     
# <a href="https://github.com/NITRO-AI/NitroFE">  NitroFE ( Nitro Feature Engineering )  </a>
