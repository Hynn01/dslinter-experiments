#!/usr/bin/env python
# coding: utf-8

# # <center> 📊Comprehensive EDA 📈 </center>
# 
#   ![](http://www.brazilfooty.com/wp-content/uploads/2018/12/201812-numbers.jpg)

# # 🎬 Introduction

# ## <span style="color:#0096FF;"> Tabular Playground Series - May 2022. </span>
#  The May edition of the 2022 Tabular Playground series binary classification problem that includes a number of different feature interactions. This competition is an opportunity to explore various methods for identifying and exploiting these feature interactions.

# <span style="color:#0096FF;"> The goal of this notebook is to explore the data visually and extract meaningful insights from it. </span>

# # ⚖ Evaluation

# *  Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target. 
# * ROC curve, is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
# * [Area under the curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)
# 

# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/ROC_curves.svg/1024px-ROC_curves.svg.png)

# ![](http://arogozhnikov.github.io/images/roc_curve.gif)

# The graphical way to compare output of two classifiers is ROC curve, which is built by checking all possible thresholds. For each threshold tpr and fpr are computed.
# 
# After checking all possible thresholds, we get the ROC curve. When ROC curve coincides with diagonal — this is the worst situation, because two distributions coincide. The higher ROC curve — the better discrimination between signal and background.
# 
# If at every point ROC curve of classifier A is higher than curve of classifier B, we are sure to say that in any application classifier A is better.
# 
# 

# # 📚 Libraries

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


class clr:
    S = '\033[1m' + '\033[96m'
    E = '\033[0m'


# In[ ]:


test_filepath = "../input/tabular-playground-series-may-2022/test.csv"
train_filepath = "../input/tabular-playground-series-may-2022/train.csv"
sample_filepath = "../input/tabular-playground-series-may-2022/sample_submission.csv"

test_data = pd.read_csv(test_filepath, index_col=0)
train_data = pd.read_csv(train_filepath, index_col=0)
sample_data = pd.read_csv(sample_filepath, index_col=0)


# # 🔢 Data Exploration

# In[ ]:


# data.shape
print(clr.S+"Test data shape.")
print(clr.S+"Number of Rows: "+clr.E, test_data.shape[0])
print(clr.S+"Number of Columns: "+clr.E, test_data.shape[1])


# In[ ]:


print(clr.S+"Number of missing data:",sum(test_data.isna().sum()))


# In[ ]:


# data.shape
print(clr.S+"Training data shape.")
print(clr.S+"Number of Rows: "+clr.E, train_data.shape[0])
print(clr.S+"Number of Columns: "+clr.E, train_data.shape[1])


# In[ ]:


print(clr.S+"Number of missing data:",sum(train_data.isna().sum()))


# In[ ]:


print(clr.S+"Training data colummn names: "+clr.E, train_data.columns.tolist())
print('\n')
print(clr.S+"Testing data colummn names: "+clr.E, test_data.columns.tolist())


# In[ ]:


test_data.head()


# In[ ]:


train_data.head()


# In[ ]:


print(clr.S+"")
train_data.info()


# <span style="color:#0096FF;">We can see that **f_27** is object type data.</span>

# In[ ]:


display(train_data[['f_27']].value_counts())


# In[ ]:


train_data.iloc[:, :-1].describe().T.sort_values(by='std' , ascending = False)                     .style.background_gradient(cmap='GnBu')                     .bar(subset=["max"], color='#BB0000')                     .bar(subset=["mean",], color='green')


# In[ ]:


print(clr.E+"Null Values:"+clr.S)
print(train_data.isna().sum().sort_values(ascending = False))


# ### <span style="color:#0096FF;">Target:</span>

# In[ ]:


target_df = pd.DataFrame(train_data['target'].value_counts()).reset_index()
target_df.columns = ['target', 'count']
fig = px.bar(data_frame =target_df, 
             x = 'target',
             y = 'count'
            ) ;
fig.update_traces(marker_color =['#58D68D','#DE3163'], 
                  marker_line_color='rgb(0,0,0)',
                  marker_line_width=2,)
fig.update_layout(title = "Target Distribution",
                  template = "plotly_white",
                  title_x = 0.5)
print(clr.S+"Percentage of Target = 0: {:.2f} %".format(target_df["count"][0] *100 / train_data.shape[0]))
print(clr.S+"Percentage of Target = 1: {:.2f} %".format(target_df["count"][1]* 100 / train_data.shape[0]))
fig.show();


# ### <span style="color:#0096FF;">Sample:</span>

# In[ ]:


sample_data.head()


# In[ ]:


sample_data.describe()


# # 🥵Heatmap

# In[ ]:


fig, ax =  plt.subplots(figsize=(30, 13))

colormap = plt.cm.YlGnBu
sns.heatmap(train_data.corr(),annot=True, fmt=".2f", cmap=colormap, annot_kws={"size": 12}, cbar_kws={"shrink": .2},vmin=-0.2 ,vmax=1)
plt.show();


# ### <span style="color:#0096FF;">Now lets look at correlations of features with target up close .</span>

# In[ ]:


fig, ax =  plt.subplots(figsize=(20, 3))

colormap = plt.cm.Set1
res = sns.heatmap(train_data.corr()[-1:],annot=True, fmt=".2f", cmap=colormap, annot_kws={"size": 18,'rotation': 90},vmin=-0.2 ,vmax=1)
res.set_yticklabels(res.get_ymajorticklabels(), weight="bold")
res.set_xticklabels(res.get_xmajorticklabels(), weight="bold")
plt.show();


# ### 💡Insights:
# <span style="color:#0096FF;">From the heatmap, we can clearly see which columns are correlated to target.</span>
# * Column "f_21" is most correlated to target.
# * Column "f_19" is most negatively correlated to target. It is negatively correlated, meaning with value increase of "f_19", target value decreases.
# 

# ### <span style="color:#0096FF;">Lets look at relation between "f_28" and "f_03" as an example.</span>

# In[ ]:


fig, ax =  plt.subplots(figsize=(30, 13))
sns.scatterplot(x="f_03", y="f_28", data=train_data, hue ="target")
plt.show();


# <span style="color:#0096FF;">Looks like cosmic background radiation, No?.</span>

# # 📡Distribution

# <span style="color:#0096FF;">Showing distribution on each feature that are available in train and test dataset. As there are 31 features, it will be broken down into 16 features for each sections. Yellow represents train dataset while Green represent test dataset.</span>

# ### <span style="color:#0096FF;">Feature: f_00 to f_15.</span>

# In[ ]:


get_ipython().run_cell_magic('time', '', 'background_color = "#f6f5f5"\n\nplt.rcParams[\'figure.dpi\'] = 600\nfig = plt.figure(figsize=(10, 10), facecolor=\'#f6f5f5\')\ngs = fig.add_gridspec(5, 5)\ngs.update(wspace=0.3, hspace=0.3)\n\nrun_no = 0\nfor row in range(0, 4):\n    for col in range(0, 4):\n        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])\n        locals()["ax"+str(run_no)].set_facecolor(background_color)\n        for s in ["top","right"]:\n            locals()["ax"+str(run_no)].spines[s].set_visible(False)\n        run_no += 1  \n\nfeatures = list(train_data.columns[1:17])\n\nbackground_color = "#f6f5f5"\n\nrun_no = 0\nfor col in features:\n    sns.kdeplot(ax=locals()["ax"+str(run_no)], x=train_data[col],shade=True, zorder=2, alpha=1, linewidth=1, color=\'#ffd514\')\n    locals()["ax"+str(run_no)].grid(which=\'major\', axis=\'x\', zorder=0, color=\'#EEEEEE\', linewidth=0.4)\n    locals()["ax"+str(run_no)].grid(which=\'major\', axis=\'y\', zorder=0, color=\'#EEEEEE\', linewidth=0.4)\n    locals()["ax"+str(run_no)].set_ylabel(\'\')\n    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=4, fontweight=\'bold\')\n    locals()["ax"+str(run_no)].tick_params(labelsize=4, width=0.5)\n    locals()["ax"+str(run_no)].xaxis.offsetText.set_fontsize(4)\n    locals()["ax"+str(run_no)].yaxis.offsetText.set_fontsize(4)\n    run_no += 1\n\nrun_no = 0\nfor col in features:\n    sns.kdeplot(ax=locals()["ax"+str(run_no)], x=test_data[col],shade=True, zorder=2, alpha=1, linewidth=1, color=\'#76C4AE\')\n    locals()["ax"+str(run_no)].grid(which=\'major\', axis=\'x\', zorder=0, color=\'#EEEEEE\', linewidth=0.4)\n    locals()["ax"+str(run_no)].grid(which=\'major\', axis=\'y\', zorder=0, color=\'#EEEEEE\', linewidth=0.4)\n    locals()["ax"+str(run_no)].set_ylabel(\'\')\n    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=4, fontweight=\'bold\')\n    locals()["ax"+str(run_no)].tick_params(labelsize=4, width=0.5)\n    locals()["ax"+str(run_no)].xaxis.offsetText.set_fontsize(4)\n    locals()["ax"+str(run_no)].yaxis.offsetText.set_fontsize(4)\n    run_no += 1\n\nplt.show();')


# ### <span style="color:#0096FF;">Feature: f_16 to f_30, except f_27 (Object type)</span>

# In[ ]:


get_ipython().run_cell_magic('time', '', 'background_color = "#f6f5f5"\n\nplt.rcParams[\'figure.dpi\'] = 600\nfig = plt.figure(figsize=(10, 10), facecolor=\'#f6f5f5\')\ngs = fig.add_gridspec(5, 5)\ngs.update(wspace=0.3, hspace=0.3)\n\nrun_no = 0\nfor row in range(0, 4):\n    for col in range(0, 4):\n        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])\n        locals()["ax"+str(run_no)].set_facecolor(background_color)\n        for s in ["top","right"]:\n            locals()["ax"+str(run_no)].spines[s].set_visible(False)\n        run_no += 1  \n\nfeatures = [\'f_16\', \'f_17\', \'f_18\', \'f_19\', \'f_20\', \'f_21\', \'f_22\', \'f_23\', \'f_24\', \'f_25\', \'f_26\', \'f_26\',\'f_28\', \'f_29\', \'f_30\', \'f_30\']\n\nbackground_color = "#f6f5f5"\n\nrun_no = 0\nfor col in features:\n    sns.kdeplot(ax=locals()["ax"+str(run_no)], x=train_data[col],shade=True, zorder=2, alpha=1, linewidth=1, color=\'#ffd514\')\n    locals()["ax"+str(run_no)].grid(which=\'major\', axis=\'x\', zorder=0, color=\'#EEEEEE\', linewidth=0.4)\n    locals()["ax"+str(run_no)].grid(which=\'major\', axis=\'y\', zorder=0, color=\'#EEEEEE\', linewidth=0.4)\n    locals()["ax"+str(run_no)].set_ylabel(\'\')\n    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=4, fontweight=\'bold\')\n    locals()["ax"+str(run_no)].tick_params(labelsize=4, width=0.5)\n    locals()["ax"+str(run_no)].xaxis.offsetText.set_fontsize(4)\n    locals()["ax"+str(run_no)].yaxis.offsetText.set_fontsize(4)\n    run_no += 1\n\nrun_no = 0\nfor col in features:\n    sns.kdeplot(ax=locals()["ax"+str(run_no)], x=test_data[col],shade=True, zorder=2, alpha=1, linewidth=1, color=\'#76C4AE\')\n    locals()["ax"+str(run_no)].grid(which=\'major\', axis=\'x\', zorder=0, color=\'#EEEEEE\', linewidth=0.4)\n    locals()["ax"+str(run_no)].grid(which=\'major\', axis=\'y\', zorder=0, color=\'#EEEEEE\', linewidth=0.4)\n    locals()["ax"+str(run_no)].set_ylabel(\'\')\n    locals()["ax"+str(run_no)].set_xlabel(col, fontsize=4, fontweight=\'bold\')\n    locals()["ax"+str(run_no)].tick_params(labelsize=4, width=0.5)\n    locals()["ax"+str(run_no)].xaxis.offsetText.set_fontsize(4)\n    locals()["ax"+str(run_no)].yaxis.offsetText.set_fontsize(4)\n    run_no += 1\n\nplt.show();')


# ### 💡Insights:
# * <span style="color:#0096FF;">All features distribution on train and test dataset are almost similar .</span>

# ### <span style="color:#0096FF;">DistPlot: Train features w.r.t Target.</span>

# In[ ]:


columns = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07',
       'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16',
       'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25',
       'f_26','f_28', 'f_29', 'f_30']


# In[ ]:


get_ipython().run_cell_magic('time', '', 'L = 30\nnrow= int(np.ceil(L/6))\nncol= 6\n\nremove_last= (nrow * ncol) - L\n\nfig, ax = plt.subplots(nrow, ncol,figsize=(24, 30))\n#ax.flat[-remove_last].set_visible(False)\nfig.subplots_adjust(top=0.95)\ni = 1\nfor feature in columns:\n    plt.subplot(nrow, ncol, i)\n    ax = sns.kdeplot(train_data[feature], shade=True, palette=\'RdBu\',  alpha=0.5, hue= train_data[\'target\'], multiple="stack")\n    plt.xlabel(feature, fontsize=9)\n    i += 1\nplt.suptitle(\'DistPlot: Train features w.r.t Target\', fontsize=20, weight ="bold")\nplt.show();')


# ### 👀 Observation:
# * From the above figure we see how target value is distributed for each feature.

#  **<span style="color:#0096FF;">If this notebook was helpful to you, do upvote it.</span>**

# ![](https://i.pinimg.com/originals/31/53/2d/31532d7d378053de3b8bf23c6e7bfae3.gif)

# ### Some useful notebooks:
# * [How to Work w. Million-row Datasets Like a Pro](https://www.kaggle.com/code/bextuychiev/how-to-work-w-million-row-datasets-like-a-pro)
# * [How to beat the heck out of XGBoost with LightGBM](https://www.kaggle.com/code/bextuychiev/how-to-beat-the-heck-out-of-xgboost-with-lightgbm)
