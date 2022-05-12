#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\n     \ndiv.h2 { background-color: #159957;\n         background-image: linear-gradient(120deg, #155799, #159957);\n         text-align: left;\n         color: white;              \n         padding:9px;\n         padding-right: 100px; \n         font-size: 20px; \n         max-width: 1500px; \n         margin: auto; \n         margin-top: 40px; }\n                                                                  \nbody {font-size: 12px;}    \n     \n                                                 \ndiv.h3 {color: #159957; \n        font-size: 18px; \n        margin-top: 20px; \n        margin-bottom:4px;}\n   \n                                      \ndiv.h4 {color: #159957;\n        font-size: 15px; \n        margin-top: 20px; \n        margin-bottom: 8px;}\n                                           \n                                      \nspan.note {font-size: 5;\n           color: gray; \n           font-style: italic;}\n  \n                                      \nhr {\n    display: block; \n    color: gray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}\n  \n                                      \nhr.light {\n    display: block; \n    color: lightgray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}   \n    \n                                      \ntable.dataframe th \n{\n    border: 1px darkgray solid;\n    color: black;\n      <table align="left">\n    ...\n  </table>\n    background-color: white;\n}\n    \n                                      \ntable.dataframe td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 11px;\n    text-align: center;\n} \n          \n                                      \ntable.rules th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 11px;\n    align: left;\n}\n       \n                                      \ntable.rules td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 13px;\n    text-align: center;\n} \n   \n                                      \n                                      \ntable.rules tr.best\n{\n    color: green;\n}    \n    \n                                      \n.output { \n    align-items: left; \n}\n        \n                                      \n.output_png {\n    display: table-cell;\n    text-align: left;\n    margin:auto;\n}                                          \n                                                                                                                                       \n</style>  ')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all 
# files under the input directory
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output 
# when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the
# current session
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_context('notebook')
from cycler import cycler
from IPython.display import display
import datetime
from io import StringIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# sns.set_style("whitegrid")
import matplotlib_inline.backend_inline
from IPython.display import set_matplotlib_formats
import matplotlib
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
# set_matplotlib_formats('retina')
plt.rcParams['savefig.facecolor']='white'
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
# from IPython.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))
# plt.rcParams['figure.dpi'] = 150
from sklearn.metrics import roc_auc_score
from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import gridplot  #firstly import gridplot
from bokeh.io import output_file, show
from bokeh.plotting import figure
from jinja2.utils import markupsafe
from markupsafe import Markup
# toolbar_location=None,  tools="")
output_notebook(hide_banner=True)
# import holoviews as hv
# from holoviews import opts
# hv.extension('bokeh')
import warnings
warnings.filterwarnings('ignore')

from bokeh.layouts import gridplot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from xgboost  import XGBClassifier
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
path = "../input/tabular-playground-series-may-2022/"
# if os.path.isfile(path):
train_df = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test_df = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
sub_df = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
# else:
#     train_df = pd.read_csv("train.csv")
#     test_df = pd.read_csv("test.csv")
# sub_df = pd.read_csv("sample_submission.csv")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int_features   = list(train_df.select_dtypes(include=['int']).columns)
int_features.remove('id')
int_features.remove('target')
string_feature = ['f_27']
float_features = list(train_df.select_dtypes(include=['float']).columns)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# <div class="h2"><i><center>Introduction</center></i></div>

# <div class="h3"><i>1. &ensp; Background</i></div>

# In[ ]:


data = ("""ID,Parameter,Value,Description
1, - Total Observations - ,"1,600,000",train and test sets combined
2,Observations (train),"900,000", 56.25% of total observations
3,Observations (test),"700,000",43.75% of total observations
4, - Total Features -,32,(does not include the target)
5,Features (Integers),15, Note: One of the integer features is 'id' - a tagged unique value for every observations (helpful)
6,Features (Float), 16,ex
7, Features (Object/String), 1, ex: ACACCADCEB (always 10 characters)
8, Missing Values, 0, (doesn't appear to be any missing values)
9,Submission Criteria, AUC, Submissions evaluated on area under the ROC curve (between the predicted probability and the observed target)
""")

mapper = pd.read_csv(StringIO(data))

d = dict(selector="th", props=[('text-align', 'center')])

mapper = mapper.set_index('ID')

mapper.style.set_properties(**{'width':'18em', 'text-align':'center'})        .set_table_styles([d])


# * 1.6M total observations, broken out to 900k training and 700k testing obs.  Total observations split to 56.25% training and 43.75% testing data ? (kinda odd breakout but we move on)

# *looking at four samples of the training data:*

# In[ ]:


display(train_df.head(4).T)


# * Somewhat unusual dataset.  Looks like we have a combination of integer features, floating features, and one object/string feature.  

# <div class="h3"><i>2. &ensp; EDA</i></div>

# **Integer Features**

# *breaking out the integer features to their value counts:*

# In[ ]:



# number of unique values in the integer columns
counts = train_df[int_features].nunique()
sorted_int_features = sorted(int_features, key=lambda x: counts[int_features.index(x)])
p = figure(x_range=sorted_int_features, 
           height=300, 
           width=500,
           title="Integer Features - Unique Value Counts per feature",
           toolbar_location=None, 
           tools="")
p.vbar(x=int_features, 
       top=counts, 
       width=0.6, 
       color = 'royalblue')
p.title.text_font_size = '9pt'
p.title.text_font_style = "normal"
p.xgrid.grid_line_color = None
p.y_range.start = 0
p.yaxis.major_tick_line_color = None 
p.yaxis.minor_tick_line_color = None  
p.xaxis.major_tick_line_color = None
p.yaxis.major_tick_line_color = None  
show(p)


# *creating tables for each discrete (integer) feature: (i.e. the feature, the list of unique values seen within that feature, and its percentage breakout seen)*

# In[ ]:



# print ibm["Close"].to_string(header=False)
print('\nPercentage Breakout - Value Counts:\n')
for col in int_features:
    print('--- feature:', col, '---')
    display( round(train_df[col].value_counts(normalize=True), 3).to_frame())
    print("")
    


# * `f_29` appears to be a binary, while `f_30` is ternary
# * what i think is somewhat unusual is that every feature otherwise has instances of value counts that are quite low, i.e. they might have four values that are barely used...

# **Continuous (float) Features**

# In[ ]:


# output_notebook()
# from random import seed
# from random import randint
# seed(1)
# x_value=[]
# for i in range(20):
#   x_value.append(i) #fill x with random values.
# y_one = [x**0.5 for x in x_value] #sqrt(x)
# y_two =  [x**2 for x in x_value] #x^2

# # gridplot to show graphs of x^2 and sqrt(x)

# # paramters of figure
# # plot_width - The width of the solution space for plotting.
# # plot_height - The height of the solution space for plotting.
# # title - This refers to the main heading of our graph.
# # x_axis_label - This shows what does x-axis represent.
# # y_axis_label - This shows what does y-axis represent.

# p1 = figure(title="Bokeh Grid plot Example", x_axis_label='x_value', y_axis_label='y_value',plot_width=500, plot_height=500)
# p1.circle(x_value,y_one,size=14,color='red')

# p2 = figure(title="Bokeh Grid plot Example", x_axis_label='x_value', y_axis_label='y_value',plot_width=500, plot_height=500)
# p2.circle(x_value,y_two,size=14, color='blue')

# p3 = gridplot([[p1,p2]], toolbar_location=None)
# show(p3)

# #Comparison of two plots shown as a grid plot.

sns.set_style("white")
fig, axs = plt.subplots(4,4, figsize=(6, 6), dpi=100)
# fig, axs = plt.subplots(4, 4, figsize=(6,8))
sns.histplot(data=train_df, x="f_00", kde=True, color ='xkcd:lightish blue', ax=axs[0, 0])
sns.histplot(data=train_df, x="f_01", kde=True, color ='xkcd:lightish blue', ax=axs[0, 1])
sns.histplot(data=train_df, x="f_02", kde=True, color ='xkcd:lightish blue',  ax=axs[0, 2])
sns.histplot(data=train_df, x="f_03", kde=True, color ='xkcd:lightish blue',  ax=axs[0, 3])
sns.histplot(data=train_df, x="f_04", kde=True, color ='xkcd:lightish blue', ax=axs[1, 0])
sns.histplot(data=train_df, x="f_05", kde=True, color ='xkcd:lightish blue', ax=axs[1, 1])
sns.histplot(data=train_df, x="f_06", kde=True, color ='xkcd:lightish blue', ax=axs[1, 2])
sns.histplot(data=train_df, x="f_19", kde=True, color ='xkcd:lightish blue', ax=axs[1, 3])
sns.histplot(data=train_df, x="f_20", kde=True, color ='xkcd:lightish blue', ax=axs[2, 0])
sns.histplot(data=train_df, x="f_21", kde=True, color ='xkcd:lightish blue', ax=axs[2, 1])
sns.histplot(data=train_df, x="f_22", kde=True, color ='xkcd:lightish blue', ax=axs[2, 2])
sns.histplot(data=train_df, x="f_23", kde=True, color ='xkcd:lightish blue', ax=axs[2, 3])
sns.histplot(data=train_df, x="f_24", kde=True, color ='xkcd:lightish blue', ax=axs[3, 0])
sns.histplot(data=train_df, x="f_25", kde=True, color ='xkcd:lightish blue', ax=axs[3, 1])
sns.histplot(data=train_df, x="f_26", kde=True, color ='xkcd:lightish blue', ax=axs[3, 2])
sns.histplot(data=train_df, x="f_28", kde=True, color ='xkcd:lightish blue', ax=axs[3, 3])
sns.despine(top=True, right=True, left=True, bottom=True)
# plt.grid(linestyle='--', alpha=0.03)
for ax in axs.flat:
    ax.set(ylabel='')
    ax.set_yticks([])
plt.tight_layout()
plt.show();


# *interactivity between features (`f_00` and `f_01`):*

# In[ ]:


# plt.rcParams['savefig.facecolor']='white'
sns.jointplot(data=train_df,
             x="f_00",
             y='f_01',
             height=6,
             kind='hex',
             color="#4CB391")
plt.ylim(-3,3)
plt.xlim(-3,3)
# plt.title('\nResale Price vs Floor Area (Interaction)',fontsize=12, loc="center")
# plt.xlabel('Floor Area in sq_m', fontsize=11)
# plt.ylabel('Resale Price in S$)', fontsize=11)
# plt.savefig('hex_price_vs_floor_area_updated2.png', dpi=400, transparent=False);
# plt.tight_layout()
plt.show();


# *highlighting the first group of very similiar distributions:*

# In[ ]:



def highlight_greaterthan_neg_5(s):
    if s["min"] > -5:
        return ['background-color: lightgrey']*7
    else:
        return ['background-color: white']*7
    
def highlight_chunk2(s):
    if s["min"] > -12 and s["min"] < -14:
        return ['background-color: lightgrey']*7
    else:
        return ['background-color: white']*7
    
float_features_data = train_df[float_features].describe().T
float_features_data = float_features_data.drop(labels = 'count', axis=1)
float_features_data = float_features_data[['mean', 'min','max','std','25%','50%','75%']]
float_features_data = float_features_data.round(4)
float_features_data.style.set_caption('Continue Features - Key Values')
float_features_data.style.apply(highlight_greaterthan_neg_5, axis=1)


# In[ ]:



sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(5,3))
sns.distplot(train_df.f_01, 
             kde=False, 
             color = 'royalblue',
             # color = "#d7191c",
             kde_kws={'bw':1, "linewidth":.5, 'color':"red"},  #  #ffffbf"},
             bins=60,
             hist_kws={"linewidth": .5, 'edgecolor':'black', 'alpha':.75, "rwidth":0.75})
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_yticklabels([])
plt.title('      Feature 00 Distribution', fontsize=11, loc="left")
ax.set(xlabel="")
plt.axvline(x=0, color='black', linestyle="-", linewidth=.8)
plt.xlim([-3.5, 3.5])
# plt.text(71.2, .15, r'6-foot mark', {'color': 'black', 'fontsize': 10})
plt.grid(linestyle='--', alpha=0.03)
plt.tight_layout()
plt.show();

fig, ax = plt.subplots(figsize=(5,3))
sns.distplot(train_df.f_01, 
             kde=False, 
             color = 'royalblue',
             # color = "#d7191c",
             kde_kws={'bw':1, "linewidth":.5, 'color':"red"},  #  #ffffbf"},
             bins=60,
             hist_kws={"linewidth": .5, 'edgecolor':'black', 'alpha':.75, "rwidth":0.75})
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_yticklabels([])
plt.title('      Feature 01 Distribution', fontsize=11, loc="left")
# plt.title('\nPlayer Height Distribution\n', fontsize=12, loc="left")
ax.set(xlabel="")
plt.axvline(x=0, color='black', linestyle="-", linewidth=.8)
plt.xlim([-3.5, 3.5])
# plt.text(71.2, .15, r'6-foot mark', {'color': 'black', 'fontsize': 10})
plt.grid(linestyle='--', alpha=0.03)
plt.tight_layout()
plt.show();

fig, ax = plt.subplots(figsize=(5,3))
sns.distplot(train_df.f_02, 
             kde=False, 
             color = 'royalblue',
             # color = "#d7191c",
             kde_kws={'bw':1, "linewidth":.5, 'color':"red"},  #  #ffffbf"},
             bins=60,
             hist_kws={"linewidth": .5, 'edgecolor':'black', 'alpha':.75, "rwidth":0.75})
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_yticklabels([])
plt.title('      Feature 02 Distribution', fontsize=11, loc="left")
# plt.title('\nPlayer Height Distribution\n', fontsize=12, loc="left")
ax.set(xlabel="")
plt.axvline(x=0, color='black', linestyle="-", linewidth=.8)
plt.xlim([-3.5, 3.5])
# plt.text(71.2, .15, r'6-foot mark', {'color': 'black', 'fontsize': 10})
plt.grid(linestyle='--', alpha=0.03)
plt.tight_layout()
plt.show();

fig, ax = plt.subplots(figsize=(5,3))
sns.distplot(train_df.f_03, 
             kde=False, 
             color = 'royalblue',
             # color = "#d7191c",
             kde_kws={'bw':1, "linewidth":.5, 'color':"red"},  #  #ffffbf"},
             bins=60,
             hist_kws={"linewidth": .5, 'edgecolor':'black', 'alpha':.75, "rwidth":0.75})
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_yticklabels([])
plt.title('      Feature 03 Distribution', fontsize=11, loc="left")
# plt.title('\nPlayer Height Distribution\n', fontsize=12, loc="left")
ax.set(xlabel="")
plt.axvline(x=0, color='black', linestyle="-", linewidth=.8)
plt.xlim([-3.5, 3.5])
# plt.text(71.2, .15, r'6-foot mark', {'color': 'black', 'fontsize': 10})
plt.grid(linestyle='--', alpha=0.03)
plt.tight_layout()
plt.show();

fig, ax = plt.subplots(figsize=(5,3))
sns.distplot(train_df.f_04, 
             kde=False, 
             color = 'royalblue',
             # color = "#d7191c",
             kde_kws={'bw':1, "linewidth":.5, 'color':"red"},  #  #ffffbf"},
             bins=60,
             hist_kws={"linewidth": .5, 'edgecolor':'black', 'alpha':.75, "rwidth":0.75})
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_yticklabels([])
plt.title('      Feature 04 Distribution', fontsize=11, loc="left")
# plt.title('\nPlayer Height Distribution\n', fontsize=12, loc="left")
ax.set(xlabel="")
plt.axvline(x=0, color='black', linestyle="-", linewidth=.8)
plt.xlim([-3.5, 3.5])
# plt.text(71.2, .15, r'6-foot mark', {'color': 'black', 'fontsize': 10})
plt.grid(linestyle='--', alpha=0.03)
plt.tight_layout()
plt.show();

fig, ax = plt.subplots(figsize=(5,3))
sns.distplot(train_df.f_05, 
             kde=False, 
             color = 'royalblue',
             # color = "#d7191c",
             kde_kws={'bw':1, "linewidth":.5, 'color':"red"},  #  #ffffbf"},
             bins=60,
             hist_kws={"linewidth": .5, 'edgecolor':'black', 'alpha':.75, "rwidth":0.75})
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_yticklabels([])
plt.title('      Feature 05 Distribution', fontsize=11, loc="left")
# plt.title('\nPlayer Height Distribution\n', fontsize=12, loc="left")
ax.set(xlabel="")
plt.axvline(x=0, color='black', linestyle="-", linewidth=.8)
plt.xlim([-3.5, 3.5])
# plt.text(71.2, .15, r'6-foot mark', {'color': 'black', 'fontsize': 10})
plt.grid(linestyle='--', alpha=0.03)
plt.tight_layout()
plt.show();

fig, ax = plt.subplots(figsize=(5,3))
sns.distplot(train_df.f_06, 
             kde=False, 
             color = 'royalblue',
             # color = "#d7191c",
             kde_kws={'bw':1, "linewidth":.5, 'color':"red"},  #  #ffffbf"},
             bins=60,
             hist_kws={"linewidth": .5, 'edgecolor':'black', 'alpha':.75, "rwidth":0.75})
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_yticklabels([])
plt.title('      Feature 06 Distribution', fontsize=11, loc="left")
# plt.title('\nPlayer Height Distribution\n', fontsize=12, loc="left")
ax.set(xlabel="")
plt.axvline(x=0, color='black', linestyle="-", linewidth=.8)
plt.xlim([-3.5, 3.5])
# plt.text(71.2, .15, r'6-foot mark', {'color': 'black', 'fontsize': 10})
plt.grid(linestyle='--', alpha=0.03)
plt.tight_layout()
plt.show();


# * extremely similiar in appearance...
# * which means that they have no real interactiivty with one another...

# *Q-Q Plot for gaussian check:*

# In[ ]:


import statsmodels.api as sm
fig = sm.qqplot(train_df['f_00'], line='45', color='green')
plt.title('  Feature 00 Q-Q Plot', fontsize=11, loc="left")
plt.show(); 

import statsmodels.api as sm
fig = sm.qqplot(train_df['f_01'], line='45')
plt.title('  Feature 01 Q-Q Plot', fontsize=11, loc="left")
plt.show(); 

fig = sm.qqplot(train_df['f_02'], line='45')
plt.title('  Feature 02 Q-Q Plot', fontsize=11, loc="left")
plt.show(); 

fig = sm.qqplot(train_df['f_03'], line='45')
plt.title('  Feature 03 Q-Q Plot', fontsize=11, loc="left")
plt.show(); 

fig = sm.qqplot(train_df['f_04'], line='45')
plt.title('  Feature 04 Q-Q Plot', fontsize=11, loc="left")
plt.show(); 

fig = sm.qqplot(train_df['f_05'], line='45')
plt.title('  Feature 05 Q-Q Plot', fontsize=11, loc="left")
plt.show(); 

fig = sm.qqplot(train_df['f_06'], line='45')
plt.title('  Feature 06 Q-Q Plot', fontsize=11, loc="left")
plt.show(); 


# * In a Q-Q plot, the x-axis displays the **theoretical** quantiles (where it would be if it were in fact a perfect normal distribution).   The y-axis displays my actual data.  Translation:  If its very close to the red line, its very very likely to be a normal distribution. 
# * Thats about as normal of a distribution set as i've ever seen...

# <div class="h3"><i>3. &ensp; Correlation</i></div>

# In[ ]:


plt.figure(figsize=(11,11))
new_column_headers=["%02d" % x for x in range(27)]
new_column_headers = new_column_headers + ['28','29','30', 'tgt']
correlation_df = train_df.copy()
correlation_df = correlation_df.drop(labels = 'id', axis=1)
correlation_df = correlation_df.drop(labels = 'f_27', axis=1)
correlation_df.columns = new_column_headers
corr_results = correlation_df.corr()
sns.heatmap(corr_results,
                 fmt='.2f', 
                 annot = True, 
                 vmin=-1,
                 vmax=1, 
                 center= 0, 
                 cmap= 'seismic', 
                 linecolor='white', 
                 linewidth=.2, 
                 cbar = False,
                 annot_kws={"size": 8})
plt.xticks(rotation=0, ha='center')
plt.yticks(rotation=0, ha='center')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.title('\nCorrelation Matrix (features:  f_00 thru f_30)\n\nRed: positive correlation                Blue: negative correlation\n\n', fontsize=10)
# plt.savefig('correlation_matrix_baseline.png', 
#             bbox_inches='tight',
#             pad_inches=0.2, 
#             dpi=500)
plt.tight_layout()
plt.show();


# * So we appear to see really three main areas:
#   * Center square
#   * Bottom right square
#   * Bottom strip 

# *examining the center 'box':*

# In[ ]:


new_column_headers=["%02d" % x for x in range(27)]
new_column_headers = new_column_headers + ['28','29','30', 'tgt']
correlation_df = train_df.copy()
correlation_df = correlation_df.drop(labels = 'id', axis=1)
correlation_df = correlation_df.drop(labels = 'f_27', axis=1)
plt.figure(figsize=(6,6))
c2k = ['f_07','f_08','f_09','f_10','f_11','f_12',
       'f_13','f_14','f_15','f_16','f_17','f_18']
sns.heatmap(correlation_df[c2k].corr(), 
                 fmt='.3f', 
                 annot = True, 
                 vmin=-1,
                 vmax=1, 
                 center= 0, 
                 cmap= 'seismic', 
                 linecolor='white',
                 cbar=False,
                 linewidth=.6, 
                 annot_kws={"size": 7.5})
plt.yticks(rotation=0, ha='right')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('\nCorrelation Matrix (middle square)\n', fontsize=10)
plt.tight_layout()
plt.show();


# *examining the bottom right 'box':*

# In[ ]:


new_column_headers=["%02d" % x for x in range(27)]
new_column_headers = new_column_headers + ['28','29','30', 'tgt']
correlation_df = train_df.copy()
correlation_df = correlation_df.drop(labels = 'id', axis=1)
correlation_df = correlation_df.drop(labels = 'f_27', axis=1)
plt.figure(figsize=(6,6))
c2k = ['f_19', 'f_20','f_21','f_22','f_23','f_24','f_25','f_26']
sns.heatmap(correlation_df[c2k].corr(), 
                 fmt='.3f', 
                 annot = True, 
                 vmin=-1,
                 vmax=1, 
                 center= 0, 
                 cmap= 'seismic', 
                 linecolor='white',
                 cbar=False,
                 linewidth=.3, 
                 annot_kws={"size": 7.5})
plt.yticks(rotation=0, ha='right')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('\nCorrelation Matrix (bottom right square)\n', fontsize=10)
plt.tight_layout()
plt.show();


# <div class="h3"><i>4. &ensp; Model</i></div>

# In[ ]:



# proper way 
# train_df_no_target = train_df.copy()
# train_df_no_target = train_df_no_target.drop(labels = 'target', axis=1)
# train_df_no_target = train_df_no_target.drop(labels = 'id', axis=1)
# train_df_no_target = train_df_no_target.drop(labels = 'f_27', axis=1)
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(train_df_no_target, 
#                                                    train_df['target'], 
#                                                    test_size = 0.20,
#                                                    random_state = 2022)

# train_df_no_target = train_df.copy()
# train_df_no_target = train_df_no_target.drop(labels = 'target', axis=1)
# train_df_no_target = train_df_no_target.drop(labels = 'id', axis=1)
# train_df_no_target = train_df_no_target.drop(labels = 'f_27', axis=1)
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(train_df_no_target, 
#                                                    train_df['target'], 
#                                                    test_size = 0.20,
#                                                    random_state = 2022)


# In[ ]:


# params = {'n_estimators'    : 5001,
#           'max_depth'       : 5,
#           'learning_rate'   : 0.10,
#           'random_state'    : 2022,
#           'eval_metric'     : 'auc',
#           'objective'       : 'binary:logistic',
#           'tree_method'     : 'gpu_hist'}

# xgb = XGBClassifier(**params)
# xgb.fit(X_train, y_train, eval_set = [(X_val, y_val)], 
#        verbose = 1000)


# In[ ]:


# params = {'n_estimators'    : 5001,
#           'max_depth'       : 5,
#           'learning_rate'   : 0.10,
#           'random_state'    : 2022,
#           'eval_metric'     : 'auc',
#           'objective'       : 'binary:logistic',
#           'tree_method'     : 'gpu_hist'}

# xgb = XGBClassifier(**params)
# xgb.fit(X_train, y_train, eval_set = [(X_val, y_val)], 
#        verbose = 1000)

# [0]	validation_0-auc:0.63394
# [1000]	validation_0-auc:0.92842
# [2000]	validation_0-auc:0.93317
# [3000]	validation_0-auc:0.93425
# [4000]	validation_0-auc:0.93437
# [5000]	validation_0-auc:0.93438


# In[ ]:


# pred_y = xgb.predict_proba(X_train)[:, 1]
# roc_auc_score(y_train, pred_y)


# In[ ]:


# [0]	validation_0-auc:0.63394
# [1000]	validation_0-auc:0.92842
# [2000]	validation_0-auc:0.93317
# [3000]	validation_0-auc:0.93425
# [4000]	validation_0-auc:0.93437
# [4999]	validation_0-auc:0.93438

# submission_df = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
# test_df = test_df.drop(['id', 'f_27'], axis=1)
# submission_df['target'] = xgb.predict_proba(test_df)[:, 1]
# submission_df.to_csv('submission.csv', index=False)
# submission_df.head()


# <br><br><br>
