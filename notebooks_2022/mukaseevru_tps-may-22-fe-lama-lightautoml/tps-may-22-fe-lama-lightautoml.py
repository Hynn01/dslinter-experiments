#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.core.display import display, HTML, Javascript

# ----- Notebook Theme -----
color_map = ['#16a085', '#e8f6f3', '#d0ece7', '#a2d9ce', '#73c6b6', '#45b39d', 
                        '#16a085', '#138d75', '#117a65', '#0e6655', '#0b5345']

prompt = color_map[-1]
main_color = color_map[0]
strong_main_color = color_map[1]
custom_colors = [strong_main_color, main_color]

css_file = ''' 

div #notebook {
background-color: white;
line-height: 20px;
}

#notebook-container {
%s
margin-top: 2em;
padding-top: 2em;
border-top: 4px solid %s; /* light orange */
-webkit-box-shadow: 0px 0px 8px 2px rgba(224, 212, 226, 0.5); /* pink */
    box-shadow: 0px 0px 8px 2px rgba(224, 212, 226, 0.5); /* pink */
}

div .input {
margin-bottom: 1em;
}

.rendered_html h1, .rendered_html h2, .rendered_html h3, .rendered_html h4, .rendered_html h5, .rendered_html h6 {
color: %s; /* light orange */
font-weight: 600;
}

div.input_area {
border: none;
    background-color: %s; /* rgba(229, 143, 101, 0.1); light orange [exactly #E58F65] */
    border-top: 2px solid %s; /* light orange */
}

div.input_prompt {
color: %s; /* light blue */
}

div.output_prompt {
color: %s; /* strong orange */
}

div.cell.selected:before, div.cell.selected.jupyter-soft-selected:before {
background: %s; /* light orange */
}

div.cell.selected, div.cell.selected.jupyter-soft-selected {
    border-color: %s; /* light orange */
}

.edit_mode div.cell.selected:before {
background: %s; /* light orange */
}

.edit_mode div.cell.selected {
border-color: %s; /* light orange */

}
'''
def to_rgb(h): 
    return tuple(int(h[i:i+2], 16) for i in [0, 2, 4])

main_color_rgba = 'rgba(%s, %s, %s, 0.1)' % (to_rgb(main_color[1:]))
open('notebook.css', 'w').write(css_file % ('width: 95%;', main_color, main_color, main_color_rgba, main_color,  main_color, prompt, main_color, main_color, main_color, main_color))

def nb(): 
    return HTML("<style>" + open("notebook.css", "r").read() + "</style>")
nb()


# <img src="https://raw.githubusercontent.com/AILab-MLTools/LightAutoML/master/imgs/LightAutoML_logo_big.png" alt="LightAutoML logo" style="width:70%;"/>

# # LightAutoML baseline
# 
# Official LightAutoML github repository is [here](https://github.com/AILab-MLTools/LightAutoML). 
# 
# ### Do not forget to put upvote for the notebook and the ⭐️ for github repo if you like it using the button below - one click for you, great pleasure for us ☺️ 

# In[ ]:


s = '<iframe src="https://ghbtns.com/github-btn.html?user=AILab-MLTools&repo=LightAutoML&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="LightAutoML GitHub"></iframe>'
HTML(s)


# ## 0. Prerequisites

# ### 0.0. install LightAutoML

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install -U lightautoml\n!pip install --upgrade pandas')


# ### 0.1. Import libraries
# 
# Here we will import the libraries we use in this kernel:
# - Standard python libraries for timing, working with OS etc.
# - Essential python DS libraries like numpy, pandas, scikit-learn and torch (the last we will use in the next cell)
# - LightAutoML modules: presets for AutoML, task and report generation module

# In[ ]:


# Standard python libraries
import os
import time

# Essential DS libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco

pd.set_option('display.max_columns', None)


# ### 0.2. Constants
# 
# Here we setup the constants to use in the kernel:
# - `N_THREADS` - number of vCPUs for LightAutoML model creation
# - `RANDOM_STATE` - random seed for better reproducibility
# - `TEST_SIZE` - houldout data part size 
# - `TIMEOUT` - limit in seconds for model to train
# - `TARGET_NAME` - target column name in dataset

# In[ ]:


N_THREADS = 4 
RANDOM_STATE = 21
TIMEOUT = 8 * 3600
TARGET_NAME = 'target'


# ### 0.3. Imported models setup
# 
# For better reproducibility fix numpy random seed with max number of threads for Torch (which usually try to use all the threads on server):

# In[ ]:


np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)


# ### 0.4. Data loading
# Let's check the data we have:

# In[ ]:


INPUT_DIR = '../input/tabular-playground-series-may-2022/'


# In[ ]:


train_data = pd.read_csv(INPUT_DIR + 'train.csv')
print(train_data.shape)
train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


test_data = pd.read_csv(INPUT_DIR + 'test.csv')
print(test_data.shape)
test_data.head()


# In[ ]:


test_data.info()


# In[ ]:


submission = pd.read_csv(INPUT_DIR + 'sample_submission.csv')
print(submission.shape)
submission.head()


# ### 0.5. Feature engineering and reduce memory
# 
# Thanks to [Ambrosm's notebook](https://www.kaggle.com/code/ambrosm/tpsmay22-keras-quickstart) for the new features.
# 
# Let's make some new features:

# In[ ]:


def feature_engineering(data):
    for i in range(10):
        data[f'f_27_{i}'] = data['f_27'].str.get(i).apply(ord) - ord('A')
    data['f_27_unique_characters'] = data['f_27'].apply(lambda x: len(set(x)))
    return data

def reduce_memory(data, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = data.memory_usage().sum() / 1024 ** 2
    for column in data.columns:
        column_type = data[column].dtypes
        if column_type in numerics:
            c_min = data[column].min()
            c_max = data[column].max()
            if str(column_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[column] = data[column].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[column] = data[column].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[column] = data[column].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[column] = data[column].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[column] = data[column].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[column] = data[column].astype(np.float32)
                else:
                    data[column] = data[column].astype(np.float64)
    end_mem = data.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Memory usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100*(start_mem - end_mem) / start_mem))
    return data


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfor data in [train_data, test_data]:\n    data = feature_engineering(data)\n    data = reduce_memory(data)')


# In[ ]:


train_data.head()


# ### 0.6. Normalization
# Let's make our data more beautiful

# In[ ]:


from sklearn.preprocessing import RobustScaler

features = [feature for feature in train_data.columns if feature not in ['id', 'f_27', 'target']]
rb = RobustScaler()
train_data[features] = rb.fit_transform(train_data[features])
test_data[features] = rb.transform(test_data[features])


# # 1. Task definition

# ### 1.1. Task type
# 
# On the cell below we create Task object - the class to setup what task LightAutoML model should solve with specific loss and metric if necessary (more info can be found [here](https://lightautoml.readthedocs.io/en/latest/generated/lightautoml.tasks.base.Task.html#lightautoml.tasks.base.Task) in our documentation):

# In[ ]:


task = Task('binary', metric='auc', loss='logloss')


# ### 1.2. Feature roles setup
# To solve the task, we need to setup columns roles. The **only role you must setup is target role**, everything else (drop, numeric, categorical, group, weights etc.) is up to user - LightAutoML models have automatic columns typization inside:

# In[ ]:


roles = {'target': TARGET_NAME,
         'drop': ['id', 'f_27']
         }


# ### 1.3. LightAutoML model creation - TabularAutoML preset

# In next the cell we are going to create LightAutoML model with `TabularAutoML` class - preset with default model structure like in the image below:
# 
# <img src="https://github.com/AILab-MLTools/LightAutoML/raw/master/imgs/tutorial_blackbox_pipeline.png" alt="TabularAutoML preset pipeline" style="width:75%;"/>
# 
# in just several lines. Let's discuss the params we can setup:
# - `task` - the type of the ML task (the only **must have** parameter)
# - `timeout` - time limit in seconds for model to train
# - `cpu_limit` - vCPU count for model to use
# - `reader_params` - parameter change for Reader object inside preset, which works on the first step of data preparation: automatic feature typization, preliminary almost-constant features, correct CV setup etc. For example, we setup `n_jobs` threads for typization algo, `cv` folds and `random_state` as inside CV seed.
# 
# **Important note**: `reader_params` key is one of the YAML config keys, which is used inside `TabularAutoML` preset. [More details](https://github.com/AILab-MLTools/blob/master/lightautoml/automl/presets/tabular_config.yml) on its structure with explanation comments can be found on the link attached. Each key from this config can be modified with user settings during preset object initialization. To get more info about different parameters setting (for example, ML algos which can be used in `general_params->use_algos`) please take a look at our [article on TowardsDataScience](https://towardsdatascience.com/lightautoml-preset-usage-tutorial-2cce7da6f936).
# 
# Moreover, to receive the automatic report for our model we will use `ReportDeco` decorator and work with the decorated version in the same way as we do with usual one. 

# In[ ]:


automl = TabularAutoML(task = task,
                       timeout = TIMEOUT,
                       cpu_limit = N_THREADS,
                       reader_params = {'n_jobs': N_THREADS, 'random_state': RANDOM_STATE}
                      )


# # 2. AutoML training

# To run autoML training use fit_predict method:
# - `train_data` - Dataset to train.
# - `roles` - Roles dict.
# - `verbose` - Controls the verbosity: the higher, the more messages.
#         <1  : messages are not displayed;
#         >=1 : the computation process for layers is displayed;
#         >=2 : the information about folds processing is also displayed;
#         >=3 : the hyperparameters optimization process is also displayed;
#         >=4 : the training process for every algorithm is displayed;
# 
# Note: out-of-fold prediction is calculated during training and returned from the fit_predict method

# In[ ]:


get_ipython().run_cell_magic('time', '', "\noof_pred = automl.fit_predict(train_data, roles=roles, verbose=3)\nprint(f'oof_pred:\\n{oof_pred}\\nShape = {oof_pred.shape}')")


# In[ ]:


oof_pred_train = oof_pred.data[:, 0]


# In[ ]:


print(f'OOF score: {roc_auc_score(train_data[TARGET_NAME].values, (oof_pred_train > 0.5).astype(int))}')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfast_fi = automl.get_feature_scores('fast')\nfast_fi.set_index('Feature')['Importance'].plot.bar(figsize=(20, 10), grid=True)")


# # 5. Predict for test dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntest_pred = automl.predict(test_data)\nprint(f'Prediction for test data:\\n{test_pred}\\nShape = {test_pred.shape}')")


# In[ ]:


test_data[TARGET_NAME] = test_pred.data[:, 0]


# In[ ]:


submission[TARGET_NAME] = test_data[TARGET_NAME]
submission


# Save submissions to .csv

# In[ ]:


submission.to_csv('lightautoml_prob.csv', index=False)


# # Additional materials

# - [Official LightAutoML github repo](https://github.com/AILab-MLTools/LightAutoML)
# - [LightAutoML documentation](https://lightautoml.readthedocs.io/en/latest)
# - [LightAutoML tutorials](https://github.com/AILab-MLTools/LightAutoML/tree/master/examples/tutorials)
# - LightAutoML course:
#     - [Part 1 - general overview](https://ods.ai/tracks/automl-course-part1) 
#     - [Part 2 - LightAutoML specific applications](https://ods.ai/tracks/automl-course-part2)
#     - [Part 3 - LightAutoML customization](https://ods.ai/tracks/automl-course-part3)
# - [OpenDataScience AutoML benchmark leaderboard](https://ods.ai/competitions/automl-benchmark/leaderboard)

# ### If you still like the notebook, do not forget to put upvote for the notebook and the ⭐️ for github repo if you like it using the button below - one click for you, great pleasure for us ☺️

# In[ ]:


s = '<iframe src="https://ghbtns.com/github-btn.html?user=AILab-MLTools&repo=LightAutoML&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="LightAutoML GitHub"></iframe>'
HTML(s)

