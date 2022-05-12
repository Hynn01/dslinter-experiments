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


# <img src="https://github.com/sberbank-ai-lab/LightAutoML/raw/master/imgs/LightAutoML_logo_big.png" alt="LightAutoML logo" style="width:70%;"/>

# # LightAutoML baseline
# 
# Official LightAutoML github repository is [here](https://github.com/sberbank-ai-lab/LightAutoML). 
# 
# ### Do not forget to put upvote for the notebook and the ⭐️ for github repo if you like it - one click for you, great pleasure for us ☺️ 

# In[ ]:


s = '<iframe src="https://ghbtns.com/github-btn.html?user=sberbank-ai-lab&repo=LightAutoML&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="LightAutoML GitHub"></iframe>'
HTML(s)


# ## This notebook is the updated copy of our [Tutorial_1 from the GIT repository](https://github.com/sberbank-ai-lab/LightAutoML/blob/master/examples/tutorials/Tutorial_1_basics.ipynb). Please check our [tutorials folder](https://github.com/sberbank-ai-lab/LightAutoML/blob/master/examples/tutorials) if you are interested in other examples of LightAutoML functionality.

# ## 0. Prerequisites

# ### 0.0. install LightAutoML

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install -U lightautoml')


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install -U pandas')


# ### 0.1. Import libraries
# 
# Here we will import the libraries we use in this kernel:
# - Standard python libraries for timing, working with OS etc.
# - Essential python DS libraries like numpy, pandas, scikit-learn and torch (the last we will use in the next cell)
# - LightAutoML modules: `TabularAutoML` preset for AutoML model creation and Task class to setup what kind of ML problem we solve (binary/multiclass classification or regression)

# In[ ]:


# Standard python libraries
import os
import time

# Essential DS libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


# ### 0.2. Constants
# 
# Here we setup the constants to use in the kernel:
# - `N_THREADS` - number of vCPUs for LightAutoML model creation
# - `N_FOLDS` - number of folds in LightAutoML inner CV
# - `RANDOM_STATE` - random seed for better reproducibility
# - `TEST_SIZE` - houldout data part size 
# - `TIMEOUT` - limit in seconds for model to train
# - `TARGET_NAME` - target column name in dataset

# In[ ]:


N_THREADS = 4
N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 1800 # equal to 10 minutes
TARGET_NAME = 'final_price'


# ### 0.3. Imported models setup
# 
# For better reproducibility fix numpy random seed with max number of threads for Torch (which usually try to use all the threads on server):

# In[ ]:


np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)


# ### 0.4. Data loading
# Let's check the data we have:

# In[ ]:


INPUT_DIR = '../input/sf-dst-predict-car-price/'


# In[ ]:


train_data = pd.read_csv(INPUT_DIR + 'train_data.csv')
print(train_data.shape)
train_data.head()


# In[ ]:


test_data = pd.read_csv(INPUT_DIR + 'test_data.csv')
print(test_data.shape)
test_data.head()


# In[ ]:


submission = pd.read_csv(INPUT_DIR + 'sample_submission.csv')
print(submission.shape)
submission.head()


# ### Step 0.5. Some user feature preparation
# 
# Be creative :)

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef create_expert_feats(data):\n    pass\n\ncreate_expert_feats(train_data)\ncreate_expert_feats(test_data)')


# ### 0.6. Data splitting for train-holdout
# As we have only one file with target values, we can split it into 80%-20% for holdout usage:

# In[ ]:


tr_data, te_data = train_test_split(
    train_data, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE
)

print(f'Data splitted. Parts sizes: tr_data = {tr_data.shape}, te_data = {te_data.shape}')

tr_data.head()


# # 1. Task definition

# ### 1.1. Task type
# 
# On the cell below we create Task object - the class to setup what task LightAutoML model should solve with specific loss and metric if necessary (more info can be found [here](https://lightautoml.readthedocs.io/en/latest/pages/modules/generated/lightautoml.tasks.base.Task.html#lightautoml.tasks.base.Task) in our documentation):

# In[ ]:


task = Task('reg', loss = 'mae', metric = 'mae')


# ### 1.2. Feature roles setup

# To solve the task, we need to setup columns roles. The **only role you must setup is target role**, everything else (drop, numeric, categorical, group, weights etc.) is up to user - LightAutoML models have automatic columns typization inside:

# In[ ]:


roles = {
    'target': TARGET_NAME,
    'drop': ['row_ID']
}


# ### 1.3. LightAutoML model creation - TabularAutoML preset

# In next the cell we are going to create LightAutoML model with `TabularAutoML` class - preset with default model structure like in the image below:
# 
# <img src="https://github.com/sberbank-ai-lab/LightAutoML/raw/master/imgs/tutorial_blackbox_pipeline.png" alt="TabularAutoML preset pipeline" style="width:85%;"/>
# 
# in just several lines. Let's discuss the params we can setup:
# - `task` - the type of the ML task (the only **must have** parameter)
# - `timeout` - time limit in seconds for model to train
# - `cpu_limit` - vCPU count for model to use
# - `reader_params` - parameter change for Reader object inside preset, which works on the first step of data preparation: automatic feature typization, preliminary almost-constant features, correct CV setup etc. For example, we setup `n_jobs` threads for typization algo, `cv` folds and `random_state` as inside CV seed.
# 
# **Important note**: `reader_params` key is one of the YAML config keys, which is used inside `TabularAutoML` preset. [More details](https://github.com/sberbank-ai-lab/LightAutoML/blob/master/lightautoml/automl/presets/tabular_config.yml) on its structure with explanation comments can be found on the link attached. Each key from this config can be modified with user settings during preset object initialization. To get more info about different parameters setting (for example, ML algos which can be used in `general_params->use_algos`) please take a look at our [article on TowardsDataScience](https://towardsdatascience.com/lightautoml-preset-usage-tutorial-2cce7da6f936).
# 
# Moreover, to receive the automatic report for our model we can use `ReportDeco` decorator and work with the decorated version in the same way as we do with usual one (more details in [this tutorial](https://github.com/sberbank-ai-lab/LightAutoML/blob/master/examples/tutorials/Tutorial_1_basics.ipynb))

# In[ ]:


automl = TabularAutoML(
    task = task, 
    timeout = TIMEOUT,
    cpu_limit = N_THREADS,
    reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE}
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


get_ipython().run_cell_magic('time', '', 'oof_pred = automl.fit_predict(tr_data, roles = roles, verbose = 1)')


# In[ ]:


print(automl.create_model_str_desc())


# # 3. Prediction on holdout and model evaluation
# 
# We now have the trained model called `automl` and it's time to see what MAE score it can receive on train and holdout data:

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nte_pred = automl.predict(te_data)\nprint(f'Prediction for te_data:\\n{te_pred}\\nShape = {te_pred.shape}')")


# In[ ]:


print(f'TRAIN out-of-fold score: {mean_absolute_error(tr_data[TARGET_NAME].values, oof_pred.data[:, 0])}')
print(f'HOLDOUT score: {mean_absolute_error(te_data[TARGET_NAME].values, te_pred.data[:, 0])}')


# # 4. Feature importances calculation 
# 
# For feature importances calculation we have 2 different methods in LightAutoML:
# - Fast (`fast`) - this method uses feature importances from feature selector LGBM model inside LightAutoML. It works extremely fast and almost always (almost because of situations, when feature selection is turned off or selector was removed from the final models with all GBM models). no need to use new labelled data.
# - Accurate (`accurate`) - this method calculate *features permutation importances* for the whole LightAutoML model based on the **new labelled data**. It always works but can take a lot of time to finish (depending on the model structure, new labelled dataset size etc.).

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Fast feature importances calculation\nfast_fi = automl.get_feature_scores('fast')\nfast_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Accurate feature importances calculation (Permutation importances) -  can take long time to calculate on bigger datasets\naccurate_fi = automl.get_feature_scores('accurate', te_data, silent = False)")


# In[ ]:


accurate_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)


# # 5. Predict for test dataset
# 
# We are also ready to predict for our test competition dataset and submission file creation:

# In[ ]:


test_pred = automl.predict(test_data)
print(f'Prediction for te_data:\n{test_pred}\nShape = {test_pred.shape}')


# In[ ]:


submission[TARGET_NAME] = test_pred.data[:, 0]
submission.to_csv('lightautoml_tabularautoml_mae_timeout.csv', index = False)


# # Additional materials

# - [Official LightAutoML github repo](https://github.com/sberbank-ai-lab/LightAutoML)
# - [LightAutoML documentation](https://lightautoml.readthedocs.io/en/latest)
# - [LightAutoML tutorials](https://github.com/sberbank-ai-lab/LightAutoML/tree/master/examples/tutorials)
# - LightAutoML course:
#     - [Part 1 - general overview](https://ods.ai/tracks/automl-course-part1) 
#     - [Part 2 - LightAutoML specific applications](https://ods.ai/tracks/automl-course-part2)
#     - [Part 3 - LightAutoML customization](https://ods.ai/tracks/automl-course-part3)
# - [OpenDataScience AutoML benchmark leaderboard](https://ods.ai/competitions/automl-benchmark/leaderboard)

# ### If you still like the notebook, do not forget to put upvote for the notebook and the ⭐️ for github repo if you like it using the button below - one click for you, great pleasure for us ☺️

# In[ ]:


s = '<iframe src="https://ghbtns.com/github-btn.html?user=sberbank-ai-lab&repo=LightAutoML&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="LightAutoML GitHub"></iframe>'
HTML(s)

