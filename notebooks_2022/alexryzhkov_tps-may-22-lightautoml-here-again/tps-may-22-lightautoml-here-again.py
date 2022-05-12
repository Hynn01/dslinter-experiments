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


# <img src="https://github.com/AILab-MLTools/LightAutoML/raw/master/imgs/LightAutoML_logo_big.png" alt="LightAutoML logo" style="width:70%;"/>

# # LightAutoML baseline
# 
# Official LightAutoML github repository is [here](https://github.com/AILab-MLTools/LightAutoML). 
# 
# ### Do not forget to put upvote for the notebook and the ⭐️ for github repo if you like it - one click for you, great pleasure for us ☺️ 

# In[ ]:


s = '<iframe src="https://ghbtns.com/github-btn.html?user=AILab-MLTools&repo=LightAutoML&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="LightAutoML GitHub"></iframe>'
HTML(s)


# ## This notebook is the updated copy of our [Tutorial_1 from the GIT repository](https://github.com/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_1_basics.ipynb). Please check our [tutorials folder](https://github.com/AILab-MLTools/LightAutoML/blob/master/examples/tutorials) if you are interested in other examples of LightAutoML functionality.

# ## 0. Prerequisites

# ### 0.0. install LightAutoML

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip3 install -U lightautoml\n\n# QUICK WORKAROUND FOR PROBLEM WITH PANDAS\n!pip3 install -U pandas')


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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

# For NN training
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.preprocessing import StandardScaler


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
N_FOLDS = 10
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 8 * 3600 # equal to 8 hours
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


test_data = pd.read_csv(INPUT_DIR + 'test.csv')
print(test_data.shape)
test_data.head()


# In[ ]:


submission = pd.read_csv(INPUT_DIR + 'sample_submission.csv')
print(submission.shape)
submission.head()


# ### 0.5. Feature engineering
# Let's prepare some features from the text feature `f_27` and top importance features compositions:

# In[ ]:


def create_tf_idf_feats(corpus, ngram_range = (1, 1), max_features = None):
    vectorizer = TfidfVectorizer(analyzer = 'char', lowercase = False, 
                                 ngram_range = ngram_range, max_features = max_features)
    X = vectorizer.fit_transform(corpus).todense()
    char_mapper = {y:x for x, y in vectorizer.vocabulary_.items()}
    column_names = ['tfidf_{}'.format(char_mapper[i]) for i in range(len(char_mapper))]
    return pd.DataFrame(X, columns = column_names)

# Create all texts array
all_texts = pd.concat([train_data[['f_27']], test_data[['f_27']]]).reset_index(drop = True)
corpus = all_texts['f_27'].values

# Calculate TF-IDF features for unigrams and top-20 bigrams
all_texts = pd.concat([all_texts, 
                      create_tf_idf_feats(corpus),
                      create_tf_idf_feats(corpus, (2,2), 20)], axis = 1)

# Create features for unique characters
for i in range(10):
    all_texts[f'char_{i}'] = all_texts['f_27'].str.get(i).map(ord) - ord('A')
    
# How many unique characters are in each text?
all_texts['unique_chars_cnt'] = all_texts['f_27'].map(lambda x: len(set(x)))

# How often the text occurs in the whole dataset
all_texts['value_frequency'] = all_texts['f_27'].map(all_texts['f_27'].value_counts() / len(all_texts))

all_texts.drop(columns = ['f_27'], inplace = True)


# In[ ]:


all_texts.head()


# In[ ]:


train_data = pd.concat([train_data,
                       all_texts.iloc[:len(train_data), :]], axis = 1)
test_data = pd.concat([test_data,
                       all_texts.iloc[len(train_data):, :].reset_index(drop = True)], axis = 1)

print(train_data.shape, test_data.shape)


# In[ ]:


train_data.head()


# In[ ]:


del all_texts


# In[ ]:


for data in [train_data, test_data]:
    for i,j in [(26, 21), (26, 22), (21, 22), (26, 30), (26, 25),
               (22, 30), (22, 25), (21, 30), (21, 25), (30, 25)]:
        data['f_{}_plus_f_{}'.format(i, j)] = data['f_{}'.format(i)] + data['f_{}'.format(j)]
        data['f_{}_minus_f_{}'.format(i, j)] = data['f_{}'.format(i)] - data['f_{}'.format(j)]


# In[ ]:


print([col for col in train_data.columns if col.startswith('tfidf_')])


# ### 0.6. Dense neural net probabilities
# As the data is created from GAN, which is neural network, let's use simple neural network to get these dependencies from the data:

# In[ ]:


def nn_model(n_feats):
    inputs = Input(shape=(n_feats))
    x = Dense(64, activation='relu')(inputs)
    #x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    #x = Dropout(0.1)(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, x)
    return model


def fit_model(X_train, y_train, 
              X_valid, y_valid, 
              X_test):
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_valid)
    X_te = scaler.transform(X_test)
    valid_data = (X_va, y_valid)

    lr = ReduceLROnPlateau(monitor = "val_loss", factor = 0.75, 
                           patience = 10, verbose = 0)
    es = EarlyStopping(monitor = "val_loss", patience = 20, 
                       verbose = 1, restore_best_weights = True)
    model = nn_model(X_train.shape[1])
    model.compile(optimizer = Adam(learning_rate = 0.01),
                  loss = BinaryCrossentropy())

    history = model.fit(X_tr, y_train, 
                        validation_data = valid_data, 
                        epochs = 500,
                        verbose = 0,
                        batch_size = 4096,
                        shuffle = True,
                        callbacks = [lr, es])
    
    y_valid_pred = model.predict(X_va).reshape(1, -1)[0]
    y_test_pred = model.predict(X_te).reshape(1, -1)[0]

    return y_valid_pred, y_test_pred


features = [x for x in train_data.columns if x not in ['id', TARGET_NAME, 'f_27']]
skf = StratifiedKFold(n_splits = N_FOLDS, random_state = RANDOM_STATE, shuffle = True)
NN_OOF_PRED = np.zeros(len(train_data))
NN_TEST_PRED = np.zeros(len(test_data))
    
y = train_data[TARGET_NAME].values
X_test = test_data[features]
N_START = 5
for it in range(N_START):
    for fold, (train_idx, valid_idx) in enumerate(skf.split(y, y)):
        X_train = train_data.iloc[train_idx, :][features]
        X_valid = train_data.iloc[valid_idx, :][features]
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        val_pred, test_pred = fit_model(X_train, y_train, 
                                        X_valid, y_valid, 
                                        X_test)

        print('ITER = {} FOLD {} score {:.5f}'.format(it, fold, roc_auc_score(y_valid, val_pred)))

        NN_OOF_PRED[valid_idx] += val_pred / N_START
        NN_TEST_PRED += test_pred / N_FOLDS / N_START
    print('AFTER ITER {} NN OOF score {:.5f}'.format(it, roc_auc_score(y, NN_OOF_PRED)))
    
print('NN OOF score {:.5f}'.format(roc_auc_score(y, NN_OOF_PRED)))


# In[ ]:


train_data['NN'] = NN_OOF_PRED
test_data['NN'] = NN_TEST_PRED


# # 1. Task definition

# ### 1.1. Task type
# 
# On the cell below we create Task object - the class to setup what task LightAutoML model should solve with specific loss and metric if necessary (more info can be found [here](https://lightautoml.readthedocs.io/en/latest/pages/modules/generated/lightautoml.tasks.base.Task.html#lightautoml.tasks.base.Task) in our documentation):

# In[ ]:


task = Task('binary',)


# ### 1.2. Feature roles setup

# To solve the task, we need to setup columns roles. The **only role you must setup is target role**, everything else (drop, numeric, categorical, group, weights etc.) is up to user - LightAutoML models have automatic columns typization inside:

# In[ ]:


roles = {
    'target': TARGET_NAME,
    'drop': ['id', 'f_27'] + [col for col in train_data.columns if col.startswith('tfidf_')]
}


# ### 1.3. LightAutoML model creation - TabularAutoML preset

# In next the cell we are going to create LightAutoML model with `TabularAutoML` class - preset with default model structure like in the image below:
# 
# <img src="https://github.com/AILab-MLTools/LightAutoML/raw/master/imgs/tutorial_blackbox_pipeline.png" alt="TabularAutoML preset pipeline" style="width:85%;"/>
# 
# in just several lines. Let's discuss the params we can setup:
# - `task` - the type of the ML task (the only **must have** parameter)
# - `timeout` - time limit in seconds for model to train
# - `cpu_limit` - vCPU count for model to use
# - `reader_params` - parameter change for Reader object inside preset, which works on the first step of data preparation: automatic feature typization, preliminary almost-constant features, correct CV setup etc. For example, we setup `n_jobs` threads for typization algo, `cv` folds and `random_state` as inside CV seed.
# 
# **Important note**: `reader_params` key is one of the YAML config keys, which is used inside `TabularAutoML` preset. [More details](https://github.com/AILab-MLTools/LightAutoML/blob/master/lightautoml/automl/presets/tabular_config.yml) on its structure with explanation comments can be found on the link attached. Each key from this config can be modified with user settings during preset object initialization. To get more info about different parameters setting (for example, ML algos which can be used in `general_params->use_algos`) please take a look at our [article on TowardsDataScience](https://towardsdatascience.com/lightautoml-preset-usage-tutorial-2cce7da6f936).
# 
# Moreover, to receive the automatic report for our model we can use `ReportDeco` decorator and work with the decorated version in the same way as we do with usual one (more details in [this tutorial](https://github.com/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_1_basics.ipynb))

# In[ ]:


automl = TabularAutoML(
    task = task, 
    timeout = TIMEOUT,
    cpu_limit = N_THREADS,
    general_params = {'use_algos': [['lgb']]},
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


get_ipython().run_cell_magic('time', '', 'oof_pred = automl.fit_predict(train_data, roles = roles, verbose = 3)')


# In[ ]:


print(automl.create_model_str_desc())


# In[ ]:


print(f'TRAIN out-of-fold score: {roc_auc_score(train_data[TARGET_NAME].values, oof_pred.data[:, 0])}')


# # 3. Feature importances calculation 
# 
# For feature importances calculation we have 2 different methods in LightAutoML:
# - Fast (`fast`) - this method uses feature importances from feature selector LGBM model inside LightAutoML. It works extremely fast and almost always (almost because of situations, when feature selection is turned off or selector was removed from the final models with all GBM models). no need to use new labelled data.
# - Accurate (`accurate`) - this method calculate *features permutation importances* for the whole LightAutoML model based on the **new labelled data**. It always works but can take a lot of time to finish (depending on the model structure, new labelled dataset size etc.).

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Fast feature importances calculation\nfast_fi = automl.get_feature_scores('fast')\ntop_3_features = fast_fi['Feature'].values[:3]\nfast_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)")


# In[ ]:


fast_fi.head()


# In[ ]:


automl.plot_pdp(test_data.sample(10000), feature_name=top_3_features[0])


# In[ ]:


automl.plot_pdp(test_data.sample(10000), feature_name=top_3_features[1])


# In[ ]:


automl.plot_pdp(test_data.sample(10000), feature_name=top_3_features[2])


# In[ ]:


#%%time

# Accurate feature importances calculation (Permutation importances) -  can take long time to calculate on bigger datasets
# accurate_fi = automl.get_feature_scores('accurate', te_data, silent = False)
# accurate_fi.set_index('Feature')['Importance'].plot.bar(figsize = (30, 10), grid = True)


# # 4. Predict for test dataset
# 
# We are also ready to predict for our test competition dataset and submission file creation:

# In[ ]:


test_pred = automl.predict(test_data)
print(f'Prediction for te_data:\n{test_pred}\nShape = {test_pred.shape}')


# In[ ]:


submission[TARGET_NAME] = test_pred.data[:, 0]
submission.to_csv('lightautoml_tabularautoml.csv', index = False)


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

