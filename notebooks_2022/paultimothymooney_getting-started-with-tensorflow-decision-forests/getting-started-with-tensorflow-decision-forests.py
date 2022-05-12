#!/usr/bin/env python
# coding: utf-8

# # Getting started with TensorFlow Decision Forests
# 

# # Introduction

# We'll be working with the [Tabular Playground Series May 2022](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/data) Kaggle Dataset.  It is a tabular dataset with 900,000 rows and 33 columns (318MB .CSV training dataset + 247MB .CSV test set) that is suitable for training algorithms to solve binary classification problems (in this case to determine if a machine is in a state of "0" or "1" based off of input sensor data).  
# 
# We'll be using [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests) (TF-DF) to train our model.  TensorFlow Decision Forests is a TensorFlow wrapper for the [Yggdrasil Decision Forests C++ libraries](https://github.com/google/yggdrasil-decision-forests).  TF-DF makes it very easy to train, serve and interpret various Decision Forest models such as [RandomForests](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel) and [GrandientBoostedTrees](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel).  These types of decision forest models require minimal pre-processing of the data and are great when working with tabular datasets and/or small datasets (especially if you just want a quick baseline result to compare against).

# Step 1: Import Python packages

# In[ ]:


get_ipython().system('pip install tensorflow_decision_forests')


# In[ ]:


# Import Python packages
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Define helper functions
def plot_tfdf_model_training_curves(model):
    # This function was adapted from the following tutorial:
    # https://www.tensorflow.org/decision_forests/tutorials/beginner_colab
    logs = model.make_inspector().training_logs()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy (out-of-bag)")

    plt.subplot(1, 2, 2)
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Logloss (out-of-bag)")

    plt.show()


# In[ ]:


import tensorflow_decision_forests as tfdf
print("TensorFlow Decision Forests v" + tfdf.__version__)


# Step 2: Identify the location of the data

# In[ ]:


# print list of all data and files attached to this notebook
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Step 3: Load the data

# In[ ]:


# load to pandas dataframe (for data exploration)
train_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')

# load to tensorflow dataset (for model training)
train_tfds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="target")
test_tfds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)


# Step 4: Explore the data

# In[ ]:


# print column names
print(train_df.columns)


# In[ ]:


# preview first few rows of data
train_df.head(10)


# In[ ]:


# print basic summary statistics
train_df.describe()


# In[ ]:


# check for missing values
sns.heatmap(train_df.isnull(), cbar=False)


# # RandomForest

# Step 5: Train a [Random Forest](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) Model
# 
# 
# 
# > "A Random Forest is a collection of deep CART decision trees trained independently and without pruning. Each tree is trained on a random subset of the original training dataset (sampled with replacement).
# > 
# > The algorithm is unique in that it is robust to overfitting, even in extreme cases e.g. when there is more features than training examples.
# > 
# > It is probably the most well-known of the Decision Forest training algorithms"
# 
# 
# 
# 
#  ~ Quoted from [TFDF RandomForest documentation ](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel)

# One neat thing about TF-DF is that in addition to having a default set of hyper-parameters, you are also provided with a list of additional hyper-parameter choices to consider.  This makes it a lot easier to optimize model performance because you do not have to do this expensive hyper-parameter optimization step all by yourself.

# In[ ]:


print(tfdf.keras.RandomForestModel.predefined_hyperparameters())


# In[ ]:


# Train the model
rf_model = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1",
                                        compute_oob_variable_importances=True)
rf_model.fit(x=train_tfds)
model = rf_model


# In[ ]:


# Visualize the model
# Currently this step works in the Kaggle Notebook Editor but unfortunately displays an empty/blank visualization in the Notebook Viewer
tfdf.model_plotter.plot_model_in_colab(model, tree_idx=0, max_depth=3)


# Step 6: Evaluate your Random Forest Model

# In[ ]:


plot_tfdf_model_training_curves(model)


# In[ ]:


inspector = model.make_inspector()
inspector.evaluation()


# In[ ]:


print("Model type:", inspector.model_type())
print("Objective:", inspector.objective())
print("Evaluation:", inspector.evaluation())


# 80% accuracy is not a bad baseline result given how quickly we put this together (and with so few lines of code).

# Step 7: Investigate variable importances for the RandomForest model

# In[ ]:


# Adapted from https://www.tensorflow.org/decision_forests/tutorials/advanced_colab
# See list of inspector methods from:
# [field for field in dir(inspector) if not field.startswith("_")]
print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
  print("\t", importance)


# In[ ]:


inspector.variable_importances()["MEAN_DECREASE_IN_ACCURACY"]


# In[ ]:


inspector.variable_importances()["MEAN_DECREASE_IN_AUC_2_VS_OTHERS"]


# # GradientBoostedTrees

# Step 8: Train a [GradientBoostedTrees](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) model.  GradientBoostedTrees often perform better than the RandomForests we were using previously.
# 
# 
# 
# > "A GBT (Gradient Boosted Tree) is a set of shallow decision trees trained sequentially. Each tree is trained to predict and then "correct" for the errors of the previously trained trees (more precisely each tree predict the gradient of the loss relative to the model output)"
# 
# 
# 
#  ~ Quoted from [TFDF GradientBoostedTrees documentation ](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel)
# 

# In[ ]:


# As mentioned previously, TF-DF gives you lots of different "default" hyper-parameter settings to choose from.
print(tfdf.keras.GradientBoostedTreesModel.predefined_hyperparameters())


# In[ ]:


# Train the model
gb_model = tfdf.keras.GradientBoostedTreesModel(hyperparameter_template="benchmark_rank1")
gb_model.fit(x=train_tfds)
model = gb_model


# In[ ]:


# Visualize the model
# Currently this step works in the Kaggle Notebook Editor but unfortunately displays an empty/blank visualization in the Notebook Viewer
tfdf.model_plotter.plot_model_in_colab(model, tree_idx=0, max_depth=3)


# Step 9: Evaluate your GradientBoostedTrees  Model

# In[ ]:


plot_tfdf_model_training_curves(model)


# In[ ]:


inspector = model.make_inspector()
inspector.evaluation()


# In[ ]:


print("Model type:", inspector.model_type())
print("Objective:", inspector.objective())
print("Evaluation:", inspector.evaluation())


# 85% accuracy is not a bad baseline result given how quickly we put this together (and with so few lines of code).

# Step 10: Investigate variable importances for the GradientBoostedTrees model

# In[ ]:


# Adapted from https://www.tensorflow.org/decision_forests/tutorials/advanced_colab
# See list of inspector methods from:
# [field for field in dir(inspector) if not field.startswith("_")]
print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
  print("\t", importance)


# In[ ]:


inspector.variable_importances()["SUM_SCORE"]


# In[ ]:


tfdf.model_plotter.plot_model_in_colab(model, tree_idx=0, max_depth=3)


# Step 11: Submit your results

# In[ ]:


sample_submission_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv')
sample_submission_df.head(5)


# In[ ]:


# Adapted from https://www.kaggle.com/code/rhythmcam/tps-may-20-xgboost-baseline
sample_submission_df['target'] = np.round(model.predict(test_tfds),2)
sample_submission_df.to_csv('/kaggle/working/submission.csv', index=False)
sample_submission_df.head()


# TF-DF makes it very easy to find lots of useful information about your model.  For example, the following code cell provides a tremendous amount of information with just a single line of code.  You can preview the output of this code cell by clicking on the "show output" button below.

# In[ ]:


model.summary()


# # Conclusion

# [TensorFlow Decision Forests](https://www.tensorflow.org/decision_forests) (TF-DF) made it  quick and easy to train our RandomForest and GradientBoostedTrees models.  These types of decision forest models require minimal pre-processing of the data and are great when working with tabular datasets and/or small datasets (especially if you just want a quick baseline result to compare against).  Some of my favorite parts about  working with TF-DF were: (1) I was able to train a GradientBoostedTrees model with only a few lines of code; (2) there were lots of different default hyper-parameter options that I could choose from; (3) it was easy to visualize the structure/architecture of my models; and (4) it was easy to explore what features were most important to my model (to interpret and explain its decisions).
# 
# 
# We worked with the [Tabular Playground Series May 2022](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/data) Kaggle Dataset.  It was a tabular dataset with 900,000 rows and 33 columns that contained data from industrial sensors, designed t be used to determine whether that piece of industrial equipment was in a state of  "0" or "1".
# 
# We were able to solve this task with an accuracy of ~85% which is not a bad baseline result given how quickly we were able to put this together (and with so few lines of code).
# 
# To learn more about TF-DF visit https://www.tensorflow.org/decision_forests.
# 
# Next steps?
#  - Click on the "copy & edit" button in the top right corner of this notebook
#  - Experiment and try to increase the score.  My recommendation would be to focus on the [feature engineering and feature selection](https://www.kaggle.com/learn/feature-engineering) steps, as these steps were omitted from this tutorial (for the sake of brevity)
#  - Make a submission to https://www.kaggle.com/competitions/tabular-playground-series-may-2022

# References:
#  - https://www.tensorflow.org/decision_forests/tutorials/beginner_colab
#  - https://www.tensorflow.org/decision_forests/tutorials/intermediate_colab
#  - https://www.tensorflow.org/decision_forests/tutorials/advanced_colab

# In[ ]:




