#!/usr/bin/env python
# coding: utf-8

# # About this Notebook
# 
# In this notebook, i will try to cover following topics in as much detail as possible:
# * `Domain Knowledge`: In general, domain knowledge is important when posing the question you want answered from data, as well as understanding the limitations of the data. If you don't understand these things (e.g., if you're working with a client on a new problem), then an essential skill is being able to ask the right questions to tease these things out.
# * `Basic statistics on the dataset`
# * `Extensive EDA`
# * `Visualization of spatial maps`
# * `Sample Submission`
# 
# 
# **This kernel will be a work in Progress,and I will keep on updating it as the competition progresses**
#     
# **<span style="color:Red">Please upvote this kernel if you like it . It motivates me to produce more quality content :)**

# ![](https://images.squarespace-cdn.com/content/5a0b0eb1ace864b50eccea63/1564620684180-ECGGLZMYN4W05AICZ07K/brainscansscience.jpg?content-type=image%2Fjpeg)

# # Domain Knowledge
# 
# So lets start with the domain knowledge and Address the few important questions
# 
# 
# ### Q1) What is the motivation behind this competition?
# 
# Human brain research is among the most complex areas of study for scientists. We know that age and other factors can affect its function and structure, but more research is needed into what specifically occurs within the brain. With much of the research using MRI scans, data scientists are well positioned to support future insights. In particular, neuroimaging specialists look for measurable markers of behavior, health, or disorder to help identify relevant brain regions and their contribution to typical or symptomatic effects.
# 
# The competition(TReNDS) is meant to encourage approaches able to predict age plus additional continuous individual-level assessment values, given multimodal brain features such as 3D functional spatial maps from resting-state functional MRI, static functional network connectivity (FNC) matrices, and source-based morphometry (SBM) loading values from structural MRI. For this task, one of the largest datasets of unbiased multimodal brain imaging features is made available. Given a set of multimodal imaging features, the developed predictors should output age and assessment predictions.
# 
# 
# 
# 
# 
# ### Q2) What is Neuroimaging?
# 

# In[ ]:



from IPython.display import IFrame, YouTubeVideo
YouTubeVideo('N2apCx1rlIQ',width=600, height=400)


# ### Q3) What is an fMRI scan and how does it work??
# 
# An fMRI scan is a functional magnetic resonance imaging scan that measures and maps the brain’s activity. An fMRI scan uses the same technology as an MRI scan. An MRI is a noninvasive test that uses a strong magnetic field and radio waves to create an image of the brain. The image an MRI scan produces is just of organs/tissue, but an fMRI will produce an image showing the blood flow in the brain. By showing the blood flow it will display which parts of the brain are being stimulated.
# 
# ![](https://www.sciencealert.com/images/articles/processed/fmri-scanss_1024.jpg)
# 
# 
# * [Important Read](https://www.jameco.com/Jameco/workshop/HowItWorks/what-is-an-fmri-scan-and-how-does-it-work.html)
# 
# 
# 

# ### Q4) How Features Were Obtained (FMRI image Data Collection)?
# 
# An unbiased strategy was utilized to obtain the provided features. This means that a separate, unrelated large imaging dataset was utilized to learn feature templates. Then, these templates were "projected" onto the original imaging data of each subject used for this competition using spatially constrained independent component analysis (scICA) via group information guided ICA (GIG-ICA).
# 
# The first set of features are source-based morphometry (SBM) loadings. These are subject-level weights from a group-level ICA decomposition of gray matter concentration maps from structural MRI (sMRI) scans.
# 
# The second set are static functional network connectivity (FNC) matrices. These are the subject-level cross-correlation values among 53 component timecourses estimated from GIG-ICA of resting state functional MRI (fMRI).
# 
# The third set of features are the component spatial maps (SM). These are the subject-level 3D images of 53 spatial networks estimated from GIG-ICA of resting state functional MRI (fMRI).

# In[ ]:


YouTubeVideo('3fNf8KX1AlQ',width=600, height=400)


# ### Q5) What are we doing in this competition?
# 
# In this challenge, kagglers will predict `age` and `assessment values` from two domains using features derived from brain MRI images as inputs.
# 
# 
# 
# 

# ### Q6) How the dataset look likes?
# 
# **Files**
# * `fMRI_train` - a folder containing 53 3D spatial maps for train samples in .mat format
# * `fMRI_test` - a folder containing 53 3D spatial maps for test samples in .mat format
# * `fnc.csv` - static FNC correlation features for both train and test samples
# * `loading.csv` - sMRI SBM loadings for both train and test samples
# * `train_scores.csv` - age and assessment values for train samples
# * `sample_submission.csv` - a sample submission file in the correct format
# 
# * `reveal_ID_site2.csv` - a list of subject IDs whose data was collected with a different scanner than the train samples
# * `fMRI_mask.nii` - a 3D binary spatial map
# * `ICN_numbers.txt` - intrinsic connectivity network numbers for each fMRI spatial map; matches FNC names
# 
# The .mat files for this competition can be read in python using `h5py`, and the `.nii` file can be read in python using nilearn.
# ****

# ### Q7) What do we need to predict?
# 
# We need to predict values for following output variables:
# 
# * `age`
# * `domain1_var1`
# * `domain1_var2`
# * `domain2_var1`
# * `domain2_var2`

# ### Q8) What are expectations of the competition Host?
# 
# Hosts are expecting models which generalize well on data from a different scanner/site (site 2). All subjects from site 2 were assigned to the test set, so their scores are not available. While there are fewer site 2 subjects than site 1 subjects in the test set, the total number of subjects from site 2 will not be revealed until after the end of the competition. To make it more interesting, the IDs of some site 2 subjects have been revealed below. Use this to inform your models about site effects. Site effects are a form of bias. To generalize well, models should learn features that are not related to or driven by site effects.
# 
# 
# 
# 

# ## Important blogs, research papers and kaggle discussions
# 
# * https://www.kaggle.com/c/trends-assessment-prediction/discussion/145818
# * https://www.kaggle.com/c/trends-assessment-prediction/discussion/145791
# * https://www.kaggle.com/c/trends-assessment-prediction/discussion/145597
# * https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3181933/
# * https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0066572
# * https://www.kaggle.com/c/trends-assessment-prediction/discussion/146205

# # Preliminaries
# Now Let's Begin by Importing the data

# In[ ]:


get_ipython().system('pip install joypy --progress-bar off')


# In[ ]:


import os


import random
import seaborn as sns
import cv2
# General packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL

import nilearn as nl
import nilearn.plotting as nlplt
import nibabel as nib
import h5py
import plotly.graph_objs as go
from IPython.display import Image, display
import joypy
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


os.listdir('/kaggle/input/trends-assessment-prediction/')


# ## Loading Dataset
# 

# In[ ]:


BASE_PATH = '../input/trends-assessment-prediction'

# image and mask directories
train_data_dir = f'{BASE_PATH}/fMRI_train'
test_data_dir = f'{BASE_PATH}/fMRI_test'


print('Reading data...')
loading_data = pd.read_csv(f'{BASE_PATH}/loading.csv')
train_data = pd.read_csv(f'{BASE_PATH}/train_scores.csv')
sample_submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')
print('Reading data completed')


# The dataset comprises of following important files:
# * `train_scores.csv`: This file has `age`, `domain1_var1`, `domain1_var2`, `domain2_var1`, `domain2_var2` as important feature variables.
# * `loading.csv`: This file has `IC_01` to `IC_29` as important feature variables.

# In[ ]:


display(train_data.head())
print("Shape of train_data :", train_data.shape)


# In[ ]:


display(loading_data.head())
print("Shape of loading_data :", loading_data.shape)


# ## Checking for Null values 
# 
# #### train_data

# In[ ]:


# checking missing data
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()


# **inference**
# 
# * `domain1_var2` and `domain1_var1` has 438 missing values each.
# * `domain2_var2` and `domain2_var1` has 39 missing values each
# 

# #### loading_data

# In[ ]:


total = loading_data.isnull().sum().sort_values(ascending = False)
percent = (loading_data.isnull().sum()/loading_data.isnull().count()*100).sort_values(ascending = False)
missing_loading_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_loading_data.head()


# ## EDA

# ### Let's Start with distribution of variables in train_data
# 
# For Visualization purpose i have rounded the values to their nearest integer

# In[ ]:


def plot_bar(df, feature, title='', show_percent = False, size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.barplot(np.round(df[feature].value_counts().index).astype(int), df[feature].value_counts().values, alpha=0.8, palette='Set2')

    plt.title(title)
    if show_percent:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(100*height/total),
                    ha="center", rotation=45) 
    plt.xlabel(feature, fontsize=12, )
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()


# In[ ]:


plot_bar(train_data, 'age', 'age count and %age plot', show_percent=True, size=4)


# **inference**
# * Top 5 most frequent ages are 57, 60, 54, 55, 50
# * Most of the patients lie between the age group 22 to 77.

# In[ ]:


def plot_bar(df, feature, title='', show_percent = False, size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.barplot(np.round(df[feature].value_counts().index).astype(int), df[feature].value_counts().values, alpha=0.8, palette='Set2')

    plt.title(title)
    if show_percent:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(100*height/total),
                    ha="center", rotation=45) 
    plt.xlabel(feature, fontsize=12, )
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()


# In[ ]:


### Age count Distribution
for col in train_data.columns[2:]:
    plot_bar(train_data, col, f'{col} count plot', size=4)


# **inference**
# * `domain1_var1` has 64 unique values when rounded to nearest integers, 23 is the most frequent.
# * `domain1_var2` has 81 unique values when rounded to nearest integers, 28 is the most frequent.
# * `domain2_var1` has 76 unique values when rounded to nearest integers, 16 is the most frequent.
# * `domain2_var2` has 83 unique values when rounded to nearest integers, 22 is the most frequent.

# ### Heatmap showing correlation between train_data features

# In[ ]:


temp_data =  train_data.drop(['Id'], axis=1)

plt.figure(figsize = (12, 8))
sns.heatmap(temp_data.corr(), annot = True, cmap="RdYlGn")
plt.yticks(rotation=0) 

plt.show()


# **inference**
# * `age` and `domain1_var1` has a correlation value of 0.34, which is quite significant and shows a positive correlation between thease two variables.

# ### Heatmap showing correlation between loading_data features

# In[ ]:


temp_data =  loading_data.drop(['Id'], axis=1)

plt.figure(figsize = (20, 20))
sns.heatmap(temp_data.corr(), annot = True, cmap="RdYlGn")
plt.yticks(rotation=0) 

plt.show()


# **inference**
# 
# The above heatmap shows very high correlation between some features. For example
# * `IC_13` and `IC_14` has a correlation value as high as 0.55
# * `IC_10` and `IC_22` are also very highly negative correlated with correlation value -0.54

# In[ ]:


temp_data =  loading_data.drop(['Id'], axis=1)
# Create correlation matrix
correl = temp_data.corr().abs()

# Select upper triangle of correlation matrix
upper = correl.where(np.triu(np.ones(correl.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.5
to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]

print('Very high correlated features: ', to_drop)


# ### plot showing distribution of loading_data features 

# In[ ]:


# Draw Plot
import joypy

targets= loading_data.columns[1:]


plt.figure(figsize=(16,10), dpi= 80)
fig, axes = joypy.joyplot(loading_data, column=list(targets), ylim='own', figsize=(14,10))

# Decoration
plt.title('Distribution of features IC_01 to IC_29', fontsize=22)
plt.show()


# 
# ## 4. Loading and Exploring Spatial Maps
# Here we will see, how to load and display subject spatial map information for fMRI spatial maps. In general the spatial maps are saved as 4-D tensors
# 
# 𝑖∈ℝ𝑋×𝑌×𝑍×𝐾
#  
# where  𝑋 ,  𝑌 , and  𝑍  are the three spatial dimensions of the volume, and  𝐾  is the number of independent components.

# **File Format**
# 
# The subject spatial maps have been saved in .mat files using the v7.3 flag, so they must be loaded as h5py datasets, and a nifti file must be used to set the headers for display purposes. We have included the load_subject function, which takes a subject .mat filename, and the loaded nilearn image to use for setting the headers.
# 
# 

# In[ ]:


# Download the ch2better template image for display
get_ipython().system('wget https://github.com/Chaogan-Yan/DPABI/raw/master/Templates/ch2better.nii')


# In[ ]:


"""
    Load and display a subject's spatial map
"""

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
mask_filename = f'{BASE_PATH}/fMRI_mask.nii'
subject_filename = '../input/trends-assessment-prediction/fMRI_train/10004.mat'
smri_filename = 'ch2better.nii'
mask_niimg = nl.image.load_img(mask_filename)


def load_subject(filename, mask_niimg):
    """
    Load a subject saved in .mat format with
        the version 7.3 flag. Return the subject
        niimg, using a mask niimg as a template
        for nifti headers.
        
    Args:
        filename    <str>            the .mat filename for the subject data
        mask_niimg  niimg object     the mask niimg object used for nifti headers
    """
    subject_data = None
    with h5py.File(subject_filename, 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])
    subject_niimg = nl.image.new_img_like(mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)
    return subject_niimg
subject_niimg = load_subject(subject_filename, mask_niimg)
print("Image shape is %s" % (str(subject_niimg.shape)))
num_components = subject_niimg.shape[-1]
print("Detected {num_components} spatial maps".format(num_components=num_components))


# ### Displaying all Components in a Probability Atlas
# First, we will display the 53 spatial maps in one complete atlas using the `nilearn` `plot_prob_atlas` function. These maps will be overlaid on a structural MRI template.

# In[ ]:


nlplt.plot_prob_atlas(subject_niimg, bg_img=smri_filename, view_type='filled_contours', draw_cross=False,title='All %d spatial maps' % num_components, threshold='auto')


# ### Displaying Individual Component Maps
# 
# Additionally, we can separately display each of the 53 maps to get a more complete view
# of individual component structure.

# In[ ]:


grid_size = int(np.ceil(np.sqrt(num_components)))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*10, grid_size*10))
[axi.set_axis_off() for axi in axes.ravel()]
row = -1
for i, cur_img in enumerate(nl.image.iter_img(subject_niimg)):
    col = i % grid_size
    if col == 0:
        row += 1
    nlplt.plot_stat_map(cur_img, bg_img=smri_filename, title="IC %d" % i, axes=axes[row, col], threshold=3, colorbar=False)


# ## Sample Submission

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

get_ipython().system('pip install pycaret --quiet')


# In[ ]:


from pycaret.regression import *


# In[ ]:


BASE_PATH = '../input/trends-assessment-prediction'

fnc_df = pd.read_csv(f"{BASE_PATH}/fnc.csv")
loading_df = pd.read_csv(f"{BASE_PATH}/loading.csv")
labels_df = pd.read_csv(f"{BASE_PATH}/train_scores.csv")


# In[ ]:


fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")
labels_df["is_train"] = True
df = df.merge(labels_df, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()


# In[ ]:


target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
test_df = test_df.drop(target_cols + ['is_train'], axis=1)

# Giving less importance to FNC features since they are easier to overfit due to high dimensionality.
FNC_SCALE = 1/500
test_df[fnc_features] *= FNC_SCALE


# In[ ]:


target_models_dict = {
    'age': 'age_br',
    'domain1_var1':'domain1_var1_ridge',
    'domain1_var2':'domain1_var2_svm',
    'domain2_var1':'domain2_var1_ridge',
    'domain2_var2':'domain2_var2_svm',
}


# In[ ]:


## load PyCaret models

for index, target in enumerate(target_cols):
    model_name = target_models_dict[target]
    model = load_model(f'../input/pycaret-trends-models/{model_name}', platform = None, authentication = None, verbose=True)

    predictions = predict_model(model, data=test_df)
    test_df[target] = predictions['Label'].values


# #### Create Submission
# 

# In[ ]:


sub_df = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == test_df.shape[0]*5

sub_df.to_csv("submission1.csv", index=False)
sub_df.head()


# ## What's Next
# * Better Visualization 

# ## My other works for this competition
# 
# If you like my work, please take a look at my other works which i have done in this competition so far
# * https://www.kaggle.com/rohitsingh9990/trends-pycaret-training-inference

# # References
# 
# * https://pycaret.org/regression/
# * https://www.kaggle.com/bbradt/loading-and-exploring-spatial-maps
# * Few ideas taken from https://www.kaggle.com/tanulsingh077/prostate-cancer-in-depth-understanding-eda-model
# 

# # END NOTES
# This notebook is work in progress. 
# I will keep on updating this kernel with my new findings and learning in order to help everyone who has just started in this competition.
# 
# **<span style="color:Red">Please upvote this kernel if you like it . It motivates me to produce more quality content :)**  
