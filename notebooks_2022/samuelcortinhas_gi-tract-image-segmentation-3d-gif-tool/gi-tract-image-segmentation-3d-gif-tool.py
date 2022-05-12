#!/usr/bin/env python
# coding: utf-8

# <br>
# 
# <br><center><img src="https://storage.googleapis.com/kaggle-competitions/kaggle/27923/logos/header.png?t=2021-06-02-20-30-25" width=100%></center>
# 
# <h2 style="text-align: center; font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: underline; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">UWM - GI Tract Image Segmentation Challenge - EDA</h2>
# <h5 style="text-align: center; font-family: Verdana; font-size: 12px; font-style: normal; font-weight: bold; text-decoration: None; text-transform: none; letter-spacing: 1px; color: black; background-color: #ffffff;">CREATED BY: DARIEN SCHETTLER, <br> EDITTED BY SAMUEL CORTINHAS</h5>

# **Acknowledgement:**
# * The [original notebook](https://www.kaggle.com/code/dschettler8845/uwm-gi-tract-image-segmentation-eda/notebook) was created by [Darien Schettler](https://www.kaggle.com/dschettler8845). If you haven't seen his work already then go and give him an upvote please!
# * My contribution is in section 4.9 where I extend the 3D GIFs to work on a case by case basis, showing all the days simultaneously.

# <p id="toc"></p>
# 
# <br><br>
# 
# <h1 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; color: teal; background-color: #ffffff;">TABLE OF CONTENTS</h1>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;"><a href="#imports">0&nbsp;&nbsp;&nbsp;&nbsp;IMPORTS</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;"><a href="#background_information">1&nbsp;&nbsp;&nbsp;&nbsp;BACKGROUND INFORMATION</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;"><a href="#setup">2&nbsp;&nbsp;&nbsp;&nbsp;SETUP</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;"><a href="#helper_functions">3&nbsp;&nbsp;&nbsp;&nbsp;HELPER FUNCTIONS</a></h3>
# 
# ---
# 
# <h3 style="text-indent: 10vw; font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: navy; background-color: #ffffff;"><a href="#create_dataset">4&nbsp;&nbsp;&nbsp;&nbsp;DATASET EXPLORATION</a></h3>
# 
# ---
# 

# <br>
# 
# <a id="imports"></a>
# 
# <h1 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; background-color: #ffffff; color: teal;" id="imports">0&nbsp;&nbsp;IMPORTS&nbsp;&nbsp;&nbsp;&nbsp;<a href="#toc">&#10514;</a></h1>

# In[ ]:


print("\n... IMPORTS STARTING ...\n")

print("\n\tVERSION INFORMATION")
# Machine Learning and Data Science Imports
import tensorflow as tf; print(f"\t\t– TENSORFLOW VERSION: {tf.__version__}");
import tensorflow_hub as tfhub; print(f"\t\t– TENSORFLOW HUB VERSION: {tfhub.__version__}");  # trained models
import tensorflow_addons as tfa; print(f"\t\t– TENSORFLOW ADDONS VERSION: {tfa.__version__}"); # community contributions
import pandas as pd; pd.options.mode.chained_assignment = None;
import numpy as np; print(f"\t\t– NUMPY VERSION: {np.__version__}");
import sklearn; print(f"\t\t– SKLEARN VERSION: {sklearn.__version__}");
from sklearn.preprocessing import RobustScaler, PolynomialFeatures  # Generate polynomial and interaction features.
from pandarallel import pandarallel; pandarallel.initialize();  # parallelerise pandas
from sklearn.model_selection import GroupKFold, StratifiedKFold
from scipy.spatial import cKDTree  # kd-tree is a space partitioning function

# # RAPIDS
# import cudf, cupy, cuml
# from cuml.neighbors import NearestNeighbors
# from cuml.manifold import TSNE, UMAP

# Built In Imports
from kaggle_datasets import KaggleDatasets
from collections import Counter
from datetime import datetime
from glob import glob # pathnames
import warnings
import requests
import hashlib
import imageio # read image data
import IPython # interactive shells
import sklearn
import urllib # working with urls
import zipfile # work with zip files
import pickle # save trained models
import random
import shutil # work with files
import string
import json # work with json data
import math
import time
import gzip # work with gzips
import ast # abstract syntax trees
import sys # system specific functions
import io
import os
import gc
import re

# Visualization Imports
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm; tqdm.pandas(); # progress bars
import plotly.express as px
import seaborn as sns
from PIL import Image, ImageEnhance
import matplotlib; print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}");
from matplotlib import animation, rc; rc('animation', html='jshtml')
import plotly
import PIL # python image library
import cv2 # OpenCV - image processing for computer vision

import plotly.io as pio
print(pio.renderers)

def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    
print("\n\n... IMPORTS COMPLETE ...\n")


# <br>
# 
# <a id="background_information"></a>
# 
# <h1 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; color: teal; background-color: #ffffff;" id="background_information">1&nbsp;&nbsp;BACKGROUND INFORMATION&nbsp;&nbsp;&nbsp;&nbsp;<a href="#toc">&#10514;</a></h1>
# 
# ---
# 

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">1.1 BASIC COMPETITION INFORMATION</h3>
# 
# ---
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">PRIMARY TASK DESCRIPTION</b>
# 
# In this competition, you’ll create a model to automatically segment the stomach and intestines on MRI scans. The MRI scans are from actual cancer patients who had 1-5 MRI scans on separate days during their radiation treatment. You'll base your algorithm on a dataset of these scans to come up with creative deep learning solutions that will help cancer patients get better care.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">BASIC BACKGROUND INFORMATION</b>
# 
# In 2019, an estimated 5 million people were diagnosed with a cancer of the gastro-intestinal tract worldwide. Of these patients, about half are eligible for radiation therapy, usually delivered over 10-15 minutes a day for 1-6 weeks. Radiation oncologists try to deliver high doses of radiation using X-ray beams pointed to tumors while avoiding the stomach and intestines. With newer technology such as integrated magnetic resonance imaging and linear accelerator systems, also known as MR-Linacs, <b>oncologists are able to visualize the daily position of the tumor and intestines, <mark>which can vary day to day</mark></b>. 
# 
# In these scans, radiation oncologists must manually outline the position of the stomach and intestines in order to adjust the direction of the x-ray beams to increase the dose delivery to the tumor and avoid the stomach and intestines. This is a time-consuming and labor intensive process that can prolong treatments from 15 minutes a day to an hour a day, which can be difficult for patients to tolerate—unless deep learning could help automate the segmentation process. <b><mark>A method to segment the stomach and intestines would make treatments much faster and would allow more patients to get more effective treatment.</mark></b>
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">COMPETITION HOST INFORMATION</b>
# 
# The UW-Madison Carbone Cancer Center is a pioneer in MR-Linac based radiotherapy, and has treated patients with MRI guided radiotherapy based on their daily anatomy since 2015. UW-Madison has generously agreed to support this project which provides anonymized MRIs of patients treated at the UW-Madison Carbone Cancer Center. The University of Wisconsin-Madison is a public land-grant research university in Madison, Wisconsin. The Wisconsin Idea is the university's pledge to the state, the nation, and the world that their endeavors will benefit all citizens.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">VISUAL EXPLANATION</b>
# 
# <center><img src="https://lh5.googleusercontent.com/zbBUgbj1jyZxyu3r1vr5zKKr8yK1hSdwAM3HpD_n6j2W-5-wKP3ZRusi_3yskSgnC-tMRKqOEtLycbLkTWCJAUe4Cylv_VsW81DYI4ray02uZLeSnlzAuZRIU7L2Q0KURYSMqFI"></center><br>
# 
# <i>The tumor above (pink thick line) is close to the stomach (red thick line). High doses of radiation are directed to the tumor while avoiding the stomach. Dose levels are represented by colour. Higher doses are represented by red and lower doses are represented by green.</i><br>
# 
# <br><center><img src="https://www.humonc.wisc.edu/wp-content/uploads/2017/09/Bayouth_Project4_72ppi.png"></center><br>
# 
# <i>MRI is an excellent imaging modality for visualization of soft tissues. This is particularly useful for tumors of the abdomen, such as pancreatic cancer shown below.  The left image shows the patient’s anatomy during exhale, while the image on the right shows the anatomical change during a maximum inspiration breath hold (MIBH). In the MIBH image we can see motion of nearly all the soft tissue, providing us superior ability to align the tumor during our treatment delivery. We are analyzing the clinical impact of using these treatment planning and delivery techniques and our patient’s ability to comply with self-guided breathing maneuvers.<b><a href="https://www.humonc.wisc.edu/research/medical-physics_research/mr-guided-radiation-therapy-research-2/">[REF]</a></b></i>
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">COMPETITION IMPACT STATEMENT</b>
# 
# Cancer takes enough of a toll. If successful, you'll enable radiation oncologists to safely deliver higher doses of radiation to tumors while avoiding the stomach and intestines. This will make cancer patients' daily treatments faster and allow them to get more effective treatment with less side effects and better long-term cancer control.

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">1.2 COMPETITION EVALUATION</h3>
# 
# ---
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">GENERAL EVALUATION INFORMATION</b>
# 
# This competition is evaluated on the mean <a href="https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient"><b>Dice coefficient</b></a> and <a href="https://github.com/scipy/scipy/blob/master/scipy/spatial/_hausdorff.pyx"><b>3D Hausdorff distance</b></a>. The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by:
# 
# $$
# \frac{2 * |X \cap Y|}{|X| + |Y|}
# $$
# 
# where $X$ is the predicted set of pixels and $Y$ is the ground truth. 
# * The Dice coefficient is defined to be $1$ when both $X$ and $Y$ are empty. 
# * The leaderboard score is the <b>mean of the Dice coefficients for each image in the test set.</b>
# 
# Hausdorff distance is a method for calculating the distance between segmentation objects A and B, by calculating the furthest point on object A from the nearest point on object B. For 3D Hausdorff, we construct 3D volumes by combining each 2D segmentation with slice depth as the Z coordinate and then find the Hausdorff distance between them. **(In this competition, the slice depth for all scans is set to 1.)** <a href="https://github.com/scipy/scipy/blob/master/scipy/spatial/_hausdorff.pyx"><b>The scipy code for Hausdorff is linked</b></a>. The expected / predicted pixel locations are normalized by image size to create a bounded 0-1 score.
# 
# <br>
#     
# ---
# 
# <b>NOTE: The two metrics are combined during evaluation!</b>
# 
# * <b>Weight of 0.4 for the Dice metric</b>
# * <b>Weight of 0.6 for the Hausdorff distance.</b>
# 
# ---
# 
# <br>
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">SUBMISSION FILE INFORMATION</b>
# 
# In order to reduce the submission file size, our metric uses **run-length encoding** on the pixel values.  
# * Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length
# * E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).
# * Note that, at the time of encoding, the mask should be **binary**
#     * The masks for all objects in an image are joined into a single large mask
#     * The value of **0** should indicate pixels that are not **masked**
#     * The value of **1** will indicate pixels that are **masked**.
# 
# The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.
# 
# <br>
# 
# The file should contain a header and have the following format:
# 
# ```
# id,class,predicted
# 1,large_bowel,1 1 5 1
# 1,small_bowel,1 1
# 1,stomach,1 1
# 2,large_bowel,1 5 2 17
# etc.
# ```
# 
# <br><font color="red"><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">IS THIS A CODE COMPETITION?</b></font>
# 
# <font color="red" style="font-size: 30px"><b>YES</b></font>
# 
# <br>

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">1.3 DATASET OVERVIEW</h3>
# 
# ---
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">GENERAL INFORMATION</b>
# 
# <b><mark>In this competition we are segmenting organs cells in images</mark></b>. 
# 
# The training **<mark>annotations are provided as RLE-encoded masks</mark>**, and the images are in **<mark>16-bit</mark>**, **<mark>grayscale</mark>**, **<mark>PNG format</mark>**.
# 
# Each case in this competition is represented by multiple sets of scan slices
# * Each set is identified by the day the scan took place
# * Some cases are split by time
#     * early days are in train
#     * later days are in test
# * Some cases are split by case
#     * the entirety of the case is in train or test
# 
# <b><mark>The goal of this competition is to be able to generalize to both partially and wholly unseen cases.</mark></b>
# 
# Note that, in this case, the test set is entirely unseen.
# * It is roughly 50 cases
# * It contains a varying number of days and slices, (similar to the training set)
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">FILE INFORMATION</b>
# 
# **`train.csv`** 
# - IDs and masks for all training objects.
# - **Columns**
#     * **`id`**
#         * unique identifier for object
#     * **`class`**
#         * the predicted class for the object
#     * **`EncodedPixels`**
#         * RLE-encoded pixels for the identified object
# 
# <br>
# 
# **`sample_submission.csv`**
# - A sample submission file in the correct format
# 
# <br>
# 
# **`train/`**
# - a folder of case/day folders, each containing slice images for a particular case on a given day.
# 
# <br>
# 
# <center><div class="alert alert-block alert-info" style="margin: 2em; line-height: 1.7em; font-family: Verdana;">
#     <b style="font-size: 18px;">⚠️ &nbsp; NOTE &nbsp; ⚠️</b><br><br><b style="font-size: 22px; color: darkorange"></b><br><br>The <b>image filenames</b> include 4 numbers <b>(ex. 276_276_1.63_1.63.png)</b>.<br><br>These four numbers are representative of:<ul><li><b>slice height</b> (integer in pixels)</li><li><b>slice width</b> (integer in pixels)</li><li><b>heigh pixel spacing</b> (floating point in mm)</li><li><b>width pixel spacing</b> (floating point in mm)</li></ul><br>The first two defines the resolution of the slide. The last two record the physical size of each pixel.<br><br>
# </div></center>
# 
#  
# 
# 

# <br>
# 
# <a id="background_information"></a>
# 
# <h1 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; color: teal; background-color: #ffffff;" id="setup">2&nbsp;&nbsp;SETUP&nbsp;&nbsp;&nbsp;&nbsp;<a href="#toc">&#10514;</a></h1>
# 
# ---
# 

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">2.1 ACCELERATOR DETECTION</h3>
# 
# ---

# In[ ]:


TPU=None


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">2.2 COMPETITION DATA ACCESS</h3>
# 
# ---
# 
# TPUs read data must be read directly from **G**oogle **C**loud **S**torage **(GCS)**. Kaggle provides a utility library – **`KaggleDatasets`** – which has a utility function **`.get_gcs_path`** that will allow us to access the location of our input datasets within **GCS**.<br><br>
# 
# <div class="alert alert-block alert-info" style="margin: 2em; line-height: 1.7em; font-family: Verdana;">
#     <b style="font-size: 16px;">📌 &nbsp; TIPS:</b><br><br>- If you have multiple datasets attached to the notebook, you should pass the name of a specific dataset to the <b><code>`get_gcs_path()`</code></b> function. <i>In our case, the name of the dataset is the name of the directory the dataset is mounted within.</i><br><br>
# </div>

# In[ ]:


print("\n... DATA ACCESS SETUP STARTED ...\n")

if TPU:
    # Google Cloud Dataset path to training and validation images
    DATA_DIR = KaggleDatasets().get_gcs_path('uw-madison-gi-tract-image-segmentation')
    save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    load_locally = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
else:
    # Local path to training and validation images
    DATA_DIR = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
    save_locally = None
    load_locally = None

print(f"\n... DATA DIRECTORY PATH IS:\n\t--> {DATA_DIR}")

print(f"\n... IMMEDIATE CONTENTS OF DATA DIRECTORY IS:")
for file in tf.io.gfile.glob(os.path.join(DATA_DIR, "*")): print(f"\t--> {file}")

print("\n\n... DATA ACCESS SETUP COMPLETED ...\n")


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">2.3 LEVERAGING XLA OPTIMIZATIONS</h3>
# 
# ---

# In[ ]:


'''
print(f"\n... XLA OPTIMIZATIONS STARTING ...\n")

print(f"\n... CONFIGURE JIT (JUST IN TIME) COMPILATION ...\n")
# enable XLA optmizations (10% speedup when using @tf.function calls)
tf.config.optimizer.set_jit(True)

print(f"\n... XLA OPTIMIZATIONS COMPLETED ...\n")
'''


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">2.4 BASIC DATA DEFINITIONS & INITIALIZATIONS</h3>
# 
# ---
# 

# In[ ]:


print("\n... BASIC DATA SETUP STARTING ...\n\n")

# Open the training dataframe and display the initial dataframe
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(TRAIN_CSV)

# Get all training images
all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)

print("\n... ORIGINAL TRAINING DATAFRAME... \n")
display(train_df)

TEST_DIR = os.path.join(DATA_DIR, "test")
SS_CSV   = os.path.join(DATA_DIR, "sample_submission.csv")
ss_df = pd.read_csv(SS_CSV)

# Get all testing images if there are any
all_test_images = glob(os.path.join(TEST_DIR, "**", "*.png"), recursive=True)

print("\n\n\n... ORIGINAL SUBMISSION DATAFRAME... \n")
display(ss_df)

# For debugging purposes when the test set hasn't been substituted we will know
DEBUG=len(ss_df)==0

if DEBUG:
    TEST_DIR = TRAIN_DIR
    all_test_images = all_train_images
    ss_df = train_df.iloc[:10]
    ss_df = ss_df[["id", "class"]]
    ss_df["predicted"] = ""
    
    print("\n\n\n... DEBUG SUBMISSION DATAFRAME... \n")
    display(ss_df)

    

SF2LF = {"lb":"Large Bowel","sb":"Small Bowel","st":"Stomach"}
LF2SF = {v:k for k,v in SF2LF.items()}
print(f"\n\n\n... ARE WE DEBUGGING: {DEBUG}... \n")

print("\n... BASIC DATA SETUP FINISHED ...\n\n")


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">2.5 UPDATE DATAFRAMES WITH ACCESSIBLE EXTRA INFORMATION</h3>
# 
# ---
# 
# **NOTE: I have changed the column identifiers as follows for the sake of brevity:**
# * **large_bowel** --> **lb**
# * **small_bowel** --> **sb**
# * **stomach** --> **st**

# In[ ]:


print("\n... UPDATING DATAFRAMES WITH ACCESSIBLE INFORMATION STARTED ...\n\n")

# 1. Get Case-ID as a column (str and int)
train_df["case_id_str"] = train_df["id"].apply(lambda x: x.split("_", 2)[0])
train_df["case_id"] = train_df["id"].apply(lambda x: int(x.split("_", 2)[0].replace("case", "")))

# 2. Get Day as a column
train_df["day_num_str"] = train_df["id"].apply(lambda x: x.split("_", 2)[1])
train_df["day_num"] = train_df["id"].apply(lambda x: int(x.split("_", 2)[1].replace("day", "")))

# 3. Get Slice Identifier as a column
train_df["slice_id"] = train_df["id"].apply(lambda x: x.split("_", 2)[2])

# 4. Get full file paths for the representative scans
train_df["_partial_ident"] = (TRAIN_DIR+"/"+ # /kaggle/input/uw-madison-gi-tract-image-segmentation/train/
                             train_df["case_id_str"]+"/"+ # .../case###/
                             train_df["case_id_str"]+"_"+train_df["day_num_str"]+ # .../case###_day##/
                             "/scans/"+train_df["slice_id"]) # .../slice_#### 
_tmp_merge_df = pd.DataFrame({"_partial_ident":[x.rsplit("_",4)[0] for x in all_train_images], "f_path":all_train_images})
train_df = train_df.merge(_tmp_merge_df, on="_partial_ident").drop(columns=["_partial_ident"])

# Minor cleanup of our temporary workaround
del _tmp_merge_df; gc.collect(); gc.collect()

# 5. Get slice dimensions from filepath (int in pixels)
train_df["slice_h"] = train_df["f_path"].apply(lambda x: int(x[:-4].rsplit("_",4)[1]))
train_df["slice_w"] = train_df["f_path"].apply(lambda x: int(x[:-4].rsplit("_",4)[2]))

# 6. Pixel spacing from filepath (float in mm)
train_df["px_spacing_h"] = train_df["f_path"].apply(lambda x: float(x[:-4].rsplit("_",4)[3]))
train_df["px_spacing_w"] = train_df["f_path"].apply(lambda x: float(x[:-4].rsplit("_",4)[4]))

# 7. Merge 3 Rows Into A Single Row (As This/Segmentation-RLE Is The Only Unique Information Across Those Rows)
l_bowel_train_df = train_df[train_df["class"]=="large_bowel"][["id", "segmentation"]].rename(columns={"segmentation":"lb_seg_rle"})
s_bowel_train_df = train_df[train_df["class"]=="small_bowel"][["id", "segmentation"]].rename(columns={"segmentation":"sb_seg_rle"})
stomach_train_df = train_df[train_df["class"]=="stomach"][["id", "segmentation"]].rename(columns={"segmentation":"st_seg_rle"})
train_df = train_df.merge(l_bowel_train_df, on="id", how="left")
train_df = train_df.merge(s_bowel_train_df, on="id", how="left")
train_df = train_df.merge(stomach_train_df, on="id", how="left")
train_df = train_df.drop_duplicates(subset=["id",]).reset_index(drop=True)
train_df["lb_seg_flag"] = train_df["lb_seg_rle"].apply(lambda x: not pd.isna(x))
train_df["sb_seg_flag"] = train_df["sb_seg_rle"].apply(lambda x: not pd.isna(x))
train_df["st_seg_flag"] = train_df["st_seg_rle"].apply(lambda x: not pd.isna(x))
train_df["n_segs"] = train_df["lb_seg_flag"].astype(int)+train_df["sb_seg_flag"].astype(int)+train_df["st_seg_flag"].astype(int)

# 8. Reorder columns to the a new ordering (drops class and segmentation as no longer necessary)
train_df = train_df[["id", "f_path", "n_segs",
                     "lb_seg_rle", "lb_seg_flag",
                     "sb_seg_rle", "sb_seg_flag", 
                     "st_seg_rle", "st_seg_flag",
                     "slice_h", "slice_w", "px_spacing_h", 
                     "px_spacing_w", "case_id_str", "case_id", 
                     "day_num_str", "day_num", "slice_id",]]

# 9. Display update dataframe
print("\n... UPDATED TRAINING DATAFRAME... \n")
display(train_df)


# <br>
# 
# 
# <a id="helper_functions"></a>
# 
# 
# <h1 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; color: teal; background-color: #ffffff;" id="helper_functions">
#     3&nbsp;&nbsp;HELPER FUNCTION & CLASSES&nbsp;&nbsp;&nbsp;&nbsp;<a href="#toc">&#10514;</a>
# </h1>
# 
# ---

# In[ ]:


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# modified from: https://www.kaggle.com/inversion/run-length-decoding-quick-start
def rle_decode(mask_rle, shape, color=1):
    """ TBD
    
    Args:
        mask_rle (str): run-length as string formated (start length)
        shape (tuple of ints): (height,width) of array to return 
    
    Returns: 
        Mask (np.array)
            - 1 indicating mask
            - 0 indicating background

    """
    # Split the string by space, then convert it into a integer array
    s = np.array(mask_rle.split(), dtype=int)

    # Every even value is the start, every odd value is the "run" length
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    # The image is actually flattened since RLE is a 1D "run"
    if len(shape)==3:
        h, w, d = shape
        img = np.zeros((h * w, d), dtype=np.float32)
    else:
        h, w = shape
        img = np.zeros((h * w,), dtype=np.float32)

    # The color here is actually just any integer you want!
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
        
    # Don't forget to change the image back to the original shape
    return img.reshape(shape)

# https://www.kaggle.com/namgalielei/which-reshape-is-used-in-rle
def rle_decode_top_to_bot_first(mask_rle, shape):
    """ TBD
    
    Args:
        mask_rle (str): run-length as string formated (start length)
        shape (tuple of ints): (height,width) of array to return 
    
    Returns:
        Mask (np.array)
            - 1 indicating mask
            - 0 indicating background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0]), order='F').T  # Reshape from top -> bottom first

# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    """ TBD
    
    Args:
        img (np.array): 
            - 1 indicating mask
            - 0 indicating background
    
    Returns: 
        run length as string formated
    """
    
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def flatten_l_o_l(nested_list):
    """ Flatten a list of lists """
    return [item for sublist in nested_list for item in sublist]

def load_json_to_dict(json_path):
    """ tbd """
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data

def tf_load_png(img_path):
    return tf.image.decode_png(tf.io.read_file(img_path), channels=3)

def open_gray16(_path, normalize=True, to_rgb=False):
    """ Helper to open files """
    if normalize:
        if to_rgb:
            return np.tile(np.expand_dims(cv2.imread(_path, cv2.IMREAD_ANYDEPTH)/65535., axis=-1), 3)
        else:
            return cv2.imread(_path, cv2.IMREAD_ANYDEPTH)/65535.
    else:
        # np.expand_dims turns (266,266) into (266,266,1)
        # np.tile turns (266,266,1) into (266,266,3) (1st frame copied 3 times)
        if to_rgb:
            return np.tile(np.expand_dims(cv2.imread(_path, cv2.IMREAD_ANYDEPTH), axis=-1), 3)
        else:
            return cv2.imread(_path, cv2.IMREAD_ANYDEPTH)


# <br>
# 
# 
# <a id="create_dataset"></a>
# 
# 
# <h1 style="font-family: Verdana; font-size: 24px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; color: teal; background-color: #ffffff;" id="create_dataset">
#     4&nbsp;&nbsp;DATASET EXPLORATION&nbsp;&nbsp;&nbsp;&nbsp;<a href="#toc">&#10514;</a>
# </h1>
# 
# ---

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">4.0 LOOK AT A SINGLE EXAMPLE PRIOR TO INVESTIGATION</h3>
# 
# ---
# 
# We simply do this to make sure everything is where it should be and we understand the basics of how to access all the relevant data.
# 
# We will wrap this basic exploration functionality as single function to allow for easy examination of any passed identifier

# In[ ]:


def get_overlay(img_path, rle_strs, img_shape, _alpha=0.999, _beta=0.35, _gamma=0):
    _img = open_gray16(img_path, to_rgb=True)
    _img = ((_img-_img.min())/(_img.max()-_img.min())).astype(np.float32)
    _seg_rgb = np.stack([rle_decode(rle_str, shape=img_shape, color=1) if rle_str is not None else np.zeros(img_shape, dtype=np.float32) for rle_str in rle_strs], axis=-1).astype(np.float32)
    seg_overlay = cv2.addWeighted(src1=_img, alpha=_alpha, 
                                  src2=_seg_rgb, beta=_beta, gamma=_gamma)
    return seg_overlay

def examine_id(ex_id, df=train_df, plot_overlay=True, print_meta=False, plot_grayscale=False, plot_binary_segmentation=False):
    """ Wrapper function to allow for easy visual exploration of an example """
    print(f"\n... ID ({ex_id}) EXPLORATION STARTED ...\n\n")
    demo_ex = df[df.id==ex_id].squeeze()

    if print_meta:
        print(f"\n... WITH DEMO_ID=`{DEMO_ID}` WE HAVE THE FOLLOWING DEMO EXAMPLE TO WORK FROM... \n\n")
        display(demo_ex.to_frame())

    if plot_grayscale:
        print(f"\n\n... GRAYSCALE IMAGE PLOT ...\n")
        plt.figure(figsize=(12,12))
        plt.imshow(open_gray16(demo_ex.f_path), cmap="gray")
        plt.title(f"Original Grayscale Image For ID: {demo_ex.id}", fontweight="bold")
        plt.axis(False)
        plt.show()

    if plot_binary_segmentation:
        print(f"\n\n... BINARY SEGMENTATION MASKS ...\n")
        plt.figure(figsize=(20,10))
        for i, _seg_type in enumerate(["lb", "sb", "st"]):
            if pd.isna(demo_ex[f"{_seg_type}_seg_rle"]): continue
            plt.subplot(1,3,i+1)
            plt.imshow(rle_decode(demo_ex[f"{_seg_type}_seg_rle"], shape=(demo_ex.slice_w, demo_ex.slice_h), color=1))
            plt.title(f"RLE Encoding For {SF2LF[_seg_type]} Segmentation", fontweight="bold")
            plt.axis(False)
        plt.tight_layout()
        plt.show()

    if plot_overlay:
        print(f"\n\n... IMAGE WITH RGB SEGMENTATION MASK OVERLAY ...\n")
        # We need to normalize the loaded image values to be between 0 and 1 or else our plot will look weird
        # _img = open_gray16(demo_ex.f_path, to_rgb=True)
        #_img = ((_img-_img.min())/(_img.max()-_img.min())).astype(np.float32)
        #_seg_rgb = np.stack([rle_decode(demo_ex[f"{_seg_type}_seg_rle"], shape=(demo_ex.slice_w, demo_ex.slice_h), color=1) if not pd.isna(demo_ex[f"{_seg_type}_seg_rle"]) else np.zeros((demo_ex.slice_w, demo_ex.slice_h)) for _seg_type in ["lb", "sb", "st"]], axis=-1).astype(np.float32)
        #seg_overlay = cv2.addWeighted(src1=_img, alpha=0.99, 
                                      #src2=_seg_rgb, beta=0.33, gamma=0)
        _rle_strs = [demo_ex[f"{_seg_type}_seg_rle"] if not pd.isna(demo_ex[f"{_seg_type}_seg_rle"]) else None for _seg_type in ["lb", "sb", "st"]]
        seg_overlay = get_overlay(demo_ex.f_path, _rle_strs, img_shape=(demo_ex.slice_w, demo_ex.slice_h))

        plt.figure(figsize=(12,12))
        plt.imshow(seg_overlay)
        plt.title(f"Segmentation Overlay For ID: {demo_ex.id}", fontweight="bold")
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = ["Large Bowel Segmentation Map", "Small Bowel Segmentation Map", "Stomach Segmentation Map"]
        plt.legend(handles,labels)
        plt.axis(False)
        plt.show()

    print("\n\n... SINGLE ID EXPLORATION FINISHED ...\n\n")


# In[ ]:


print("\n... SINGLE ID EXPLORATION STARTED ...\n\n")

DEMO_ID = "case123_day20_slice_0082"
demo_ex = train_df[train_df.id==DEMO_ID].squeeze()

print(f"\n... WITH DEMO_ID=`{DEMO_ID}` WE HAVE THE FOLLOWING DEMO EXAMPLE TO WORK FROM... \n\n")
display(demo_ex.to_frame())

print(f"\n\n... LET'S PLOT THE IMAGE FIRST ...\n")
plt.figure(figsize=(12,12))
plt.imshow(open_gray16(demo_ex.f_path), cmap="gray")  # show grayscale image
plt.title(f"Original Grayscale Image For ID: {demo_ex.id}", fontweight="bold")
plt.axis(False)
plt.show()

print(f"\n\n... LET'S PLOT THE 3 SEGMENTATION MASKS ...\n")

plt.figure(figsize=(20,10))
for i, _seg_type in enumerate(["lb", "sb", "st"]):
    if pd.isna(demo_ex[f"{_seg_type}_seg_rle"]): continue # skip iteration if no mask available
    plt.subplot(1,3,i+1)
    plt.imshow(rle_decode(demo_ex[f"{_seg_type}_seg_rle"], shape=(demo_ex.slice_w, demo_ex.slice_h), color=1)) # decode rle first
    plt.title(f"RLE Encoding For {SF2LF[_seg_type]} Segmentation", fontweight="bold")
    plt.axis(False)
plt.tight_layout()
plt.show()

print(f"\n\n... LET'S PLOT THE IMAGE WITH AN RGB SEGMENTATION MASK OVERLAY ...\n")

# We need to normalize the loaded image values to be between 0 and 1 or else our plot will look weird
_img = open_gray16(demo_ex.f_path, to_rgb=True)
_img = ((_img-_img.min())/(_img.max()-_img.min())).astype(np.float32)
_seg_rgb = np.stack([rle_decode(demo_ex[f"{_seg_type}_seg_rle"], shape=(demo_ex.slice_w, demo_ex.slice_h), color=1) if not pd.isna(demo_ex[f"{_seg_type}_seg_rle"]) else np.zeros((demo_ex.slice_w, demo_ex.slice_h)) for _seg_type in ["lb", "sb", "st"]], axis=-1).astype(np.float32) # shape (266,266,3)
seg_overlay = cv2.addWeighted(src1=_img, alpha=0.99, 
                              src2=_seg_rgb, beta=0.33, gamma=0.0) # linear blending: alpha*src1+beta*src2+gamma

# Note: (R = G = B) => gray value. This is why _img looks gray whilst _seg_rgb is rgb.

plt.figure(figsize=(12,12))
plt.imshow(seg_overlay)
plt.title(f"Segmentation Overlay For ID: {demo_ex.id}", fontweight="bold")
handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
labels = ["Large Bowel Segmentation Map", "Small Bowel Segmentation Map", "Stomach Segmentation Map"]
plt.legend(handles,labels)
plt.axis(False)
plt.show()

print(f"\n\n... LET'S PRINT THE RELEVANT INFORMATION ...\n")
print(f"\t--> IMAGE CASE ID              : {demo_ex.case_id}")
print(f"\t--> IMAGE DAY NUMBER           : {demo_ex.day_num}")
print(f"\t--> IMAGE SLICE WIDTH          : {demo_ex.slice_w}")
print(f"\t--> IMAGE SLICE HEIGHT         : {demo_ex.slice_h}")
print(f"\t--> IMAGE PIXEL SPACING WIDTH  : {demo_ex.px_spacing_w}")
print(f"\t--> IMAGE PIXEL SPACING HEIGHT : {demo_ex.px_spacing_h}")

print("\n\n... SINGLE ID EXPLORATION FINISHED ...\n\n")


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">4.1 INVESTIGATE THE OCCURENCE SEGMENTATION MAP TYPES</h3>
# 
# ---
# 
# It's quite apparent that not all images have segmentation maps for the various regions (stomach, large-bowel, small-bowel), so we will identify the frequency for which these occur independently... as well as the frequency for which these maps co-occur.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">OBSERVATIONS</b>
# 
# * There are **38,496** total examples.
# * It can be observed that more than half of the given examples have no annotations present!
#     * There are **21,906** (56.9046%) examples with no annotations/masks/segmentation present
#     * Inversely there are **16,590** (43.0954%) examples with one or more annotations present
# * There are **2,468** (6.41%) examples with **one annotation present**. 
# * It can be observed that the vast majority of single mask annotations are **Stomach**!
#     * Of these annotations, **2286** (~92.6%) are **Stomach**
#     * Of these annotations, **123** (~4.98%) are **Large Bowel**
#     * Of these annotations, **59** (~2.39%) are **Small Bowel**
# * There are **10,921** (28.37%) examples with **two annotations present**. 
# * It can be observed, in contrast to the single annotation examples, that the majority of annotations do NOT include stomach i.e. **'Large Bowel, Small Bowel'**!
#     * Of these annotations, **7781** (~71.3%) are **'Large Bowel, Small Bowel'**
#     * Of these annotations, **2980** (~27.3%) are **'Large Bowel, Stomach'**
#     * Of these annotations, **160** (~1.47%) are **'Small Bowel, Stomach'**
# * Finally, there are **3,201** (8.32%) examples with **all three annotations present**. 
# 
# <!--  # print(len(train_df))
# # print(len(train_df[train_df["seg_combo_str"]=="No Mask"]), len(train_df[train_df["seg_combo_str"]=="No Mask"])/len(train_df))
# # print(len(train_df[train_df["seg_combo_str"]=="Large Bowel"])/2468)
# # print(len(train_df[train_df["seg_combo_str"]=="Small Bowel"])/2468)
# # print(len(train_df[train_df["seg_combo_str"]=="Stomach"])/2468)
# # print(len(train_df[train_df["seg_combo_str"].apply(lambda x: x.count(",")==1)]))
# # print(len(train_df[train_df["seg_combo_str"]=="Large Bowel, Stomach"])/10921)
# # print(len(train_df[train_df["seg_combo_str"]=="Large Bowel, Small Bowel"])/10921)
# # print(len(train_df[train_df["seg_combo_str"]=="Small Bowel, Stomach"])/10921)
# # print(len(train_df[train_df["seg_combo_str"].apply(lambda x: x.count(",")==2)])) -->

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">4.2 INVESTIGATE THE IMAGE SIZES</h3>
# 
# ---
# 
# It's observable that not all images have the same size... however, given that, there is not that much variation between image slice sizes.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">OBSERVATIONS</b>
# * Remember, there are **38,496** total examples.
# * Globally, we can see that 3 of the image shapes are **square** while one is **rectangular** and they all fall within a fairly tight distribution of relatively small sizes
# * Of these there are **4** unique sizes:
#     * $234 \times 234$
#         * **Least frequent** image size
#         * **Smallest** image size
#         * Only **144** of the 38,496 occurences are this size (0.37%)
#     * $266 \times 266$
#         * **Most frequent** image size
#         * **Second smallest** image size
#         * **25,920** of the 38,496 occurences are this size (67.33%)
#     * $276 \times 276$
#         * **Second least frequent** image size
#         * **Second largest** image size
#         * **1,200** of the 38,496 occurences are this size (3.12%)
#     * $310 \times 360$
#         * **Second most frequent** image size
#         * **Largest** image size
#         * **11,232** of the 38,496 occurences are this size (29.17%)
# 
# 
# 
# 

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">4.3 INVESTIGATE THE PIXEL SPACING</h3>
# 
# ---
# 
# It's observable that not all images have the same pixel spacing... however, given that, there is not that much variation between pixel spacing.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">OBSERVATIONS</b>
# * Remember, there are **38,496** total examples.
# * Globally, we can see that all of the pixel spacings are **square** and that the vast majority are $1.50mm \times 1.50mm$
# * There are only **2** unique sets of pixel spacings:
#     * $1.50mm \times 1.50mm$
#         * **Most frequent** pixel spacing
#         * **Smallest** pixel spacing (barely)
#         * **37,296** of the 38,496 occurences are this size (96.88%)
#     * $1.63mm \times 1.63mm$
#         * **Least frequent** image size
#         * **Largest** pixel spacing (barely)
#         * **1,200** of the 38,496 occurences are this size (3.12%)

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">4.4 INVESTIGATE CASE IDS</h3>
# 
# ---
# 
# Here's the host description of **`case_id`**
# 
# > "Each case in this competition is represented by multiple sets of scan slices (each set is identified by the day the scan took place). Some cases are split by time (early days are in train, later days are in test) while some cases are split by case - the entirety of the case is in train or test. The goal of this competition is to be able to generalize to both partially and wholly unseen cases."
# 
# I don't really observe any oddities associated with any particular **`case_id`** values. I would probably attempt to group them when stratifying/creating-folds... however, they don't seem to perpetrate an obvious bias.
# 
# When we colour by **day**, we can see that all cases are made up (mostly) of groups of **144**, or less frequently, **80**, images from different days.

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">4.5 MASK SIZES/AREAS</h3>
# 
# ---
# 
# We know that every other number in an RLE encoding represents a run of mask... so if we add up all those numbers we get the total number of masked pixels in an image. This is much faster than opening and closing each image.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">OBSERVATIONS</b>
# 
# * It's observable that the distributions of mask area is mostly normal although it skews slightly to the smaller side...
# * All the distributions are similar although the Stomach distribution has an odd gap between 400-750 pixels.
# * It's interesting to note that, while not common, we do have some VERY large masks (>7500 pixels)
#     * Also, it's kind of funny that the biggest masks are for **small** bowel
#     

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">4.6 MASK DATASET CREATION, CLASS OVERLAP & MASK HEATMAP</h3>
# 
# ---
# 
# It's important to determine if the the masks overlap one another (**multilabel**) or not (**multiclass**). To do this, we will quickly create a dataset of **`npy`** files. During this creation process we will check for overlap.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">OBSERVATIONS</b>
# 
# * There is overlap, and while it is not that common, some images exhibit a high degree of overlap.
# * This means that we cannot frame the problem as simple categorical semantic segmentation.
# * We must instead frame the problem as multi-label semantic segmentation
# * This means our mask will take the form --> $W \times H \times 3$
#     * Where the channel dimensions are binary masks for each respective segmentation type
#     * This will allow for the masks to overlap

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">4.7 PIXEL VALUES IN OUR DATASET</h3>
# 
# ---
# 
# It's important to analyse the dataset because we will need to normalize the data to convert it into a format that is more expected for machine learning (uint8 (0-255) or float32 (0-1)). Without knowing the limits of the images, we may diminish the resolution of the data by accident when normalizing.
# 
# <br><b style="text-decoration: underline; font-family: Verdana; text-transform: uppercase;">OBSERVATIONS</b>
# 
# Interestingly the maximum value in the dataset is equiavlent to less than half of an int16 or a quarter of a uint16.
# * Max Value for UINT16
#     * **65535**
# * Max Value for INT16
#     * **32767**
# * Half of Max Value for INT16
#     * **16384**
# * Actual Max Value in the dataset
#     * **15865**

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">4.8 IDENTIFY ANY HEURISTICS OR RULES REGARDING SEGMENTATION</h3>
# 
# ---
# 
# For a given **case-id** and **day number** there are two different amounts of scans present
# * 144 slices --> 259 instances
# * 80 slices ---> 15 instances
# 
# Some other observations about our training dataset
# * There are no examples for slices number **1, 138, 139, 140, 141, 142, 143 or 144** that have any segmentation masks
# * If we break it down by organ we get the following no-value slices for each respective organ
#     * Large Bowel – **1, 138, 139, 140, 141, 142, 143, 144**
#     * Small Bowel – **1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 138, 139, 140, 141, 142, 143, 144**
#     * Stomach – **1, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144**

# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">4.9 CREATE 3D GIF FOR CASE GROUPS OF SLICES (WITH MASK!!)</h3>
# 
# ---
# 

# In[ ]:


def get_overlay(img_path, rle_strs, img_shape, _alpha=0.999, _beta=0.35, _gamma=0):
    _img = open_gray16(img_path, to_rgb=True)
    _img = ((_img-_img.min())/(_img.max()-_img.min())).astype(np.float32)
    _seg_rgb = np.stack([rle_decode(rle_str, shape=img_shape, color=1) if (rle_str is not None and not pd.isna(rle_str)) else np.zeros(img_shape, dtype=np.float32) for rle_str in rle_strs], axis=-1).astype(np.float32)
    seg_overlay = cv2.addWeighted(src1=_img, alpha=_alpha, 
                                  src2=_seg_rgb, beta=_beta, gamma=_gamma)
    return seg_overlay


# In[ ]:


# Multiple animations simultaneously
def case_animation(case_id, day_nums=True, df=train_df):
    # Day info
    if day_nums:
        days = np.sort(df[df['case_id']==case_id]['day_num'].unique()) # Show all days
    else:
        days = np.sort(day_nums.unique()) # Show subset of days
    n_days = len(days)
    
    # Loop over each day
    for index, day_num in enumerate(days):
        # Desired subset
        sub_df = df[(df.case_id==case_id) & (df.day_num==day_num)]
        
        # Metadata
        f_paths  = sub_df.f_path.tolist()
        lb_rles  = sub_df.lb_seg_rle.tolist()
        sb_rles  = sub_df.sb_seg_rle.tolist()
        st_rles  = sub_df.st_seg_rle.tolist()
        slice_ws = sub_df.slice_w.tolist()
        slice_hs = sub_df.slice_h.tolist()
        
        # Images stacked together
        animation_arr = np.stack([
            get_overlay(img_path=_f, rle_strs=(_lb, _sb, _st), img_shape=(_w, _h)) \
            for _f, _lb, _sb, _st, _w, _h in \
            zip(f_paths, lb_rles, sb_rles, st_rles, slice_ws, slice_hs)
        ], axis=0)
        
        # Images can have different sizes between days
        if index==0:
            animation_arr1 = animation_arr
        elif index==1:
            animation_arr2 = animation_arr
        elif index==2:
            animation_arr3 = animation_arr
        elif index==3:
            animation_arr4 = animation_arr
        elif index==4:
            animation_arr5 = animation_arr
        elif index==5:
            animation_arr6 = animation_arr
            
    # Initialise plot sizes
    if n_days==1:
        fig, (ax1) = plt.subplots(1,1)
        fig.set_figheight(4)
        fig.set_figwidth(4)
    elif n_days==2:
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_figheight(4)
        fig.set_figwidth(8)
    elif n_days==3:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        fig.set_figheight(4)
        fig.set_figwidth(12)
    elif n_days==4:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
        fig.set_figheight(6) # gifs get truncated if size is too large (on kaggle)
        fig.set_figwidth(6)
    elif n_days==5:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)
        fig.set_figheight(6)
        fig.set_figwidth(9)
    elif n_days==6:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)
        fig.set_figheight(6)
        fig.set_figwidth(9)
        
    # Initialise plots
    frame_min=1e6
    if n_days>=1:
        ax1.axis('off')
        im1 = ax1.imshow(animation_arr1[0])
        ax1.title.set_text(f"Day {days[0]}")
        frame_min = min(frame_min,animation_arr1.shape[0])
    if n_days>=2:
        ax2.axis('off')
        im2 = ax2.imshow(animation_arr2[0])
        ax2.title.set_text(f"Day {days[1]}")
        frame_min = min(frame_min,animation_arr2.shape[0])
    if n_days>=3:
        ax3.axis('off')
        im3 = ax3.imshow(animation_arr3[0])
        ax3.title.set_text(f"Day {days[2]}")
        frame_min = min(frame_min,animation_arr3.shape[0])
    if n_days>=4:
        ax4.axis('off')
        im4 = ax4.imshow(animation_arr4[0])
        ax4.title.set_text(f"Day {days[3]}")
        frame_min = min(frame_min,animation_arr4.shape[0])
    if n_days>=5:
        ax5.axis('off')
        ax6.axis('off')
        im5 = ax5.imshow(animation_arr5[0])
        ax5.title.set_text(f"Day {days[4]}")
        frame_min = min(frame_min,animation_arr5.shape[0])
    if n_days==6:
        im6 = ax6.imshow(animation_arr6[0])
        ax6.title.set_text(f"Day {days[5]}")
        frame_min = min(frame_min,animation_arr6.shape[0])
    
    # Set overall title
    fig.suptitle(f"3D Animation for Case {case_id}", fontweight="bold")
    
    # Animate function
    def animate_func(i):
        out=[]
        if n_days>=1:
            im1.set_array(animation_arr1[i])
            out.append(im1)
        if n_days>=2:
            im2.set_array(animation_arr2[i])
            out.append(im2)
        if n_days>=3:
            im3.set_array(animation_arr3[i])
            out.append(im3)
        if n_days>=4:
            im4.set_array(animation_arr4[i])
            out.append(im4)
        if n_days>=5:
            im5.set_array(animation_arr5[i])
            out.append(im5)
        if n_days>=6:
            im6.set_array(animation_arr6[i])
            out.append(im6)
        return out
    plt.close()
    
    return animation.FuncAnimation(fig, animate_func, frames = frame_min, interval = 100)


# **Examples**

# In[ ]:


# Example with 1 day side-to-side
#case_animation(case_id=16)

# Example with 2 days side-by-side
#case_animation(case_id=33)

# Example with 3 days side-by-side
anim1 = case_animation(case_id=149)
anim1


# In[ ]:


# Example with 4 days side-by-side
#case_animation(case_id=139)

# Example with 5 days side-by-side
#case_animation(case_id=122)

# Example with 6 days side-by-side
anim2=case_animation(case_id=36)
anim2


# In[ ]:


# Save gifs
get_ipython().system('pip install imagemagick')

anim1.save('case149.gif', writer='imagemagick')
anim2.save('case36.gif', writer='imagemagick')


# <br>
# 
# <h3 style="font-family: Verdana; font-size: 20px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: teal; background-color: #ffffff;">4.10 SCANS WITH ERRORS</h3>
# 
# ---
# 
# * [**Paul G**](https://www.kaggle.com/pgeiger) identified two cases with errors in the segmentation masks in [**this discussion post**](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/319963)
#   * Case 7
#       * Day 0
# 
# <img src="https://i.ibb.co/M8p8Xfk/case7-day0-slice-0096.png">
# 
#   * Case 81
#       * Day 30
#       
# <img src="https://i.ibb.co/jkdcdzR/case81-day30-slice-0096.png">
