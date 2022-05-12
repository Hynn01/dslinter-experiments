#!/usr/bin/env python
# coding: utf-8

# # <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;text-align:center">IMPORT</p></div>

# In[ ]:


print("\n... IMPORTS STARTING ...\n")

print("\n\tVERSION INFORMATION")
# Machine Learning and Data Science Imports
import tensorflow as tf; print(f"\t\t– TENSORFLOW VERSION: {tf.__version__}");
import tensorflow_hub as tfhub; print(f"\t\t– TENSORFLOW HUB VERSION: {tfhub.__version__}");
import tensorflow_addons as tfa; print(f"\t\t– TENSORFLOW ADDONS VERSION: {tfa.__version__}");
import pandas as pd; pd.options.mode.chained_assignment = None;
import numpy as np; print(f"\t\t– NUMPY VERSION: {np.__version__}");
import sklearn; print(f"\t\t– SKLEARN VERSION: {sklearn.__version__}");
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from pandarallel import pandarallel; pandarallel.initialize();
from sklearn.model_selection import GroupKFold, StratifiedKFold
from scipy.spatial import cKDTree

# # RAPIDS
# import cudf, cupy, cuml
# from cuml.neighbors import NearestNeighbors
# from cuml.manifold import TSNE, UMAP

# Built In Imports
from kaggle_datasets import KaggleDatasets
from collections import Counter
from datetime import datetime
from glob import glob
import warnings
import requests
import hashlib
import imageio
import IPython
import sklearn
import urllib
import zipfile
import pickle
import random
import shutil
import string
import json
import math
import time
import gzip
import ast
import sys
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
from tqdm.notebook import tqdm; tqdm.pandas();
import plotly.express as px
import seaborn as sns
from PIL import Image, ImageEnhance
import matplotlib; print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}");
from matplotlib import animation, rc, gridspec 
rc('animation', html='jshtml')
import plotly
import PIL
import cv2

import plotly.io as pio
print(pio.renderers)

def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    
print("\n\n... IMPORTS COMPLETE ...\n")


# # <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;text-align:center">SETUP</p></div>
# ****

# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">ACCELERATOR DETECTION</p></div>

# In[ ]:


print(f"\n... ACCELERATOR SETUP STARTING ...\n")

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()  
except ValueError:
    TPU = None

if TPU:
    print(f"\n... RUNNING ON TPU - {TPU.master()}...")
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    strategy = tf.distribute.experimental.TPUStrategy(TPU)
else:
    print(f"\n... RUNNING ON CPU/GPU ...")
    # Yield the default distribution strategy in Tensorflow
    #   --> Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy() 

# What Is a Replica?
#    --> A single Cloud TPU device consists of FOUR chips, each of which has TWO TPU cores. 
#    --> Therefore, for efficient utilization of Cloud TPU, a program should make use of each of the EIGHT (4x2) cores. 
#    --> Each replica is essentially a copy of the training graph that is run on each core and 
#        trains a mini-batch containing 1/8th of the overall batch size
N_REPLICAS = strategy.num_replicas_in_sync
    
print(f"... # OF REPLICAS: {N_REPLICAS} ...\n")

print(f"\n... ACCELERATOR SETUP COMPLTED ...\n")


# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">LEVERAGING XLA OPTIMIZATIONS</p></div>

# In[ ]:


print(f"\n... XLA OPTIMIZATIONS STARTING ...\n")

print(f"\n... CONFIGURE JIT (JUST IN TIME) COMPILATION ...\n")
# enable XLA optmizations (10% speedup when using @tf.function calls)
tf.config.optimizer.set_jit(True)

print(f"\n... XLA OPTIMIZATIONS COMPLETED ...\n")


# # <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;text-align:center">CONFIGURATION and HANDING FUNCTIONS</p></div>
# ****

# In[ ]:


class CFG:
    EPOCHS = 30
    DEMO_ID = "case123_day20_slice_0082"
    DEMO_CASE = 134
    NPY_DIR = "/kaggle/working/npy_files"


# In[ ]:


def preprocessing_df(df, globbed_file_list, is_test=False):
    # 1. Get Case-ID as a column (str and int)
    df["case_id_str"] = df.id.apply(lambda x : x.split("_",2)[0])
    df["case_id"] = df.case_id_str.apply(lambda x : int(x.replace("case","")))

    # 2. Get Date as a column
    df["day_num_str"] = df.id.apply(lambda x : x.split("_",2)[1])
    df["day_num"] = df.day_num_str.apply(lambda x : int(x.replace("day","")))

    # 3. Get Slide Identifier as a column
    df["slice_id"] = df.id.apply(lambda x : x.split("_",2)[2])

    # 4. Get full file paths for the representative scans
    df["_partial_ident"] = (globbed_file_list[0].rsplit("/",4)[0]+"/"+ # /kaggle/input/uw-madison-gi-tract-image-segmentation/train/
                                 df["case_id_str"]+"/"+ # .../case###/
                                 df["case_id_str"]+"_"+ df["day_num_str"]+ # .../case###_day##/
                                 "/scans/"+
                                 df["slice_id"]) # .../slice_#### 
    _tmp_merge_df = pd.DataFrame({"_partial_ident":[x.rsplit("_",4)[0] for x in globbed_file_list], "f_path":globbed_file_list})
    df = df.merge(_tmp_merge_df, on="_partial_ident").drop(columns=["_partial_ident"])

    # 5. Get slice dimensions from filepath (int in pixels)
    df["slice_h"] = df.f_path.apply(lambda x: int(x[:-4].split("_")[3]))
    df["slice_w"] = df.f_path.apply(lambda x: int(x[:-4].split("_")[4]))

    # 6. Pixel spacing from filepath (float in mm)
    df["px_spacing_h"] = df.f_path.apply(lambda x: float(x[:-4].split("_")[5]))
    df["px_spacing_w"] = df.f_path.apply(lambda x: float(x[:-4].split("_")[6]))
    if not is_test:
        # 7. Merge 3 Rows Into A Single Row (As This/Segmentation-RLE Is The Only Unique Information Across Those Rows)
        l_bowel_df = df[df["class"]=="large_bowel"][["id","segmentation"]].rename(columns={"segmentation":"lb_seg_rle"})
        s_bowel_df = df[df["class"]=="small_bowel"][["id", "segmentation"]].rename(columns={"segmentation":"sb_seg_rle"})
        stomach_df = df[df["class"]=="stomach"][["id", "segmentation"]].rename(columns={"segmentation":"st_seg_rle"})

        df = df.merge(l_bowel_df, on="id", how="left")
        df = df.merge(s_bowel_df, on="id", how="left")
        df = df.merge(stomach_df, on="id", how="left")

        df = df.drop_duplicates(subset=["id",]).reset_index(drop=True)

        df["lb_seg_flag"] = df.lb_seg_rle.apply(lambda x: not pd.isna(x))
        df["sb_seg_flag"] = df.sb_seg_rle.apply(lambda x: not pd.isna(x))
        df["st_seg_flag"] = df.st_seg_rle.apply(lambda x: not pd.isna(x))

        df["n_segs"] = df.lb_seg_flag.astype(int)+df.sb_seg_flag.astype(int)+df.st_seg_flag.astype(int)
    # 8. Reorder columns to the a new ordering (drops class and segmentation as no longer necessary)
    new_col_order = ["id", "f_path", "n_segs",
                     "lb_seg_rle", "lb_seg_flag",
                     "sb_seg_rle", "sb_seg_flag", 
                     "st_seg_rle", "st_seg_flag",
                     "slice_h", "slice_w", "px_spacing_h", 
                     "px_spacing_w", "case_id_str", "case_id", 
                     "day_num_str", "day_num", "slice_id",]
    if is_test: new_col_order.insert(1, "class")
    new_col_order = [_c for _c in new_col_order if _c in df.columns]
    df = df[new_col_order]
    # 9. Display update dataframe
    display(df)
    return df


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

    # The image image is actually flattened since RLE is a 1D "run"
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
        if to_rgb:
            return np.tile(np.expand_dims(cv2.imread(_path, cv2.IMREAD_ANYDEPTH), axis=-1), 3)
        else:
            return cv2.imread(_path, cv2.IMREAD_ANYDEPTH)


# In[ ]:


def get_overlay(img_path, rle_strs, img_shape, _alpha=0.999, _beta=0.35, _gamma=0):
    _img = open_gray16(img_path, to_rgb=True)
    _img = ((_img-_img.min())/(_img.max()-_img.min())).astype(np.float32)
    _seg_rgb = np.stack([rle_decode(rle_str, shape=img_shape, color=1)                          if (rle_str is not None and not pd.isna(rle_str))                          else np.zeros(img_shape, dtype=np.float32) for rle_str in rle_strs], axis=-1).astype(np.float32)
    seg_overlay = cv2.addWeighted(src1=_img, alpha=_alpha, 
                                  src2=_seg_rgb, beta=_beta, gamma=_gamma)
    return seg_overlay

def examine_id(ex_id, df, plot_overlay=True, print_meta=False, plot_grayscale=False, plot_binary_segmentation=False):
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


def get_seg_combo_str(row):
    seg_str_list = []
    if row["lb_seg_flag"]: seg_str_list.append("Large Bowel")
    if row["sb_seg_flag"]: seg_str_list.append("Small Bowel")
    if row["st_seg_flag"]: seg_str_list.append("Stomach")
    if len(seg_str_list)>0:
        return ", ".join(seg_str_list)
    else:
        return "No Mask"


# In[ ]:


def plot_case(case_id, df, day=None, _figsize=(20,30), n_cols=16):
    # Initialize
    case_df = df[df.case_id==case_id]
    
    if day is not None:
        _case_df = case_df[(case_df.day_num==day) | (case_df.day_num_str==str(day))]
        if len(_case_df)>0:
            approx_shrink = len(_case_df)/len(case_df)
            case_df=_case_df
            _figsize = (_figsize[0], int(np.ceil(1.25*_figsize[1]*approx_shrink)))
        else:
            print("There are no valid samples for the passed `day`. Reverting to all days in case.")
        del _case_df
    
    n_ex = len(case_df)
    
    print("...Preparing...")
    # Get relevant data
    case_paths = case_df["f_path"].tolist()
    case_rles = [[_rle if not pd.isna(_rle) else None for _rle in _rles] for _rles in case_df[["lb_seg_rle", "sb_seg_rle", "st_seg_rle"]].values.tolist()]
    case_img_shapes = [(_w,_h) for _w,_h in zip(case_df["slice_w"].tolist(), case_df["slice_h"].tolist())]
    all_overlays = [get_overlay(img_path, rle_strs, img_shape) for img_path, rle_strs, img_shape in zip(case_paths, case_rles, case_img_shapes)]
    
    print("...Plotting...")    
    # Plot
    plt.figure(figsize=_figsize)
    n_rows = int(np.ceil(n_ex/n_cols))
    
    gs = gridspec.GridSpec(n_rows, n_cols,
         wspace=0.0, hspace=0.0, 
         top=1.-0.5/(n_rows+1), bottom=0.5/(n_rows+1), 
         left=0.5/(n_cols+1), right=1-0.5/(n_cols+1))
    
    for i in range(n_rows):
        if len(all_overlays)==0: break
        for j in range(n_cols):
            if len(all_overlays)==0: break
            ax=plt.subplot(gs[i,j])
            ax.imshow(all_overlays.pop())
            ax.axis(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        
    print("...Displaying...")    
    plt.show()


# In[ ]:


def get_mask_area(rle):
    return sum([int(x) for x in rle.split()[1::2]])


# In[ ]:


def is_overlap(_arr):
    return _arr.sum(axis=-1).max()>1
    
def make_seg_mask(row, output_dir="/kaggle/working/npy_files", check_overlap=False):
    slice_shape = (row.slice_w, row.slice_h)
    if not pd.isna(row.lb_seg_rle):
        lb_mask = rle_decode(row.lb_seg_rle, slice_shape, )
    else:
        lb_mask = np.zeros(slice_shape)
    if not pd.isna(row.sb_seg_rle):
        sb_mask = rle_decode(row.sb_seg_rle, slice_shape)
    else:
        sb_mask = np.zeros(slice_shape)
    if not pd.isna(row.st_seg_rle):
        st_mask = rle_decode(row.st_seg_rle, slice_shape)
    else:
        st_mask = np.zeros(slice_shape)
    mask_arr = np.stack([lb_mask, sb_mask, st_mask], axis=-1).astype(np.uint8)
    np.save(f"./npy_files/{row.id}_mask", mask_arr)
    
    if check_overlap: 
        if is_overlap(mask_arr): 
            return np.where(mask_arr.sum(axis=-1)>1, 1, 0).sum()
        else:
            return 0


# In[ ]:


def get_image_vals(row):
    _img = cv2.imread(row.f_path, -1)
    _nonzero_px_count = np.count_nonzero(_img)
    
    row["nonzero_num_pxs"] = _nonzero_px_count
    row["max_px_value"] = _img.max()
    row["min_px_value"] = _img.min()
    row["mean_px_value"] = _img.mean()
    row["nonzero_mean_px_value"] = _img.sum()/_nonzero_px_count
    
    return row


# In[ ]:


def create_animation(case_id, day_num, df):
    
    sub_df = df[(df.case_id==case_id) & (df.day_num==day_num)]
    
    f_paths  = sub_df.f_path.tolist()
    lb_rles  = sub_df.lb_seg_rle.tolist()
    sb_rles  = sub_df.sb_seg_rle.tolist()
    st_rles  = sub_df.st_seg_rle.tolist()
    slice_ws = sub_df.slice_w.tolist()
    slice_hs = sub_df.slice_h.tolist()
    
    animation_arr = np.stack([get_overlay(img_path=_f, 
                                          rle_strs=(_lb, _sb, _st), 
                                          img_shape=(_w, _h)) \
                                for _f, _lb, _sb, _st, _w, _h in zip(f_paths, lb_rles, sb_rles, st_rles, slice_ws, slice_hs)
                                ], axis=0)
    
    fig = plt.figure(figsize=(8,8))
    
    plt.axis('off')
    im = plt.imshow(animation_arr[0])
    plt.title(f"3D Animation for Case {case_id} on Day {day_num}", fontweight="bold")
    
    def animate_func(i):
        im.set_array(animation_arr[i])
        return [im]
    plt.close()
    
    return animation.FuncAnimation(fig, animate_func, frames = animation_arr.shape[0], interval = 1000//12)


# # <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;text-align:center">DATASET</p></div>

# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">COMPETITION DATA ACCESS</p></div>

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


# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">BASIC DATA DEFINITIONS & INITIALIZATIONS</p></div> 

# In[ ]:


print("\n... BASIC DATA SETUP STARTING ...\n\n")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
train_full = pd.read_csv(TRAIN_CSV)
# Get all training images
all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)

print("\n... ORIGINAL TRAINING DATAFRAME... \n")
display(train_full)


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
    ss_df = train_full.iloc[:10]
    ss_df = ss_df[["id", "class"]]
    ss_df["predicted"] = ""
    
    print("\n\n\n... DEBUG SUBMISSION DATAFRAME... \n")
    display(ss_df)

SF2LF = {"lb":"Large Bowel","sb":"Small Bowel","st":"Stomach"}
LF2SF = {v:k for k,v in SF2LF.items()}

print("\n... BASIC DATA SETUP FINISHED ...\n\n")


# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">UPDATE DATAFRAMES WITH ACCESSIBLE EXTRA INFORMATION</p></div>  
# ****
# I have changed the column identifiers as follows for the sake of brevity:
# 
# * large_bowel --> lb
# * small_bowel --> sb
# * stomach --> st

# In[ ]:


train_df = train_full.copy()
print("\n... UPDATED TRAINING DATAFRAME... \n")
train_df = preprocessing_df(train_df, all_train_images)
print("\n\n\n... UPDATED SUBMISSION DATAFRAME... \n")
ss_df = preprocessing_df(ss_df, all_test_images, is_test=True)
print("\n... UPDATING DATAFRAMES WITH ACCESSIBLE INFORMATION FINISHED ...\n\n")


# # <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;text-align:center">DATASET EXPLORATION</p></div>

# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">LOOK AT A SINGLE EXAMPLE PRIOR TO INVESTIGATION</p></div>  

# In[ ]:


print("\n... SINGLE ID EXPLORATION STARTED ...\n\n")
demo_ex = train_df[train_df.id == CFG.DEMO_ID].squeeze()
print(f"\n... WITH DEMO_ID=`{CFG.DEMO_ID}` WE HAVE THE FOLLOWING DEMO EXAMPLE TO WORK FROM... \n\n")
display(demo_ex.to_frame())


# In[ ]:


print(f"\n\n... LET'S PLOT THE IMAGE FIRST ...\n")
plt.figure(figsize=(12,12))
plt.imshow(open_gray16(demo_ex.f_path), cmap="gray")
plt.title(f"Original Grayscale Image For ID: {demo_ex.id}", fontweight="bold")
plt.axis(False)
plt.show()


# In[ ]:


print(f"\n\n... LET'S PLOT THE 3 SEGMENTATION MASKS ...\n")

plt.figure(figsize=(20,10))
for i, _seg_type in enumerate(["lb", "sb", "st"]):
    if pd.isna(demo_ex[f"{_seg_type}_seg_rle"]): continue
    plt.subplot(1,3,i+1)
    plt.imshow(rle_decode(demo_ex[f"{_seg_type}_seg_rle"], shape=(demo_ex.slice_w, demo_ex.slice_h), color=1))
    plt.title(f"RLE Encoding For {SF2LF[_seg_type]} Segmentation", fontweight="bold")
    plt.axis(False)
plt.tight_layout()
plt.show()


# In[ ]:


print(f"\n\n... LET'S PLOT THE IMAGE WITH AN RGB SEGMENTATION MASK OVERLAY ...\n")

# We need to normalize the loaded image values to be between 0 and 1 or else our plot will look weird
_img = open_gray16(demo_ex.f_path, to_rgb=True)
_img = ((_img-_img.min())/(_img.max()-_img.min())).astype(np.float32)
_seg_rgb = np.stack([rle_decode(demo_ex[f"{_seg_type}_seg_rle"], shape=(demo_ex.slice_w, demo_ex.slice_h), color=1) if not pd.isna(demo_ex[f"{_seg_type}_seg_rle"]) else np.zeros((demo_ex.slice_w, demo_ex.slice_h)) for _seg_type in ["lb", "sb", "st"]], axis=-1).astype(np.float32)
seg_overlay = cv2.addWeighted(src1=_img, alpha=0.99, 
                              src2=_seg_rgb, beta=0.33, gamma=0.0)

plt.figure(figsize=(12,12))
plt.imshow(seg_overlay)
plt.title(f"Segmentation Overlay For ID: {demo_ex.id}", fontweight="bold")
handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
labels = ["Large Bowel Segmentation Map", "Small Bowel Segmentation Map", "Stomach Segmentation Map"]
plt.legend(handles,labels)
plt.axis(False)
plt.show()


# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">INVESTIGATE THE OCCURENCE SEGMENTATION MAP TYPES</p></div>  

# In[ ]:


train_df["seg_combo_str"] = train_df.progress_apply(get_seg_combo_str, axis=1)

fig = px.histogram(train_df, 
                   train_df["n_segs"].astype(str), 
                   color="seg_combo_str", 
                   title="<b>Number of Segmentation Masks Per Image</b>", 
                   labels={"x":"Number of Segmentation Masks Per Image", 
                           "seg_combo_str":"<b>Segmentation Masks Present</b>"})
fig.show()


# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">INVESTIGATE THE IMAGE SIZES</p></div>  

# In[ ]:


fig = px.scatter(train_df.drop_duplicates(subset=["slice_w", "slice_h"]), 
                 x="slice_w", 
                 y="slice_h", 
                 size=train_df.groupby(["slice_w", "slice_h"])["id"].transform("count").iloc[train_df.drop_duplicates(subset=["slice_w", "slice_h"]).index], 
                 color="("+train_df.drop_duplicates(subset=["slice_w", "slice_h"])["slice_w"].astype(str)+","+train_df.drop_duplicates(subset=["slice_w", "slice_h"])["slice_h"].astype(str)+")", 
                 title="<b>Bubble Chart Showing The Various Image Sizes</b>",
                 labels={"color":"<b>Size Legend</b>", 
                         "size":"<b>Number Of Observations</b>",
                         "slice_h":"<b>Image Slice Height (pixels)</b>",
                         "slice_w":"<b>Image Slice Width (pixels)</b>"},
                 size_max=160)
fig.show()


# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">INVESTIGATE THE PIXEL SPACING</p></div> 

# In[ ]:


fig = px.scatter(train_df.drop_duplicates(subset=["px_spacing_w", "px_spacing_h"]), 
                 x="px_spacing_w", 
                 y="px_spacing_h", 
                 size=train_df.groupby(["px_spacing_w", "px_spacing_h"])["id"].transform("count").iloc[train_df.drop_duplicates(subset=["px_spacing_w", "px_spacing_h"]).index], 
                 color="("+train_df.drop_duplicates(subset=["px_spacing_w", "px_spacing_h"])["px_spacing_w"].astype(str)+","+train_df.drop_duplicates(subset=["px_spacing_w", "px_spacing_h"])["px_spacing_h"].astype(str)+")", 
                 title="<b>Bubble Chart Showing The Various Pixel Spacings</b>",
                 labels={"color":"<b>Pixel Spacing Sets Legend</b>", 
                         "size":"<b>Number Of Observations</b>",
                         "px_spacing_h":"<b>Pixel Spacing Height (mm)</b>",
                         "px_spacing_w":"<b>Pixel Spacing Width (mm)</b>"},
                 size_max=160)
fig.show()


# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">INVESTIGATE CASE IDS</p></div>  

# In[ ]:


fig = px.histogram(train_df, 
                   train_df.case_id.astype(str), 
                   color="day_num_str", 
                   title="<b>Distribution Of Images Per Case ID</b>", 
                   labels={"x":"<b>Case ID</b>", "day_num_str": "<b>The Day The Scan Took Place</b>"}, 
                   text_auto=True, 
                   width=2000)
fig.show()


# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">VISUALIZE A SIMPLE CASE</p></div>  

# In[ ]:


print(f"\n\n... PLOTTING DEMO CASE ID #{CFG.DEMO_CASE} ...\n\n")
plot_case(CFG.DEMO_CASE, df=train_df)


# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">MASK SIZES/AREAS</p></div>  

# In[ ]:


train_df["lb_seg_area"] = train_df.lb_seg_rle.apply(lambda x: None if pd.isna(x) else get_mask_area(x))
train_df["sb_seg_area"] = train_df.sb_seg_rle.apply(lambda x: None if pd.isna(x) else get_mask_area(x))
train_df["st_seg_area"] = train_df.st_seg_rle.apply(lambda x: None if pd.isna(x) else get_mask_area(x))

fig = px.histogram(train_df, 
                   ["lb_seg_area", "sb_seg_area", "st_seg_area"], 
                   title="<b>Mask Areas Overlaid</b>", 
                   barmode="overlay",
                   labels={"value":"<b>Mask Area</b>"})
fig.show()


# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">MASK DATASET CREATION, CLASS OVERLAP & MASK HEATMAP</p></div>  

# In[ ]:


if not os.path.isdir(CFG.NPY_DIR): os.makedirs(CFG.NPY_DIR, exist_ok=True)
train_df["seg_overlap_area"] = train_df.progress_apply(lambda x: make_seg_mask(x, output_dir=CFG.NPY_DIR, check_overlap=True), axis=1)

print("\n... LET'S EXAMINE THE IMAGE WITH THE HIGHEST AMOUNT OF OVERLAP ...\n")

examine_id(train_df[train_df.seg_overlap_area==train_df.seg_overlap_area.max()].id.values[0], train_df)

fig = px.histogram(train_df[train_df.seg_overlap_area>0], 
                   "seg_overlap_area", 
                   color="seg_combo_str", 
                   nbins=50,
                   log_y=True, 
                   title="<b>Distribution of Non-Zero Segmentation Overlaps <sub>(Count Is Logarithmic)</sub></b>",  
                   labels={"seg_overlap_area":"<b>Area of Mask Overlap</b>", 
                           "seg_combo_str":"<b>Segmentation Masks In Image</b>"})
fig.update_layout(legend=dict(yanchor="top",
                              y=0.99,
                              xanchor="right",
                              x=0.995
                             )
                 )
fig.show()


# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">HEATMAP for SEGMENTATION MASK</p></div>  

# In[ ]:


heatmap = np.zeros((256,256,3), dtype=np.float32)
for _, _row in tqdm(train_df.iterrows(), total=len(train_df)):
    if (_row.lb_seg_flag or _row.sb_seg_flag or _row.st_seg_flag):
        _mask = cv2.resize(np.load(f"./npy_files/{_row.id}_mask.npy"), (256,256), interpolation=cv2.INTER_NEAREST)
        heatmap+=_mask


# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">PIXEL VALUES</p></div>  
# ****
# Knowing the range of Pixel values to normalize the data to convert it into the format that is expected in ML. 
# 
# OBSERVATIONS: 
# 
# The maximum value in the dataset(15865.0) is equiavlent to less than half of an int16 or a quarter of a uint16(16384).
# 
# 

# In[ ]:


train_df = train_df.progress_apply(get_image_vals, axis=1)

print(f"\n\n\n... UPDATED TRAIN DATAFRAME ...\n")
display(train_df.head())
print("\n\n")

for _c in ["nonzero_num_pxs", "max_px_value", "min_px_value", "mean_px_value", "nonzero_mean_px_value"]:
    print(f"\n... STATS FOR COLUMN --> `{_c}`...")
    print(f"\t--> MIN  VAL: {train_df[_c].min():.1f}")
    print(f"\t--> MEAN VAL: {train_df[_c].mean():.1f}")
    print(f"\t--> MAX  VAL: {train_df[_c].max():.1f}")


# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">IDENTIFY ANY HEURISTICS OR RULES REGARDING SEGMENTATION</p></div>  
# ****
# For a given case-id and day number there are two different amounts of scans present
# 
# * 144 slices --> 259 instances
# * 80 slices ---> 15 instances

# In[ ]:


train_df["slice_count"] = train_df.id.apply(lambda x: int(x.rsplit("_", 1)[-1]))
print("\n... CASE-ID/DAY-NUM SLICE INFORMATION ...\n")
train_df.groupby(["case_id", "day_num"])["slice_count"].max().value_counts()


# In[ ]:


slice_to_occurence_df = train_df.groupby("slice_count")[["lb_seg_flag", "sb_seg_flag", "st_seg_flag"]].sum().reset_index()
fig = px.bar(slice_to_occurence_df, 
             x="slice_count", 
             y=["lb_seg_flag", "sb_seg_flag", "st_seg_flag"],
             orientation="v", 
             labels={"slice_count":"<b>Slice Number</b>", "value":"<b>Number Of Examples</b>",}, 
             title="<b>Number of Examples Per Example For Our 3 Organs</b>")
fig.update_layout(legend_title="<b>Organ Type Legend</b>")
fig.show()


# In[ ]:


print("\n... WHICH SLICES ARE ALWAYS BLANK (NO SEG) BY LABEL ...\n")
keep_slice_blank_map = {_sh_lbl:slice_to_occurence_df[slice_to_occurence_df[f"{_sh_lbl}_seg_flag"]==0].slice_count.to_list() for _sh_lbl in ["lb", "sb", "st"]}
keep_slice_blank_map


# ## <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:70%;text-align:left">CREATE 3D GIF FOR CASE>DAY GROUPS OF SLICES (WITH MASK)</p></div> 

# In[ ]:


case_id=115
day_num=0
df=train_df
sub_df = df[(df.case_id==case_id) & (df.day_num==day_num)]
    
f_paths  = sub_df.f_path.tolist()
lb_rles  = sub_df.lb_seg_rle.tolist()
sb_rles  = sub_df.sb_seg_rle.tolist()
st_rles  = sub_df.st_seg_rle.tolist()
slice_ws = sub_df.slice_w.tolist()
slice_hs = sub_df.slice_h.tolist()


# In[ ]:


create_animation(case_id=115, 
                 day_num=0, 
                 df=train_df)


# # <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;text-align:center">REFERENCE</p></div>
# ****
# https://www.kaggle.com/code/dschettler8845/uwm-gi-tract-image-segmentation-eda

# # <div style="color:white;display:fill;border-radius:5px;background-color:#75B7BF;letter-spacing:0.1px;overflow:hidden"><p style="padding:20px;color:white;overflow:hidden;margin:0;font-size:100%;text-align:center">To be Continued</p></div>
