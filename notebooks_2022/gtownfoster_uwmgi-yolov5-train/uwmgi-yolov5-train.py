#!/usr/bin/env python
# coding: utf-8

# # [UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation)
# > Detect stomach, large bowell, and small bowell in scans
# 
# <img src="https://storage.googleapis.com/kaggle-competitions/kaggle/27923/logos/header.png?t=2021-06-02-20-30-25">

# # 🛠 Install Libraries

# In[ ]:


# !pip install -qU wandb
# !add-apt-repository ppa:ubuntu-toolchain-r/test -y
# !apt-get update
# !apt-get upgrade libstdc++6 -y
# !pip install scikit-learn==1.0.1


# # 📚 Import Libraries

# In[ ]:


from itertools import groupby
import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()
import pandas as pd
import os
import pickle
import cv2
from multiprocessing import Pool
import matplotlib.pyplot as plt
# import cupy as cp
import ast
import glob
from PIL import Image, ImageDraw, ImageFont
import shutil
import sys
sys.path.append('../input/tensorflow-great-barrier-reef')

from joblib import Parallel, delayed
import glob

from IPython.display import display, HTML
import random
from matplotlib import animation, rc
rc('animation', html='jshtml')


# ## References
# This notebook is heavily based off these three resources specifically the yolov5 notebook. I want to thank them for providing me the resources to create this notebook.
# * This notebook is based on a yolov5 notebook from a previous kaggle competition found [here.](https://www.kaggle.com/code/awsaf49/great-barrier-reef-yolov5-train)
# * The data has been converted from [this](https://www.kaggle.com/datasets/awsaf49/uwmgi-coco-dataset) dataset.
# * The helper functions for displaying the data was taken from [this](https://blog.paperspace.com/train-yolov5-custom-data/) article.
# 
# 

# ## Please Upvote if you find this Helpful

# # ⭐ WandB
# <img src="https://camo.githubusercontent.com/dd842f7b0be57140e68b2ab9cb007992acd131c48284eaf6b1aca758bfea358b/68747470733a2f2f692e696d6775722e636f6d2f52557469567a482e706e67" width=600>
# 
# Weights & Biases (W&B) is MLOps platform for tracking our experiemnts. We can use it to Build better models faster with experiment tracking, dataset versioning, and model management. Some of the cool features of W&B:
# 
# * Track, compare, and visualize ML experiments
# * Get live metrics, terminal logs, and system stats streamed to the centralized dashboard.
# * Explain how your model works, show graphs of how model versions improved, discuss bugs, and demonstrate progress towards milestones.
# 

# # Data
# * `uwmgi yolov5 coco` - Dataset containing data in yolov5 format.
# * `data.yaml` - Provides paths to yolov5 algorithm to train and valid sets
# * `annotations files` - In .txt formats for yolov5

# # 🖼Visualization

# In[ ]:


class_name_to_id_mapping = {"Stomach": 0,
                           "Small Bowell": 1,
                           "Large Bowell": 2
                           }
class_id_to_color_mapping = {0: "Black",
                             1: "Red",
                             2: "Blue"
    }
# Find labels with boxes to display
files = glob.glob(f"../input/uwmgi-yolov5-coco/uwmgi/train/labels/*")
annotations = []
for a in files:
    if os.stat(a).st_size > 0:
        annotations.append(a)


# In[ ]:





# In[ ]:


random.seed(0)

class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    plotted_image = ImageDraw.Draw(image)
    transformed_annotations = []
    if len(annotations) > 0:
        transformed_annotations = np.copy(annotations)
        transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
        transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 

        transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
        transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
        transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
        transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)), width=4)
        
        plotted_image.text((x0-20, y0-10), class_id_to_name_mapping[(int(obj_cls))], font=font, fill=class_id_to_color_mapping[(int(obj_cls))])
    image = image.resize((1024,1024))
    plt.imshow(np.array(image))
    plt.show()

# Get any random annotation file 
annotation_file = random.sample(annotations, 5)
for ann in annotation_file:
    with open(ann, "r") as file:
        annotation_list = file.read().split("\n")[:-1]

        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x ] for x in annotation_list]

    #Get the corresponding image file
    image_file = ann.replace("labels", "images").replace("txt", "jpg")
    assert os.path.exists(image_file)

    #Load the image
    image = Image.open(image_file)

    #Plot the Bounding Box
    plot_bounding_box(image, annotation_list)


# In[ ]:





# # 📦 [YOLOv5](https://github.com/ultralytics/yolov5/)
# <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash.jpg" width=800>

# In[ ]:


get_ipython().run_line_magic('cd', '/kaggle/working')
get_ipython().system('rm -r /kaggle/working/yolov5')
# !git clone https://github.com/ultralytics/yolov5 # clone
get_ipython().system('cp -r /kaggle/input/yolov5-lib-ds /kaggle/working/yolov5')
get_ipython().run_line_magic('cd', 'yolov5')
get_ipython().run_line_magic('pip', 'install -qr requirements.txt  # install')

from yolov5 import utils
display = utils.notebook_init()  # check


# In[ ]:


# Weights & Biases  (optional)
import wandb
wandb.login(anonymous='must')


# # 🚅 Training

# In[ ]:


# Train YOLOv5s on COCO128 for 3 epochs
get_ipython().system('python train.py --img 512--batch 16--epochs 10--data /kaggle/input/uwmgi-yolov5-coco/uwmgi/data.yaml--weights yolov5l.pt --workers 2')


# # ✨ Overview
# ![image.png](attachment:14c7fea9-9a96-45de-a620-675270d74c8d.png)

# ## Output Files

# In[ ]:


get_ipython().system('ls runs/train/exp')


# # 📈 Class Distribution

# In[ ]:


plt.figure(figsize = (10,10))
plt.axis('off')
plt.imshow(plt.imread('runs/train/exp/labels_correlogram.jpg'));


# In[ ]:


plt.figure(figsize = (10,10))
plt.axis('off')
plt.imshow(plt.imread('runs/train/exp/labels.jpg'));


# # 🔭 Batch Image

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize = (10, 10))
plt.imshow(plt.imread('runs/train/exp/train_batch0.jpg'))

plt.figure(figsize = (10, 10))
plt.imshow(plt.imread('runs/train/exp/train_batch1.jpg'))

plt.figure(figsize = (10, 10))
plt.imshow(plt.imread('runs/train/exp/train_batch2.jpg'))


# ## GT Vs Pred

# In[ ]:


fig, ax = plt.subplots(3, 2, figsize = (2*9,3*5), constrained_layout = True)
for row in range(3):
    ax[row][0].imshow(plt.imread(f'runs/train/exp/val_batch{row}_labels.jpg'))
    ax[row][0].set_xticks([])
    ax[row][0].set_yticks([])
    ax[row][0].set_title(f'runs/train/exp/val_batch{row}_labels.jpg', fontsize = 12)
    
    ax[row][1].imshow(plt.imread(f'runs/train/exp/val_batch{row}_pred.jpg'))
    ax[row][1].set_xticks([])
    ax[row][1].set_yticks([])
    ax[row][1].set_title(f'runs/train/exp/val_batch{row}_pred.jpg', fontsize = 12)
plt.show()


# # 🔍 Result

# ## Score Vs Epoch

# In[ ]:


plt.figure(figsize=(30,15))
plt.axis('off')
plt.imshow(plt.imread('runs/train/exp/results.png'));


# ## Confusion Matrix

# In[ ]:


plt.figure(figsize=(12,10))
plt.axis('off')
plt.imshow(plt.imread('runs/train/exp/confusion_matrix.png'));


# ## Metrics

# In[ ]:


for metric in ['F1', 'PR', 'P', 'R']:
    print(f'Metric: {metric}')
    plt.figure(figsize=(12,10))
    plt.axis('off')
    plt.imshow(plt.imread(f'runs/train/exp/{metric}_curve.png'));
    plt.show()


# ## Please Upvote if you find this Helpful
# And thanks again to the resources mentioned above.

# <img src="https://www.pngall.com/wp-content/uploads/2018/04/Under-Construction-PNG-File.png">
