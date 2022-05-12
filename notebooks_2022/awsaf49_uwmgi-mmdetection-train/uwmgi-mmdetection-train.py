#!/usr/bin/env python
# coding: utf-8

# # [UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/)
# > Track healthy organs in medical scans to improve cancer treatment
# 
# <img src="https://storage.googleapis.com/kaggle-competitions/kaggle/27923/logos/header.png?t=2021-06-02-20-30-25">

# # 📒 Notebooks
# 📌 **UNet**:
# * Train: [UWMGI: Unet [Train] [PyTorch]](https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/)
# * Infer: [UWMGI: Unet [Infer] [PyTorch]](https://www.kaggle.com/code/awsaf49/uwmgi-unet-infer-pytorch/)
# 
# 📌 **Data/Dataset**:
# * Data: [UWMGI: Mask Data](https://www.kaggle.com/datasets/awsaf49/uwmgi-mask-data)
# * Dataset: [UWMGI: Mask Dataset](https://www.kaggle.com/datasets/awsaf49/uwmgi-mask-dataset)

# # 🛠 Install Libraries

# In[ ]:


# dependencies
get_ipython().system('pip install -q /kaggle/input/mmdet-lib-ds-v2/torch-1.7.0%2Bcu110-cp37-cp37m-linux_x86_64.whl')
get_ipython().system('pip install -q /kaggle/input/mmdet-lib-ds-v2/torchvision-0.8.0-cp37-cp37m-manylinux1_x86_64.whl')
get_ipython().system('pip install -q /kaggle/input/mmdet-lib-ds-v2/yapf-0.32.0-py2.py3-none-any.whl')
get_ipython().system('pip install -q /kaggle/input/mmdet-lib-ds-v2/pycocotools-2.0.3/pycocotools-2.0.3.tar')
get_ipython().system('pip install -q /kaggle/input/mmdet-lib-ds-v2/mmcv_full-1.4.2-cp37-cp37m-manylinux1_x86_64.whl')
get_ipython().system('pip install -q /kaggle/input/mmdet-lib-ds-v2/addict-2.4.0-py3-none-any.whl')


#wandb
get_ipython().system('pip install -qU wandb')


# # 📚 Import Libraries

# In[ ]:


from itertools import groupby
from pycocotools import mask as mutils
from pycocotools.coco import COCO
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import wandb
from PIL import Image
import gc

from glob import glob
import matplotlib.pyplot as plt


# # ⭐ WandB
# <img src="https://camo.githubusercontent.com/dd842f7b0be57140e68b2ab9cb007992acd131c48284eaf6b1aca758bfea358b/68747470733a2f2f692e696d6775722e636f6d2f52557469567a482e706e67" width=600>
# 
# Weights & Biases (W&B) is MLOps platform for tracking our experiemnts. We can use it to Build better models faster with experiment tracking, dataset versioning, and model management. Some of the cool features of W&B:
# 
# * Track, compare, and visualize ML experiments
# * Get live metrics, terminal logs, and system stats streamed to the centralized dashboard.
# * Explain how your model works, show graphs of how model versions improved, discuss bugs, and demonstrate progress towards milestones.
# 

# In[ ]:


import wandb

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("WANDB")
    wandb.login(key=api_key)
    anonymous = None
except:
    anonymous = "must"
    print('To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')


# # 📖 Meta Data
# 
# * Files
#     * train.csv - IDs and masks for all training objects.
#     * sample_submission.csv - a sample submission file in the correct format
#     * train - a folder of case/day folders, each containing slice images for a particular case on a given day.
# 
# > Note that the image filenames include 4 numbers (ex. `276_276_1.63_1.63.png`). These four numbers are slice height / width (integers in pixels) and heigh/width pixel spacing (floating points in mm). The first two defines the resolution of the slide. The last two record the physical size of each pixel.
# 
# * Columns
#     * id - unique identifier for object
#     * class - the predicted class for the object
#     * EncodedPixels - RLE-encoded pixels for the identified object

# In[ ]:


ROOT = '/kaggle/input/uw-madison-gi-tract-image-segmentation'
DATA_DIR = '/kaggle/input/uwmgi-coco-dataset'
config = 'configs/uwmgi/custom_config.py'


# Train Data
df = pd.read_csv('/kaggle/input/uwmgi-mask-dataset/train.csv')
df['segmentation'] = df.segmentation.fillna('')
df['rle_len'] = df.segmentation.map(len) # length of each rle mask
df['mask_path'] = df.mask_path.str.replace('/png/','/np').str.replace('.png','.npy')

df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index() # rle list of each id
df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index()) # total length of all rles of each id

df = df.drop(columns=['segmentation', 'class', 'rle_len'])
df = df.groupby(['id']).head(1).reset_index(drop=True)
df = df.merge(df2, on=['id'])
df['empty'] = (df.rle_len==0) # empty masks
df.head()


# # 🌈 Visualization
# 

# In[ ]:


def load_img(path, ch=3):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if ch==3:
        img = np.tile(img[...,None], [1, 1, 3]) # gray to rgb
    img = img.astype('float32') # original is uint16
    mx = np.max(img)
    if mx:
        img/=mx # scale image to [0, 1]
    return img

def load_msk(path):
    msk = np.load(path)
    msk = msk.astype('float32')
    msk/=255.0
    return msk


# In[ ]:


coco = COCO(f'{DATA_DIR}/annotations_valid.json')
imgIds = coco.getImgIds()

imgsData = coco.loadImgs(imgIds[100:105:2])
_,axs = plt.subplots(len(imgsData),2,figsize=(20,10 * len(imgsData)))
for imgData, ax in zip(imgsData, axs):
    img = load_img(f'{DATA_DIR}/valid2017/'+imgData['file_name'])
    annIds = coco.getAnnIds(imgIds=[imgData['id']])
    anns = coco.loadAnns(annIds)
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(img)
    ax[1].axis('off')
    plt.sca(ax[1])
    coco.showAnns(anns, draw_bbox=True)
plt.tight_layout()
plt.show()


# # 📦 MMDetetection
# <div align=center><img src="https://raw.githubusercontent.com/open-mmlab/mmdetection/master/resources/mmdet-logo.png" width=500></div>
# 
# Here are some cool facts about **MMDet**,
# * It comes with bunch of sota models. You may get tired trying them out ;)
# * Easy to comstomize and Deploy.
# * It has built-in **wandb** integration. So, we can easily track our training.
# * Error/Result Analysis is easier.

# In[ ]:


# mmdet
get_ipython().system('rm -r /kaggle/working/mmdetection')
get_ipython().system('cp -r /kaggle/input/mmdet-repo-ds /kaggle/working/mmdetection')
get_ipython().run_line_magic('cd', 'mmdetection')
get_ipython().system('pip install -q -e .')


# # ⚙️ Configuration
# > You can tune following parameters for better result. 
# 
# * **Num Classes** 
# * **Score-Theshold**
# * **IoU**
# * **Wandb**
# ```
# dict(type='WandbLoggerHook', # this is where magic happens ;)
#      init_kwargs=dict(project='sartorius',
#                       name=f'mask_rcnn_r50',
#                       config={'config':mask_rcnn_r50_fpn_1x_coco,
#                       'comment':'baseline01',},
#                       entity=None)) # this is where magic happens
# ```
# * **Augmentation** 
#     * **Flip**
#     * **Multi-Scale**
#     * **PhotoMetricDistortion**
# * **Batch Size**

# In[ ]:


get_ipython().system('mkdir -p configs/uwmgi')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'configs/uwmgi/custom_config.py', "\n# model settings\nmodel = dict(\n    type='MaskRCNN',\n    backbone=dict(\n        type='ResNet',\n        depth=50,\n        num_stages=4,\n        out_indices=(0, 1, 2, 3),\n        frozen_stages=1,\n        norm_cfg=dict(type='BN', requires_grad=True),\n        norm_eval=True,\n        style='pytorch',\n        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),\n    neck=dict(\n        type='FPN',\n        in_channels=[256, 512, 1024, 2048],\n        out_channels=256,\n        num_outs=5),\n    rpn_head=dict(\n        type='RPNHead',\n        in_channels=256,\n        feat_channels=256,\n        anchor_generator=dict(\n            type='AnchorGenerator',\n            scales=[8],\n            ratios=[0.5, 1.0, 2.0],\n            strides=[4, 8, 16, 32, 64]),\n        bbox_coder=dict(\n            type='DeltaXYWHBBoxCoder',\n            target_means=[.0, .0, .0, .0],\n            target_stds=[1.0, 1.0, 1.0, 1.0]),\n        loss_cls=dict(\n            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),\n        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),\n    roi_head=dict(\n        type='StandardRoIHead',\n        bbox_roi_extractor=dict(\n            type='SingleRoIExtractor',\n            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),\n            out_channels=256,\n            featmap_strides=[4, 8, 16, 32]),\n        bbox_head=dict(\n            type='Shared2FCBBoxHead',\n            in_channels=256,\n            fc_out_channels=1024,\n            roi_feat_size=7,\n            num_classes=3,\n            bbox_coder=dict(\n                type='DeltaXYWHBBoxCoder',\n                target_means=[0., 0., 0., 0.],\n                target_stds=[0.1, 0.1, 0.2, 0.2]),\n            reg_class_agnostic=False,\n            loss_cls=dict(\n                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),\n            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),\n        mask_roi_extractor=dict(\n            type='SingleRoIExtractor',\n            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),\n            out_channels=256,\n            featmap_strides=[4, 8, 16, 32]),\n        mask_head=dict(\n            type='FCNMaskHead',\n            num_convs=4,\n            in_channels=256,\n            conv_out_channels=256,\n            num_classes=3,\n            loss_mask=dict(\n                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),\n    # model training and testing settings\n    train_cfg=dict(\n        rpn=dict(\n            assigner=dict(\n                type='MaxIoUAssigner',\n                pos_iou_thr=0.7,\n                neg_iou_thr=0.3,\n                min_pos_iou=0.3,\n                match_low_quality=True,\n                ignore_iof_thr=-1),\n            sampler=dict(\n                type='RandomSampler',\n                num=256,\n                pos_fraction=0.5,\n                neg_pos_ub=-1,\n                add_gt_as_proposals=False),\n            allowed_border=-1,\n            pos_weight=-1,\n            debug=False),\n        rpn_proposal=dict(\n            nms_pre=2000,\n            max_per_img=1000,\n            nms=dict(type='nms', iou_threshold=0.7),\n            min_bbox_size=0),\n        rcnn=dict(\n            assigner=dict(\n                type='MaxIoUAssigner',\n                pos_iou_thr=0.5,\n                neg_iou_thr=0.5,\n                min_pos_iou=0.5,\n                match_low_quality=True,\n                ignore_iof_thr=-1),\n            sampler=dict(\n                type='RandomSampler',\n                num=512,\n                pos_fraction=0.25,\n                neg_pos_ub=-1,\n                add_gt_as_proposals=True),\n            mask_size=28,\n            pos_weight=-1,\n            debug=False)),\n    test_cfg=dict(\n        rpn=dict(\n            nms_pre=1000,\n            max_per_img=1000,\n            nms=dict(type='nms', iou_threshold=0.7),\n            min_bbox_size=0),\n        rcnn=dict(\n            score_thr=0.05,\n            nms=dict(type='nms', iou_threshold=0.5),\n            max_per_img=100,\n            mask_thr_binary=0.5)))\n\n# dataset settings\ndataset_type = 'CocoDataset'\nclasses = ('large_bowel', 'small_bowel', 'stomach') # Added\ndata_root = '/kaggle/input/uwmgi-coco-dataset/' # Modified\nimg_norm_cfg = dict(\n    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)\ntrain_pipeline = [\n    dict(type='LoadImageFromFile'),\n    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),\n    dict(type='Resize',\n         img_scale=[(640, 640),(512, 512),(384, 384)], # [(1280, 1280), (1152, 1152), (1024, 1024)],\n         multiscale_mode='value',\n         keep_ratio=True),\n    dict(type='RandomFlip', direction=['horizontal', 'vertical'], flip_ratio=0.5), # augmentation starts\n    dict(type='RandomShift', shift_ratio=0.25, max_shift_px=32),\n    dict(type='PhotoMetricDistortion',\n         brightness_delta=32, contrast_range=(0.5, 1.5),\n         saturation_range=(0.5, 1.5), hue_delta=18),\n    dict(type='Normalize', **img_norm_cfg),\n    dict(type='Pad', size_divisor=32),\n    dict(type='DefaultFormatBundle'),\n    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),\n]\ntest_pipeline = [\n    dict(type='LoadImageFromFile'),\n    dict(\n        type='MultiScaleFlipAug',\n        img_scale=(640, 640), # (1280, 1280),\n        flip=False,\n        transforms=[\n            dict(type='Resize', keep_ratio=True),\n            dict(type='RandomFlip'),\n            dict(type='Normalize', **img_norm_cfg),\n            dict(type='Pad', size_divisor=32),\n            dict(type='ImageToTensor', keys=['img']),\n            dict(type='Collect', keys=['img']),\n        ])\n]\ndata = dict(\n    samples_per_gpu=16, # BATCH_SIZE\n    workers_per_gpu=2,\n    train=dict(\n        type=dataset_type,\n        ann_file=data_root + 'annotations_train.json', # Modified\n        img_prefix=data_root + 'train2017/', # Modified\n        classes=classes, # Added\n        pipeline=train_pipeline),\n    val=dict(\n        type=dataset_type,\n        ann_file=data_root + 'annotations_valid.json', # Modified\n        img_prefix=data_root + 'valid2017/', # Modified\n        classes=classes, # Added\n        pipeline=test_pipeline),\n    test=dict(\n        type=dataset_type,\n        ann_file=data_root + 'annotations_valid.json', # Modified\n        img_prefix=data_root + 'valid2017/', # Modified\n        classes=classes, # Added\n        pipeline=test_pipeline))\nevaluation = dict(interval=1,\n                  metric=['bbox','segm'], # bbox, segm\n                  save_best='segm_mAP')\n\n\n\n# optimizer\noptimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)\noptimizer_config = dict(grad_clip=None)\n# learning policy\n# lr_config = dict(\n#     policy='step',\n#     warmup='linear',\n#     warmup_iters=400,\n#     warmup_ratio=0.001,\n#     step=[3, 6, 9, 12])\nlr_config = dict(\n    policy='CosineAnnealing',\n    warmup='linear',\n    warmup_iters=800,\n    warmup_ratio=1.0 / 10,\n    min_lr_ratio=1e-5)\nrunner = dict(type='EpochBasedRunner', max_epochs=15)\n\n# default_runtime\ncheckpoint_config = dict(interval=-1)\n# yapf:disable\nlog_config = dict(\n    interval=10,\n    hooks=[\n        dict(type='TextLoggerHook'),\n        dict(type='WandbLoggerHook', # wandb logger\n             init_kwargs=dict(project='uwmgi-mmdet-public',\n                              name=f'mask_rcnn-resnet50-640x640-fold0',\n                              config={'config':'mask_rcnn_r50_fpn_1x_coco',\n                                      'exp_name':'baseline',\n                                      'comment':'mask_rcnn-resnet50-640x640-multi',\n                                      'batch_size':16,\n                                      'lr':0.020\n                                     },\n                              group='ask_rcnn-resnet50-640x640-multi',\n                              entity=None)),\n#         dict(type='TensorboardLoggerHook')\n    ])\n# yapf:enable\ncustom_hooks = [dict(type='NumClassCheckHook')]\n\ndist_params = dict(backend='nccl')\nlog_level = 'INFO'\nload_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'\nresume_from = None\nworkflow = [('train', 1)]")


# # 🚅 Training

# In[ ]:


get_ipython().system('python tools/train.py {config}')


# In[ ]:


ls work_dirs/custom_config


# # ✨ Overview
# ![image.png](attachment:27f272fb-91f4-478a-85bf-bba3707b9cb6.png)

# # 🔎 Result Analysis

# ## Best Checkpoint

# In[ ]:


ckpt  = glob('work_dirs/custom_config/best_segm_mAP_epoch_*.pth')[0]
ckpt


# ## Test Model

# In[ ]:


get_ipython().system('python tools/test.py {config} {ckpt} --eval segm --out /kaggle/working/result.pkl')


# ## Analyze

# In[ ]:


import pickle
with open('/kaggle/working/result.pkl', 'rb') as f:
    data = pickle.load(f)
    
with open('/kaggle/working/result.pkl', 'wb') as f:
    pickle.dump(data[:200], f)


# In[ ]:


get_ipython().system('python tools/analysis_tools/analyze_results.py        {config}        /kaggle/working/result.pkl        /kaggle/working/result       --show-score-thr 0.40')


# ## Utils

# In[ ]:


def plot_batch(paths,row = 3, col = 2, scale=1):
    plt.figure(figsize=(col*5*scale, row*5*scale))
    for i, path in enumerate(paths[:row*col]):
        plt.subplot(row, col, i+1)
        img = load_img(path, ch=None)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# # 👍 Good Ones

# In[ ]:


good_paths = glob('/kaggle/working/result/good/*')
plot_batch(good_paths)


# # 👎 Bad Ones

# In[ ]:


bad_paths = glob('/kaggle/working/result/bad/*')
plot_batch(bad_paths)


# # ✂️ Remove Files

# In[ ]:


get_ipython().system('cp -r work_dirs /kaggle/working')
get_ipython().run_line_magic('cd', '/kaggle/working')
get_ipython().system('rm -r /kaggle/working/mmdetection')


# # 💡 Reference
