#!/usr/bin/env python
# coding: utf-8

# # EDA🔎 and baseline with Lightning⚡Flash & DeepLab-v3 & albumentations
# 
# see: https://lightning-flash.readthedocs.io/en/stable/reference/semantic_segmentation.html

# In[ ]:


get_ipython().system('pip uninstall -y torchtext')
# !pip install -q --upgrade torch torchvision
get_ipython().system('pip install -q "lightning-flash[image]" "torchmetrics<0.8" --no-index --find-links ../input/demo-flash-semantic-segmentation/frozen_packages')
get_ipython().system('pip install -q -U timm segmentation-models-pytorch --no-index --find-links ../input/demo-flash-semantic-segmentation/frozen_packages')
get_ipython().system("pip install -q 'kaggle-imsegm' --no-index --find-links ../input/tract-segm-eda-3d-interactive-viewer/frozen_packages")

get_ipython().system(' pip list | grep torch')
get_ipython().system(' pip list | grep lightning')
get_ipython().system(' nvidia-smi -L')


# In[ ]:


import os, glob
import pandas as pd
import matplotlib.pyplot as plt

DATASET_FOLDER = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
df_train = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))
display(df_train.head())


# In[ ]:


all_imgs = glob.glob(os.path.join(DATASET_FOLDER, "train", "case*", "case*_day*", "scans", "*.png"))
all_imgs = [p.replace(DATASET_FOLDER, "") for p in all_imgs]

print(f"images: {len(all_imgs)}")
print(f"annotated: {len(df_train['id'].unique())}")


# ## 🔎 Explore and enrich dataset
# 
# Take the input train table and parse some additiona informations

# In[ ]:


from pprint import pprint
from kaggle_imsegm.data import extract_tract_details

pprint(extract_tract_details(df_train['id'].iloc[0], DATASET_FOLDER))

df_train[['Case','Day','Slice', 'image', 'image_path', 'height', 'width']] = df_train['id'].apply(
    lambda x: pd.Series(extract_tract_details(x, DATASET_FOLDER))
)
display(df_train.head())


# Compare timeseries and stack sizes

# ## Browse the 3D image
# 
# see the full version (without importing own package) in https://www.kaggle.com/code/jirkaborovec/tract-segm-eda-3d-data-browser

# In[ ]:


from ipywidgets import interact, IntSlider
from kaggle_imsegm.data import load_volume_from_images, create_tract_segm
from kaggle_imsegm.visual import show_tract_volume

CASE = 108
DAY = 10
IMAGE_FOLDER = os.path.join(DATASET_FOLDER, "train", f"case{CASE}", f"case{CASE}_day{DAY}", "scans")
vol = load_volume_from_images(img_dir=IMAGE_FOLDER)
print(vol.shape)

df_ = df_train[(df_train["Case"] == CASE) & (df_train["Day"] == DAY)]
segm = create_tract_segm(df_vol=df_, vol_shape=vol.shape)

def interactive_show(volume):
    vol_shape = volume.shape
    interact(
        lambda x, y, z: plt.show(show_tract_volume(volume, segm, z, y, x)),
        z=IntSlider(min=0, max=vol_shape[0], step=5, value=int(vol_shape[0] / 2)),
        y=IntSlider(min=0, max=vol_shape[1], step=5, value=int(vol_shape[1] / 2)),
        x=IntSlider(min=0, max=vol_shape[2], step=5, value=int(vol_shape[2] / 2)),
    )


# In[ ]:


interactive_show(vol)


# ## Prepare flatten dataset

# In[ ]:


DATASET_IMAGES = "/kaggle/temp/dataset-flash/images"
DATASET_SEGMS = "/kaggle/temp/dataset-flash/segms"

for rdir in (DATASET_IMAGES, DATASET_SEGMS):
    for sdir in ("train", "val"):
        os.makedirs(os.path.join(rdir, sdir), exist_ok=True)


# In[ ]:


df_train['Case_Day'] = [f"case{r['Case']}_day{r['Day']}" for _, r in df_train.iterrows()]

CASES_DAYS = list(df_train['Case_Day'].unique())
VAL_SPLIT = 0.1
VAL_CASES_DAYS = CASES_DAYS[-int(VAL_SPLIT * len(CASES_DAYS)):]

print(f"all case-day: {len(CASES_DAYS)}")
print(f"val case-day: {len(VAL_CASES_DAYS)}")


# In[ ]:


import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from kaggle_imsegm.data import preprocess_tract_scan

LABELS = sorted(df_train["class"].unique())
print(LABELS)

def _chose_sfolder(df_, val_cases_days=VAL_CASES_DAYS) -> str:
    case, day = df_.iloc[0][["Case", "Day"]]
    case_day = f"case{case}_day{day}"
    return 'val' if case_day in val_cases_days else 'train'


# In[ ]:


from tqdm.auto import tqdm

_args = dict(
    dir_data=os.path.join(DATASET_FOLDER, "train"),
    dir_imgs=DATASET_IMAGES,
    dir_segm=DATASET_SEGMS,
    labels=LABELS,
)
df_train["Case_Day"] = [f"case{r['Case']}_day{r['Day']}" for _, r in df_train.iterrows()]
_= Parallel(n_jobs=6)(
    delayed(preprocess_tract_scan)(dfg, sfolder=_chose_sfolder(dfg), **_args)
    for _, dfg in tqdm(df_train.groupby("Case_Day"))
)


# In[ ]:


spl_imgs = glob.glob(os.path.join(DATASET_IMAGES, "*", "*.png"))[:3]
fig, axarr = plt.subplots(ncols=2, nrows=len(spl_imgs), figsize=(6, 3 * len(spl_imgs)))

for i, img in enumerate(spl_imgs):
    segm = img.replace(DATASET_IMAGES, DATASET_SEGMS)
    axarr[i, 0].imshow(plt.imread(img))
    axarr[i, 1].imshow(plt.imread(segm))
plt.tight_layout()


# ## Lightning⚡Flash
# 
# lets follow the Semantinc segmentation example: https://lightning-flash.readthedocs.io/en/stable/reference/semantic_segmentation.html

# In[ ]:


import torch

import flash
from flash.core.data.utils import download_data
from flash.image import SemanticSegmentation, SemanticSegmentationData


# ### 1. Create the DataModule

# In[ ]:


from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union
import albumentations as alb
from flash.core.data.io.input_transform import InputTransform
from flash.image.segmentation.input_transform import prepare_target, remove_extra_dimensions
from kaggle_imsegm.augment import FlashAlbumentationsAdapter

@dataclass
class SemanticSegmentationInputTransform(InputTransform):
    # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation

    image_size: Tuple[int, int] = (128, 128)

    def train_per_sample_transform(self) -> Callable:
        return FlashAlbumentationsAdapter([
            alb.Resize(*self.image_size),
            alb.VerticalFlip(p=0.5),
            alb.HorizontalFlip(p=0.5),
            alb.RandomRotate90(p=0.5),
            alb.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.03, rotate_limit=5, p=1.),
            alb.GaussNoise(var_limit=(0.001, 0.005), mean=0, per_channel=False, p=1.0),
            alb.OneOf([
                alb.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                alb.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            ], p=0.25),
            #alb.ElasticTransform(p=1, alpha=100, sigma=100 * 0.05, alpha_affine=100 * 0.03),
            #alb.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        ])

    def per_sample_transform(self) -> Callable:
        return FlashAlbumentationsAdapter([alb.Resize(*self.image_size)])

    def predict_input_per_sample_transform(self) -> Callable:
        return FlashAlbumentationsAdapter([alb.Resize(*self.image_size)])

    def target_per_batch_transform(self) -> Callable:
        return prepare_target

    def predict_per_batch_transform(self) -> Callable:
        return remove_extra_dimensions

    def serve_per_batch_transform(self) -> Callable:
        return remove_extra_dimensions


# In[ ]:


datamodule = SemanticSegmentationData.from_folders(
    train_folder=os.path.join(DATASET_IMAGES, 'train'),
    train_target_folder=os.path.join(DATASET_SEGMS, 'train'),
    val_folder=os.path.join(DATASET_IMAGES, 'val'),
    val_target_folder=os.path.join(DATASET_SEGMS, 'val'),
    #val_split=0.1,
    train_transform=SemanticSegmentationInputTransform,
    val_transform=SemanticSegmentationInputTransform,
    # predict_transform=SemanticSegmentationInputTransform,
    transform_kwargs=dict(image_size=(256, 256)),
    num_classes=len(LABELS) + 1,
    batch_size=24,
    num_workers=3,
)


# In[ ]:


# datamodule.show_train_batch()

fig, axarr = plt.subplots(ncols=2, nrows=5, figsize=(8, 20))
running_i = 0

for batch in datamodule.train_dataloader():
    print(batch.keys())
    for i in range(len(batch['input'])):
        segm = batch['target'][i].numpy()
        if np.sum(segm) == 0 or np.max(segm) <= 1:
            continue
        img = np.rollaxis(batch['input'][i].cpu().numpy(), 0, 3)
        axarr[running_i, 0].imshow(img)
        axarr[running_i, 1].imshow(segm)
        running_i += 1
        if running_i >= 5:
            break
    if running_i >= 5:
        break


# ### 2. Build the task

# In[ ]:


model = SemanticSegmentation(
    backbone="resnext50_32x4d",
    head="deeplabv3",
    pretrained=False,
    optimizer="AdamW",
    learning_rate=1e-2,
    num_classes=datamodule.num_classes,
)


# ### 3. Create the trainer and finetune the model

# In[ ]:


import pytorch_lightning as pl

logger = pl.loggers.CSVLogger(save_dir='logs/')
trainer = flash.Trainer(
    max_epochs=10,
    logger=logger,
    # precision=16,
    gpus=torch.cuda.device_count(),
#     limit_train_batches=0.2,
#     limit_val_batches=0.2,
)


# In[ ]:


# Train the model
trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")

# Save the model!
trainer.save_checkpoint("semantic_segmentation_model.pt")


# In[ ]:


import seaborn as sn

metrics = pd.read_csv(f'{trainer.logger.log_dir}/metrics.csv')
del metrics["step"]
metrics.set_index("epoch", inplace=True)
display(metrics.dropna(axis=1, how="all").head())
g = sn.relplot(data=metrics, kind="line")
plt.gcf().set_size_inches(12, 4)
plt.grid()


# ### 4. Segment a few images!

# In[ ]:


sample_imgs = glob.glob(os.path.join(DATASET_FOLDER, "test", "**", "*.png"), recursive=True)
if not sample_imgs:
    sample_imgs = glob.glob(os.path.join(DATASET_FOLDER, "train", "**", "*.png"), recursive=True)
print(f"images: {len(sample_imgs)}")
sample_imgs = sample_imgs[:5]

datamodule = SemanticSegmentationData.from_files(
    predict_files=sample_imgs,
    predict_transform=SemanticSegmentationInputTransform,
    transform_kwargs=dict(image_size=(256, 256)),
    batch_size=3,
)


# In[ ]:


fig, axarr = plt.subplots(ncols=5, nrows=len(sample_imgs), figsize=(15, 3 * len(sample_imgs)))
running_i = 0
for preds in trainer.predict(model, datamodule=datamodule):
    for pred in preds:
        # print(pred.keys())
        img = np.rollaxis(pred['input'].cpu().numpy(), 0, 3)
        axarr[running_i, 0].imshow(img)
        for j, seg in enumerate(pred['preds'].cpu().numpy()):
            axarr[running_i, j + 1].imshow(seg)
        running_i += 1


# In[ ]:


fig, axarr = plt.subplots(ncols=2, nrows=len(sample_imgs), figsize=(8, 4 * len(sample_imgs)))
running_i = 0
for preds in trainer.predict(model, datamodule=datamodule, output="labels"):
    for pred in preds:
        # print(pred)
        img = plt.imread(sample_imgs[running_i])
        axarr[running_i, 0].imshow(img, cmap="gray")
        axarr[running_i, 1].imshow(pred)
        running_i += 1


# # Inference

# In[ ]:


model = SemanticSegmentation.load_from_checkpoint(
    "semantic_segmentation_model.pt"
)


# In[ ]:


df_pred = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
sfolder = "test"
display(df_pred.head())

if df_pred.empty:
    sfolder = "train"
    df_pred = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))
    df_pred = df_pred[df_pred["id"].str.startswith("case123_day")]

os.makedirs(os.path.join(DATASET_IMAGES, sfolder), exist_ok=True)


# In[ ]:


from pprint import pprint
from kaggle_imsegm.data import extract_tract_details

pprint(extract_tract_details(df_pred['id'].iloc[0], DATASET_FOLDER, folder=sfolder))

df_pred[['Case','Day','Slice', 'image', 'image_path', 'height', 'width']] = df_pred['id'].apply(
    lambda x: pd.Series(extract_tract_details(x, DATASET_FOLDER, folder=sfolder))
)
df_pred["Case_Day"] = [f"case{r['Case']}_day{r['Day']}" for _, r in df_pred.iterrows()]
display(df_pred.head())


# ## Predictions for test scans

# In[ ]:


from joblib import Parallel, delayed
from kaggle_imsegm.data import preprocess_tract_scan

_args = dict(
    dir_data=os.path.join(DATASET_FOLDER, sfolder),
    dir_imgs=DATASET_IMAGES,
    dir_segm=None,
    labels=LABELS,
    sfolder=sfolder,
)
test_scans = Parallel(n_jobs=6)(
    delayed(preprocess_tract_scan)(dfg, **_args)
    for _, dfg in df_pred.groupby("Case_Day")
)


# In[ ]:


import numpy as np
from itertools import chain
from kaggle_imsegm.mask import rle_encode

preds = []
for test_imgs in test_scans:
    dm = SemanticSegmentationData.from_files(
        predict_files=test_imgs,
        predict_transform=SemanticSegmentationInputTransform,
        transform_kwargs=dict(image_size=(256, 256)),
        num_classes=len(LABELS) + 1,
        batch_size=10,
        num_workers=3,
    )
    pred = trainer.predict(model, datamodule=dm, output="labels")
    pred = list(chain(*pred))
    for img, seg in zip(test_imgs, pred):
        rle = rle_encode(np.array(seg)) if np.sum(seg) > 1 else {}
        name, _ = os.path.splitext(os.path.basename(img))
        id_ = "_".join(name.split("_")[:4])
        preds += [{"id": id_, "class": lb, "predicted": rle.get(i + 1, "")} for i, lb in enumerate(LABELS)]

df_pred = pd.DataFrame(preds)
display(df_pred[df_pred["predicted"] != ""].head())


# ## Finalize submissions

# In[ ]:


df_ssub = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
del df_ssub['predicted']
df_pred = df_ssub.merge(df_pred, on=['id','class'])

df_pred[['id', 'class', 'predicted']].to_csv("submission.csv", index=False)

get_ipython().system('head submission.csv')

