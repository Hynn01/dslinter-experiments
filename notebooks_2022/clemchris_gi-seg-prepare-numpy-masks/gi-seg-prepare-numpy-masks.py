#!/usr/bin/env python
# coding: utf-8

# # GI-Seg Prepare NumPy Masks
# 
# Let's create a dataset where masks are stored as uint8 NumPy arrays to make their processing easier.
# 
# This is basically just a refactor and small extension of Awsaf's great [UWMGI: Mask Data](https://www.kaggle.com/code/awsaf49/uwmgi-mask-data) and [UWMGI: 2.5D stride=2 Data](https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-stride-2-data) notebooks so give them an upvote if you find this useful.
# 
# ## Link to dataset
# [UW-Madison GI Tract Image Segmentation Masks](https://www.kaggle.com/datasets/clemchris/uw-madison-gi-tract-image-segmentation-masks)
# 
# ## Link to Train & Infer Notebook
# [GI-Seg PyTorch ⚡ Train & Infer](https://www.kaggle.com/code/clemchris/gi-seg-pytorch-train-infer)
# 
# ## Sources
# - Awsaf's [UWMGI: Mask Data](https://www.kaggle.com/code/awsaf49/uwmgi-mask-data)
# - Awsaf's [UWMGI: 2.5D stride=2 Data](https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-stride-2-data)
# - Awsaf's [UWMGI: Unet [Train] [PyTorch]](https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch)

# # Imports

# In[ ]:


import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.notebook import tqdm


# # Paths & Settings

# In[ ]:


KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"
OUTPUT_DIR = KAGGLE_DIR / "working"

INPUT_DATA_DIR = INPUT_DIR / "uw-madison-gi-tract-image-segmentation"
N_SPLITS = 5
RANDOM_SEED = 2022

# For 2.5D Data
CHANNELS = 3
STRIDE = 2

OUTPUT_DATA_DIR = OUTPUT_DIR / INPUT_DATA_DIR.stem
OUTPUT_DATA_DIR.mkdir(exist_ok=True)

DEBUG = True


# # Preprocess Train Data

# ## Load Train Data

# In[ ]:


train_df = pd.read_csv(INPUT_DATA_DIR / "train.csv")

if DEBUG:
    train_df = train_df.head(999)
    print(f"{len(train_df)}")
    display(train_df.head())


# ## Merge IDs

# In[ ]:


def merge_ids(df):
    df["segmentation"] = df.segmentation.fillna("")
    df["rle_len"] = df.segmentation.map(len)

    # Aggregate classes
    df2 = df.groupby(["id"])["class"].agg(list).to_frame().reset_index()
    df2 = df2.rename(columns={"class": "classes"})

    # Aggregate segmentations
    df2 = df2.merge(df.groupby(["id"])["segmentation"].agg(list).to_frame().reset_index())
    
    # Sum RLE Lengths
    df2 = df2.merge(df.groupby(["id"])["rle_len"].agg(sum).to_frame().reset_index())

    df = df.drop(columns=["class", "segmentation", "rle_len"])
    df = df.groupby(["id"]).head(1).reset_index(drop=True)
    df = df.merge(df2, on=["id"])
    
    df["empty"] = (df.rle_len==0)

    return df


# In[ ]:


train_df = merge_ids(train_df)

if DEBUG:
    print(f"{len(train_df)}")
    display(train_df[~train_df["empty"]].head())


# ## Extract metadata from id

# In[ ]:


def extract_metadata_from_id(df):
    df[["case", "day", "slice"]] = df["id"].str.split("_", n=2, expand=True)

    df["case"] = df["case"].str.replace("case", "").astype(int)
    df["day"] = df["day"].str.replace("day", "").astype(int)
    df["slice"] = df["slice"].str.replace("slice_", "").astype(int)

    return df


# In[ ]:


train_df = extract_metadata_from_id(train_df)

if DEBUG:
    print(f"{len(train_df)}")
    display(train_df.head())


# ## Create Folds

# In[ ]:


def create_folds(df, n_splits, random_seed):
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df["empty"], groups=df["case"])):
        df.loc[val_idx, "fold"] = fold

    return df


# In[ ]:


train_df = create_folds(train_df, N_SPLITS, RANDOM_SEED)

if DEBUG:
    print(f"{len(train_df)}")
    display(train_df.head())


# ## Add Image Path

# In[ ]:


def add_image_path(df, data_dir, stage="train"):
    image_paths = [str(path) for path in (data_dir / stage).rglob("*.png")]
    path_df = pd.DataFrame(image_paths, columns=["image_path"])

    path_df = extract_metadata_from_path(path_df)

    df = df.merge(path_df, on=["case", "day", "slice"])

    return df


def extract_metadata_from_path(path_df):
    path_df[["parent", "case_day", "scans", "file_name"]] = path_df["image_path"].str.rsplit("/", n=3, expand=True)

    path_df[["case", "day"]] = path_df["case_day"].str.split("_", expand=True)
    path_df["case"] = path_df["case"].str.replace("case", "")
    path_df["day"] = path_df["day"].str.replace("day", "")

    path_df[["slice", "width", "height", "spacing", "spacing_"]] = (
        path_df["file_name"].str.replace("slice_", "").str.replace(".png", "").str.split("_", expand=True)
    )
    path_df = path_df.drop(columns=["parent", "case_day", "scans", "file_name", "spacing_"])

    numeric_cols = ["case", "day", "slice", "width", "height", "spacing"]
    path_df[numeric_cols] = path_df[numeric_cols].apply(pd.to_numeric)

    return path_df


# In[ ]:


train_df = add_image_path(train_df, INPUT_DATA_DIR)

if DEBUG:
    print(f"{len(train_df)}")
    display(train_df.head())


# ## Add Mask Path

# In[ ]:


def add_mask_path(df):
    df["mask_path"] = df["image_path"].str.replace(".png", ".npy").str.replace("scans", "masks")
    df["mask_path"] = df["mask_path"].str.replace("input", "working")
    return df


# In[ ]:


train_df = add_mask_path(train_df)

if DEBUG:
    print(f"{len(train_df)}")
    display(train_df.head())


# ## Add Image Paths (for 2.5D)

# In[ ]:


def add_image_paths(df, channels, stride):
    for i in range(channels):
        df[f"image_path_{i:02}"] = df.groupby(["case", "day"])["image_path"].shift(-i * stride).fillna(method="ffill")
    
    image_path_columns = [f"image_path_{i:02d}" for i in range(channels)]
    df["image_paths"] = df[image_path_columns].values.tolist()
    df = df.drop(columns=image_path_columns)
    
    return df


# In[ ]:


train_df = add_image_paths(train_df, CHANNELS, STRIDE)

if DEBUG:
    print(f"{len(train_df)}")
    display(train_df.head())


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# # Create NumPy Masks

# ## Helpers

# In[ ]:


def load_mask(row):
    shape = (row.height, row.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)

    for i, rle in enumerate(row.segmentation):
        if rle:
            mask[..., i] = rle_decode(rle, shape[:2])

    return mask * 255


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


def save_array(file_path, array):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, array)


# ## Save Masks

# In[ ]:


for row in tqdm(train_df.itertuples(), total=len(train_df)):
    mask = load_mask(row)
    save_array(row.mask_path, mask)


# # Visualize

# ## Helpers

# In[ ]:


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = image.astype("float32")  # original is uint16
    image /= image.max()
    return image


def load_images(paths):
    images = [load_image(path) for path in paths]
    images = np.stack(images, axis=-1)
    return images


def load_mask(path):
    mask = np.load(path)
    mask = mask.astype("float32")
    mask /= 255.0
    return mask
    
    
def show_image(image, mask=None):
    plt.imshow(image, cmap="bone")

    if mask is not None:
        plt.imshow(mask, alpha=0.5)

        handles = [
            Rectangle((0, 0), 1, 1, color=_c) for _c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
        ]
        labels = ["Stomach", "Large Bowel", "Small Bowel"]

        plt.legend(handles, labels)

    plt.axis("off")
    

def show_grid(train_df, nrows, ncols):
    fig, _ = plt.subplots(figsize=(5 * ncols, 5 * nrows))

    train_df_sampled = train_df[~train_df["empty"]].sample(n=nrows * ncols)
    for i, row in enumerate(train_df_sampled.itertuples()):

        image = load_images(row.image_paths)
        #image = load_image(row.image_path)

        mask = load_mask(row.mask_path)

        plt.subplot(nrows, ncols, i + 1)
        plt.tight_layout()
        plt.title(row.id)

        show_image(image, mask)


# ## Show grid

# In[ ]:


nrows = 4
ncols = 4
show_grid(train_df, nrows, ncols)


# # Fix Mask Path & Save Preprocessed Train DataFrame

# In[ ]:


train_df["mask_path"] = train_df["mask_path"].str.replace("working/uw-madison-gi-tract-image-segmentation", "input/uw-madison-gi-tract-image-segmentation-masks")

train_df.to_csv(OUTPUT_DATA_DIR / "train_preprocessed.csv", index=False)


# # Create Archive & Cleanup

# In[ ]:


shutil.make_archive(
    base_name=f"{OUTPUT_DATA_DIR}-masks",
    format="zip",
    root_dir=OUTPUT_DATA_DIR,
)


# In[ ]:


rm /kaggle/working/uw-madison-gi-tract-image-segmentation -rf

