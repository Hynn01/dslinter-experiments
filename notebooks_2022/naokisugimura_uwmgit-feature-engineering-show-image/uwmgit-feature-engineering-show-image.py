#!/usr/bin/env python
# coding: utf-8

# # Import

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
import cv2
from matplotlib.patches import Rectangle


# In[ ]:


df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')


# In[ ]:


df


# # RLE decoder and encoder

# Inspired by [UWMGI: Mask Data](https://www.kaggle.com/code/awsaf49/uwmgi-mask-data)

# In[ ]:


def rle_decoder(size, mask):
    mask_list = np.asarray(mask.split(), dtype=int)
    starts = mask_list[0::2] - 1
    lengths = mask_list[1::2]
    ends = starts + lengths
    img = np.zeros(size[0]*size[1], dtype=np.uint8)
    for i, j in zip(starts, ends):
        img[i:j] = 1
    return img.reshape(size)  # Return  the image which annotation value is 1
    

def rle_encoder(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# # Feature Engineering

# Inspired by [Detailed data visualization for beignners](https://www.kaggle.com/code/subinek/detailed-data-visualization-for-beignners)

# return data
# 
# * Path
# * CaseNum_Day
# * SliceNum
# * Case
# * Day
# * Slice
# * Height
# * Width
# * id
# * class
# * segmentation (encode)

# In[ ]:


def feat_eng():
    #Generate the list of images
    images_list = glob('../input/uw-madison-gi-tract-image-segmentation/train/*/*/scans/*.png')

    #extract details from the path
    images_metadata = pd.DataFrame({'Path':images_list})


    
    #split the path to get individual parameters
    path_split = images_metadata['Path'].str.split('/',n=7,expand=True)
    

    #we need to extract [5] and [7]
    images_metadata['CaseNum_Day'] = path_split[5]
    images_metadata['SliceNum'] = path_split[7]

    #Resplitting to extract case, day, slice, height and width
    case_split = images_metadata['CaseNum_Day'].str.split('_',n=2, expand=True)
    images_metadata['Case'] = case_split[0].str[4:].astype(int)
    images_metadata['Day'] = case_split[1].str[3:].astype(int)

    #Resplitting to extract slice, height and width
    fileName_split = images_metadata['SliceNum'].str.split('_',n=6, expand=True)
    images_metadata['Slice'] = fileName_split[1].astype(int)
    images_metadata['Height'] = fileName_split[2].astype(int)
    images_metadata['Width'] = fileName_split[3].astype(int)
    
    images_metadata['id'] = path_split[5] + '_slice_' + fileName_split[1]
    
    # merge train.csv and images_metadata
    df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')
    
    df = pd.merge(images_metadata, df, on='id')
    
    # remove null mask row
    df = df[df['segmentation'].notnull()] 
    df = df.reset_index(drop=True)
    
    a = df.groupby(['id'])['segmentation'].apply(list)
    b = df.groupby(['id'])['class'].apply(list)
    
    dataframe = pd.DataFrame({'id':df.id.unique()})
    dataframe = pd.merge(dataframe, a, on = 'id')    
    dataframe = pd.merge(dataframe, b, on = 'id')    
    dataframe = pd.merge(dataframe, df, on = 'id')   
    dataframe = dataframe.drop(['class_y', 'segmentation_y'], axis = 1)
    dataframe = dataframe.rename(columns={'segmentation_x': 'segmentation', 'class_x': 'class'})

    return  dataframe


# In[ ]:


a = feat_eng()
a


# # show image

# Inspired by [UWMGI: Mask Data](https://www.kaggle.com/code/awsaf49/uwmgi-mask-data)

# In[ ]:


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype('float32') # original is uint16
    img = (img - img.min())/(img.max() - img.min())*255.0 # scale image to [0, 255]
    img = img.astype('uint8')
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def shows(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
#     plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='bone')
    
    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = [ "Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles,labels)
    plt.axis('off')


# In[ ]:


def show_img(idx):
    train_df = feat_eng()
#     classes = dict(zip(train_df['class'].unique(), [0,1,2]))
    row = train_df.iloc[idx]
    path = row['Path']
    size = (row['Height'], row['Width'])
    part = row['class']
    seg = row['segmentation']
    img = load_img(path)
    seg_dict = dict(zip(part,seg))
    ans = 0
    zero_img = np.zeros(size, dtype=np.uint8)
    if 'large_bowel' in list(seg_dict.keys()):
        label = seg_dict['large_bowel']
        label = rle_decoder(size, label)
        l_label = np.stack([label, zero_img, zero_img], axis=-1)
        ans += l_label
    if 'small_bowel' in list(seg_dict.keys()):
        label = seg_dict['small_bowel']
        label = rle_decoder(size, label)
        s_label = np.stack([zero_img, label, zero_img], axis=-1)
        ans += s_label
    if 'stomach' in list(seg_dict.keys()):
        label = seg_dict['stomach']
        label = rle_decoder(size, label)
        st_label = np.stack([zero_img, zero_img, label], axis=-1)
        ans += st_label
    ans *=255
    plt.title(f'{part}')
    shows(img, ans)


# In[ ]:


show_img(1)


# # datasets

# In[ ]:


def mklabel(seg_dict, size):
    ans = 0
    zero_img = np.zeros(size, dtype=np.uint8)
    if 'large_bowel' in list(seg_dict.keys()):
        label = seg_dict['large_bowel']
        label = rle_decoder(size, label)
        l_label = np.stack([label, zero_img, zero_img], axis=-1)
        ans += l_label
    if 'small_bowel' in list(seg_dict.keys()):
        label = seg_dict['small_bowel']
        label = rle_decoder(size, label)
        s_label = np.stack([zero_img, label, zero_img], axis=-1)
        ans += s_label
    if 'stomach' in list(seg_dict.keys()):
        label = seg_dict['stomach']
        label = rle_decoder(size, label)
        st_label = np.stack([zero_img, zero_img, label], axis=-1)
        ans += st_label
    return label


# In[ ]:


from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor

class UWMGITdatasets(Dataset):
    def __init__(self, transform = None):
        self.df = feat_eng()
        self.img_path = self.df['Path']
        self.classes = self.df['class']
        self.seg = self.df['segmentation']
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        self.seg_dict = dict(zip(self.classes[idx], self.seg[idx]))
        self.size = (self.df.loc[idx,'Height'], self.df.loc[idx,'Width'])
        self.label = mklabel(seg_dict = self.seg_dict, size = self.size)
        self.img = read_image(self.img_path[idx])
        self.label = ToTensor(self.label)
        self.label = self.img + self.label
        
#         if self.transform:
#             out_data = self.transform(out_data)

        return self.img, self.label


# In[ ]:


a = feat_eng()


# In[ ]:


d = a.loc[1,'Path']


# In[ ]:


d


# In[ ]:


read_image(d)


# In[ ]:




