#!/usr/bin/env python
# coding: utf-8

# [![](https://img.youtube.com/vi/FnV0thLS1Fs/0.jpg)](https://www.youtube.com/watch?v=FnV0thLS1Fs)

# lunana  
# last update 2022 05 05  
# ゆっくりしていってね

# # Data list  
# * [**train.csv**](#train.csv)  
# * [**train_image**](#train_image)  
# * [**test.csv**](#test.csv)  
# * [**sample_submission.csv**](#sample_submission.csv)

# **霊夢:今日は画像のコンペだね。  
# 魔理沙:まずは概略を見るぞ。**
# 
# **Reimu: Today is an image competition.  
# Marisa: First, let's take a look at the outline.**

# 2019年には、世界中で推定500万人が胃腸管の癌と診断されました。これらの患者のうち、約半数が放射線療法の対象であり、通常、1日10〜15分かけて1〜6週間投与されます。放射線腫瘍医は、胃や腸を避けながら、腫瘍に向けられたX線ビームを使用して高線量の放射線を照射しようとします。統合磁気共鳴画像法やMR-Linacsとしても知られる線形加速器システムなどの新しい技術により、腫瘍学者は腫瘍と腸の日々の位置を視覚化することができます。これは日々変化する可能性があります。これらのスキャンでは、放射線腫瘍医は、X線ビームの方向を調整して腫瘍への線量送達を増やし、胃と腸を避けるために、胃と腸の位置を手動で輪郭を描く必要があります。これは時間と労力を要するプロセスであり、治療を1日15分から1時間に延長できます。これは、ディープラーニングがセグメンテーションプロセスの自動化に役立つ場合を除いて、患者が耐えるのが難しい場合があります。胃と腸をセグメント化する方法は、治療をはるかに速くし、より多くの患者がより効果的な治療を受けることを可能にします。
# 
# UW-マディソンカーボンがんセンターは、MR-Linacベースの放射線治療のパイオニアであり、2015年以来、毎日の解剖学的構造に基づいてMRIガイド放射線治療で患者を治療してきました。UW-マディソンは、治療を受けた患者の匿名MRIを提供するこのプロジェクトを支援することに寛大に同意しました。 UW-マディソンカーボンがんセンターで。ウィスコンシン大学マディソン校は、ウィスコンシン州マディソンにある公立のランドグラント研究大学です。ウィスコンシンのアイデアは、州、国、そして世界に対する大学の誓約であり、彼らの努力はすべての市民に利益をもたらすでしょう。
# 
# このコンテストでは、MRIスキャンで胃と腸を自動的にセグメント化するモデルを作成します。MRIスキャンは、放射線治療中の別々の日に1〜5回のMRIスキャンを受けた実際の癌患者からのものです。これらのスキャンのデータセットに基づいてアルゴリズムを作成し、がん患者がより良いケアを受けるのに役立つ創造的な深層学習ソリューションを考え出します。  
# 
# In 2019, an estimated 5 million people were diagnosed with a cancer of the gastro-intestinal tract worldwide. Of these patients, about half are eligible for radiation therapy, usually delivered over 10-15 minutes a day for 1-6 weeks. Radiation oncologists try to deliver high doses of radiation using X-ray beams pointed to tumors while avoiding the stomach and intestines. With newer technology such as integrated magnetic resonance imaging and linear accelerator systems, also known as MR-Linacs, oncologists are able to visualize the daily position of the tumor and intestines, which can vary day to day. In these scans, radiation oncologists must manually outline the position of the stomach and intestines in order to adjust the direction of the x-ray beams to increase the dose delivery to the tumor and avoid the stomach and intestines. This is a time-consuming and labor intensive process that can prolong treatments from 15 minutes a day to an hour a day, which can be difficult for patients to tolerate—unless deep learning could help automate the segmentation process. A method to segment the stomach and intestines would make treatments much faster and would allow more patients to get more effective treatment.
# 
# The UW-Madison Carbone Cancer Center is a pioneer in MR-Linac based radiotherapy, and has treated patients with MRI guided radiotherapy based on their daily anatomy since 2015. UW-Madison has generously agreed to support this project which provides anonymized MRIs of patients treated at the UW-Madison Carbone Cancer Center. The University of Wisconsin-Madison is a public land-grant research university in Madison, Wisconsin. The Wisconsin Idea is the university's pledge to the state, the nation, and the world that their endeavors will benefit all citizens.
# 
# In this competition, you’ll create a model to automatically segment the stomach and intestines on MRI scans. The MRI scans are from actual cancer patients who had 1-5 MRI scans on separate days during their radiation treatment. You'll base your algorithm on a dataset of these scans to come up with creative deep learning solutions that will help cancer patients get better care.

# **霊夢:Sartoriusコンペとそっくりだな。**  　
# 
# **Reimu: It's just like the Sartorius competition.**
# 
# https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation

# # train.csv

# **霊夢:次はtrainデータを見てみよう**  
# 
# **Reimu: Let's look at the train data next**

# train.csv - IDs and masks for all training objects.train - a folder of case/day folders, each containing slice images for a particular case on a given day.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from os import listdir
import cv2
from PIL import Image


# In[ ]:


df_train=pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')
df_train.head()


# In[ ]:


len(df_train)


# In[ ]:


df_train['class'].unique()


# In[ ]:


df_train.loc[194]['segmentation']


# In[ ]:


df_train.loc[194]['id']


# # train_image

# **霊夢:train画像を見てみよう**
# 
# **Reimu: Let's take a look at the train image**

# train - a folder of case/day folders, each containing slice images for a particular case on a given day.

# In[ ]:


im=Image.open('../input/uw-madison-gi-tract-image-segmentation/train/case114/case114_day0/scans/slice_0006_360_310_1.50_1.50.png')
im_list = np.asarray(im)
plt.imshow(im_list)
plt.show()


# In[ ]:


im=Image.open('../input/uw-madison-gi-tract-image-segmentation/train/case114/case114_day0/scans/slice_0023_360_310_1.50_1.50.png')
im_list = np.asarray(im)
plt.imshow(im_list)
plt.show()


# In[ ]:


im=Image.open('../input/uw-madison-gi-tract-image-segmentation/train/case123/case123_day20/scans/slice_0065_266_266_1.50_1.50.png')
im_list = np.asarray(im)
plt.imshow(im_list)
plt.show()


# **霊夢:画像に色付けをしてどこが大腸か小腸かわかるようにするよ。  
# 魔理沙:Awsaf氏のコードを参考にするよ。**  
# 
# **Reimu: I'll color the image so you can see where the large intestine or the small intestine is.  
# Marisa: I'll refer to Mr. Awsaf's code.**
# 
# https://www.kaggle.com/code/awsaf49/uwmgi-mask-data

# In[ ]:


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape) 

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[ ]:


import tensorflow as tf
from tqdm import tqdm
tqdm.pandas()
from matplotlib.patches import Rectangle
from glob import glob


# In[ ]:


def get_metadata(row):
    data = row['id'].split('_')
    case = int(data[0].replace('case',''))
    day = int(data[1].replace('day',''))
    slice_ = int(data[-1])
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row

def path2info(row):
    path = row['image_path']
    data = path.split('/')
    slice_ = int(data[-1].split('_')[1])
    case = int(data[-3].split('_')[0].replace('case',''))
    day = int(data[-3].split('_')[1].replace('day',''))
    width = int(data[-1].split('_')[2])
    height = int(data[-1].split('_')[3])
    row['height'] = height
    row['width'] = width
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row


# In[ ]:


def id2mask(id_):
    idf = df_train[df_train['id']==id_]
    wh = idf[['height','width']].iloc[0]
    shape = (wh.height, wh.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(['large_bowel', 'small_bowel', 'stomach']):
        cdf = idf[idf['class']==class_]
        rle = cdf.segmentation.squeeze()
        if len(cdf) and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask

def rgb2gray(mask):
    pad_mask = np.pad(mask, pad_width=[(0,0),(0,0),(1,0)])
    gray_mask = pad_mask.argmax(-1)
    return gray_mask

def gray2rgb(mask):
    rgb_mask = tf.keras.utils.to_categorical(mask, num_classes=4)
    return rgb_mask[..., 1:].astype(mask.dtype)


# In[ ]:


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype('float32') 
    img = (img - img.min())/(img.max() - img.min())*255.0 
    img = img.astype('uint8')
    return img

def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='bone')
    
    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = [ "Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles,labels)
    plt.axis('off')


# In[ ]:


df_train = df_train.progress_apply(get_metadata, axis=1)
df_train.head()


# In[ ]:


paths = glob('/kaggle/input/uw-madison-gi-tract-image-segmentation/train/*/*/*/*')
path_df = pd.DataFrame(paths, columns=['image_path'])
path_df = path_df.progress_apply(path2info, axis=1)
df_train = df_train.merge(path_df, on=['case','day','slice'])
df_train.head()


# In[ ]:


row=1; col=4
plt.figure(figsize=(5*col,5*row))
for i, id_ in enumerate(df_train[~df_train.segmentation.isna()].sample(frac=1.0)['id'].unique()[:row*col]):
    print(id_)
    img = load_img(df_train[df_train['id']==id_].image_path.iloc[0])
    mask = id2mask(id_)*255
    show_img(img, mask=mask)
    plt.tight_layout()


# **魔理沙:縦に4つ表示してるけど、表示できたからまあ良し。**  
# 
# **Marisa: I'm displaying four vertically, but it's okay because I was able to display them.**

# # sample_submission.csv

# **魔理沙:最後にsample_submissionを見てみよう**  
# 
# **Marisa: Finally, let's take a look at sample_submission**

# sample_submission.csv - a sample submission file in the correct format

# 提出ファイル  
# 送信ファイルのサイズを減らすために、メトリックはピクセル値にランレングスエンコーディングを使用します。セグメンテーションのインデックスの完全なリストを送信する代わりに、開始位置と実行長を含む値のペアを送信します。たとえば、「1 3」は、ピクセル1から開始し、合計3ピクセル（1,2,3）を実行することを意味します。  
# 
# エンコード時には、マスクはバイナリである必要があることに注意してください。つまり、画像内のすべてのオブジェクトのマスクが1つの大きなマスクに結合されます。値0はマスクされていないピクセルを示し、値1はマスクされているピクセルを示します。  
# 
# 競技形式では、スペースで区切られたペアのリストが必要です。たとえば、「1 3 10 5」は、ピクセル1、2、3、10、11、12、13、14がマスクに含まれることを意味します。メトリックは、ペアがソートされ、正であり、デコードされたピクセル値が複製されていないことを確認します。ピクセルには上から下、次に左から右に番号が付けられます。1はピクセル（1,1）、2はピクセル（2,1）などです。  
# 
# ファイルにはヘッダーが含まれ、次の形式である必要があります。  
# 
# id、class、predicted  
# 1、large_bowel、1 1 5 1  
# 1、small_bowel、1 1  
# 1、stomach、1 1  
# 2、large_bowel、1 5217 
# など。

# In[ ]:


sumple_submission=pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv')
sumple_submission


# **霊夢:今回はここまでです。**
# 
# **Reimu: That's all for this time.**
