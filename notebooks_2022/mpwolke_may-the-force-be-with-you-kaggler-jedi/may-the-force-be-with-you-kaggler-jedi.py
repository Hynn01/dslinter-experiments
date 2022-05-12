#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# It isn't a mask, it's just a doormat. 
# 
# ![](https://http2.mlstatic.com/D_NQ_NP_631359-MLB44181441120_112020-O.jpg)produto.mercadolivre.com.br

# In[ ]:


from fastai.vision.all import *
from fastai.vision.widgets import * 


# In[ ]:


get_ipython().system('cp -r ../input/images-from-star-wars-movies/')


# In[ ]:


path = Path('./images-from-star-wars-movies')
fns = get_image_files(path)
failed = verify_images(fns)
failed.map(Path.unlink);


# In[ ]:


starwars = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(224))


# In[ ]:


#dls = starwars.dataloaders(path) #Saved for next time since it returned "'NoneType' object is not iterable"


# In[ ]:


from fastai.vision.all import *
from fastai.imports import *
from fastai.vision.data import *
from fastai import *
import numpy as np
import fastai
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


path = Path("/kaggle/input/images-from-star-wars-movies")
path.ls()


# In[ ]:


np.random.seed(42)
data = ImageDataLoaders.from_folder(path, train=".", valid_pct=0.2, item_tfms=RandomResizedCrop(512, min_scale=0.75),
                                    bs=32,batch_tfms=[*aug_transforms(size=256, max_warp=0), Normalize.from_stats(*imagenet_stats)],num_workers=0)


# In[ ]:


data.show_batch(nrows=6, figsize=(7,8))#nrows 3were clumsy overlapping


# In[ ]:


data.show_batch(nrows=4, figsize=(12,16))#nrows =2 were clumsy/overlapping


# In[ ]:


data.show_batch(nrows=5, figsize=(7,8))#nrows 1 clumsy overlapping


# In[ ]:


path2 = Path("/kaggle/input/images-from-star-wars-movies/Episode VI - Return of the Jedi")
path2.ls()


# In[ ]:


np.random.seed(42)
data2 = ImageDataLoaders.from_folder(path2, train=".", valid_pct=0.2, item_tfms=RandomResizedCrop(512, min_scale=0.75),
                                    bs=32,batch_tfms=[*aug_transforms(size=256, max_warp=0), Normalize.from_stats(*imagenet_stats)],num_workers=0)


# In[ ]:


data2.show_batch(nrows=5, figsize=(7,8))


# In[ ]:


data2.show_batch(nrows=6, figsize=(7,8))#nrows were 2 before clumsy/overlapping


# In[ ]:


data2.show_batch(nrows=6, figsize=(7,8))


# In[ ]:


def _add1(x): return x+1
dumb_tfm = RandTransform(enc=_add1, p=0.5)
start,d1,d2 = 2,False,False
for _ in range(40):
    t = dumb_tfm(start, split_idx=0)
    if dumb_tfm.do: test_eq(t, start+1); d1=True
    else:           test_eq(t, start)  ; d2=True
assert d1 and d2
dumb_tfm


# #Below: That's the Darth Vader funniest image.

# In[ ]:


from PIL import Image

img = Image.open("../input//images-from-star-wars-movies/Episode VII - The Force Awakens/Episode VII - The Force Awakens_69.jpg")
img


# In[ ]:


_,axs = subplots(1,2)
show_image(img, ctx=axs[0], title='original')
show_image(img.flip_lr(), ctx=axs[1], title='flipped');


# ![](https://pics.me.me/monday-may-the-force-be-with-you-23465972.png)me.me

# #Thank you Michal Bogacz, michau96, for this wonderful Dataset.
