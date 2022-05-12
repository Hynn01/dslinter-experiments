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


# # STEP:1 Download images of alligator and crocodile

# In[ ]:


from fastcore.all import *
import time

def search_images(term, max_images=200):
    url = 'https://duckduckgo.com/'
    res = urlread(url,data={'q':term})
    searchObj = re.search(r'vqd=([\d-]+)\&', res)
    requestUrl = url + 'i.js'
    params = dict(l='us-en', o='json', q=term, vqd=searchObj.group(1), f=',,,', p='1', v7exp='a')
    urls,data = set(),{'next':1}
    while len(urls)<max_images and 'next' in data:
        data = urljson(requestUrl,data=params)
        urls.update(L(data['results']).itemgot('image'))
        requestUrl = url + data['next']
        time.sleep(0.2)
    return L(urls)[:max_images]


# Let's start by searching for a alligator photo and seeing what kind of result we get. We'll start by getting URLs from a search:

# In[ ]:


urls=search_images('alligator photos',max_images=10)
urls[1]


# ...and then download a URL and take a look at it:
# 
# 

# In[ ]:


from fastdownload import download_url
dest='alligator.jpg'
download_url(urls[0],dest,show_progress=False)

from fastai.vision.all import *
im=Image.open(dest)
im.to_thumb(256,256)



# Now let's do the same with "crocodile photos":
# 
# 

# In[ ]:


download_url(search_images('crocodile photos',max_images=10)[1])


# In[ ]:


download_url(search_images('crocodile photos',max_images=1)[0] ,'crocodile.jpg',show_progress=False)
Image.open('crocodile.jpg').to_thumb(256,256)


# Our searches seem to be giving reasonable results, so let's grab 200 examples of each of "alligator" and "crocodile" photos, and save each group of photos to a different folder:
# 
# 

# In[ ]:


searches='alligator','crocodile'
path=Path('alligator_or_not')

for o in searches:
    dest=(path/o)
    dest.mkdir(exist_ok=True,parents=True)
    download_images(dest,urls=search_images(f'{o} photo'))
    resize_images(path/o, max_size=400,dest=path/o)
    


# # Step 2: Train our model

# Some photos might not download correctly which could cause our model training to fail, so we'll remove them:

# 

# In[ ]:


failed=verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)


# To train a model, we'll need DataLoaders, which is an object that contains a training set (the images used to create a model) and a validation set (the images used to check the accuracy of a model -- not used during training). In fastai we can create that easily using a DataBlock, and view sample images from it:

# In[ ]:


dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n=6)


# Now we're ready to train our model. The fastest widely used computer vision model is resnet18. You can train this in a few minutes, even on a CPU! (On a GPU, it generally takes under 10 seconds...)
# 
# fastai comes with a helpful fine_tune() method which automatically uses best practices for fine tuning a pre-trained model, so we'll use that.

# In[ ]:


learn=vision_learner(dls,resnet18,metrics=error_rate)
learn.fine_tune(3)


# Generally when I run this I see 100% accuracy on the validation set (although it might vary a bit from run to run).
# 
# "Fine-tuning" a model means that we're starting with a model someone else has trained using some other dataset (called the pretrained model), and adjusting the weights a little bit so that the model learns to recognise your particular dataset. In this case, the pretrained model was trained to recognise photos in imagenet, and widely-used computer vision dataset with images covering 1000 categories) For details on fine-tuning and why it's important, check out the free fast.ai course.

# 

# # Step 3: Use our model (and build your own!)

# In[ ]:


is_alligator,pred_idx,probs=learn.predict(PILImage.create('alligator.jpg'))
print(f'This is a:{is_alligator}.')
print(f'probability it is  a alligator:{probs[pred_idx]:.4f}')


# In[ ]:


probs


# In[ ]:




