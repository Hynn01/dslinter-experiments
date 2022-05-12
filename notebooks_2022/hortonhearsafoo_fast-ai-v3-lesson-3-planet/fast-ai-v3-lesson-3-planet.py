#!/usr/bin/env python
# coding: utf-8

# ## Multi-label prediction with Planet Amazon dataset

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *


# In[ ]:


path = Path('/kaggle/input/')
path.ls()


# ## Multiclassification

# Contrary to the pets dataset studied in last lesson, here each picture can have multiple labels. If we take a look at the csv file containing the labels (in 'train_v2.csv' here) we see that each 'image_name' is associated to several tags separated by spaces.

# In[ ]:


df = pd.read_csv(path/'train_v2.csv')
df.head()


# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageItemList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.

# In[ ]:


tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# We use parentheses around the data block pipeline below, so that we can use a multiline statement without needing to add '\\'.

# In[ ]:


np.random.seed(42)
src = (ImageItemList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')
       .random_split_by_pct(0.2)
       .label_from_df(sep=' '))


# In[ ]:


data = (src.transform(tfms, size=128)
        .databunch(num_workers=0).normalize(imagenet_stats))


# `show_batch` still works, and show us the different labels separated by `;`.

# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# To create a `Learner` we use the same function as in lesson 1. Our base architecture is resnet34 again, but the metrics are a little bit differeent: we use `accuracy_thresh` instead of `accuracy`. In lesson 1, we determined the predicition for a given class by picking the final activation that was the biggest, but here, each activation can be 0. or 1. `accuracy_thresh` selects the ones that are above a certain threshold (0.5 by default) and compares them to the ground truth.
# 
# As for Fbeta, it's the metric that was used by Kaggle on this competition. See [here](https://en.wikipedia.org/wiki/F1_score) for more details.

# In[ ]:


arch = models.resnet50


# In[ ]:


acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = create_cnn(data, arch, metrics=[acc_02, f_score], model_dir='/tmp/models')


# We use the LR Finder to pick a good learning rate.

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# Then we can fit the head of our network.

# In[ ]:


lr = 0.01


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-1-rn50')


# ...And fine-tune the whole model:

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


learn.save('stage-2-rn50')


# In[ ]:


data = (src.transform(tfms, size=256)
        .databunch(num_workers=0).normalize(imagenet_stats))

learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr=1e-2/2


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-1-256-rn50')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-2-256-rn50')


# You won't really know how you're going until you submit to Kaggle, since the leaderboard isn't using the same subset as we have for training. But as a guide, 50th place (out of 938 teams) on the private leaderboard was a score of `0.930`.

# In[ ]:


learn.export()


# ## fin

# (This section will be covered in part 2 - please don't ask about it just yet! :) )

# In[ ]:


test = ImageItemList.from_folder(path/'test-jpg').add(ImageItemList.from_folder(path/'test-jpg-additional'))
len(test)


# In[ ]:


learn = load_learner(path, test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


thresh = 0.2
labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]


# In[ ]:


labelled_preds[:5]


# In[ ]:


fnames = [f.name[:-4] for f in learn.data.test_ds.items]


# In[ ]:


df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])


# In[ ]:


df.to_csv(path/'submission.csv', index=False)


# Private Leaderboard score: 0.9296 (around 80th)
