#!/usr/bin/env python
# coding: utf-8

# ## Regression with BIWI head pose dataset

# This is a more advanced example to show how to create custom datasets and do regression with images. Our task is to find the center of the head in each image. The data comes from the [BIWI head pose dataset](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html#db), thanks to Gabriele Fanelli et al. We have converted the images to jpeg format, so you should download the converted dataset from [this link](https://s3.amazonaws.com/fast-ai-imagelocal/biwi_head_pose.tgz).

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *


# ## Getting and converting the data

# In[ ]:


path = untar_data(URLs.BIWI_HEAD_POSE)


# In[ ]:


cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6); cal


# In[ ]:


fname = '09/frame_00667_rgb.jpg'


# In[ ]:


def img2txt_name(f): return path/f'{str(f)[:-7]}pose.txt'


# In[ ]:


img = open_image(path/fname)
img.show()


# In[ ]:


ctr = np.genfromtxt(img2txt_name(fname), skip_header=3); ctr


# In[ ]:


def convert_biwi(coords):
    c1 = coords[0] * cal[0][0]/coords[2] + cal[0][2]
    c2 = coords[1] * cal[1][1]/coords[2] + cal[1][2]
    return tensor([c2,c1])

def get_ctr(f):
    ctr = np.genfromtxt(img2txt_name(f), skip_header=3)
    return convert_biwi(ctr)

def get_ip(img,pts): return ImagePoints(FlowField(img.size, pts), scale=True)


# In[ ]:


get_ctr(fname)


# In[ ]:


ctr = get_ctr(fname)
img.show(y=get_ip(img, ctr), figsize=(6, 6))


# ## Creating a dataset

# In[ ]:


data = (PointsItemList.from_folder(path)
        .split_by_valid_func(lambda o: o.parent.name=='13')
        .label_from_func(get_ctr)
        .transform(get_transforms(), tfm_y=True, size=(120,160))
        .databunch(num_workers=0).normalize(imagenet_stats)
       )


# In[ ]:


data.show_batch(3, figsize=(9,6))


# ## Train model

# In[ ]:


learn = create_cnn(data, models.resnet34)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 2e-2


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.load('stage-1');


# In[ ]:


learn.show_results()


# ## Data augmentation

# In[ ]:


tfms = get_transforms(max_rotate=20, max_zoom=1.5, max_lighting=0.5, max_warp=0.4, p_affine=1., p_lighting=1.)

data = (PointsItemList.from_folder(path)
        .split_by_valid_func(lambda o: o.parent.name=='13')
        .label_from_func(get_ctr)
        .transform(get_transforms(), tfm_y=True, size=(120,160))
        .databunch(num_workers=0).normalize(imagenet_stats)
       )


# In[ ]:


def _plot(i,j,ax):
    x,y = data.train_ds[0]
    x.show(ax, y=y)

plot_multi(_plot, 3, 3, figsize=(8,6))

