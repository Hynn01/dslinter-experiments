#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.vision.gan import *


# ## LSun bedroom data

# For this lesson, we'll be using the bedrooms from the [LSUN dataset](http://lsun.cs.princeton.edu/2017/). The full dataset is a bit too large so we'll use a sample from [kaggle](https://www.kaggle.com/jhoward/lsun_bedroom).

# In[ ]:


path = Path('/kaggle/input/sample/data0/lsun/bedroom')


# We then grab all the images in the folder with the data block API. We don't create a validation set here for reasons we'll explain later. It consists of random noise of size 100 by default (can be changed below) as inputs and the images of bedrooms as targets. That's why we do `tfm_y=True` in the transforms, then apply the normalization to the ys and not the xs.

# In[ ]:


def get_data(bs, size):
    return (GANItemList.from_folder(path, noise_sz=100)
               .no_split()
               .label_from_func(noop)
               .transform(tfms=[[crop_pad(size=size, row_pct=(0,1), col_pct=(0,1))], []], size=size, tfm_y=True)
               .databunch(bs=bs, num_workers=0)
               .normalize(stats = [torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])], do_x=False, do_y=True))


# We'll begin with a small side and use gradual resizing.

# In[ ]:


data = get_data(128, 64)


# In[ ]:


data.show_batch(rows=5)


# ## Models

# GAN stands for [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf) and were invented by Ian Goodfellow. The concept is that we will train two models at the same time: a generator and a critic. The generator will try to make new images similar to the ones in our dataset, and the critic will try to classify real images from the ones the generator does. The generator returns images, the critic a single number (usually 0. for fake images and 1. for real ones).
# 
# We train them against each other in the sense that at each step (more or less), we:
# 1. Freeze the generator and train the critic for one step by:
#   - getting one batch of true images (let's call that `real`)
#   - generating one batch of fake images (let's call that `fake`)
#   - have the critic evaluate each batch and compute a loss function from that; the important part is that it rewards positively the detection of real images and penalizes the fake ones
#   - update the weights of the critic with the gradients of this loss
#   
#   
# 2. Freeze the critic and train the generator for one step by:
#   - generating one batch of fake images
#   - evaluate the critic on it
#   - return a loss that rewards posisitivly the critic thinking those are real images; the important part is that it rewards positively the detection of real images and penalizes the fake ones
#   - update the weights of the generator with the gradients of this loss
#   
# Here, we'll use the [Wassertein GAN](https://arxiv.org/pdf/1701.07875.pdf).

# We create a generator and a critic that we pass to `gan_learner`. The noise_size is the size of the random vector from which our generator creates images.

# In[ ]:


generator = basic_generator(in_size=64, n_channels=3, n_extra_layers=1)
critic    = basic_critic   (in_size=64, n_channels=3, n_extra_layers=1)


# In[ ]:


learn = GANLearner.wgan(data, generator, critic, switch_eval=False,
                        opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0., model_dir='/tmp/models')


# In[ ]:


learn.fit(5,2e-4)


# In[ ]:


# had to restrict to 5 epochs because of Kaggle Kernel limits, uncomment below for full training
# learn.fit(30,2e-4)


# In[ ]:


learn.gan_trainer.switch(gen_mode=True)
learn.show_results(ds_type=DatasetType.Train, rows=16, figsize=(8,8))


# In[ ]:




