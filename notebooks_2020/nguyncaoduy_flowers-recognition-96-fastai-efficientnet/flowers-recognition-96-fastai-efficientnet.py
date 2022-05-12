#!/usr/bin/env python
# coding: utf-8

# * Normally, we can use a common model such as **ResNet** for this task!
# * However, I want to test out **EfficientNet** with fastai this time! :))

# In[ ]:


# Download EfficientNet from LukeMK
get_ipython().system(' pip install efficientnet-pytorch')


# In[ ]:


# Importing the libraries
from fastai.vision import *
from efficientnet_pytorch import EfficientNet


# In[ ]:


# Define the path
path = Path('/kaggle/input/flowers-recognition/flowers')
path.ls()


# In[ ]:


# Create the data using fastai's Datablock API
src = (ImageList.from_folder(path)
                .split_by_rand_pct(0.2, seed=42)
                .label_from_folder()
                .transform(get_transforms(), size=300))

data = src.databunch(bs=8).normalize(imagenet_stats)


# In[ ]:


# Let's see some training examples
data.show_batch(rows=2, figsize=(9, 6))


# * EfficientNet comes with a variety of sub-models **from b0 to b7**
# * The larger the model, the higher the amount of **width, depth, resolution, and dropout**
# * We start out with image size of **224x224**, so let's use **b0** first as a baseline
# * We can use bigger images later with larger models

# | Coefficient | Width | Depth | Resolution | Dropout | Last layer |
# |:-----------:|:-----:|:-----:|:----------:|:-------:|:----------:|
# |      b0     |  1.0  |  1.0  |     224    |   0.2   |1280|
# |      b1     |  1.0  |  1.1  |     240    |   0.2   |1280|
# |      b2     |  1.1  |  1.2  |     260    |   0.3   |1408|
# |      b3     |  1.2  |  1.4  |     300    |   0.3   |1536|
# |      b4     |  1.4  |  1.8  |     380    |   0.4   |1792|
# |      b5     |  1.6  |  2.2  |     456    |   0.4   |2048|
# |      b6     |  1.8  |  2.6  |     528    |   0.5   |2304|
# |      b7     |  2.0  |  3.1  |     600    |   0.5   |2560|

# In[ ]:


# Replace the fully connected layer at the end to fit our task
# Pre-trained model based on adversarial training
arch = EfficientNet.from_pretrained("efficientnet-b3", advprop=True)
arch._fc = nn.Linear(1536, data.c)


# In[ ]:


# Define custom loss function
loss_func = LabelSmoothingCrossEntropy()


# In[ ]:


# Define the model
learn = Learner(data, arch, loss_func=loss_func, metrics=accuracy, model_dir='/kaggle/working')


# In[ ]:


# Train the model using 1 Cycle policy
learn.fit_one_cycle(3, slice(1e-3))


# In[ ]:


# Unfreeze the model and retrain
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-5))


# In[ ]:


# Let's see the result
learn.show_results(rows=2, figsize=(9, 6))


# * In conclusion, our model was able to reach **96-97% accuracy** using b3 with 300x300 images!
# * However, it seemed that the model **consumed a lot of memory and training time!** 
# * I was not able to train with b7 using 600x600 images. 
# * I'm pretty sure that the result **will be even better** if I could use that!
# * Hope that I will be able to find out the issue next time. :((
# * The good thing is that the model **converges very fast after just 1-2 epochs**! :))
