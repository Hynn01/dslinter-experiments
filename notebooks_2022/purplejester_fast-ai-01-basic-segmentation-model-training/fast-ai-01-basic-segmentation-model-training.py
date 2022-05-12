#!/usr/bin/env python
# coding: utf-8

# # Building a Basic Segmentation Model With fast.ai
# 
# Some time ago, a new iteration of the famous [fast.ai course](https://course.fast.ai) has started, and I'm participating there as an international fellow. Though I'm not new to Machine Learning in general and Deep Learning in particular, I think that the course provides a great chance to try another problems' solving approach and shares interesting insights about the topic. 
# 
# The `fastai` library is developed to be an end-to-end deep learning solution that helps to handle data, train a model, and verify the quality of predictions. It provides classes that help users to deal with various forecasting tasks: classification, regression, image segmentation, etc. In this notebook, we'll follow the first chapter of the [Deep Learning for Coders with fastai and PyTorch](https://www.oreilly.com/library/view/deep-learning-for/9781492045519/) book and [Lecture 1 of the course](https://course.fast.ai/videos/?lesson=1). However, instead of training a classification model, we'll train an image segmentation model. I decided to try a different task compared to what is described in the lesson to see how easy it is to switch from one problem to another.
# 
# For this purpose, we'll use [UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation) dataset. In this kernel, we only train a basic model, without doing any inference or submitting the predicted result. Also, we don't use the original dataset, but [it's pre-processed version](https://www.kaggle.com/datasets/purplejester/uwm-images-and-masks) that includes the same samples and the original dataset plus decoded segmentation masks. 
# 
# # Imports and Setup
# 

# In[ ]:


from pathlib import Path
from fastai.vision.all import *
from fastcore.all import *

set_seed(1, reproducible=True)

DEBUG = True
DATASET_SIZE = 3000 if DEBUG else 38496  # read only a small subset of data to train some basic model
DATA_ROOT = Path("/kaggle/input/uwm-images-and-masks")


# # Reading the data
# 
# In order to read the data, we'll use an instance of `DataBlock` class. The class reads data using `get_items` and `get_y` functions. The former parses a folder with images and returns paths, and the later uses these paths to derive a label. In our case, we transform a path to an image into a path to its segmentation mask. 

# In[ ]:


def get_items(source_dir: Path):
    return get_image_files(source_dir.joinpath("images"))[:DATASET_SIZE]

def get_y(fn: Path):
    return fn.parent.parent.joinpath("masks").joinpath(f"{fn.stem}.png")

def get_combined_code(*class_names):
    assert 1 <= len(class_names) <= 3
    combined_code = 0
    for class_name in class_names:
        combined_code |= MASK_CODES[class_name]
    label = "+".join(class_names)
    return label, combined_code


# As the classes shown in segmentation masks can overlapp (I didn't check the data carefully, but account for this possibility just in case), the [pre-processed dataset](https://www.kaggle.com/datasets/purplejester/uwm-images-and-masks/settings) includes multiple segmentation codes.

# In[ ]:


MASK_CODES = {"large_bowel": 0b001, "small_bowel": 0b010, "stomach": 0b100}

COMBINED_MASK_CODES = dict([
    ("none", 0),
    *list(MASK_CODES.items()),
    get_combined_code("large_bowel", "small_bowel"),
    get_combined_code("small_bowel", "stomach"),
    get_combined_code("large_bowel", "stomach"),
    get_combined_code("large_bowel", "small_bowel", "stomach"),
])

list(COMBINED_MASK_CODES)


# Finally, the created functions and mask codes combined together into a single object responsible for reading the data and collating it into training batches.

# In[ ]:


seg = DataBlock(blocks=(ImageBlock, MaskBlock(list(COMBINED_MASK_CODES))),
                get_items=get_items,
                get_y=get_y,
                splitter=RandomSplitter(valid_pct=0.2),
                item_tfms=[Resize(192, method="squash")])

dls = seg.dataloaders(DATA_ROOT, bs=64)

dls.show_batch()


# Ok, it works! So we've done everything right. Though I am not quite sure what transformations are applied to the images. As you can see, some of them seem to be filled with noise. Is it a data quality issue, or just a kind of normalization applied to the samples?
# 
# # Model Training
# 
# This part is my favourite one: we fine-tune a segmentation model with two lines of code. I would say that this kind of interactivity and quick training loop setup isn't easy to achive with other libraries, like `ignite` or `pytorch-lightning`.

# In[ ]:


learn = unet_learner(dls, resnet18, metrics=DiceMulti)
learn.fine_tune(3)


# That's it! Not bad for a few lines of code. Of course, I've also spent a bit of time to figure out how to do this, but [the documentation](https://docs.fast.ai) (in addition to the course and the book) gives plenty of tips.
# 
# Now we're ready to interpret model's predictions. There is a special class for this purpose called `SegmentationInterpretation`. It should help us to see cases when the model performs good, and where it performs bad.

# In[ ]:


interp = SegmentationInterpretation.from_learner(learn)


# In[ ]:


interp.show_results([5, 15, 25, 50])


# In[ ]:


interp.plot_top_losses(4)


# As the interpreter shows, the model makes some reasonable predictions. However, we also see samples that model fails to forecast successfully. Again, it is the same problem that was revealed previously, so probably we should to review the input data before we can proceed.
# 
# # To Be Continued...
# 
# Ok, it was quite easy to start developing a segmentation model using the `fastai` library. Next, we should figure out data quality issues and see if we can get a better result on the same small subset of data. [In the next notebook](https://www.kaggle.com/code/purplejester/fast-ai-02-histogram-equalization), we make sure that dataset samples look more reasonable and train a new model to see if it helps to improve the quality of its forecast.
