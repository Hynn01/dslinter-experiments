#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install monai')


# ## MONAI is a PyTorch-based, open-source framework for deep learning in healthcare imaging, part of PyTorch Ecosystem.
# 
# I took its `HausdorffDistanceMetric` for reference, and implement a simple function to calculate the Hausdorff Distance for this competition. Compared to `scipy.spatial.distance.directed_hausdorff`, it performs much faster, and can get the same results.
# 
# The following are the implementations and the comparisions of these two ways:

# In[ ]:


from scipy.spatial.distance import directed_hausdorff
from monai.metrics.utils import get_mask_edges, get_surface_distance
import numpy as np


# In[ ]:


# normal way, using scipy's directed_hausdorff
def compute_hausdorff_scipy(pred, gt, max_dist):
    if np.all(pred == gt):
        return 0.0
    dist = directed_hausdorff(np.argwhere(pred), np.argwhere(gt))[0]
    if dist > max_dist:  # when gt is all 0s, may get inf.
        return 1.0
    return dist / max_dist

# faster way, using monai
def compute_hausdorff_monai(pred, gt, max_dist):
    if np.all(pred == gt):
        return 0.0
    (edges_pred, edges_gt) = get_mask_edges(pred, gt)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 0.0
    dist = surface_distance.max()
    if dist > max_dist:
        return 1.0
    return dist / max_dist


# In[ ]:


# test consistency

image_shape = (256, 256)
max_distance = np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)

pred = np.random.randint(0, high=2, size=image_shape)
label = np.random.randint(0, high=2, size=image_shape)
print("result for random pred and label: ")
print(compute_hausdorff_scipy(pred, label, max_distance))
print(compute_hausdorff_monai(pred, label, max_distance))
print("result for random pred and empty label: ")
label_empty = np.zeros(image_shape)
print(compute_hausdorff_scipy(pred, label_empty, max_distance))
print(compute_hausdorff_monai(pred, label_empty, max_distance))


# In[ ]:


#test speed
import time
start = time.time()
for i in range(100):
    compute_hausdorff_scipy(pred, label, max_distance)
scipy_time = time.time() - start
print(f"time used for scipy way: {scipy_time:.4f} seconds")


# In[ ]:


#test speed
import time
start = time.time()
for i in range(100):
    compute_hausdorff_monai(pred, label, max_distance)
monai_time = time.time() - start
print(f"time used for monai way: {monai_time:.4f} seconds")


# In[ ]:


print(scipy_time/monai_time)

