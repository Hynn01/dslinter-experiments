#!/usr/bin/env python
# coding: utf-8

# Run SfM and visualize reconstruction + ground truth.
# 
# Based on notebook https://colab.research.google.com/drive/1MrVs9b8aQYODtOGkoaGNF9Nji3sbCNMQ from this repo https://github.com/cvg/Hierarchical-Localization

# In[ ]:


get_ipython().system('git clone --recursive https://github.com/cvg/Hierarchical-Localization/')
import os
os.chdir("./Hierarchical-Localization")
get_ipython().system('python -m pip install -e .')
get_ipython().system('curl https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat --create-dirs -o /kaggle/working/Hierarchical-Localization/third_party/netvlad/VGG16-NetVLAD-Pitts30K.mat')
get_ipython().system('pip install --upgrade --quiet plotly')

import pandas as pd
import numpy as np
import tqdm, tqdm.notebook
tqdm.tqdm = tqdm.notebook.tqdm  # notebook-friendly progress bars
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, pairs_from_retrieval
from hloc.utils import viz_3d
import plotly.graph_objects as go


# # Run SfM
# Takes about 20 minutes for british museum

# In[ ]:


scene = "british_museum"

images = Path(f'/kaggle/input/image-matching-challenge-2022/train/{scene}/images')

outputs = Path('outputs/sfm/')
sfm_pairs = outputs / 'pairs-netvlad.txt'
sfm_dir = outputs / 'sfm_superpoint+superglue'

retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']


retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path, verbose=False)

fig = viz_3d.init_figure()

viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', points=True, cameras=False)
viz_3d.plot_reconstruction(fig, model, color='rgba(0,255,0,0.5)', points=False, cameras=True)

fig.show()


# # Align to ground truth

# In[ ]:


cal_df = pd.read_csv(f"/kaggle/input/image-matching-challenge-2022/train/{scene}/calibration.csv")

name_index_dict = {}

for k, v in model.images.items():
    name_index_dict[v.name] = k
    
name_location_dict = {}
cameras = []
for index, row in cal_df.iterrows():
    R = np.array([float(x) for x in row.rotation_matrix.split(" ")]).reshape(3, 3)
    t = np.array([float(x) for x in row.translation_vector.split(" ")])
    K = np.array([float(x) for x in row.camera_intrinsics.split(" ")]).reshape(3, 3)
    
    image_name = row.image_id + ".jpg"
    name_location_dict[image_name] = t

    camera = {"R": R, "t": t, "K": K, "name": name_index_dict[image_name]}
    cameras.append(camera)

model.align_robust(list(name_location_dict.keys()), list(name_location_dict.values()), len(name_location_dict))


# # Visualize
# 
# Green cameras: reconstruction cameras
# 
# Blue cameras: ground truth cameras
# 
# Yellow lines: matching cameras

# In[ ]:


fig = viz_3d.init_figure()

viz_3d.plot_reconstruction(fig, model, color='rgba(0,255,0,0.5)', points=False, cameras=True)

# Plot ground truth cameras
for camera in cameras:
    cam = camera.copy()
    
    # Not so sure about this transform
    cam["R"] = np.matmul(camera["R"], np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])).T
    
    viz_3d.plot_camera(fig, color='rgba(0,0,255,0.5)', **cam)

# Plot lines connecting reconstruction cameras and ground truth cameras
lines = []
for image in model.images.values():
    t1 = image.projection_center()
    t2 = name_location_dict[image.name]
    x = np.array([t1[0], t2[0]])
    y = np.array([t1[1], t2[1]])
    z = np.array([t1[2], t2[2]])
    line = go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='rgb(255,255,0)', width=1), name=image.name)
    lines.append(line)
fig.add_traces(lines)

fig.update_layout(showlegend=False)
fig.show()


# In[ ]:




