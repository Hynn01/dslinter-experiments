#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys, os, csv
from PIL import Image
import cv2, gc
import matplotlib.pyplot as plt
import torch
sys.path.append('/kaggle/input/imc2022-dependencies/DKM/')

dry_run = False


# In[ ]:


get_ipython().system('mkdir -p pretrained/checkpoints')
get_ipython().system('cp /kaggle/input/imc2022-dependencies/pretrained/dkm.pth pretrained/checkpoints/dkm_base_v11.pth')

get_ipython().system('pip install -f /kaggle/input/imc2022-dependencies/wheels --no-index einops')
get_ipython().system('cp -r /kaggle/input/imc2022-dependencies/DKM/ /kaggle/working/DKM/')
get_ipython().system('cd /kaggle/working/DKM/; pip install -f /kaggle/input/imc2022-dependencies/wheels -e . ')


# In[ ]:


import torch
if not torch.cuda.is_available():
    print('You may want to enable the GPU switch?')
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.hub.set_dir('/kaggle/working/pretrained/')
from dkm import dkm_base
model = dkm_base(pretrained=True, version="v11").to(device).eval()
# model.load_state_dict(torch.load(WEIGHTS))


# In[ ]:


def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''
    
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


# In[ ]:


src = '/kaggle/input/image-matching-challenge-2022/'

test_samples = []
with open(f'{src}/test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        test_samples += [row]

if dry_run:
    for sample in test_samples:
        print(sample)


# In[ ]:


F_dict = {}
for i, row in enumerate(test_samples):
    sample_id, batch_id, image_1_id, image_2_id = row

    img1 = cv2.imread(f'{src}/test_images/{batch_id}/{image_1_id}.png') 
    img2 = cv2.imread(f'{src}/test_images/{batch_id}/{image_2_id}.png')
        
    img1PIL = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2PIL = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    
    dense_matches, dense_certainty = model.match(img1PIL, img2PIL)
    dense_certainty = dense_certainty.sqrt()
    sparse_matches, sparse_certainty = model.sample(dense_matches, dense_certainty, 2000)
    
    mkps1 = sparse_matches[:, :2]
    mkps2 = sparse_matches[:, 2:]
    
    h, w, c = img1.shape
    mkps1[:, 0] = ((mkps1[:, 0] + 1)/2) * w
    mkps1[:, 1] = ((mkps1[:, 1] + 1)/2) * h

    h, w, c = img2.shape
    mkps2[:, 0] = ((mkps2[:, 0] + 1)/2) * w
    mkps2[:, 1] = ((mkps2[:, 1] + 1)/2) * h

    F, mask = cv2.findFundamentalMat(mkps1, mkps2, cv2.USAC_MAGSAC, 0.3, 0.9999, 25_000)

    
    good = F is not None and F.shape == (3,3)
    
    if good:
        F_dict[sample_id] = F
    else:
        F_dict[sample_id] = np.zeros((3, 3))
        continue

    gc.collect()    

with open('submission.csv', 'w') as f:
    f.write('sample_id,fundamental_matrix\n')
    for sample_id, F in F_dict.items():
        f.write(f'{sample_id},{FlattenMatrix(F)}\n')

if dry_run:
    get_ipython().system('cat submission.csv')

