#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Modules
import numpy as np 
import pandas as pd
import csv
import cv2
import gc
import torch
import sys

# SOLD2
sys.path.append('../input/sold2linematching')
from sold2.model.line_matcher import LineMatcher
from sold2.misc.visualize_util import plot_images, plot_lines, plot_line_matches, plot_color_line_matches, plot_keypoints


# In[ ]:


# Set Torch - Device
if not torch.cuda.is_available():
    print('You may want to enable the GPU switch?')    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## Test Data

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


# ## Setup SOLD2

# In[ ]:


ckpt_path = '../input/sold2linematching/pretrained/sold2_wireframe.tar'
mode = 'dynamic'  # 'dynamic' or 'static'

# Initialize the line matcher
config = {
    'model_cfg': {
        'model_name': "lcnn_simple",
        'model_architecture': "simple",
        # Backbone related config
        'backbone': "lcnn",
        'backbone_cfg': {
            'input_channel': 1, # Use RGB images or grayscale images.
            'depth': 4,
            'num_stacks': 2,
            'num_blocks': 1,
            'num_classes': 5
        },
        # Junction decoder related config
        'junction_decoder': "superpoint_decoder",
        'junc_decoder_cfg': {},
        # Heatmap decoder related config
        'heatmap_decoder': "pixel_shuffle",
        'heatmap_decoder_cfg': {},
        # Descriptor decoder related config
        'descriptor_decoder': "superpoint_descriptor",
        'descriptor_decoder_cfg': {},
        # Shared configurations
        'grid_size': 8,
        'keep_border_valid': True,
        # Threshold of junction detection
        'detection_thresh': 0.0153846, # 1/65
        'max_num_junctions': 300,
        # Threshold of heatmap detection
        'prob_thresh': 0.5,
        # Weighting related parameters
        'weighting_policy': mode,
        # [Heatmap loss]
        'w_heatmap': 0.,
        'w_heatmap_class': 1,
        'heatmap_loss_func': "cross_entropy",
        'heatmap_loss_cfg': {
            'policy': mode
        },
        # [Heatmap consistency loss]
        # [Junction loss]
        'w_junc': 0.,
        'junction_loss_func': "superpoint",
        'junction_loss_cfg': {
            'policy': mode
        },
        # [Descriptor loss]
        'w_desc': 0.,
        'descriptor_loss_func': "regular_sampling",
        'descriptor_loss_cfg': {
            'dist_threshold': 8,
            'grid_size': 4,
            'margin': 1,
            'policy': mode
        },
    },
    'line_detector_cfg': {
        'detect_thresh': 0.25,  # depending on your images, you might need to tune this parameter
        'num_samples': 512,     # Original: 64
        'sampling_method': "local_max",
        'inlier_thresh': 0.9,
        "use_candidate_suppression": True,
        "nms_dist_tolerance": 3.,
        "use_heatmap_refinement": True,
        "heatmap_refine_cfg": {
            "mode": "local",
            "ratio": 0.2,
            "valid_thresh": 1e-3,
            "num_blocks": 20,
            "overlap_ratio": 0.5
        }
    },
    'multiscale': False,
    'line_matcher_cfg': {
        'cross_check': True,
        'num_samples': 5,
        'min_dist_pts': 8,
        'top_k_candidates': 10,
        'grid_size': 4
    }
}

# Create SOLD2 Matcher
sold2_matcher = LineMatcher(config["model_cfg"], ckpt_path, device, config["line_detector_cfg"], config["line_matcher_cfg"], config["multiscale"])


# ## Support Functions

# In[ ]:


def FlattenMatrix(M, num_digits = 8):
    '''Convenience function to write CSV files.'''    
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])

def get_fundamental_matrix(kpts1, kpts2):    
    if len(kpts1) > 7:
        F, inliers = cv2.findFundamentalMat(kpts1, 
                                            kpts2, 
                                            cv2.USAC_MAGSAC, 
                                            ransacReprojThreshold = 0.20, 
                                            confidence = 0.99999, 
                                            maxIters = 100000)
        return F, inliers
    else:
        return np.zeros((3, 3)), None
    

def load_image(image_path):
    img = cv2.imread(image_path, 0)
    
    # Scale Factor...recommended to scale images to between 400 - 800
    scale_factor = 800 / max(img.shape[0], img.shape[1]) 
    w = int(img.shape[1] * scale_factor)
    h = int(img.shape[0] * scale_factor)
    
    # Resize
    img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
    img = (img / 255.).astype(float)
    
    return img


# With the SOLD2 model we eventually have a set of matched lines. When using Ransac (or one of its variants..) in the process to calculate a fundamental matrix we can't use the lines. So what we instead do is use the start and endpoints of a line as unique keypoints.

# In[ ]:


def get_keypoints(batch_id, img_id1, img_id2, plot = False):
    image_fpath_1 = f'{src}/test_images/{batch_id}/{img_id1}.png'
    image_fpath_2 = f'{src}/test_images/{batch_id}/{img_id2}.png'
    
    # Process Image 1
    img1 = load_image(image_fpath_1)
    torch_img1 = torch.tensor(img1, dtype=torch.float)[None, None]
    
    # Process Image 2
    img2 = load_image(image_fpath_2)
    torch_img2 = torch.tensor(img2, dtype=torch.float)[None, None]

    # Match the lines
    outputs = sold2_matcher([torch_img1, torch_img2])
    line_seg1 = outputs["line_segments"][0]
    line_seg2 = outputs["line_segments"][1]
    matches = outputs["matches"]

    # Get Valid Matches
    valid_matches = matches != -1
    match_indices = matches[valid_matches]
    matched_lines1 = line_seg1[valid_matches][:, :, ::-1]
    matched_lines2 = line_seg2[match_indices][:, :, ::-1]
    
    # Plot the matches
    if plot:
        plot_images([img1, img2], ['Image1 - detected lines', 'Image2 - detected lines'])
        plot_lines([line_seg1[:, :, ::-1], line_seg2[:, :, ::-1]], ps=3, lw=2)
        plot_images([img1, img2], ['Image1 - matched lines', 'Image2 - matched lines'])
        plot_color_line_matches([matched_lines1, matched_lines2], lw=2)

    # Get start and end point of matched lines as regular keypoints
    mkpts1 = matched_lines1.reshape(matched_lines1.shape[0] * 2, 2)
    mkpts2 = matched_lines2.reshape(matched_lines2.shape[0] * 2, 2)
    
    return mkpts1, mkpts2


# ## Predictions

# In[ ]:


f_matrix_dict = {}
for i, row in enumerate(test_samples):
    sample_id, batch_id, img_id1, img_id2 = row

    # Get SOLD2 Keypoints
    plot = False
    if i < 3: plot = True
    mkpts1, mkpts2 = get_keypoints(batch_id, img_id1, img_id2, plot)
    
    # Get Fundamental matrix
    f_matrix_dict[sample_id], _ = get_fundamental_matrix(mkpts1, mkpts2)
    
    # Mem Cleanup
    gc.collect()


# ## Create Submission

# In[ ]:


# Write Submission File   
with open('submission.csv', 'w') as f:
    f.write('sample_id,fundamental_matrix\n')
    for sample_id, F in f_matrix_dict.items():
                
        f.write(f'{sample_id},{FlattenMatrix(F)}\n')


# In[ ]:


# Summary
sub = pd.read_csv('submission.csv')
sub.head()

