#!/usr/bin/env python
# coding: utf-8

# # About
# 
# According to [evaluation page](https://www.kaggle.com/competitions/foursquare-location-matching/overview/evaluation):
# 
# >... **Places always self-match**, so the list of matches for an id **should always contain that id**.
# 
# So a prediction using only self-match may be the most simple baseline. Let's Check! 

# # Prepare

# ## import

# In[ ]:


from __future__ import annotations  # for Type Hint
from pathlib import Path

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from tqdm.notebook import tqdm, trange

from matplotlib import pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
pd.set_option("max_rows", 500)


# ## set constants

# In[ ]:


ROOT = Path.cwd().parent
INPUT = ROOT / "input"
DATA = INPUT / "foursquare-location-matching"
WORK = ROOT / "working"


# ## read data

# In[ ]:


train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
# pairs = pd.read_csv(DATA / "pairs.csv")
smpl_sub = pd.read_csv(DATA / "sample_submission.csv")


# In[ ]:


train.head()


# # Check Train

# ## create matches by PoI
# 
# Places in training data which have the same point-of-interest(PoI) share the same `matches`. 

# In[ ]:


matches_sets_by_poi: dict[str, dict[str]] = dict()

for place_id, poi in tqdm(train[["id", "point_of_interest"]].values):
    if poi in matches_sets_by_poi:
        matches_sets_by_poi[poi].add(place_id)
    else:
        matches_sets_by_poi[poi] = {place_id,}


# In[ ]:


num_matches_by_poi = {k: len(v) for k, v in matches_sets_by_poi.items()}


# ## check score by only self-match

# In[ ]:


score_df = train[["id", "point_of_interest"]].copy()

score_df["num_pred_matches"] = 1  # self
score_df["num_true_matches"] = score_df["point_of_interest"].map(num_matches_by_poi)
score_df["num_union"] = score_df["num_true_matches"]  # matches include self
score_df["num_intersection"] = 1  # pred and true only share self
score_df["IoU"] = score_df["num_intersection"] / score_df["num_union"]


# In[ ]:


score_df.head()


# In[ ]:


# mean IoU score
score_df["IoU"].mean()


# Hmm... this seems too high to me, isn't it?
# 
# Let me check the mean IoU by `num_true_matchs`

# In[ ]:


iou_by_num_matchs = score_df.groupby("num_true_matches").agg(
    num_places=("id", "count"),
    mean_IoU=("IoU", "mean"),
).reset_index()


# In[ ]:


iou_by_num_matchs


# In[ ]:


fig = plt.figure(figsize=(20, 10))
fig.subplotpars.update(wspace=0.4, hspace=0.6)

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
_ = iou_by_num_matchs.plot(kind="bar", x="num_true_matches", y="num_places", ax=ax1)
_ = iou_by_num_matchs.plot(kind="bar", x="num_true_matches", y="mean_IoU", ax=ax2)

_ = ax1.set_title("num_true_matches - num_places", fontsize=20)
_ = ax2.set_title("num_true_matches - mean_IoU", fontsize=20)


# In[ ]:


iou_by_num_matchs.query("num_true_matches <= 2").num_places.sum() / iou_by_num_matchs.num_places.sum()


# I see, there are many places which match only one or two places.
# 
# These places make score higher ... but is it just a facade?

# In[ ]:


score_df.query("num_true_matches <= 2").IoU.mean()


# In[ ]:


score_df.query("num_true_matches > 2").IoU.mean()


# # Make Submission
# 
# Finally, make a submission using only self-match

# In[ ]:


sub = smpl_sub.copy()
sub["matches"] = sub["id"]

sub.to_csv("submission.csv", index=False)


# In[ ]:


sub.head()


# # EOF
