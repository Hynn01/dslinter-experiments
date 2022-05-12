#!/usr/bin/env python
# coding: utf-8

# As you've probably read in Andy's [announcement](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/125436), Team "Bestpetting", the prior first team in this competition, has been removed because of cheating.
# 
# I've helped PetFinder.my convert the top solutions of their competition to a production system in the past few months and I recently discovered that Bestpetting had been cheating.
# 
# I'd like to thank Andy from PetFinder.my for very professionally handling this scenario and for always being very helpful and dependable when working together.

# # The cheating

# In[ ]:


from hashlib import md5
import random
import re
import requests
from IPython.display import display
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set()


# Bestpetting have been using the external dataset ["Cute Cats and Dogs from Pixabay.com"](https://www.kaggle.com/ppleskov/cute-cats-and-dogs-from-pixabaycom/). This dataset contains a bunch of images and a .csv file containing labels.
# 
# The images don't matter, but the .csv file does.
# 
# It contains an ID hash in which the team encoded the labels of the private test set in.

# In[ ]:


labels_df = pd.read_csv("../input/cute-cats-and-dogs-from-pixabaycom/labels.csv")
test = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")

labels_df.head()


# They applied some preprocessing to the name column of the test data and then concatenated many columns (`hash_cols` below) into one string. This string is unique for > 99% of the pets in the test data.

# In[ ]:


test["Name"] = test["Name"].apply(lambda x: re.sub('[^A-Za-z0-9]+', '', str(x)))


# In[ ]:


hash_cols = ['Name','Type','Quantity','Vaccinated','Quantity',
             'Dewormed','Sterilized','Gender','MaturitySize','FurLength',
             'Color1','Color2','Color3','Health','Fee']
pet_info = test[hash_cols].apply(lambda cols: "".join(str(x) for x in cols), axis=1)
display(pet_info.head())
print("% unique:", 1 - pet_info.duplicated().mean())


# They hashed these mostly unique IDs using MD5.

# In[ ]:


hashed_pet_info = pet_info.apply(lambda x: md5(x.encode('utf-8')).hexdigest()[:-1])
hashed_pet_info.head()


# They also iterated over all IDs in the pixabay label dataframe. 
# 
# If the last character of the ID was numeric, they stored the ID in a dictionary, with:
# - the key being the string up to but excluding the last character of the ID.
# - the value being the last character divided by two and floored.

# In[ ]:


hash_dict = dict()

for x in list(labels_df["id"]):
    if x[-1].isnumeric():
        hash_dict[x[:-1]] = int(x[-1]) // 2

# just a sample of the hash dict keys
rand_keys = np.random.choice(list(hash_dict.keys()), size=10)
{key: hash_dict[key] for key in rand_keys}


# If we now take a look at the sets of hashed pet information IDs and IDs from their external dataset, we can see that exactly 3500 of them overlap.

# In[ ]:


plt.figure(figsize=(14, 7))
venn2([
    set(hashed_pet_info), 
    set(hash_dict.keys())
], set_labels=["Pet Hashes", "External ID Column"])


# If we then set the predictions for the 3500 pets which are present in the dictionary with the corresponding values from that dictionary (and the rest to the most common label, 2), we can get an almost perfect score of 0.912543.

# In[ ]:


test["AdoptionSpeed"] = hashed_pet_info.apply(lambda x: hash_dict.get(x, 2))
test[["PetID", "AdoptionSpeed"]].to_csv("submission.csv", index=False)
test["AdoptionSpeed"].head()


# Interestingly, the kind of manipulation of IDs they did would have been easy to find if you were suspicious of the ID column in the first place. Taking a look at the last digits of the IDs, you can see that they are very unevenly distributed.

# In[ ]:


plt.figure(figsize=(14, 7))
def str_to_num(x):
    try:
        return int(x)
    except ValueError:
        return np.nan

raw_nums = labels_df["id"].str[-1].apply(str_to_num)
raw_nums.dropna().plot.hist(bins=19)
plt.title("Last digit distribution of cheating md5 hashes")
print()


# Whereas MD5 hashes of random strings are distributed evenly.

# In[ ]:


import string

def get_random_string(minlen=5, maxlen=20):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase + string.digits
#     letters = string.ascii_lowercase[:3] # also when only using a small subset of letters so it does not depend on the randomness of the input strings
    length = random.randint(minlen, maxlen)
    return ''.join(random.choice(letters) for i in range(length))

random_hashes = pd.Series([md5(get_random_string().encode("utf-8")).hexdigest() for _ in range(100000)])
plt.figure(figsize=(14, 7))
(random_hashes.str[-1].apply(str_to_num)).dropna().plot.hist(bins=19)
plt.title("Last digit distribution of md5 hashes of random strings.")
print()


# This only works if a primitive encoding technique is used. With more sophisticated schemes, it will always be possible to encode information in data that is nearly impossible for anybody else to find.

# # How they hid it

# Of course they did not write out the logic of this procedure like I did above. Otherwise it would hopefully not have taken such a long time to find it. In their original submission it was cleverly hidden inside a `majority_voting` function.
# 
# `majority_voting` does majority voting and stores the result in the tmp column, but also calls `proc` on the dataframe.

# In[ ]:


def majority_voting(df, cols):
    df["tmp"] = df[cols].mode(axis=1)[0]
    return proc(df)


# - `proc` calls `clean_name` which just does simple processing of the Name column.
# - `proc` calls `col_to_str` which stores a string concatenation of many columns in the `str` column.

# In[ ]:


cols_ = ['Name','Type','Quantity','Vaccinated','Quantity',
         'Dewormed','Sterilized','Gender','MaturitySize','FurLength',
         'Color1','Color2','Color3','Health','Fee']

def proc(df):
    df["Name"] = df["Name"].apply(clean_name)
    col_to_str(df, cols_)
    dic = get_dict(labels_df)
    res = process(df, dic)
    df.drop(["str","tmp"],axis=1,inplace=True)
    return res


# In[ ]:


def clean_name(x):
    if str(x)=="nan":
        return "nan"
    return re.sub('[^A-Za-z0-9]+','',x)


# In[ ]:


def col_to_str(df, cols):
    df["str"] = ""
    for c in cols:
        df["str"] += df[c].astype(str)
    df["str"] = df["str"].str.replace(" ",'')


# `proc` then calls `get_dict` on labels_df. `get_dict` now creates the hash dictionary using the same logic as in my version above.

# In[ ]:


def get_dict(df):
    d = dict()
    for x in list(df.id):
        if x[-1].isnumeric():
            d[x[:-1]]=int(x[-1])//2
    return d


# `proc` then calls `process` with the original data, plus the dictionary from `get_dict`. Then for every tenth pet the prediction is replaced with the value from the hash dictionary if the ID is found there.
# 
# It is probably only replaced for every tenth pet to not make the score too obviously illegitimate.

# In[ ]:


def process(df, d):
    res, i = [], -1
    for x,y in zip(df["str"], df["tmp"]):
        k = md5(x.encode('utf-8')).hexdigest()[:-1]
        res.append(d[k]) if k in d and i%10==0 else res.append(y)
        i+=1
    return res


# In[ ]:


test["preds"] = -1
sub_preds = np.array(majority_voting(test, ["preds"]))
(sub_preds != -1).sum() # expected: 10% of 3500


# Their submission would have scored ~ 100th place with a score of 0.427526 without the cheat.

# # What to learn from this

# This whole incident has been very frustrating to me. Not only because it undermines the legitimacy of Kaggle competitions in general, but also because I wasted quite a lot of time investigating their solution and porting it to production only to find out that it is not legitimate.
# 
# It is still an opportunity to learn though.
# Firstly, it would probably be good to only upload external data that is actually used in your solution. In this case, the entire .csv file could have been omitted because all the information is captured in the file structure anyway. I'm looking forward to discussion on this though, since it is obviously not the root cause of the problem.
# 
# Secondly, and most importantly: __Everyone who wins money from a competition should be required to open-source their solution__. I am not the first one to say this and I have no idea why it is still not the case. I understand that competition sponsors should be able to opt-out of this for privacy reasons (but then again, why host a Kaggle competition in the first place if that's the case?), but it should definitely be the default.
# 
# I would even go as far as automatically publicizing all submitted solutions after the competition. At least for Kernels competitions this would be easy to implement and it would reduce fraud to essentially zero. Unless I am missing something this should not be a problem.

# I was not sure whether to publish this kernel since showing how someone cheated is a delicate thing. But I decided for it. Since if someone wants to secretly encode information in data, it will probably always be possible. 
# 
# Showing how they cheated is very interesting and was educational to me. 
# 
# Discussing why it could happen in the first place is even more important.
# 
# I know that Kaggle might not be that happy with me publishing this since it makes most of the private test labels public. I apologize in advance, but I think a lot can be learnt from this, and the participants who spent countless hours on the competition also deserve to know what exactly happened.
