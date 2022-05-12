#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
get_ipython().system('pip -q install transformers ')
from transformers import AutoTokenizer, AutoModel , AutoConfig
get_ipython().system('pip -q  install datasets ')
from datasets import * 


# In[ ]:


# when we are dealing with multilingual datasets they might have multiple configs. 
from datasets import get_dataset_config_names
xtreme_subsets = get_dataset_config_names("xtreme")
print(f"Xtreme is having different {len(xtreme_subsets)} configurations")


# In[ ]:


# filtering the dataset based on what we want to load German corpus we are passing de code to load_dataset fucntion 
[x for x in xtreme_subsets if x.startswith("PAN")][:5]


# In[ ]:


# loading the de dataset
load_dataset("xtreme", name ="PAN-X.de")


# In[ ]:


# since we have to perform NER on german french italian and english downloading those dataset to 
from collections import defaultdict
from datasets import DatasetDict

langs = ["de", "fr" , "it", "en"]
fracs = [0.629, 0.228 , 0.084, 0.059]

panx_ch = defaultdict(DatasetDict)

for lang, frac in zip(langs, fracs):
    # Load monolingual corpus
    ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
    # Shuffle and downsample each split according to spoken proportion
    for split in ds:
            panx_ch[lang][split] = (
            ds[split]
            .shuffle(seed=0)
            .select(range(int(frac * ds[split].num_rows))))


# In[ ]:


# looking out for how many rows does each dataset have  with ds.num_rows

pd.DataFrame({lang: [panx_ch[lang]["train"].num_rows] for lang in langs},
index=["Number of training examples"])


# In[ ]:


element = panx_ch["de"]["train"][0]
for key, value in element.items():
    print(f"{key}: {value}")


# In[ ]:


for key, value in panx_ch["de"]["train"].features.items():
    print(f"{key}: {value}")


# In[ ]:


tags = panx_ch["de"]["train"].features["ner_tags"].feature
print(tags)


# In[ ]:


def create_tag_names(batch):
    return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}


# In[ ]:


panx_de = panx_ch["de"].map(create_tag_names)


# In[ ]:


de_example = panx_de["train"][0]
pd.DataFrame([de_example["tokens"], de_example["ner_tags_str"]],
['Tokens', 'Tags'])


# In[ ]:


# checking the distribution
from collections import Counter
split2freqs = defaultdict(Counter)
for split, dataset in panx_de.items():
    for row in dataset["ner_tags_str"]:
        for tag in row:
            if tag.startswith("B"):
                tag_type = tag.split("-")[1]
                split2freqs[split][tag_type] += 1
pd.DataFrame.from_dict(split2freqs, orient="index")


# # WORK UNDER PROGRESS 

# In[ ]:




