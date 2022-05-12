#!/usr/bin/env python
# coding: utf-8

# ## Importing required libraries

# In[ ]:


import numpy as np 
import pandas as pd 
get_ipython().system('pip -q install transformers ')
from transformers import AutoTokenizer , AutoModel , AutoConfig 


# ## visualizing scaled dot attention with bertviz. 
# link : https://github.com/jessevig/bertviz

# In[ ]:


model = "bert-base-uncased"


# In[ ]:


# downloading bertviz library 
get_ipython().system(' pip -q install bertviz')
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(model)
model = BertModel.from_pretrained(model)


# In[ ]:


text = "Apple mobile is awesome"
show(model, "bert", tokenizer , text, display_mode = "dark", layer = 0 , head = 8 )


# In[ ]:




