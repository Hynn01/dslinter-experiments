#!/usr/bin/env python
# coding: utf-8

# <img src="https://i.imgur.com/ZsuSumT.png" />

# # Summary
# In this Notebook, I finetune GPT-2 on Quotes dataset using gpt-2-simple library.
# 
# 
# 
# 
# Dataset : Quotes-500k <br>
# Also, Generated text(Quotes) from trained Language Model. Though, most of the time it doesn't make any sense .
# 
# 
# 

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import warnings
warnings.filterwarnings("ignore")        


# # Installing Library

# In[ ]:


get_ipython().system('pip install gpt-2-simple')


# In[ ]:


get_ipython().system('pip install tensorflow-gpu==1.14.0')


# In[ ]:


import gpt_2_simple as gpt2


# In[ ]:


model_name = "355M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
#     gpt2.download_gpt2(model_name=model_name)
    # model is saved into current directory under /models/124M/


# # Finetuning Model

# In[ ]:



file_name = "/kaggle/input/quotes-500k/train.txt"

# sess = gpt2.start_tf_sess()
# gpt2.finetune(sess,
#               file_name,
#               model_name=model_name,
#               steps=470)   # steps is max number of training steps

# This Two linnes finetune the model. For detailed training check notebook version 3


# In[ ]:


# gpt2.generate(sess)
# This line generates Text. In our case which is quotes.


# # Generated Text (Quotes)

# The only thing that's ever worth worth doing is to keep doing it.<br><br>
# I had a dream one day that I believed was real. I had a dream that I believed was real.<br><br>
# If you are serious about your work, you have to do something about it. And no matter how much you try, you can never stop thinking about it.<br><br>
# There are a lot of factors that influence whether you succeed or not.<br><br>
# I have learned from my mistakes. I have learned from my failures. I have learned from my mistakes and I have learned from my successes. And then, I have learned that the best thing I can do is to keep moving forward.<br><br>
# If you're serious about your work, you have to do something about it. And no matter how much you try, you can never stop thinking about it.<br><br>
# I don’t have the record to prove it, but I believe that one of the most important things you can do is to keep moving forward.<br><br>
# We have been taught that it’s a shame to fail. It’s a shame to fail. But we haven’t been taught that it’s a shame to keep moving forward.<br><br>
# You can’t listen to a man who doesn’t have confidence in you if you don’t have confidence in yourself.<br><br>
# Making love with a good man is like making love with a good working woman.<br><br>
# The best thing you can do is to keep moving forward.<br><br>
# I have learned that the best thing you can do is to keep moving forward.<br><br>
# I believe that the best thing you can do is to keep moving forward.<br><br>
# Hate is the most wonderful thing you can see. It is a wonderful thing.<br><br>
# A struggle is not easy. It is hard to build a life of happiness. It is hard to make love with someone who won't give you his heart.<br><br>
# We have been taught that we have to be the best people we can be to make love.<br><br>
# Some of us will be disappointed by the outcome of our relationships. But if you can be disappointed by yourself, you don’t need to be disappointed in others. You can reach out and make love with someone who is capable of making love with you.<br><br>
# It's great to be a rebel. If you don’t have anything to gain or lose by being a rebel, you don’t need to be a rebel.<br><br>
# The best thing you can do is to keep moving forward.<br><br>
# The best thing you can do is to keep moving forward.<br><br>
# If you're serious about your work, you have to do something about it. And no matter how much you try, you can never stop thinking about it.<br><br>
# I have learned that the best thing you can do is to keep moving forward.<br><br>
# I’m not saying you’re a bad person, but I’m saying you’re not a good person.<br><br>
# When you have everything you need to make love, you should take the best possible care of it.<br><br>
# I want to hear more about this kind of thing, and I want to hear all the things I’m willing to do to make love, and I’m not saying I’m not a good person, but I’m not a good person.<br><br>
# It's not how many times you make love that counts, it's how many times you feel you’re making love with someone who is capable of making love with you.<br><br>
# I never thought I was going to have a good time with him. I had a bad experience.<br><br>
# I’ve learned that the best thing you can do is to keep moving forward.<br><br>
# I’m not saying you’re a bad person, but I’m saying you’re not a good person.<br><br>
# When a man is serious about his work, he will find ways to make love with people who are capable of making love with him.<br><br>
# The best thing you can do is to keep moving forward.<br><br>
# If you're serious about your work, you have to do something about it. And no matter how much you try, you can never stop thinking about it.<br><br>
# If you're serious about your work, you have to do something about it. And no matter how much you try, you can never stop thinking about it.<br><br>
# He is in love with a beautiful girl who is going to be his wife for the rest of his life.<br><br>
# My heart is a fragile thing, and my heart is fragile.<br><br>
# If you're serious about your work, you have to do something about it. And no matter how much you try, you can never stop thinking about it.
# I am not saying that I am an idealist, or that I am looking for perfection. I am simply saying that I am a person gone through with a heart that is always moving, a heart that is always moving.<br><br>
# 
# 

# In[ ]:





# In[ ]:




