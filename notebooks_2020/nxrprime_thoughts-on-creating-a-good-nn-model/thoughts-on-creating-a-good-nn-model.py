#!/usr/bin/env python
# coding: utf-8

# # Thoughts on creating a good NN model
# 
# I know a lot of people have been struggling with how to build a good NN model, so here I compile all the tips and information pertaining to NNs in this competition. 
# 
# ---
# 
# There are 6 principal ideas to boost your NN.

# ---
# 
# # 1. Extreme Sensitivity to 0s
# 
# ---
# 
# This is probably the biggest issue for any NN, that an NN is extremely sensitive to 0s. My Transformer model scored 0.80 due to the presence of a lot of zeroes in the data. Other models are probably not as sensitive, but the higher the complexity of the model is (my Transformer model was titanic), the more likely your model is to be extremely sensitive to 0s. 
# 
# You can keep some amount of 0s, just make sure that about 60-70 percent of your NA values stay 0. Because trust me, keeping some amount of 0s is good for your model, but you can't keep too many zeroes and at the same time, ironically, you can't keep too few zeros. 
# 
# You could feasibly try average imputation, but that doesn't exactly work always. Average imputation is great and all, but it theoretically won't help so much.
# 
# Or you could try **random imputation**, which randomly selects values from the column other than 0s and fills in 0s with that. Here's a simple Python class to do that:

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np


# In[ ]:


class RandomImputer():
    def __init__(self, dataset):
        self.df = dataset
        self.cols = dataset.columns
        self.shape2 = dataset.shape[1]
    def impute(self):
        shape = self.shape2 / 10
        shape = shape * 3
        for i in self.df.columns():
            
            df[i] = df.head(shape).fillna(np.random.sample(np.zonzero(np.array(df[i]))))


# --- 
# 
# # 2. Seasonality in NNs
# 
# ---
# 
# You can use TensorFlow Probability (TFP) to compute seasonality in an NN. TensorFlow Probability is a great library that can help you compute statistical models on GPUs. For PyTorch lovers like myself, Open.ai has an alternative called Pyro, although the documentation is abstruse and virtually nonexistant at points. For this reason, I like to use TensorFlow Probability although I know I am better with PyTorch.
# 
# Here is a code snippet that shows you how to use seasonality from the TensorFlow Probability documentation:
# ```
# tfp.sts.Seasonal( num_seasons, num_steps_per_season=1, allow_drift=True, drift_scale_prior=None, initial_effect_prior=None, constrain_mean_effect_to_zero=True, observed_time_series=None, name=None )
# ```

# Now here we put it in practice with TensorFlow's high-level Keras API and a Sequential model to top it all off:

# In[ ]:


import tensorflow_probability as tp
import tensorflow as tf


# In[ ]:


model = tf.keras.models.Sequential(
    tf.keras.layers.Dense(1), # pretty much useless, just to serve the purpose of display
    tp.sts.Seasonal(3) # just an example
)


# ---
# 
# # 3. Categorical Embeddings
# 
# ---
# 
# The power combo of Categorical Embeddings and batch normalization can take your NN a pretty long way. Treating your categorical variables as embeddings is an extremely powerful concept for you NN. It is pretty powerful when combined with a batch normalization layer as well. I am still trying to figure out whether categorical embeddings are better than one-hot encoding, but both seem to  give similar results with categorical embeddings + batch norm doing better, albeit slightly.
# 
# I don't believe that my example will work so well, as what I use depends a lot on positional encodings. I recommend you check out MichaelMayer 's kernel, it really is a work of art.

# ---
# 
# # 4. The Proper Loss Function
# 
# ---
# 
# This is by far one of the most important things while building an NN. The loss function is pivotal to this comp, and what I have chosen has been carefully picked from one of the following samples.
# 
# Here's the best loss function(s) for some model types:
# 
# *Note: The following have been evaluated on Pytorch. Feel free to expiriment on your own.*
# 
# * **LSTM**
#     + 1. CrossEntropyLoss()
#     + 2. PoissonNLLLoss()
#     + 3. MSELoss()
# * **1-D CNN**
#     + 1. MSELoss()
#     + 2. CrossEntropyLoss()
#     + 3. PoissonNLLLoss()

# ---
# 
# # 5. Recursive and Fourier-based features
# 
# ---
# 
# I would really like to talk about Fourier-based features, because I have worked on them quite a lot and they work well if you try random imputation. There are a few Fourier features that I would like to explain here, as well as provide the mathematical equations.
# 
# Unfortunately, I cannot explain everything in so much detail so I suggest you visit the [official Scipy fourier transform documentation](https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html).
# 
# Recursive and fourier features work very well, with shifts of multilples of 7 in both cases being very fruitful. Many other people have used recursive features, but Fourier features are ones that I have used in my kernel [on preprocessing as well as feature engineering for this competition](https://www.kaggle.com/nxrprime/preprocessing-fe).

# ---
# 
# # 6. Be Creative with you NN Models
# 
# ---
# 
# Albeit this hint is less of technical nature that the previously provided ones, I really suggest you try something novel or hybridized. You have a huge list of options ahead of you like Transformer, Transformer-XL, ArgoNET, CapsuleNet (yes, CapsuleNet)... LRCN, there are so many models for you to put your own spin on.
# 
# For encoder-decoder structures, try positional encoding. For models like LSTMs and LRCNs, try to add seasonality. Literally, novel architectures won m4 (check Slaweks Smyl's paper on his ES-RNN, my intelligence pales in comparison to his so try to do what you can on your own). 
# 
# The TensorFlow ecosystem is a great way to incorporate your own ideas into models, I mean you can add Bernoulli models, Markov Chain models, and even TensorFlow Lattice models. 
# 
# **At the end, Slaweks Smyl won M4 by him putting a nice spin on RNNs. I recommend you be creative most of all.**
