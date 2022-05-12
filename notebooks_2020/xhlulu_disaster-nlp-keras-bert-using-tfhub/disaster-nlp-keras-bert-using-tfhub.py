#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# 
# I've seen a lot of people pooling the output of BERT, then add some Dense layers. I also saw various learning rates for fine-tuning. In this kernel, I wanted to try some ideas that were used in the original paper that did not appear in many public kernel. Here are some examples:
# * *No pooling, directly use the CLS embedding*. The original paper uses the output embedding for the `[CLS]` token when it is finetuning for classification tasks, such as sentiment analysis. Since the `[CLS]` token is the first token in our sequence, we simply take the first slice of the 2nd dimension from our tensor of shape `(batch_size, max_len, hidden_dim)`, which result in a tensor of shape `(batch_size, hidden_dim)`.
# * *No Dense layer*. Simply add a sigmoid output directly to the last layer of BERT, rather than experimenting with different intermediate layers.
# * *Fixed learning rate, batch size, epochs, optimizer*. As specified by the paper, the optimizer used is Adam, with a learning rate between 2e-5 and 5e-5. Furthermore, they train the model for 3 epochs with a batch size of 32. I wanted to see how well it would perform with those default values.
# 
# I also wanted to share this kernel as a **concise, reusable, and functional example of how to build a workflow around the TF2 version of BERT**. Indeed, it takes less than **50 lines of code to write a string-to-tokens preprocessing function and model builder**.
# 
# ## References
# 
# * Source for `bert_encode` function: https://www.kaggle.com/user123454321/bert-starter-inference
# * All pre-trained BERT models from Tensorflow Hub: https://tfhub.dev/s?q=bert

# In[ ]:


# We will use the official tokenization script created by the Google team
get_ipython().system('wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py')


# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

import tokenization


# # Helper Functions

# In[ ]:


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


# In[ ]:


def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# # Load and Preprocess
# 
# - Load BERT from the Tensorflow Hub
# - Load CSV files containing training data
# - Load tokenizer from the bert layer
# - Encode the text into tokens, masks, and segment flags

# In[ ]:


get_ipython().run_cell_magic('time', '', 'module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"\nbert_layer = hub.KerasLayer(module_url, trainable=True)')


# In[ ]:


train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


# In[ ]:


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


# In[ ]:


train_input = bert_encode(train.text.values, tokenizer, max_len=160)
test_input = bert_encode(test.text.values, tokenizer, max_len=160)
train_labels = train.target.values


# # Model: Build, Train, Predict, Submit

# In[ ]:


model = build_model(bert_layer, max_len=160)
model.summary()


# In[ ]:


checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint],
    batch_size=16
)


# In[ ]:


model.load_weights('model.h5')
test_pred = model.predict(test_input)


# In[ ]:


submission['target'] = test_pred.round().astype(int)
submission.to_csv('submission.csv', index=False)

