#!/usr/bin/env python
# coding: utf-8

# > **Since I received great response from the community for my [Original BERT kernel](https://www.kaggle.com/abhinand05/bert-for-humans-tutorial-baseline) in the [TF QA Competition](https://www.kaggle.com/c/tensorflow2-question-answering) and even some people reached out asking me to do a similar kernel for the [Google QUEST competition ](https://www.kaggle.com/c/google-quest-challenge) (as they are kinda similar as well), I was really motivated and here I am with another BERT for Humans thing, hope you enjoy it.**

# ![](http://)<font size=4 color='#25171A'>This is a two part Notebook</font>
# 
# <a href="#Comprehensive-BERT-Tutorial">1. Comprehensive BERT Tutorial</a> <br>
# <a href="#Code-Implementation-in-Tensorflow-2.0">2. Implementation in Tensorflow 2.0</a>
# <br>
# 
# > **Note:** The main objective of this notebook is to provide a **baseline for this competition with some explanation about BERT**. I decided to wite such a notebook because I didn't find anything quite like this when I started out at NLP Competitions. I hope beginners can benefit from this notebook. Even if you're a non-beginner there might be some elements in this notebook you may be interested in.
# 
# <br>

# <font size=4 color='red'>If you like this approach please give this kernel an UPVOTE to show your appreciation</font>

# # Comprehensive BERT Tutorial
# 
# ## Introduction
# <font size="3" color='#003249'>So if you're like me just starting out at NLP after spending a few months building Computer Vision models as a beginner then surely this kernel has something in store for you.</font>
# <br>
# 
# So if you're like me just starting out at NLP after spending a few months building Computer Vision models as a beginner then surely this kernel has something in store for you.
# 
# I had a hard time wrapping my head around this all new bleeding-edge, state-of-the-art NLP model BERT, I had to dig through a lot of articles to truly grasp what BERT is all about, I'll share my understanding of BERT in this notebook.
# 
# ![](https://cdn-images-1.medium.com/max/1000/1*-oQKmzvHrzqeSQEnM9f_kQ.png)
# 
# ## References and Credits:
# This notebook wouldn't have been possible without these amazing resources. Most of the text and figures used in this notebooks are taken from the below mentioned resources, combining everything into one.
# 1. [BERT for Dummies step by step tutorial by Michel Kana](https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03)
# 2. [Demystifying BERT: Groundbreaking NLP Framework by Mohd Sanad Zaki Rizvi](https://www.analyticsvidhya.com/blog/2019/09/demystifying-bert-groundbreaking-nlp-framework/)
# 3. [A visual guide to using BERT by Jay Alammar](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
# 4. [BERT Fine tuning By Chris McCormick and Nick Ryan](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
# 5. [How to use BERT in Kaggle competitions - Reddit Thread](https://www.reddit.com/r/MachineLearning/comments/ao23cp/p_how_to_use_bert_in_kaggle_competitions_a/)
# 6. [BERT GitHub repository](https://github.com/google-research/bert)
# 7. [BERT - SOTA NLP model Explained by Rani Horev](https://www.kdnuggets.com/2018/12/bert-sota-nlp-model-explained.html)
# 8. [YOUTUBE - BERT Pretranied Deep Bidirectional Transformers for Language Understanding algorithm by Danny Luo](https://www.youtube.com/watch?v=BhlOGGzC0Q0)
# 9. [State-of-the-art pre-training for natural language processing with BERT by Javed Quadrud-Din](https://blog.insightdatascience.com/using-bert-for-state-of-the-art-pre-training-for-natural-language-processing-1d87142c29e7)
# 
# 
# ## Contents
# <a href="#The-BERT-Landscape">1. The BERT Landscape</a>  
# <a href="#What-is-BERT?">2. What is BERT?</a>  
# <a href="#Why-BERT-matters?">3. Why BERT Matters?</a>
# <br>
# <a href="#How-BERT-Works?">4. How BERT works?</a> <br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#1.-Architecture-of-BERT">4.1 Architecture of BERT</a>   
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#2.-Preprocessing-Text-for-BERT">4.2 Preprocessing text for BERT</a>   
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#3.-Pre-training">4.3 Pre-training</a>   
# <a href="#5.-Fine-Tuning_Techniques-for-BERT">5. Fine Tuning Techniques for BERT</a> <br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#5.1-Sequence-Classification-Tasks">5.1 Sequence Classification Tasks</a>   
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#5.2-Sentence-Pair-Classification-Tasks">5.2 Sentence Pair Classification Tasks</a>   
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#5.3-Question-Answering-Tasks">5.3 Question Answering Tasks</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#5.4-Single-Sentence-Tagging-Tasks">5.4 Single Sentence Tagging  Tasks</a><br> 
# &nbsp;&nbsp;&nbsp;&nbsp;<a href="#5.5-Hyperparameter-Tuning">5.5 Hyperparameter Tuning</a><br>
# <a href="#6.-BERT-Benchmarks-on-Question-Answering-Tasks">6. BERT Benchmarks on Question/Answering Tasks</a> <br>
# <a href="#7.-Key-Takeaways">7. Key Takeaways</a> <br>
# <a href="#8.-Conclusion">8. Conclusion</a>

# ## 1. The BERT Landscape
# > BERT is a deep learning model that has given state-of-the-art results on a wide variety of natural language processing tasks. It stands for **Bidirectional Encoder Representations for Transformers**. It has been pre-trained on Wikipedia and BooksCorpus and requires (only) task-specific fine-tuning.
# 
#  It has caused a stir in the Machine Learning community by presenting state-of-the-art results in a wide variety of NLP tasks, including **Question Answering (SQuAD v1.1)**, **Natural Language Inference (MNLI)**, and others.
#  
#  It’s not an exaggeration to say that BERT has significantly altered the NLP landscape. Imagine using a single model that is trained on a large unlabelled dataset to achieve State-of-the-Art results on 11 individual NLP tasks. And all of this with little fine-tuning. That’s BERT! It’s a tectonic shift in how we design NLP models.
# 
# BERT has inspired many recent NLP architectures, training approaches and language models, such as Google’s TransformerXL, OpenAI’s GPT-2, XLNet, ERNIE2.0, RoBERTa, etc.

# ## 2. What is BERT?
# It is basically a bunch of Transformer encoders stacked together (not the whole Transformer architecture but just the encoder). The concept of bidirectionality is the key differentiator between BERT and its predecessor, OpenAI GPT. BERT is bidirectional because its self-attention layer performs self-attention on both directions.
# 
# There are a few things I want to explain in this section.
# 
# * First, It’s easy to get that BERT stands for Bidirectional Encoder Representations from Transformers. Each word here has a meaning to it and we will encounter that one by one. For now, **the key takeaway from this line is – BERT is based on the Transformer architecture.** 
# 
# * Second, BERT is **pre-trained on a large corpus of unlabelled text** including the entire Wikipedia(that’s 2,500 million words!) and Book Corpus (800 million words). This pretraining step is really important for BERT's success. This is because as we train a model on a large text corpus, our model starts to pick up the deeper and intimate understandings of how the language works. This knowledge is the swiss army knife that is useful for almost any NLP task.
# 
# * Third, BERT is a **deeply bidirectional** model. Bidirectional means that BERT learns information from both the left and the right side of a token’s context during the training phase.
# 
# This bidirectional understanding is crucial to take NLP models to the next level. Let's see an example to understand what it really means. There may be two sentences having the same word but their meaning may be completely different based on what comes before or after as we can see here below.
# 
# ![bidirectionalexample](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2019/09/sent_context.png)
# 
# Without taking these contexts into consideration it's impossible for machines to truly understand meanings and it may throw out trashy responses time and time again which is not really a good thing.
# 
# But BERT fixes this. Yes it does. That was one of the game changing aspect of BERT.
# 
# * Fourth, finally the biggest advantage of BERT is it brought about the **ImageNet movement** with it and the most impressive aspect of BERT is that we can fine-tune it by adding just a couple of additional output layers to create state-of-the-art models for a variety of NLP tasks.
# 

# ## 3. Why BERT matters?
# 
# Now I think it's pretty clear to you why but let's see proof, as we should always do.
# 
# ![stats](https://miro.medium.com/max/1200/0*-k_fjBnCuByNye4v)
# 
# While it’s not clear that all GLUE tasks are very meaningful, generic models based on an encoder named Transformer (Open-GPT, BERT and BigBird), closed the gap between task-dedicated models and human performance and within less than a year.

# ## 4. How BERT Works?
# Let’s look a bit closely at BERT and understand why it is such an effective method to model language. We’ve already seen what BERT can do earlier – but how does it do it? We’ll answer this pertinent question in this section.
# 
# ### 1. Architecture of BERT
# 
# BERT is a multi-layer bidirectional Transformer encoder. There are two models introduced in the paper.
# 
# * BERT base – 12 layers (transformer blocks), 12 attention heads, and 110 million parameters.
# * BERT Large – 24 layers, 16 attention heads and, 340 million parameters.
# 
# For an in-depth understanding of the building blocks of BERT (aka Transformers), you should definitely check [this awesome post](http://jalammar.github.io/illustrated-transformer/) – The Illustrated Transformers.
# 
# *Here's a representation of BERT Architecture*
# ![arch](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2019/09/bert_encoder.png)
# 
# ### 2. Preprocessing Text for BERT
# The input representation used by BERT is able to represent a single text sentence as well as a pair of sentences (eg., Question, Answering) in a single sequence of tokens.
# 
# * The first token of every input sequence is the special classification token – **[CLS]**. This token is used in classification tasks as an aggregate of the entire sequence representation. It is ignored in non-classification tasks.
# * For single text sentence tasks, this **[CLS]** token is followed by the WordPiece tokens and the separator token – **[SEP]**.
# 
# ![](https://yashuseth.files.wordpress.com/2019/06/fig7.png)
# 
# * For sentence pair tasks, the WordPiece tokens of the two sentences are separated by another [SEP] token. This input sequence also ends with the **[SEP]** token.
# 
# * A sentence embedding indicating Sentence A or Sentence B is added to each token. Sentence embeddings are similar to token/word embeddings with a vocabulary of 2.
# 
# * A positional embedding is also added to each token to indicate its position in the sequence.
# 
# BERT developers have set a a specific set of rules to represent languages before feeding into the model.
# 
# ![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2019/09/bert_emnedding.png)
# 
# For starters, every input embedding is a combination of 3 embeddings:
# 
# * **Position Embeddings**: BERT learns and uses positional embeddings to express the position of words in a sentence. These are added to overcome the limitation of Transformer which, unlike an RNN, is not able to capture “sequence” or “order” information
# * **Segment Embeddings**: BERT can also take sentence pairs as inputs for tasks (Question-Answering). That’s why it learns a unique embedding for the first and the second sentences to help the model distinguish between them. In the above example, all the tokens marked as EA belong to sentence A (and similarly for EB)
# * **Token Embeddings**: These are the embeddings learned for the specific token from the WordPiece token vocabulary
# 
# For a given token, its input representation is constructed by **summing the corresponding token, segment, and position embeddings**.
# 
# Such a comprehensive embedding scheme contains a lot of useful information for the model.
# 
# These combinations of preprocessing steps make BERT so versatile. This implies that without making any major change in the model’s architecture, we can easily train it on multiple kinds of NLP tasks.
# 
# **Tokenization:**
# BERT uses WordPiece tokenization. The vocabulary is initialized with all the individual characters in the language, and then the most frequent/likely combinations of the existing words in the vocabulary are iteratively added.
# 
# 
# ### 3. Pre Training
# The model was trained in two tasks simultaneously:
# 
# **1. Masked Language Model** 
# 
# **2. Next Sentence Prediction.**
# 
# **Note:** I am not going to go over these two techniques in this notebook. I recommend online reading.

# ## 5. Fine Tuning Techniques for BERT
# Using BERT for a specific task is relatively straightforward. BERT can be used for a wide variety of language tasks, while only adding a small layer to the core model
# 
# ### 5.1 Sequence Classification Tasks
# The final hidden state of the **[CLS]** token is taken as the fixed-dimensional pooled representation of the input sequence. This is fed to the classification layer. The classification layer is the only new parameter added and has a dimension of K x H, where K is the number of classifier labels and H is the size of the hidden state. The label probabilities are computed with a standard softmax.
# 
# ![](https://yashuseth.files.wordpress.com/2019/06/fig1-1.png?w=460&h=400)
# 
# ### 5.2 Sentence Pair Classification Tasks
# This procedure is exactly similar to the single sequence classification task. The only difference is in the input representation where the two sentences are concatenated together.
# 
# ![](https://yashuseth.files.wordpress.com/2019/06/fig2-1.png?w=443&h=398)
# 
# ### 5.3 Question-Answering Tasks
# Question answering is a prediction task. Given a question and a context paragraph, the model predicts a start and an end token from the paragraph that most likely answers the question.
# 
# ![](https://yashuseth.files.wordpress.com/2019/06/fig6.png?w=389&h=297)
# 
# Just like sentence pair tasks, the question becomes the first sentence and paragraph the second sentence in the input sequence. There are only two new parameters learned during fine-tuning a start vector and an end vector with size equal to the hidden shape size. The probability of token i being the start of the answer span is computed as – softmax(S . K), where S is the start vector and K is the final transformer output of token i. The same applies to the end token.
# 
# ![](https://yashuseth.files.wordpress.com/2019/06/fig3.png?w=452&h=380)
# 
# ### 5.4 Single Sentence Tagging Tasks
# 
# In single sentence tagging tasks such as named entity recognition, a tag must be predicted for every word in the input. The final hidden states (the transformer output) of every input token is fed to the classification layer to get a prediction for every token. Since WordPiece tokenizer breaks some words into sub-words, the prediction of only the first token of a word is considered.
# 
# ![](https://yashuseth.files.wordpress.com/2019/06/fig4.png?w=441&h=389)
# 
# ### 5.5 Hyperparameter Tuning
# The optimal hyperparameter values are task-specific. But, the authors found that the following range of values works well across all tasks
# 
# * **Dropout** – 0.1
# * **Batch Size** – 16, 32
# * **Learning Rate (Adam)** – 5e-5, 3e-5, 2e-5
# * **Number of epochs** – 3, 4 (yeah you read it right)
# 
# The authors also observed that large datasets (> 100k labeled samples) are less sensitive to hyperparameter choice than smaller datasets.
# 

# ## 6. BERT Benchmarks on Question Answering tasks
# 
# > The Standford Question Answering Dataset (SQuAD) is a collection of 100k crowdsourced question/answer pairs (Rajpurkar et al., 2016). Given a question and a paragraph from Wikipedia containing the answer, the task is to predict the answer text span in the paragraph.
# 
# In SQUAD the big improvement in performance was achieved by BERT large. The model that achieved the highest score was an ensemble of BERT large models, augmenting the dataset with TriviaQA.
# 
# ![](https://miro.medium.com/max/558/1*CYzIm-u1-JUR2jDyPRHlQg.png)

# ## 7. Key Takeaways
# 
# > 1) Model size matters, even at huge scale. BERT_large, with 345 million parameters, is the largest model of its kind. It is demonstrably superior on small-scale tasks to BERT_base, which uses the same architecture with “only” 110 million parameters.
# 
# > 2) With enough training data, more training steps == higher accuracy. For instance, on the MNLI task, the BERT_base accuracy improves by 1.0% when trained on 1M steps (128,000 words batch size) compared to 500K steps with the same batch size.
# 
# > 3) BERT’s bidirectional approach (MLM) converges slower than left-to-right approaches (because only 15% of words are predicted in each batch) but bidirectional training still outperforms left-to-right training after a small number of pre-training steps.
# 
# ![](https://miro.medium.com/max/1576/0*KONsqvDohE7ytu_E.png)

# ## 8. Conclusion
# 
# BERT is undoubtedly a breakthrough in the use of Machine Learning for Natural Language Processing. The fact that it’s approachable and allows fast fine-tuning will likely allow a wide range of practical applications in the future. In this summary, we attempted to describe the main ideas of the paper while not drowning in excessive technical details. For those wishing for a deeper dive, we highly recommend reading the full article and ancillary articles referenced in it. 
# 
# > **Feel free to pass on any suggestion to improve this notebook in the comment section (if you have any)**

# <font size=5 color='green'>Please give this kernel an UPVOTE to show your appreciation, if you find it useful.</font>

# # Code Implementation in Tensorflow 2.0
# > Note: The code for this notebook is taken from the [public kernel](https://www.kaggle.com/akensert/bert-base-tf2-0-minimalistic/) posted by [akensert](https://www.kaggle.com/akensert)
# 
# This kernel does not explore the data. For that you could check out some of the great EDA kernels: [introduction](https://www.kaggle.com/corochann/google-quest-first-data-introduction), [getting started](https://www.kaggle.com/phoenix9032/get-started-with-your-questions-eda-model-nn) & [another getting started](https://www.kaggle.com/hamditarek/get-started-with-nlp-lda-lsa). This kernel is an example of a TensorFlow 2.0 Bert-base implementation, using TensorFow Hub. <br><br>

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow_hub as hub
import tensorflow as tf
import bert_tokenization as tokenization
import tensorflow.keras.backend as K
import gc
import os
from scipy.stats import spearmanr
from math import floor, ceil

np.set_printoptions(suppress=True)


# **1. Read data and tokenizer**
# 
# Read tokenizer and data, as well as defining the maximum sequence length that will be used for the input to Bert (maximum is usually 512 tokens)

# In[ ]:


PATH = '../input/google-quest-challenge/'
BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
tokenizer = tokenization.FullTokenizer(BERT_PATH+'/assets/vocab.txt', True)
MAX_SEQUENCE_LENGTH = 512

df_train = pd.read_csv(PATH+'train.csv')
df_test = pd.read_csv(PATH+'test.csv')
df_sub = pd.read_csv(PATH+'sample_submission.csv')
print('train shape =', df_train.shape)
print('test shape =', df_test.shape)

output_categories = list(df_train.columns[11:])
input_categories = list(df_train.columns[[1,2,5]])
print('\noutput categories:\n\t', output_categories)
print('\ninput categories:\n\t', input_categories)


# **2. Preprocessing functions**
# 
# These are some functions that will be used to preprocess the raw text data into useable Bert inputs.

# In[ ]:


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def _trim_input(title, question, answer, max_sequence_length, 
                t_max_len=30, q_max_len=239, a_max_len=239):

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+4) > max_sequence_length:
        
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len
      
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len
            
            
        if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d" 
                             % (max_sequence_length, (t_new_len+a_new_len+q_new_len+4)))
        
        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]
    
    return t, q, a

def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    
    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]

def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        t, q, a = _trim_input(t, q, a, max_sequence_length)

        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


# #### 3. Create model
# 
# `compute_spearmanr()` is used to compute the competition metric for the validation set
# <br><br>
# `CustomCallback()` is a class which inherits from `tf.keras.callbacks.Callback` and will compute and append validation score and validation/test predictions respectively, after each epoch.
# <br><br>
# `bert_model()` contains the actual architecture that will be used to finetune BERT to our dataset. It's simple, just taking the sequence_output of the bert_layer and pass it to an AveragePooling layer and finally to an output layer of 30 units (30 classes that we have to predict)
# <br><br>
# `train_and_predict()` this function will be run to train and obtain predictions

# In[ ]:


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, valid_data, test_data, batch_size=16, fold=None):

        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.test_inputs = test_data
        
        self.batch_size = batch_size
        self.fold = fold
        
    def on_train_begin(self, logs={}):
        self.valid_predictions = []
        self.test_predictions = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(
            self.model.predict(self.valid_inputs, batch_size=self.batch_size))
        
        rho_val = compute_spearmanr(
            self.valid_outputs, np.average(self.valid_predictions, axis=0))
        
        print("\nvalidation rho: %.4f" % rho_val)
        
        if self.fold is not None:
            self.model.save_weights(f'bert-base-{fold}-{epoch}.h5py')
        
        self.test_predictions.append(
            self.model.predict(self.test_inputs, batch_size=self.batch_size)
        )

def bert_model():
    
    input_word_ids = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')
    
    bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)
    
    _, sequence_output = bert_layer([input_word_ids, input_masks, input_segments])
    
    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(30, activation="sigmoid", name="dense_output")(x)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_masks, input_segments], outputs=out)
    
    return model    
        
def train_and_predict(model, train_data, valid_data, test_data, 
                      learning_rate, epochs, batch_size, loss_function, fold):
        
    custom_callback = CustomCallback(
        valid_data=(valid_data[0], valid_data[1]), 
        test_data=test_data,
        batch_size=batch_size,
        fold=None)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer)
    model.fit(train_data[0], train_data[1], epochs=epochs, 
              batch_size=batch_size, callbacks=[custom_callback])
    
    return custom_callback


# **4. Obtain inputs and targets, as well as the indices of the train/validation splits**

# In[ ]:


gkf = GroupKFold(n_splits=5).split(X=df_train.question_body, groups=df_train.question_body)

outputs = compute_output_arrays(df_train, output_categories)
inputs = compute_input_arays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)


# **5. Training, validation and testing**
# 
# Loops over the folds in gkf and trains each fold for 4 epochs --- with a learning rate of 3e-5 and batch_size of 8. A simple binary crossentropy is used as the objective-/loss-function. 

# In[ ]:


histories = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
    
    # will actually only do 3 folds (out of 5) to manage < 2h
    if fold < 3:
        K.clear_session()
        model = bert_model()

        train_inputs = [inputs[i][train_idx] for i in range(3)]
        train_outputs = outputs[train_idx]

        valid_inputs = [inputs[i][valid_idx] for i in range(3)]
        valid_outputs = outputs[valid_idx]

        # history contains two lists of valid and test preds respectively:
        #  [valid_predictions_{fold}, test_predictions_{fold}]
        history = train_and_predict(model, 
                          train_data=(train_inputs, train_outputs), 
                          valid_data=(valid_inputs, valid_outputs),
                          test_data=test_inputs, 
                          learning_rate=3e-5, epochs=4, batch_size=8,
                          loss_function='binary_crossentropy', fold=fold)

        histories.append(history)


# #### 6. Process and submit test predictions
# 
# First the test predictions are read from the list of lists of `histories`. Then each test prediction list (in lists) is averaged. Then a mean of the averages is computed to get a single prediction for each data point. Finally, this is saved to `submission.csv`

# In[ ]:


test_predictions = [histories[i].test_predictions for i in range(len(histories))]
test_predictions = [np.average(test_predictions[i], axis=0) for i in range(len(test_predictions))]
test_predictions = np.mean(test_predictions, axis=0)

df_sub.iloc[:, 1:] = test_predictions

df_sub.to_csv('submission.csv', index=False)


# <font size=4 color='#57467B'>Please give this kernel an UPVOTE to show your appreciation, if you find it useful.</font>
# <br>
# <br>
# <font size=4 color='#57467B'>Also don't forget to upvote akensert's kernel <a href='https://www.kaggle.com/akensert/bert-base-tf2-0-minimalistic'>here</a></font>
