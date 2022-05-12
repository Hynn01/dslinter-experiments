#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 

import h2o
from h2o.automl import H2OAutoML

TRAIN_PATH = "../input/tabular-playground-series-may-2022/train.csv"
TEST_PATH = "../input/tabular-playground-series-may-2022/test.csv"
SAMPLE_SUBMISSION_PATH = "../input/tabular-playground-series-may-2022/sample_submission.csv"
SUBMISSION_PATH = "submission.csv "

ID = "id"
TARGET = "target"

NEW_TRAIN_PATH = "train.csv"
NEW_TEST_PATH = "test.csv"

SEED_LIST = [7,77,777]
MAX_RUNTIME_SECS = 60 * 3


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

def create_tf_idf_feats(corpus, ngram_range = (1, 1), max_features = None):
    vectorizer = TfidfVectorizer(analyzer = 'char', lowercase = False, 
                                 ngram_range = ngram_range, max_features = max_features)
    X = vectorizer.fit_transform(corpus).todense()
    char_mapper = {y:x for x, y in vectorizer.vocabulary_.items()}
    column_names = ['tfidf_{}'.format(char_mapper[i]) for i in range(len(char_mapper))]
    return pd.DataFrame(X, columns = column_names)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Create all texts array
all_texts = pd.concat([train[['f_27']], test[['f_27']]]).reset_index(drop = True)
corpus = all_texts['f_27'].values

# https://www.kaggle.com/code/alexryzhkov/tps-may-22-lightautoml-here-again/notebook
# Calculate TF-IDF features for unigrams and top-20 bigrams
all_texts = pd.concat([all_texts, 
                      create_tf_idf_feats(corpus),
                      create_tf_idf_feats(corpus, (2,2), 20)], axis = 1)

for i in range(10):
    all_texts[f'ch{i}'] = all_texts.f_27.str.get(i).apply(ord) - ord('A')

all_texts["unique_characters"] = all_texts.f_27.apply(lambda s: len(set(s)))

# How often the text occurs in the whole dataset
all_texts['value_frequency'] = all_texts['f_27'].map(all_texts['f_27'].value_counts() / len(all_texts))

all_texts.drop(columns = ['f_27'], inplace = True)

train = pd.concat([train,
                       all_texts.iloc[:len(train), :]], axis = 1)
test = pd.concat([test,
                       all_texts.iloc[len(train):, :].reset_index(drop = True)], axis = 1)
    
train.to_csv(NEW_TRAIN_PATH,index=False)
test.to_csv(NEW_TEST_PATH,index=False)


# In[ ]:


h2o.init()

train = h2o.import_file(NEW_TRAIN_PATH)
test = h2o.import_file(NEW_TEST_PATH)

x = train.columns
y = TARGET

x.remove(y)
x.remove(ID) #remove id  

pred_test = []
for selSeed in SEED_LIST:
    aml_y = H2OAutoML(max_runtime_secs=MAX_RUNTIME_SECS, seed=selSeed)
    aml_y.train(x=x, y=y, training_frame=train)

    preds_y = aml_y.predict(test)
    pred_test.append(preds_y.as_data_frame().predict) 


# In[ ]:


from scipy import stats
modeResult = stats.mode(pred_test, axis=0)
final_test_pred = modeResult.mode

submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
submission[TARGET] = final_test_pred[0]
submission.to_csv(SUBMISSION_PATH, index=False)
submission.head()

