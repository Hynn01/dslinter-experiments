#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# In[ ]:


labels = pd.read_csv('../input/tabular-playground-series-apr-2022/train_labels.csv')
train = pd.read_csv('../input/tabular-playground-series-apr-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-apr-2022/test.csv')


# In[ ]:


labels = labels.merge(train.iloc[::60, :][['sequence', 'subject']], on='sequence', how='left')


# In[ ]:


oofs1 = np.load('../input/tpsaprilblend/oof_bilstm_do_0.974.npy')
oofs2 = np.load('../input/tpsaprilblend/oof_inception_0.971.npy')
oofs3 = pd.read_csv('../input/tpsaprilblend/oof_feature_0.98.csv').set_index('sequence').loc[labels['sequence']]['pred'].values
oofs4 = pd.read_csv('../input/tpsaprilblend/oof_tfv1_0.98.csv').set_index('sequence').loc[labels['sequence']]['pred'].values
oofs5 = pd.read_csv('../input/tpsaprilblend/oof_cat_features_0.977.csv').set_index('sequence').loc[labels['sequence']]['pred'].values
oofs5 = rankdata(oofs5) / len(oofs5)
oofs6 = pd.read_csv('../input/tpsaprilblend/oof_tps-apr22-tfv1__oof_preds__NO__goup__0.9725.csv').set_index('sequence').loc[labels['sequence']]['valid_pred'].values
oofs7 = pd.read_csv('../input/tpsaprilblend/oof_train_preds_xgb_0.972.csv').set_index('sequence').loc[labels['sequence']]['pred'].values
oofs8 = pd.read_csv('../input/tpsaprilblend/oof_tfv2_0.981.csv').set_index('sequence').loc[labels['sequence']]['pred'].values
oofs9 = np.load('../input/tpsaprilblend/oof_cnn_lstm_0.979.npy')
oofs10 = np.load('../input/tpsaprilblend/oof_resnet_0.977.npy')


# In[ ]:


oofs = [oofs1, oofs2, oofs3, oofs4, oofs5, oofs6, oofs7, oofs8, oofs9, oofs10]

for i, oof in enumerate(oofs):
    print(i, roc_auc_score(labels['state'], oof))
print('Mean AUC =', roc_auc_score(labels['state'], np.mean(oofs, axis=0)))
print('Mean-Rank AUC =', roc_auc_score(labels['state'], np.mean([rankdata(oof) for oof in oofs], axis=0)))


# In[ ]:


sub1 = pd.read_csv('../input/tpsaprilblend/sub_bilstm_do_0.974.csv')['state'].values
sub2 = pd.read_csv('../input/tpsaprilblend/sub_inception_0.971.csv')['state'].values
sub3 = pd.read_csv('../input/tpsaprilblend/sub_feature_0.98.csv')['state'].values
sub4 = pd.read_csv('../input/tpsaprilblend/sub_tfv1_0.98.csv')['state'].values
sub5 = pd.read_csv('../input/tpsaprilblend/sub_1904_2.csv')['state'].values
sub5 = rankdata(sub5) / len(sub5)
sub6 = pd.read_csv('../input/tpsaprilblend/sub_tps-apr22-tfv1_0.975.csv')['state'].values
sub7 = pd.read_csv('../input/tpsaprilblend/sub_xgboost-2500-features_0.972.csv')['state'].values
sub8 = pd.read_csv('../input/tpsaprilblend/sub_tfv2_0.981.csv')['state'].values
sub9 = pd.read_csv('../input/tpsaprilblend/sub_cnn_lstm_0.979.csv')['state'].values
sub10 = pd.read_csv('../input/tpsaprilblend/sub_resnet_0.977.csv')['state'].values


# In[ ]:


X_test = np.hstack(([sub[:, None] for sub in [sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9, sub10]]))


# In[ ]:


oof_df = pd.DataFrame(np.array(oofs).T)
oof_df['subject'] = labels['subject']
averages = oof_df.groupby(['subject']).agg(['mean', 'std', 'max', 'min', 'count'])
cols = averages.columns.values
new_cols = ['_'.join(map(str, x)) for x in cols]
averages.columns = new_cols
oof_df = oof_df.merge(averages, left_on='subject', right_index=True, how='left')
oof_df = oof_df.drop(columns=['subject'])
oof_df.head(5)


# In[ ]:


test_pred_df = pd.DataFrame(X_test)
test_pred_df['subject'] = test['subject'].values[::60]
averages = test_pred_df.groupby(['subject']).agg(['mean', 'std', 'max', 'min', 'count'])
cols = averages.columns.values
new_cols = ['_'.join(map(str, x)) for x in cols]
averages.columns = new_cols
test_pred_df = test_pred_df.merge(averages, left_on='subject', right_index=True, how='left').drop(columns=['subject'])
test_pred_df.head(5)


# In[ ]:


all_preds = []
aucs = []
for i in range(100):
    seed = i#np.random.randint(1000000)
    
#     gss = GroupShuffleSplit(n_splits=1, train_size=.95, random_state=i)
#     train_ix, val_ix = next(gss.split(np.arange(len(oofs[0])), labels['state'].values, labels['subject'].values))
#     y_train = labels['state'].values[train_ix]
#     y_val = labels['state'].values[val_ix]
    
    train_ix, val_ix, y_train, y_val = train_test_split(np.arange(len(oofs[0])), 
                                                    labels['state'].values, test_size=0.05,
                                                    random_state=seed)
    
    X_train = oof_df.values[train_ix]
    X_val = oof_df.values[val_ix]
    
    model = CatBoostClassifier(iterations=5000, verbose=0, od_type='Iter', od_wait=100, 
                               task_type="CPU", random_state=seed, eval_metric='AUC')
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    preds = rankdata(model.predict_proba(test_pred_df.values)[:, 1])
    
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(i, auc)
    aucs.append(auc)
    all_preds.append(preds)


# In[ ]:


np.mean(aucs), np.std(aucs)


# In[ ]:


new_sub = pd.read_csv('../input/tpsaprilblend/sub_1904_2.csv')
new_sub['state'] = sum(all_preds)/len(all_preds)
new_sub.to_csv('stacking.csv', index=False)


# In[ ]:




