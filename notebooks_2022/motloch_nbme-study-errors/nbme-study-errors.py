#!/usr/bin/env python
# coding: utf-8

# # Description

# In this notebook I show our error analysis routine for the NBME competition. Given OOF predictions and ground truths, we first count how many false positives/negative errors we made for each feature. At the end we then show a routine that for the selected feature (e.g. 815) displays the misannotated notes and highlights what we annotated correctly (true positive), what we missed (false negative) and what we annotated extra (false positive).

# ![Image](https://raw.githubusercontent.com/motloch/kaggle_images/main/kaggle_nbme.png)

# # Load libraries

# In[ ]:


import numpy as np
import pandas as pd
import spacy
import re


# # Load data

# Competition data

# In[ ]:


patient_notes = pd.read_csv('../input/nbme-score-clinical-patient-notes/patient_notes.csv')
features = pd.read_csv('../input/nbme-score-clinical-patient-notes/features.csv')


# Results of our OOF analysis - for each ID, we have a string of zeros and ones representing our prediction and similar string representing the ground truth. Notice these strings can be shorter than the length of the note (we truncate unnecessarily zeros at the end).

# In[ ]:


oof = pd.read_csv("../input/nbme-for-error-analysis/for_error_analysis.csv")
oof.fillna('0',inplace=True)
oof = oof.merge(patient_notes, on=['pn_num', 'case_num'], how='left')


# This is how you would get the file (starting from the OOF in [notebook](https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-inference#OOF)):

# In[ ]:


# oof = pd.read_pickle(CFG.path+'oof_df.pkl')

# truths = create_labels_for_scoring(oof)
# char_probs = get_char_probs(oof['pn_history'].values,
#                             oof[[i for i in range(CFG.max_len)]].values, 
#                             CFG.tokenizer)

# results = get_results(char_probs, th=th)
# preds = get_predictions(results)
# score = get_score(preds, truths)

# bin_preds = []
# bin_truths = []
# for pred, truth in zip(preds, truths):
#     if not len(pred) and not len(truth):
#         bin_preds.append([])          #CHANGE!
#         bin_truths.append([])         #CHANGE!
#         continue
#     length = max(np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0)
#     bin_preds.append(spans_to_binary(pred, length))
#     bin_truths.append(spans_to_binary(truth, length))
    
# for_error_analysis_df = oof[['id','case_num', 'pn_num', 'feature_num']]
# for_error_analysis_df['preds'] = 0
# for_error_analysis_df['truth'] = 0
    
# for idx in range(len(for_error_analysis_df)):
#     for_error_analysis_df['preds'][idx] = ''.join([str(int(x)) for x in bin_preds[idx]])
#     for_error_analysis_df['truth'][idx] = ''.join([str(int(x)) for x in bin_truths[idx]])
    
# for_error_analysis_df.to_csv('for_error_analysis.csv', index=False)


# # Useful functions

# In[ ]:


def has_error(IDX):
    """
    For OOF at IDX-th position, determine whether our model made a mistake.
    """
    t = oof.iloc[IDX]['truth']
    p = oof.iloc[IDX]['preds']
    if t == p:
        return False
    else:
        return True

def return_tp_fp_fn(IDX):
    """
    For OOF at IDX-th position, compare predictions made by our model with the ground truth and
    return three boolean arrays indicating whether given character is true positive/false positive/
    false negative.
    """
    t = oof.iloc[IDX]['truth']
    t = np.array([int(c) for c in t])
    p = oof.iloc[IDX]['preds']
    p = np.array([int(c) for c in p])
    tp = (t == 1)*(p == 1)
    fp = (t == 0)*(p == 1)
    fn = (t == 1)*(p == 0)
    return tp,fp,fn

def return_locations_from_array(arr):
    """
    Convert a boolean array into a list of lists of two ints, describing position of uninterrupted
    sequences of ones.
    
    Example:
    [0,0,0,1,1,1,1,0,0,1,0,0,1,1,1,0] -> [[3, 7], [9, 10], [12, 15]]
    """
    loc = []
    streak_on = False
    for i in range(len(arr)):
        if arr[i] == True and streak_on == True:
            pass
        elif arr[i] == True and streak_on == False:
            streak_on = True
            streak_start = i
        elif arr[i] == False and streak_on == False:
            pass
        elif arr[i] == False and streak_on == True:
            streak_on = False
            loc.append([streak_start, i])
    
    # Close running streak, if applicable
    if streak_on == True:
        loc.append([streak_start, len(arr) + 1])
        
    return loc


# # Summary statistics of errors

# For each feature number, we count the number of true positives, false positives and false negatives.

# In[ ]:


full_tp = 0
full_fn = 0
full_fp = 0
print('  #  FN+FP   TP     FN    FP')
print('----------------------------')
for FN in oof['feature_num'].unique():
    indices = [i for i in range(len(oof)) if oof.iloc[i]['feature_num'] == FN]
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for IDX in indices:
        tp,fp,fn = return_tp_fp_fn(IDX)
        total_tp += np.sum(tp)
        total_fn += np.sum(fn)
        total_fp += np.sum(fp)
    print(f'{FN:3d} {total_fn+total_fp:5d} {total_tp:6d} {total_fn:5d} {total_fp:5d}')
    full_tp += total_tp
    full_fp += total_fp
    full_fn += total_fn
print('----------------------------')
print(f'TOT {full_fn+full_fp:5d} {full_tp:6d} {full_fn:5d} {full_fp:5d}')


# Check F1 of our model

# In[ ]:


precision = full_tp/(full_tp + full_fp)
recall = full_tp/(full_tp + full_fn)
f1 = 2*precision*recall/(precision + recall)
f1


# # Find examples where we make a mistake

# In[ ]:


have_issues = []
for i in range(14300):
    if has_error(i):
        have_issues.append(i)
print(f'First ten problematic: {have_issues[:10]}')
print(f'Number of examples with issues: {len(have_issues)}')


# # Routine that highlights issues

# In[ ]:


def plot_errors(IDX):
    """
    For OOF note at IDX-th position, print case number/feature number header and then
    display the note with correct annotations, as well as true positive / negative.
    """
    patient_df = oof.iloc[IDX]
    print(f"\t\tcase num: {patient_df['case_num']}, feature num: {patient_df['feature_num']}")
    tp,fp,fn = return_tp_fp_fn(IDX)
    tp = return_locations_from_array(tp)
    fp = return_locations_from_array(fp)
    fn = return_locations_from_array(fn)

    ents = []

    for x in tp:
        ents.append({'start':x[0], 'end': x[1], 'label': 'ok'})
    for x in fp:
        ents.append({'start':x[0], 'end': x[1], 'label': 'extra'})
    for x in fn:
        ents.append({'start':x[0], 'end': x[1], 'label': 'missed'})

    ents.sort(key = lambda x: x.get('start'))
    doc = {
        'text' : patient_df["pn_history"],
        "ents" : ents
    }
    colors = {'ok': 'green', 'missed': 'red', 'extra': 'lightblue'} 
    options = {"colors": colors}
    spacy.displacy.render(doc, style="ent", options = options , manual=True, jupyter=True);


# # Study errors made for a particular feature

# In[ ]:


FN = 815
print(features[features['feature_num'] == FN]['feature_text'].values[0])
print('----------------------------------------------------------')
indices = [i for i in have_issues if oof.iloc[i]['feature_num'] == FN]
for IDX in indices:
    plot_errors(IDX)
    print('----------------------------------------------------------')


# In[ ]:




