#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import matplotlib.pyplot as plt

def prepare_dataset(PATH):
    sents = []
    chunks = open(PATH,'r').read().split('\n\n')
    for chunk in chunks:
        lines = chunk.split('\n')
        sent = []
        current_tag = None
        previous_tag = None
        for line in lines:
            if line != '':
                token = line.split('\t')
                previous_tag = current_tag 
                current_tag = token[1]
                sent.append((token[0],token[1]))
        sents.append(sent)
    return sents

def compute_stats(sentences):
    stats = {}
    stats['n_samples'] = len(sentences)
    stats['longest sequence'] = -1
    stats['shortest sequence'] = 99
    for sentence in sentences:
        for item in sentence:
            label = item[1][2:]
            token = item[0]
            if label != '' and label != 'O':
                if label not in stats.keys():
                    stats[label] = 0
                stats[label] += 1
        
        N = len(sentence)
        
        if N > stats['longest sequence']:
            stats['longest sequence'] = N
        
        if N < stats['shortest sequence']:
            stats['shortest sequence'] = N
    return stats
    


# In[ ]:


datasets = {}
datasets_stats = {}
for filename in os.listdir("../input/multilingual-ner-dataset"):
    lang_label = filename.split('.')[0]
    sentences = prepare_dataset('../input/multilingual-ner-dataset/' + filename)
    datasets[lang_label] = sentences
    datasets_stats[lang_label] = compute_stats(sentences)
    
print(datasets.keys())
print(datasets['gre'][0])


# In[ ]:


header = 'lang  | n_samples    |    longest sequence   |   shortest sequence'
print(header)
for key in datasets_stats.keys():
    print(f"{key}   |     {datasets_stats[key]['n_samples']}     |        {datasets_stats[key]['longest sequence']}             |      {datasets_stats[key]['shortest sequence']}  ")


# In[ ]:


from matplotlib.pyplot import figure

for key in datasets_stats.keys():
    labels = [item[0] for item in list(datasets_stats[key].items())[4:]]
    freqs = [item[1] for item in list(datasets_stats[key].items())[4:]]
    figure(figsize=(10, 4), dpi=100)
    plt.barh(labels,freqs)
    plt.title(key)
    plt.show()

