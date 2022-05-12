#!/usr/bin/env python
# coding: utf-8

# As of now, despite that only a few days have passed since this competition started, we know a lot about the structure of dataset. We know about ID's, splits, etc. etc.
# I tried to put everything we knew so far together and generate a synthetic `train.csv` dataset from scratch as similar as possible to Kaggle's `train.csv`.
# I then ran some EDA on it and also trained some of the public kernels on this synthetic dataset and compared the CV score to theirs. 
# 
# Hope you enjoy this kernel. Don't forget to upvote and share your opinions.
# 
# 
# 
# First, we generate the dataset from 512 different classification datasets by using sklearn's `make_calssification`. 

# In[12]:


from sklearn.datasets import make_classification 
import numpy as np
import pandas as pd
np.random.seed(2019)

# generate dataset 
train, target = make_classification(512, 255, n_informative=np.random.randint(33, 47), n_redundant=0, flip_y=0.08)
train = np.hstack((train, np.ones((len(train), 1))*0))

for i in range(1, 512):
    X, y = make_classification(512, 255, n_informative=np.random.randint(33, 47), n_redundant=0, flip_y=0.08)
    X = np.hstack((X, np.ones((len(X), 1))*i))
    train = np.vstack((train, X))
    target = np.concatenate((target, y))
    


# Silly column names abound;
# 
# let's add some names to our features. It must be silly and cryptic so let's have this format of `kind-color-animal-goal`. 
# 
# Note that if you want it to look more cryptic you can rename some of them and replace `goal` part with `animal` part or `animal` part with `color` part so that the competitors spend some time on that too. For example make something like `slimy-seashell-cassowary-goose` (this column actually exists in Kaggle's train.csv).
# 
# You can also add some probabilities in `np.random.choice()` to make them look as such that there is pattern.

# In[13]:


col_names = []
kind_arr = ['muggy', 'dorky', 'slimy', 'snazzy', 'frumpy', 'stealthy', 'chummy', 'hazy', 'nerdy', 'leaky', 'ugly', 'shaggy', 'flaky','squirrely', 'freaky', 'lousy', 'bluesy', 'baggy', 'greasy',
       'cranky', 'snippy', 'flabby', 'goopy', 'homey', 'homely', 'hasty','blurry', 'snoopy', 'stinky', 'bumpy', 'slaphappy', 'messy','geeky', 'crabby', 'beady', 'pasty', 'snappy', 'breezy', 'sunny',
       'cheeky', 'wiggy', 'flimsy', 'lanky', 'scanty', 'grumpy', 'chewy','crappy', 'clammy', 'tasty', 'thirsty', 'gloppy', 'gamy', 'hilly','woozy', 'squeaky', 'lovely', 'paltry', 'smelly', 'pokey','skanky', 'zippy', 'sleazy', 'queasy', 'foggy', 'wheezy', 'droopy',
       'cozy', 'skinny', 'seedy', 'stuffy', 'jumpy', 'trippy', 'woolly','gimpy', 'randy', 'silly', 'craggy', 'skimpy', 'nippy', 'whiny','boozy', 'pretty', 'sickly', 'shabby', 'surly']
color_arr = ['smalt', 'peach', 'seashell', 'harlequin', 'beige', 'cream','emerald', 'indigo', 'amaranth', 'tangerine', 'silver','chocolate', 'tan', 'plum', 'rose', 'copper', 'scarlet','cinnamon', 'cardinal', 'auburn', 'sepia', 'brass', 'eggplant',
       'ruby', 'blue', 'wisteria', 'maroon', 'tomato', 'mauve', 'pumpkin','teal', 'goldenrod', 'aquamarine', 'gamboge', 'persimmon','mustard', 'red', 'magnolia', 'chestnut', 'champagne', 'flax',
       'viridian', 'amber', 'zucchini', 'myrtle', 'lemon', 'pear',
       'xanthic', 'turquoise', 'lilac', 'amethyst', 'lime', 'pink',
       'periwinkle', 'crimson', 'burgundy', 'purple', 'rust', 'cerise',
       'khaki', 'malachite', 'violet', 'sangria', 'magenta', 'russet',
       'apricot', 'cobalt', 'platinum', 'denim', 'yellow', 'sapphire',
       'bronze', 'green', 'thistle', 'buff', 'razzmatazz', 'charcoal',
       'ultramarine', 'puce', 'carmine', 'gold', 'asparagus', 'ivory',
       'orange', 'vermilion', 'chartreuse', 'heliotrope', 'azure', 'grey',
       'jade', 'olive', 'coral', 'brown', 'cinnabar', 'lavender', 'aqua',
       'firebrick', 'corn', 'bistre', 'cyan', 'ochre', 'dandelion',
       'white']
animal_arr = ['axolotl', 'sheepdog', 'cassowary', 'chicken', 'mau', 'pinscher',
       'tarantula', 'cuttlefish', 'wolfhound', 'lizard', 'chihuahua',
       'indri', 'beetle', 'sheep', 'angelfish', 'penguin', 'wallaby',
       'oriole', 'hound', 'bonobo', 'dogfish', 'vole', 'coral', 'fowl',
       'bombay', 'bulldog', 'oyster', 'blue', 'armadillo', 'ragdoll',
       'wolverine', 'moorhen', 'otter', 'bat', 'affenpinscher', 'rat',
       'caterpillar', 'newt', 'collie', 'weasel', 'guppy', 'bullfrog',
       'alligator', 'sloth', 'moth', 'kudu', 'wasp', 'okapi', 'quoll',
       'shrew', 'walrus', 'schnauzer', 'termite', 'dragonfly', 'kakapo',
       'quetzal', 'capuchin', 'eel', 'iguana', 'zonkey', 'fousek',
       'javanese', 'leopard', 'gorilla', 'malamute', 'birman', 'donkey',
       'lionfish', 'llama', 'emu', 'koala', 'saola', 'neanderthal',
       'horse', 'mammoth', 'duck', 'peccary', 'hippopotamus',
       'grasshopper', 'dolphin', 'gharial', 'frog', 'ostrich', 'akbash',
       'bison', 'hyrax', 'capybara', 'earwig', 'cuscus', 'chinook',
       'jackal', 'hornet', 'monkey', 'bordeaux', 'reindeer', 'squid',
       'maltese', 'buffalo', 'hedgehog', 'octopus', 'swan', 'husky',
       'zebu', 'olm', 'retriever', 'lemur', 'dhole', 'rabbit', 'loon',
       'cat', 'millipede', 'flamingo', 'opossum', 'turtle', 'eagle',
       'eleuth', 'binturong', 'uguisu', 'whippet', 'tiger', 'lobster',
       'macaque', 'scorpion', 'goat', 'tapir', 'audemer', 'fox', 'molly',
       'discus', 'gopher', 'gerbil', 'civet', 'gecko', 'dog', 'squirt',
       'insect', 'tarsier', 'whale', 'paradise', 'deer', 'urchin',
       'serval', 'rhinoceros', 'numbat', 'frigatebird', 'catfish', 'kiwi',
       'bee', 'seahorse', 'beagle', 'tzu', 'dodo', 'mayfly', 'impala',
       'dachshund', 'budgerigar', 'moose', 'labradoodle', 'spider',
       'flounder', 'woodpecker', 'bobcat', 'corgi', 'buzzard', 'clam',
       'hamster', 'bandicoot', 'mandrill', 'lemming', 'snail', 'havanese',
       'hyena', 'monster']
goal_arr = ['pembus', 'ordinal', 'goose', 'distraction', 'golden', 'entropy',
       'unsorted', 'sorted', 'important', 'fimbus', 'grandmaster',
       'sumble', 'noise', 'discard', 'dummy', 'fepid', 'contributor',
       'learn', 'dataset', 'master', 'expert', 'kernel', 'hint', 'novice',
       'gaussian']
for _ in range(255):
    new_col_name = np.random.choice(kind_arr) + '-' + np.random.choice(color_arr) + '-' + np.random.choice(animal_arr) + '-' + np.random.choice(goal_arr)
    col_names.append(new_col_name)
col_names.append('wheezy-copper-turtle-magic')


# build the dataframe
train = pd.DataFrame(train, columns=col_names)
train['target'] = target


# Add mysterious IDs. 
# 
# As Yirun found in https://www.kaggle.com/c/instant-gratification/discussion/92634#latest-537452, md5 hashing seems cool.

# In[14]:


import hashlib

def generate_hashed_id(inp):
    return hashlib.md5(bytes(f'{inp}train', encoding='utf-8')).hexdigest()
    
train['id'] = train.index
train['id'] = train['id'].apply(generate_hashed_id)

# re-arrange columns
cols = [c for c in train.columns if c not in ['id', 'target']]
train = train[['id', 'target']+cols]


# Now that the dataset is ready let's take a look at it from different aspects to see if it's similar enough?
# First we're gonna look at the head of dataframe.

# In[ ]:


train.head()


# We have super balanced target classes in Kaggle's train.csv, Does sklearn generates balanced target classes? i.e. do we have super balanced target classes? Yes!

# In[ ]:


train.target.hist()


# How about the histograms? Are they similar to Kaggle's train.csv? Compare the shapes of this dataset to Bojan's EDA at https://www.kaggle.com/tunguz/instant-eda

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(train[train.columns[10]])
plt.figure()
sns.distplot(train[train.columns[210]])


# What about the correlation plot? (it was taking a long time to run, so I restricted it to only the first 50 columns) Compare the results with Allunia's EDA at https://www.kaggle.com/allunia/instant-gratification-some-eda-to-go

# In[ ]:


train_corr = train.iloc[:,:50].drop(["target", 'id'], axis=1).corr()
plt.figure(figsize=(10,10))
sns.heatmap(train_corr, vmin=-0.016, vmax=0.016, cmap="RdYlBu_r");


# Last but not least, let's filter `train.csv` by the magical feature `wheezy-copper-turtle-magic==0` and see what happens.

# In[ ]:


columns = [x for x in train.columns if x not in ["target", 'id', 'wheezy-copper-turtle-magic']]
train2 = train.loc[train['wheezy-copper-turtle-magic']==0, columns]
plt.figure(figsize=(6,6))
plt.plot(train2.std())


# Oh cool! we have those useful/useless columns in here too. Surprisingly enough, useless cols have std about 1.0 and useful columns have an STD of about 3.7. Thanks to Chris for his discussion https://www.kaggle.com/c/instant-gratification/discussion/92930#latest-538458
# 
# what about the distributions now? do they look normal after filtering?

# In[ ]:


sns.distplot(train2[train.columns[10]])
plt.figure()
sns.distplot(train2[train.columns[210]])


# Yes they do. 

# EDA shows this dataset is very similar to competition's data. 
# Now it's time to build some models on top of this dataset;
# 
# Below is the code from Chris' kernel https://www.kaggle.com/cdeotte/logistic-regression-0-800 which has CV score of 0.52994. 
# 
# Let's train it on this dataset and compare the CV score.

# In[15]:


import numpy as np, pandas as pd, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

cols = [c for c in train.columns if c not in ['id', 'target']]
oof = np.zeros(len(train))
skf = StratifiedKFold(n_splits=5, random_state=42)
   
for train_index, test_index in skf.split(train.iloc[:,1:-1], train['target']):
    clf = LogisticRegression(solver='liblinear',penalty='l2',C=1.0)
    clf.fit(train.loc[train_index][cols],train.loc[train_index]['target'])
    oof[test_index] = clf.predict_proba(train.loc[test_index][cols])[:,1]
    
auc = roc_auc_score(train['target'],oof)
print('LR without interactions scores CV =',round(auc,5))


# Below is the code from Chris' kernel https://www.kaggle.com/cdeotte/logistic-regression-0-800 which has CV score of 0.80549.
# 
# Let's train it on this dataset and compare the CV score.

# In[5]:


# INITIALIZE VARIABLES
cols.remove('wheezy-copper-turtle-magic')
interactions = np.zeros((512,255))
oof = np.zeros(len(train))

# BUILD 512 SEPARATE MODELS
for i in range(512):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index
    train2.reset_index(drop=True,inplace=True)
    
    skf = StratifiedKFold(n_splits=25, random_state=42)
    for train_index, test_index in skf.split(train2.iloc[:,1:-1], train2['target']):
        # LOGISTIC REGRESSION MODEL
        clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.05)
        clf.fit(train2.loc[train_index][cols],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train2.loc[test_index][cols])[:,1]

        
# PRINT CV AUC
auc = roc_auc_score(train['target'],oof)
print('LR with interactions scores CV =',round(auc,5))


# Below is the code from Chris' kernel https://www.kaggle.com/cdeotte/support-vector-machine-0-925 which has CV score of 0.9262.
# 
# Let's train it on this dataset and compare the CV score.

# In[ ]:


# LOAD LIBRARIES
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
# INITIALIZE VARIABLES
oof = np.zeros(len(train))
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

# BUILD 512 SEPARATE NON-LINEAR MODELS
for i in range(512):
    
    # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index
    train2.reset_index(drop=True,inplace=True)
    
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
        
    # STRATIFIED K FOLD (Using splits=25 scores 0.002 better but is slower)
    skf = StratifiedKFold(n_splits=11, random_state=42)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        # MODEL WITH SUPPORT VECTOR MACHINE
        clf = SVC(probability=True,kernel='poly',degree=4,gamma='auto')
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        
    #if i%10==0: print(i)
        
# PRINT VALIDATION CV AUC
auc = roc_auc_score(train['target'],oof)
print('CV score =',round(auc,5))


# ## Final notes
# * It is very likely that Kaggle used a similar way to generate this competition's dataset. Yet nothing's for sure and they might have tricked a few parts of it so keep searching.
# * I believe if you play with `flip_y` and `n_informative` in `make_classification` you can get closer CV scores than mine to those mentioned above.
# * I would like to express my gratitude to Chris, Bojan, and other kagglers who generously share their findings with us
# * dont forget to upvote all the kernels/discussions linked here 
# * please let me know if I used parts of your work here and forgot to link
