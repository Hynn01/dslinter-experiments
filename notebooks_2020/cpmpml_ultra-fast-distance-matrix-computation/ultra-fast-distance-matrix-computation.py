#!/usr/bin/env python
# coding: utf-8

# Distance matrix between atoms of a single molecule are used in many public kernels to compute features, for instance Coulomb interaction, Van de Walls interaction, and Yukawa interactions, 
# 
# This kernel shows how to speed up the distance matrix computation tremendously compared to the already fast version used in [coulomb_interaction - speed up!](https://www.kaggle.com/rio114/coulomb-interaction-speed-up).  The code from that kernel takes 2 minutes for all molecules.  
# 
# Here we provide a code that runs in 3 seconds, i.e. 40 times faster.
# 
# This speedup is nice, but the code optimization technique used in this kernel is rather generic and can be reused in other context.
# 
# V4 update.  @jmtest has suggested a nice improvement using einssum.  I added his version.  This brings down time to about 1.2 second.  We can do even better using numba, which brings down time to about 0.4 second. This is more than 250 faster than original code.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


structures = pd.read_csv('../input/structures.csv')

structures


# > An efficient way of computing distance matrix is provided in the [coulomb_interaction - speed up!](https://www.kaggle.com/rio114/coulomb-interaction-speed-up) kernel from Ryoji Nomura.  Please drop me a comment if the code is from another author and I'll correct the attribution.

# In[ ]:


structures_idx = structures.set_index('molecule_name')

def get_dist_matrix(df_structures_idx, molecule):
    df_temp = df_structures_idx.loc[molecule]
    locs = df_temp[['x','y','z']].values
    num_atoms = len(locs)
    loc_tile = np.tile(locs.T, (num_atoms,1,1))
    dist_mat = np.sqrt((loc_tile - loc_tile.T)**2).sum(axis=1)
    return dist_mat


# Let's see how much time this takes for all molecules

# In[ ]:


for molecule in tqdm_notebook(structures.molecule_name.unique()):
    get_dist_matrix(structures_idx, molecule)


# This takes 2 minutes.
# 
# Can we do better?
# 
# A way to accelerate code is to replace pandas operations by numpy operations.  The code above already does this in a very clever way by using the numpy tile function and vectorized distance computation.
# 
# Still, it creates a pandas dataframe for each molecule, then converts it into numpy array.  This is slow.  Also, filtering by the molecule name to extract the structure information for a given molecule is expensive.  We can speed up both by using numpy arrays up front.  Let's create the array we need.

# In[ ]:


xyz = structures[['x','y','z']].values


# This will remove the need to first create a data frame then convert it into a numpy array.  This is nice, but we need a way to extract the part relevant to a given molecule from this single numpy array.  
# 
# The original structures data frame is sorted by molecule, meaning that information for a given molecule is in a consecutive set of rows. We can then precompute the indices of each molecule segment.  
# 
# We first compute the number of rows for each molecule, then using cumsum to get indices for molecule changes.  We then store this as a numpy array where we add 0 as the first value.  The molecule names can be retrieved as the index of the series.

# In[ ]:


ss = structures.groupby('molecule_name').size()
ss = ss.cumsum()
ss


# In[ ]:


ssx = np.zeros(len(ss) + 1, 'int')
ssx[1:] = ss
ssx


# Then, to retrieve information for the i-th molecule, we slice the xyz array using the indices we just compuited.  For instance, to get the information for the 20th molecule we do:

# In[ ]:


molecule_id = 20
print(ss.index[molecule_id])
start_molecule = ssx[molecule_id]
end_molecule = ssx[molecule_id+1]
xyz[start_molecule:end_molecule]


# We can compare with the information we get from the original pandas dataframe

# In[ ]:


structures_idx.loc['dsgdb9nsd_000022'][['x', 'y', 'z']].values


# Looks good.
# 
# We can now rewrite our function using our arrays.
# 

# In[ ]:


def get_fast_dist_matrix(xyz, ssx, molecule_id):
    start_molecule, end_molecule = ssx[molecule_id], ssx[molecule_id+1]
    locs = xyz[start_molecule:end_molecule]    
    num_atoms = end_molecule - start_molecule
    loc_tile = np.tile(locs.T, (num_atoms,1,1))
    dist_mat = np.sqrt((loc_tile - loc_tile.T)**2).sum(axis=1)
    return dist_mat


# Let's check we get the same result with both techniques.  We use the smallest molecule for the sake of simplicity. It is water.

# In[ ]:


molecule_id = 2
molecule = ss.index[molecule_id]
print(molecule)
get_fast_dist_matrix(xyz, ssx, molecule_id)


# In[ ]:


get_dist_matrix(structures_idx, molecule)


# Looks good.
# 
# We can now benchmark our version.

# In[ ]:


for molecule_id in tqdm_notebook(range(structures.molecule_name.nunique())):
    get_fast_dist_matrix(xyz, ssx, molecule_id)


# We are down to 3 seconds from 2 minutes!
# 
# A nice improvement was proposed by @jmtest in the comments.  In order to benchmark our code and his, we need to use a more precise way than tqdm.  Let's use the `%time` magic. It requires us to wrap the code in a function

# In[ ]:


def ultra_fast_dist_matrices(xyz, ssx):
    for molecule_id in range(structures.molecule_name.nunique()):
        get_fast_dist_matrix(xyz, ssx, molecule_id)


# In[ ]:


get_ipython().run_line_magic('time', 'ultra_fast_dist_matrices(xyz, ssx)')


# Let's do the same for @jmtest code

# In[ ]:


def sofast_dist(xyz, ssx, molecule_id):
    start_molecule, end_molecule = ssx[molecule_id], ssx[molecule_id+1]
    locs = xyz[start_molecule:end_molecule]     
    d=locs[:,None,:]-locs
    return np.sqrt(np.einsum('ijk,ijk->ij',d,d))

def sofast_dist_matrices(xyz, ssx):
    for molecule_id in range(structures.molecule_name.nunique()):
        sofast_dist(xyz, ssx, molecule_id)


# In[ ]:


get_ipython().run_line_magic('time', 'sofast_dist_matrices(xyz, ssx)')


# almost a 3x speedup.  
# 
# Can we do better?  The ultimate weapon to accelerate vectorized operaitons in Python is to use a compiler.  Numba is my favorite because we stay in Python.  Alternatives are Cython compiler, or calling C/C++ code.  In my experience Numba provides almost the same speed as compiled C code if there are no indirect array access, which is the case here.  
# 
# Numba works by annotating code.

# In[ ]:


from numba import jit
from math import sqrt

@jit
def numba_dist_matrix(xyz, ssx, molecule_id):
    start_molecule, end_molecule = ssx[molecule_id], ssx[molecule_id+1]
    locs = xyz[start_molecule:end_molecule]     
   # return locs
    num_atoms = end_molecule - start_molecule
    dmat = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            d = sqrt((locs[i,0] - locs[j,0])**2 + (locs[i,1] - locs[j,1])**2 + (locs[i,2] - locs[j,2])**2)
            dmat[i,j] = d
            dmat[j,i] = d
    return dmat

def numba_dist_matrices(xyz, ssx):
    for molecule_id in range(structures.molecule_name.nunique()):
        numba_dist_matrix(xyz, ssx, molecule_id)


# Let's make sure our code is correct by comparing with the previous one.

# In[ ]:


molecule_id = 2
molecule = ss.index[molecule_id]
print(molecule)
numba_dist_matrix(xyz, ssx, molecule_id)


# In[ ]:


sofast_dist(xyz, ssx, molecule_id)


# Looks good

# In[ ]:


get_ipython().run_line_magic('time', 'numba_dist_matrices(xyz, ssx)')


# A further 3x speedup!  We are now about 250 times faster than the original code...
# 

# 

# It was suggested in comments that scipy distance would be faster.  Let's have a look.  scipy requires two functions to output a square matrix.

# In[ ]:


from scipy.spatial.distance import pdist, squareform

def scipy_dist_matrix(xyz, ssx, molecule_id):
    start_molecule, end_molecule = ssx[molecule_id], ssx[molecule_id+1]
    locs = xyz[start_molecule:end_molecule]     
    cmat = pdist(locs)
    dmat = squareform(cmat, force='tomatrix')
    return dmat


# 

# Let's check that we get the same result.

# In[ ]:


scipy_dist_matrix(xyz, ssx, molecule_id)


# We can now benchmark.

# In[ ]:


def scipy_dist_matrices(xyz, ssx):
    for molecule_id in range(structures.molecule_name.nunique()):
        scipy_dist_matrix(xyz, ssx, molecule_id)


# In[ ]:


get_ipython().run_line_magic('time', 'scipy_dist_matrices(xyz, ssx)')


# Results are disappointing.
# 
# If this kernel gets enough votes then I'll share another kernel with ultra fast way to compute angles and dighedral angles with numpy ;)

# @ilyivanchenko points out that there is an issue in the get_dist_matrix thagt I borrowed from [coulomb_interaction - speed up!](https://www.kaggle.com/rio114/coulomb-interaction-speed-up).

# In[ ]:


epsilon = 1e-5

def get_dist_matrix_assert(df_structures_idx, molecule):
    df_temp = df_structures_idx.loc[molecule]
    locs = df_temp[['x','y','z']].values
    num_atoms = len(locs)
    loc_tile = np.tile(locs.T, (num_atoms,1,1))
    dist_mat = np.sqrt((loc_tile - loc_tile.T)**2).sum(axis=1)
    assert np.abs(dist_mat[0,1] - np.linalg.norm(locs[0] - locs[1])) < epsilon
    return dist_mat

for molecule in tqdm_notebook(structures.molecule_name.unique()[660:]):
    try:
        get_dist_matrix_assert(structures_idx, molecule)
    except: 
        print('assertion error on', molecule)
        break
        


# Let's see if we have the issue in our code.

# In[ ]:


@jit
def numba_dist_matrix_ssert(xyz, ssx, molecule_id):
    start_molecule, end_molecule = ssx[molecule_id], ssx[molecule_id+1]
    locs = xyz[start_molecule:end_molecule]     
   # return locs
    num_atoms = end_molecule - start_molecule
    dmat = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            d = sqrt((locs[i,0] - locs[j,0])**2 + (locs[i,1] - locs[j,1])**2 + (locs[i,2] - locs[j,2])**2)
            dmat[i,j] = d
            dmat[j,i] = d
    assert np.abs(dmat[0,1] - np.linalg.norm(locs[0] - locs[1])) < epsilon
    return dmat

for molecule_id in range(structures.molecule_name.nunique()):
    numba_dist_matrix_ssert(xyz, ssx, molecule_id)


# The code runs fine.

# In[ ]:




