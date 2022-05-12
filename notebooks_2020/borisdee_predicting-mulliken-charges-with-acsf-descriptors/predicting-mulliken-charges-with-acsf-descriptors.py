#!/usr/bin/env python
# coding: utf-8

# # Introducing Atom-Centered Symmetry Functions: Application to the prediction of Mulliken charges
# 
# Greetings everyone!
# 
# One thing that you probably noticed in the [Predicting Molecular Properties](https://www.kaggle.com/c/champs-scalar-coupling/overview) challenge is that the Mulliken charges are only available for the training set. And so are the dipole moments, magnetic shield tensor, potential energies, etc.
# 
# Therefore, several attempts have been made to try and approximate these properties for the test set (see for instance this [kernel](https://www.kaggle.com/asauve/estimation-of-mulliken-charges-with-open-babel) by Alexandre).
# 
# But what if we had a way to machine learn these properties and predict them for all the atoms/molecules of the test set? This would allow us to subsequently use them as input features for the prediction of the scalar coupling.
# 
# It is actually possible to machine learn these properties using only the structural information contained in the structures.csv file. We do that using **descriptors**. You may have already heard about these descriptors (maybe because I'm mentioning them in almost all of my posts...) and it's time we took a look at them :)
# 
# For this kernel I choose to focus on the Mulliken charges, but of course this can be applied to any of the properties mentioned above, however with some subtleties that I will explain.
# 
# # 1. What is a descriptor?
# The problem that we are facing is **how to appropriately represent an atomic environment**? 
# 
# In the challenge, we are provided with the cartesian positions of each atom in the molecules. While the cartesian system may seem like a simple and unequivocal descriptor of atomic configurations, it actually suffers a major drawback: the list of coordinates is ordered **arbitrarily** and two structures might be mapped to each other by a **rotation**, **reflection**, or **translation** so that two different lists of atomic coordinates can, in fact, represent the same or very similar structures.
# 
# Therefore, **a good representation is invariant with respect to permutational, rotational, reflectional, and translational symmetries, while retaining the faithfulness of the cartesian representation**.
# 
# This is what a descriptor does. It is also called the **fingerprint** of a molecule.
# 
# During the last decade, a number of different descriptors have been established (and every month a new descriptor pops out, claiming it is superior to all the previous ones...) among which:
# - The symmetry functions by Behler and Parrinello: [*Phys. Rev. Lett.* **98**, 146401 (2007)](https://doi.org/10.1103/PhysRevLett.98.146401).
# - The bispectrum by Bartok *et al.*: [*Phys. Rev. Lett.* **104**, 136403 (2010)](https://doi.org/10.1103/PhysRevLett.104.136403).
# - The Coulomb matrix by Rupp *et al.*: [*Phys. Rev. Lett.* **108**, 058301 (2012)](https://doi.org/10.1103/PhysRevLett.108.058301).
# - And others!
# 
# The descriptors can be **global** or **local**, depending on whether they represent the entire molecule (global) or the environment around each atom (local). The symmetry functions, bispectrum, and Coulomb matrix are all local descriptors: one descriptor for each atom in the molecule. Depending on the quantity that we want to predict, we will use global or local descriptors. For instance, the Mulliken charge is a local property (since there is one charge for each atom), so that it's better to use a local descriptor. On the contrary, the potential energy of the molecule is a global property, so we might instead use a global descriptor.
# 
# # 2. Atom-Centered Symmetry Functions (ACSF)
# ACSF are local descriptors that are rather easy to understand and are very powerful because they contain a lot of information on the chemical environment around each atom. They are based on a function called the **cutoff function**.
# 
# ## 2.1. Understanding the cutoff function 
# The cutoff function is expressed as follows:
# 
# $$
# f_c(R_{ij}) = 
# \left\{
#     \begin{array}{ll}
#     0.5 \left[ \cos\left( \pi\frac{R_{ij}}{R_c} \right) + 1 \right] & {\rm for} \quad R_{ij} \leq R_c \\
#     0  & {\rm for} \quad R_{ij} > R_c \\
#     \end{array}
# \right.
# $$
# 
# where $R_{ij}$ is the distance between atoms $i$ and $j$, and if this distance is greater than a **cutoff radius** $R_c$, then the cutoff function becomes zero.
# 
# If maths is not your cup of tea, this function might look barbaric. Let's try to understand it by plotting it for different values of $R_c$:

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Definition of the cutoff function
def fc(Rij, Rc):
    y_1 = 0.5*(np.cos(np.pi*Rij[Rij<=Rc]/Rc)+1)
    y_2 = Rij[Rij>Rc]*0
    y = np.concatenate((y_1,y_2))
    return y

# Define x
x = np.arange(0, 11, 0.01)

# Plot the function with different cutoff radii
Rc_range = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

fig = plt.figure(figsize=(10,8))

for Rc in Rc_range:
    plt.plot(x, fc(x,Rc), label=f'Rc={Rc}')

plt.axis([0, 11, 0, 1.1])
plt.xticks(range(11))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('Distance from center of atom')
plt.ylabel('Cutoff Function')
plt.legend()
    
plt.show()


# So how does that work?
# 
# Let's consider one atom of a given molecule. When the distance from the center of this atom equals zero, it means that we are located on this atom. We then see that the cutoff function is maximal and equals $1$.
# 
# Then, we notice that the farther away we are going from our atom, the more the cutoff function decreases until it reaches zero when the distance reaches the cutoff radius that we have set.
# 
# > **The cutoff function can be seen as describing the importance of the environment of an atom at a given distance.**
# 
# In other words, close to the atom, the cutoff function is maximal, meaning that we consider everything close to the atom as of "high importance". 
# 
# On the contrary, far from the atom, the cutoff function has a low value, meaning that we consider everything there as of "low importance".
# 
# All ACSF are built on top of this function, which acts as a **weight function** that favors what is close to the atom and penalizes what is far. There are two types of ACSF: **radial** and **angular** functions, which describe the radial (distances) and angular (angles) distribution of the neighboring atoms, respectively.
# 
# ## 2.2. The most basic two-body radial ACSF: $G^1_i$
# $G^1_i$ is the most basic radial function that describes the environment of an atom $i$. It's just the sum of the plain cutoff functions with respect to each neighboring atom $j$:
# $$ G_i^1 = \sum\limits_{j=1}^{N_{\rm atom}} f_c(R_{ij})$$
# 
# It's called a **two-body** ACSF because it involves pairs of atoms, as opposed to the **three-body** ACSF that involves triplets of atoms. There is one $G^1$ for each atom in the molecule, and it's a real number.
# 
# > **The physical interpretation of $G^1_i$ is the coordination number of atom $i$ within the cutoff radius.**
# 
# As an example, let's consider a simple molecule: methanol CH$_3$OH.

# In[ ]:


import ase.visualize
from ase.build import molecule

# Create the methanol molecule
methanol = molecule('CH3OH')

ase.visualize.view(methanol, viewer="x3d")


# To find the ideal cutoff radius $R_c$, let's check what is the maximum distance between two atoms:

# In[ ]:


max(methanol.get_all_distances().flatten())


# We can therefore use $R_c=3.0$ and we'll be sure to include all the atoms in the neighborhood of each atom.

# In[ ]:


Rc = 3.0


# Now let's calculate $G^1_C$ for the carbon atom. First let's identify which one is the carbon atom:

# In[ ]:


methanol.get_chemical_symbols()


# It's the first on the list! Now we get the distances between this atom and every other atoms. To do that, we use the `get_all_distances()` method, which returns the distances of all of the atoms with every other atom:

# In[ ]:


all_dist = methanol.get_all_distances()
print('This is the distance matrix:')
print(all_dist)

dist_from_C = all_dist[0]
print('')
print('Distances from carbon atom to every other atoms:\n', dist_from_C)


# We're good to go! Let's calculate $G^1_C$:

# In[ ]:


G1_C = fc(dist_from_C, Rc).sum()
print('G1 for the carbon atom:', G1_C)


# Let's calculate the $G^1$ array, which contains the $G^1$ of all atoms in the molecule:

# In[ ]:


# Number of atoms in the molecules
natom = len(methanol.get_chemical_symbols())

# Definition of a vectorized cutoff function
def fc_vect(Rij, Rc):
    return np.where(Rij <= Rc, 0.5 * (np.cos(np.pi * Rij/Rc)+1), 0).sum(1)

# Calculate G1
G1 = fc_vect(all_dist, Rc)

print(G1)


# The result is one number for each atom in the molecule, with larger values indicating more atoms in the vicinity. Notice how the hydrogen atom located on the oxygen atom has the lowest $G^1=2.26953861$ because it is the atom that has the lowest number of atoms in the close vicinity.
# 
# > **Remember that the cutoff function penalizes atoms that are far and favors atoms that are close!**
# 
# The problem with the $G^1$ ACSF is that it has to be calculated with different values of the cutoff radius $R_c$ in order to be truly interesting but at the same time **cannot be used with too low values of $R_c$** (because of numerical issues that are out of the scope of this kernel). Therefore, it describes poorly the environment very close to each atom. This problem can be solved by employing instead the radial symmetry function $G^2$.
# 
# ## 2.3. Going further: the radial two-body ACSF $G_i^2$
# The $G^2$ ACSF is an improvement over $G^1$. It is expressed as:
# $$ G_i^2 = \sum\limits_{j=1}^{N_{\rm atom}}\exp\left[ -\eta\ (R_{ij}-R_s)^2 \right] \times f_c(R_{ij})$$ 
# 
# It is better than $G^1$ because it doesn't need to be evaluated at different cutoff radii. The cutoff radius can be kept at a large value for all functions, while the radial resolution is now determined by the parameter $\eta$. As for the parameter $R_s$, it can be used to improve the description of specific interatomic distances.
# 
# > **The physical interpretation of $G^2_i$ is that it quantifies atomic pair interactions at a distance $R_s$ away from atom $i$.**
# 
# In practice, $G^2$ is computed for several combinations of $\eta$ and $R_s$, therefore each atom will have a series of $G^2$ values. Contrary to $G^1$, the $G^2$ ACSF allows the region close to the reference atom to be covered. Consequently they are generally superior to functions of type $G^1$. Let's calculate $G^2$ for each atom of our methanol molecule:

# In[ ]:


# Define the G2 function
def get_G2(Rij, eta, Rs):
     return np.exp(-eta*(Rij-Rs)**2) * fc(Rij, Rc)
    
# Set a list of six eta/Rs tuples
p = [(0.4, 0.2),(0.4, 0.5),(0.4, 1.0),(0.5, 2.0),(0.5, 3.0),(0.5, 4.0)]

# Compute the six G2 corresponding to the six eta/Rs tuples
G2 = np.zeros((natom, len(p)))
for i in range(natom):
    for j, (eta, Rs) in enumerate(p):
        G2[i,j] =  get_G2(all_dist[i], eta, Rs).sum()
    
print(G2)


# The result is a **matrix**: one row for each atom, one column for each pair or parameters $\eta$ and $R_s$.
# 
# While a set of $G^1$ and $G^2$ functions can describe accurately the radial distribution of the neighboring atoms, it is not possible to distinguish different angular distributions. Therefore, additional atom-centered **angular** symmetry functions were also introduced. They are called $G^4$ and $G^5$ (for those who follow, yes there's also a $G^3$ but it's a radial function and I'm not using it in this kernel).
# 
# As with the radial functions, the angular functions yield a single real value independent of the actual number of neighbors. Their mathematical expressions is not very useful (very barbaric!) so I won't write them. All you need to know is that:
# 
# > **The physical interpretation of $G^4_{ij}$ and $G^5_{ij}$ is that they quantify the angles between all triplets of atoms in the molecule.**
# 
# OK so what now? Now we are going to use the ACSF to predict the Mulliken charges of the molecules in the test set **with a frightening MAE!**
# 
# # 3. Taking care of business: the Mulliken charges
# Why use ACSF to predict the Mulliken charges? Let's take a look at Wikipedia:
# 
# > Mulliken charges arise from the Mulliken population analysis and provide a means of estimating **partial atomic charges**.
# 
# The important words are in bold: partial atomic charges. Mulliken charges are **local** atomic properties, as opposed for instance to the potential energy, which is a **global** molecular property. So that's why the ACSF are perfectly designed to predict the Mulliken charges, though we could also have used the SOAP descriptors (but this will be the topic of another kernel! Don't be hasty, young padawan).
# 
# > **We use a local descriptor to predict a local atomic property.**
# 
# It's as simple as that.
# 
# So what I've done is that I've spent 2 weeks building **a set of 245 symmetry functions for each atom in each molecule!** I have used $G^1$ and $G^2$ for the radial part, and $G^4$ for the angular part. Below are the parameters that I've used in case you wish to build them:
# ```
# For all ACSF functions: rcut = 10.0
# 
# G2 - eta/Rs couples:
# g2_params=[[1, 2], [0.1, 2], [0.01, 2],
#            [1, 6], [0.1, 6], [0.01, 6]]
# 
# G4 - eta/ksi/lambda triplets:
# g4_params=[[1, 4,  1], [0.1, 4,  1], [0.01, 4,  1], 
#            [1, 4, -1], [0.1, 4, -1], [0.01, 4, -1]]
# 
# ```
# 
# Note that **between 50 and 100 symmetry functions** per atom were proven to be enough to accurately describe atomic local environments. But better be safe than sorry, I have added a few more.
# 
# These ACSF have been merged into the `structures.csv` file of the challenge, resulting in a 5.5GB file that we're keeping private for now until we're through with playing with these little things.
# 
# Now we're ready to train our model, which will be an `ExtraTreesRegressor`. The accuracy metrics will be the **mean absolute error**, which is OK for the Mulliken charges, I guess. I like it because it has the same units as what we're calculating.
# 
# Let's go!

# In[ ]:


import pandas as pd
import borisdee_kaggle_functions as bd

# Load all relevant files
raw_struct = pd.read_csv('../input/acsf-up-to-g4/structures_with_g4.csv')
raw_charges = pd.read_csv('../input/champs-scalar-coupling/mulliken_charges.csv')
raw_test = pd.read_csv('../input/champs-scalar-coupling/test.csv')

# We need to free a bit of memory for Kaggle servers. Kudos to artgor for this function.
raw_struct = bd.reduce_mem_usage(raw_struct)
raw_charges = bd.reduce_mem_usage(raw_charges)
raw_test = bd.reduce_mem_usage(raw_test)


# In[ ]:


# Create train and test sets from the structure file containing ACSF
raw_train = raw_struct[raw_struct['molecule_name'].isin(raw_charges['molecule_name'].unique())]
raw_train.reset_index(drop=True, inplace=True)
test = raw_struct[raw_struct['molecule_name'].isin(raw_test['molecule_name'].unique())]
display(raw_train.head(), test.head())


# In[ ]:


# Drop useless columns
columns_to_drop = ['Unnamed: 0', 'molecule_name', 'atom_index', 'atom', 'x', 'y', 'z']
raw_train = raw_train.drop(columns_to_drop, axis=1)
test = test.drop(columns_to_drop, axis=1)

# Add Mulliken charges to training set
raw_train['mulliken_charge'] = raw_charges['mulliken_charge']

# Create train and cv sets
train = raw_train.sample(frac=0.80, random_state=2019)
cv = raw_train.drop(train.index)

print('Shape of train set:', train.shape)
print('Shape of cv set:', cv.shape)


# Before running the model, let's ensure that the cv and train sets have similar statistics:

# In[ ]:


# Hypothesis Testing
df1 = pd.DataFrame(train['mulliken_charge'])
df2 = pd.DataFrame(cv['mulliken_charge'])
bd.check_statistics(df1,df2)


# We're all good. Let's continue!

# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

target = 'mulliken_charge'

# Test set 
X_test = test

# Train set
X_train = train.drop(target, axis=1)    
y_train = train[target]

# CV set
X_cv = cv.drop(target, axis=1)
y_cv = cv[target]
  
# Extra Tree
reg = ExtraTreesRegressor(n_estimators=8, max_depth=20, n_jobs=4)
reg.fit(X_train, y_train)
pred_train = reg.predict(X_train)
pred_cv = reg.predict(X_cv)
pred_test = reg.predict(X_test)

print('MAE on train set: %.2E.' %mean_absolute_error(y_train, pred_train)) 
print('MAE on cv set: %.2E.' %mean_absolute_error(y_cv, pred_cv))
print('')

# Plotiplot
plt.plot(y_cv,pred_cv,'o')
plt.plot([-1,1],[-1,1]) # perfect fit line
plt.xlabel('True')
plt.ylabel('Predicted')
plt.show()


# Our predicted Mulliken charges have a MAE of approximately $10^{-2}$! We could probably do better with a more advanced model and cross-validation scheme but it's already great as it is! We're on the same order of magnitude as the DFT here.
# 
# **Side note:** Be careful with `ExtraTreesRegressor` because it overfits like crazy if `max_depth` is left unspecified. Here we see that the MAE on the train and cv sets are similar, so that we're not overfitting.

# # Wrapping up
# Let's put everything in a nice dataframe:

# In[ ]:


tmp = raw_struct[raw_struct['molecule_name'].isin(raw_test['molecule_name'].unique())]
mulliken_charges_test=pd.DataFrame()
mulliken_charges_test['molecule_name'] = tmp['molecule_name']
mulliken_charges_test['atom_index'] = tmp['atom_index']
mulliken_charges_test['mulliken_charge'] = pred_test


# Out of curiosity, let's take a look at one random molecule:

# In[ ]:


mulliken_charges_test[mulliken_charges_test['molecule_name'] == 'dsgdb9nsd_000004']


# Looking at the predictions, there's something very interesting that you may notice: **some Mulliken charges are identical!** Is it a numerical artifact or does it have a physical meaning?
# 
# Let's visualize the molecule!

# In[ ]:


# Need to reload the initial structure file for my function to work properly...
struct = pd.read_csv('../input/champs-scalar-coupling/structures.csv')

bd.view('dsgdb9nsd_000004', struct)


# We see that the molecule is H-C≡C-H, which is a **symmetric** molecule: there's a symmetry plane right in the middle of the molecule (among others). And because of this symmetry plane, the C atoms have identical charges, as do both of the H atoms.
# 
# > **This is the power of descriptors, they reproduce the invariances with respect to permutational, rotational, reflectional, and translational symmetries.**
# 
# Therefore, no need to do any **data augmentation** in order to train your models on the same molecule that has been rotated, translated, etc. The descriptors take care of that for you :)
# 
# Finally, let's output the file so that everyone can get these predicted charges :)

# In[ ]:


mulliken_charges_test.to_csv('mulliken_charges_test_set.csv', index=False)


# I hope you enjoyed this little notebook!
# 
# Cheers
# 
# Boris D.
