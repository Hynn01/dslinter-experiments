#!/usr/bin/env python
# coding: utf-8

# # HOW TO: Easy Visualization of Molecules.
# 
# Greetings everyone!
# 
# I've seen many people ask for a simple yet elegant visualization tool for the [Predicting Molecular Properties](https://www.kaggle.com/c/champs-scalar-coupling/overview) challenge. Therefore, I'm going to explain in this Kernel how to install and use **ase**, which is a python module that allows one to work with atoms and molecules. 
# 
# It is available on gitlab: [ase](https://gitlab.com/ase/ase).
# 
# The first thing we need to do is to install **ase** on our Kernel. To do that, just click on the *Settings* tab on the right panel, then click on *Install...*, right next to *Packages*. In the *pip package name* entry, just write **ase** then hit *Install Package*.
# 
# Kaggle is going to do its things then restart the Kernel. **ase** should then be installed! Let's check this:

# In[ ]:


import ase


# It worked! 
# 
# Now let's visualize one of the molecule from the structures.csv file:

# In[ ]:


import pandas as pd

struct_file = pd.read_csv('../input/structures.csv')


# Now we select a random molecule from this file:

# In[ ]:


import random

# Select a molecule
random_molecule = random.choice(struct_file['molecule_name'].unique())
molecule = struct_file[struct_file['molecule_name'] == random_molecule]
display(molecule)


# Next we need to retrieve the atomic coordinates in a numpy array form:

# In[ ]:


# Get atomic coordinates
atoms = molecule.iloc[:, 3:].values
print(atoms)


# The last thing we need is the atomic symbols:

# In[ ]:


# Get atomic symbols
symbols = molecule.iloc[:, 2].values
print(symbols)


# Finally, let's put everything into something that **ase** can process:

# In[ ]:


from ase import Atoms
import ase.visualize

system = Atoms(positions=atoms, symbols=symbols)

ase.visualize.view(system, viewer="x3d")


# TADA!!!
# 
# You can rotate the molecule with a left click, translate it with a middle click, and zoom in or out using right click. 
# 
# All this can be summarized in a single function:

# In[ ]:


def view(molecule):
    # Select a molecule
    mol = struct_file[struct_file['molecule_name'] == molecule]
    
    # Get atomic coordinates
    xcart = mol.iloc[:, 3:].values
    
    # Get atomic symbols
    symbols = mol.iloc[:, 2].values
    
    # Display molecule
    system = Atoms(positions=xcart, symbols=symbols)
    print('Molecule Name: %s.' %molecule)
    return ase.visualize.view(system, viewer="x3d")

random_molecule = random.choice(struct_file['molecule_name'].unique())
view(random_molecule)


# I hope you enjoyed this little notebook!
# 
# Cheers
# 
# Boris D.
