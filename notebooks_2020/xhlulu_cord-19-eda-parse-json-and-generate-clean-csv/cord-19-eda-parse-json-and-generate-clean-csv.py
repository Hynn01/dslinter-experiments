#!/usr/bin/env python
# coding: utf-8

# ## About this notebook
# 
# In this notebook, I quickly explore the `biorxiv` subset of the papers. Since it is stored in JSON format, the structure is likely too complex to directly perform analysis. Thus, I not only explore the structure of those files, but I also provide the following helper functions for you to easily format inner dictionaries from each file:
# * `format_name(author)`
# * `format_affiliation(affiliation)`
# * `format_authors(authors, with_affiliation=False)`
# * `format_body(body_text)`
# * `format_bib(bibs)`
# 
# Feel free to reuse those functions for your own purpose! If you do, please leave a link to this notebook.
# 
# Throughout the EDA, I show you how to use each of those files. At the end, I show you how to generate a clean version of the `biorxiv` as well as all the other datasets, which you can directly use by choosing this notebook as a data source ("File" -> "Add or upload data" -> "Kernel Output File" tab -> search the name of this notebook).
# 
# ### Update Log
# 
# * V9: First release.
# * V10: Updated paths to include the [14k new papers](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/137474).

# In[ ]:


import os
import json
from pprint import pprint
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


# ## Helper Functions

# Unhide the cell below to find the definition of the following functions:
# * `format_name(author)`
# * `format_affiliation(affiliation)`
# * `format_authors(authors, with_affiliation=False)`
# * `format_body(body_text)`
# * `format_bib(bibs)`

# In[ ]:


def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return ", ".join(name_ls)

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body

def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []
    
    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'], 
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)


# Unhide the cell below to find the definition of the following functions:
# * `load_files(dirname)`
# * `generate_clean_df(all_files)`

# In[ ]:


def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    
    return raw_files

def generate_clean_df(all_files):
    cleaned_files = []
    
    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'], 
                           with_affiliation=True),
            format_body(file['abstract']),
            format_body(file['body_text']),
            format_bib(file['bib_entries']),
            file['metadata']['authors'],
            file['bib_entries']
        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'abstract', 'text', 
                 'bibliography','raw_authors','raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()
    
    return clean_df


# ## Biorxiv: Exploration
# 
# Let's first take a quick glance at the `biorxiv` subset of the data. We will also use this opportunity to load all of the json files into a list of **nested** dictionaries (each `dict` is an article).

# In[ ]:


biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'
filenames = os.listdir(biorxiv_dir)
print("Number of articles retrieved from biorxiv:", len(filenames))


# In[ ]:


all_files = []

for filename in filenames:
    filename = biorxiv_dir + filename
    file = json.load(open(filename, 'rb'))
    all_files.append(file)


# In[ ]:


file = all_files[0]
print("Dictionary keys:", file.keys())


# ## Biorxiv: Abstract

# The abstract dictionary is fairly simple:

# In[ ]:


pprint(file['abstract'])


# ## Biorxiv: body text

# Let's first probe what the `body_text` dictionary looks like:

# In[ ]:


print("body_text type:", type(file['body_text']))
print("body_text length:", len(file['body_text']))
print("body_text keys:", file['body_text'][0].keys())


# We take a look at the first part of the `body_text` content. As you will notice, the body text is separated into a list of small subsections, each containing a `section` and a `text` key. Since multiple subsection can have the same section, we need to first group each subsection before concatenating everything.

# In[ ]:


print("body_text content:")
pprint(file['body_text'][:2], depth=3)


# Let's see what the grouped section titles are for the example above:

# In[ ]:


texts = [(di['section'], di['text']) for di in file['body_text']]
texts_di = {di['section']: "" for di in file['body_text']}
for section, text in texts:
    texts_di[section] += text

pprint(list(texts_di.keys()))


# The following example shows what the final result looks like, after we format each section title with its content:

# In[ ]:


body = ""

for section, text in texts_di.items():
    body += section
    body += "\n\n"
    body += text
    body += "\n\n"

print(body[:3000])


# The function below lets you display the body text in one line (unhide to see exactly the same as above):

# In[ ]:


print(format_body(file['body_text'])[:3000])


# ## Biorxiv: Metadata

# Let's first see what keys are contained in the `metadata` dictionary:

# In[ ]:


print(all_files[0]['metadata'].keys())


# Let's take a look at each of the correspond values:

# In[ ]:


print(all_files[0]['metadata']['title'])


# In[ ]:


authors = all_files[0]['metadata']['authors']
pprint(authors[:3])


# The `format_name` and `format_affiliation` functions:

# In[ ]:


for author in authors:
    print("Name:", format_name(author))
    print("Affiliation:", format_affiliation(author['affiliation']))
    print()


# Now, let's take as an example a slightly longer list of authors:

# In[ ]:


pprint(all_files[4]['metadata'], depth=4)


# Here, I provide the function `format_authors` that let you format a list of authors to get a final string, with the optional argument of showing the affiliation:

# In[ ]:


authors = all_files[4]['metadata']['authors']
print("Formatting without affiliation:")
print(format_authors(authors, with_affiliation=False))
print("\nFormatting with affiliation:")
print(format_authors(authors, with_affiliation=True))


# ## Biorxiv: bibliography

# Let's take a look at the bibliography section. 

# In[ ]:


bibs = list(file['bib_entries'].values())
pprint(bibs[:2], depth=4)


# You can reused the `format_authors` function here:

# In[ ]:


format_authors(bibs[1]['authors'], with_affiliation=False)


# The following function let you format the bibliography all at once. It only extracts the title, authors, venue, year, and separate each entry of the bibliography with a `;`.

# In[ ]:


bib_formatted = format_bib(bibs[:5])
print(bib_formatted)


# ## Biorxiv: Generate CSV
# 
# In this section, I show you how to manually generate the CSV files. As you can see, it's now super simple because of the `format_` helper functions. In the next sections, I show you have to generate them in 3 lines using the `load_files` and `generate_clean_dr` helper functions.

# In[ ]:


cleaned_files = []

for file in tqdm(all_files):
    features = [
        file['paper_id'],
        file['metadata']['title'],
        format_authors(file['metadata']['authors']),
        format_authors(file['metadata']['authors'], 
                       with_affiliation=True),
        format_body(file['abstract']),
        format_body(file['body_text']),
        format_bib(file['bib_entries']),
        file['metadata']['authors'],
        file['bib_entries']
    ]
    
    cleaned_files.append(features)


# In[ ]:


col_names = [
    'paper_id', 
    'title', 
    'authors',
    'affiliations', 
    'abstract', 
    'text', 
    'bibliography',
    'raw_authors',
    'raw_bibliography'
]

clean_df = pd.DataFrame(cleaned_files, columns=col_names)
clean_df.head()


# In[ ]:


clean_df.to_csv('biorxiv_clean.csv', index=False)


# ## Generate CSV: Custom (PMC), Commercial, Non-commercial licenses

# In[ ]:


pmc_dir = '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/'
pmc_files = load_files(pmc_dir)
pmc_df = generate_clean_df(pmc_files)
pmc_df.to_csv('clean_pmc.csv', index=False)
pmc_df.head()


# In[ ]:


comm_dir = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/'
comm_files = load_files(comm_dir)
comm_df = generate_clean_df(comm_files)
comm_df.to_csv('clean_comm_use.csv', index=False)
comm_df.head()


# In[ ]:


noncomm_dir = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/'
noncomm_files = load_files(noncomm_dir)
noncomm_df = generate_clean_df(noncomm_files)
noncomm_df.to_csv('clean_noncomm_use.csv', index=False)
noncomm_df.head()

