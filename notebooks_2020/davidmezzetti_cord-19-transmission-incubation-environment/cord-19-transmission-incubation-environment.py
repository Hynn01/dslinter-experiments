#!/usr/bin/env python
# coding: utf-8

# CORD-19 Transmission, Incubation, Environment
# ======
# 
# This notebook shows the query results for a single task.
# 
# The report data is linked from the [CORD-19 Analysis with Sentence Embeddings Notebook](https://www.kaggle.com/davidmezzetti/cord-19-analysis-with-sentence-embeddings).

# # Tasks
# ---
# 

# In[ ]:


from IPython.display import display, Markdown

# Builds a markdown report
def report(file):
    display(Markdown(filename="../input/cord-19-analysis-with-sentence-embeddings/%s/%s.md" % (file, file)))

report("transmission")

