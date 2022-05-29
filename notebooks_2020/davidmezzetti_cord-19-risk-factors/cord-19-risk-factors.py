#!/usr/bin/env python
# coding: utf-8

# CORD-19 Risk Factors
# ======
# 
# This notebook shows the query results for a single task. CSV summary tables can be found in the output section.
# 
# The report data is linked from the [CORD-19 Analysis with Sentence Embeddings Notebook](https://www.kaggle.com/davidmezzetti/cord-19-analysis-with-sentence-embeddings).

# In[ ]:


from cord19reports import install

# Install report dependencies
install()


# In[ ]:


get_ipython().run_cell_magic('capture', '--no-display', 'from cord19reports import run\n\ntask = """\nid: 8\nname: risk_factors\n\n# Field definitions\nfields:\n    common: &common\n        - name: Date\n        - name: Study\n        - name: Study Link\n        - name: Journal\n        - name: Study Type\n\n    severity: &severity\n        - {name: Severe, query: $QUERY, question: What is $NAME risk number}\n        - {name: Severe lower bound, query: $QUERY, question: What is $NAME range minimum}\n        - {name: Severe upper bound, query: $QUERY, question: What is $NAME range maximum}\n        - {name: Severe p-value, query: $QUERY, question: What is the $NAME p-value}\n\n    appendix: &appendix\n        - name: Sample Size\n        - name: Sample Text\n        - name: Study Population\n        - name: Matches\n        - name: Entry\n\n    columns: &columns\n        - *common\n        - *severity\n        - *appendix\n\nAge:\n    query: +age ci\n    columns: *columns\n\nAsthma:\n    query: +asthma ci\n    columns: *columns\n\nAutoimmune disorders:\n    query: +autoimmune disorders ci\n    columns: *columns\n\nCancer:\n    query: +cancer ci\n    columns: *columns\n\nCardio- and cerebrovascular disease:\n    query: cardio and +cerebrovascular disease ci\n    columns: *columns\n\nCerebrovascular disease:\n    query: +cerebrovascular disease ci\n    columns: *columns\n\nChronic digestive disorders:\n    query: +digestive disorders ci\n    columns: *columns\n\nChronic kidney disease:\n    query: +kidney disease ckd ci\n    columns: *columns\n\nChronic liver disease:\n    query: +liver disease ci\n    columns: *columns\n\nChronic respiratory diseases:\n    query: chronic +respiratory disease ci\n    columns: *columns\n\nCOPD:\n    query: chronic obstructive pulmonary disease +copd ci\n    columns: *columns\n\nDementia:\n    query: +dementia ci\n    columns: *columns\n\nDiabetes:\n    query: +diabetes ci\n    columns: *columns\n\nDrinking:\n    query: +alcohol abuse ci\n    columns: *columns\n\nEndocrine diseases:\n    query: +endocrine disease ci\n    columns: *columns\n\nEthnicity_ Hispanic vs. non-Hispanic:\n    query: +hispanic race ci\n    columns: *columns\n\nHeart Disease:\n    query: +heart +disease ci\n    columns: *columns\n\nHeart Failure:\n    query: +heart +failure ci\n    columns: *columns\n\nHypertension:\n    query: +hypertension ci\n    columns: *columns\n\nImmune system disorders:\n    query: +immune system disorder ci\n    columns: *columns\n\nMale gender:\n    query: +male ci\n    columns: *columns\n\nNeurological disorders:\n    query: +neurological disorders ci\n    columns: *columns\n\nOverweight or obese:\n    query: +overweight obese ci\n    columns: *columns\n\nRace_ Asian vs. White:\n    query: race +asian +white ci\n    columns: *columns\n\nRace_ Black vs. White:\n    query: race +black +white ci\n    columns: *columns\n\nRace_ Other vs. White:\n    query: race +white ci\n    columns: *columns\n\nRespiratory system diseases:\n    query: +respiratory disease ci\n    columns: *columns\n\nSmoking Status:\n    query: +smoking smoker ci\n    columns: *columns\n"""\n\n# Build and display the report\nrun(task)')
