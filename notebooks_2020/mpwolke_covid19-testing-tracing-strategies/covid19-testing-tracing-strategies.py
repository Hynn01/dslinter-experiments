#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'data:image/jpeg;base64,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',width=400,height=400)


# cebm.net

# #COVID-19 DIAGNOSTIC TESTS
# 
# Serological blood tests that detect antibodies to previous infections have been one of the main demands of the coronavirus pandemic.
# 
# METHODOLOGY: Immunochromatography, RT-PCR, Immunofluorescence (FIA), Chemiluminescent immunoassay (CLIA), Elisa.
# 
# TARGET: IgM / IgG; ORF1ab genes; N; Genes E; N2; Ag; ORF1ab and N gene; RdRp and N; Rp, E, P1; ORF1ab, N, S genes; ORF1 SARS-CoV-2 gene; ORF1ab and N genes.
# 
# SAMPLE TYPES: Blood, serum, plasma; Stool, nasopharyngeal swab; Nasopharyngeal aspirate / secretion, sputum, bronchial lavage; ; Swab / nasopharyngeal aspirate, bronchoalveolar lavage; Combined nasopharynx / triple swab; Bronchial lavage, oropharyngeal secretion, nasopharyngeal secretion.
# 
# REGISTRATION VALIDITY: Standard, Emergency.
# https://translate.google.com.br/translate?hl=en&sl=pt&u=http://www.inpi.gov.br/menu-servicos/patente/tecnologias-para-covid-19/TESTES%2520PARA%2520DIAGNOSTICO&prev=search

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# The COVID Tracking Project collects and publishes the most complete testing data available for US states and territories. https://covidtracking.com/

# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'https://covidtracking.com/static/wsj-7deed4e5103e0a2f19d1af6fce01925a.png',width=400,height=400)


# Number of tests performed VS. Positive Cases - The Wall Street Journal - https://covidtracking.com/

# 4/29/2020
# 
# UFMG foresees the development of software based on artificial intelligence and machine learning to assist in the automatic detection of acute respiratory syndromes, such as Covid-19 and H1N1 (28/04/2020). Source: UFMG  
# 
# Unesp Venom and Venomous Animals Study Center (Cevap), located on the campus of Botucatu, joined the São Paulo Biological Institute, the Pan American Health Organization (PAHO), the Vital Brazil Institute, the Ezequiel Dias Foundation and Brazilian and American pharmaceutical companies to develop innovative treatment based on nano-bodies to combat Covid-19 in infected patients (04/28/2020). Source: Unesp
# 
# Large-scale genetic testing is hampered, in part, by the scarcity of solutions used to store samples and extract viral RNA from them. To overcome this difficulty, a team from the University of Washington in Seattle, developed a different procedure to detect viral RNA in swabs (04/29/2020). Source: Biorexiv
# https://translate.google.com.br/translate?hl=en&sl=pt&u=http://www.inpi.gov.br/menu-servicos/patente/tecnologias-para-covid-19/TESTES%2520PARA%2520DIAGNOSTICO&prev=search

# In[ ]:


df = pd.read_csv('../input/covid19-challenges/test_data_on.csv', encoding='ISO-8859-2')
df.head()


# 4/28/2020
# 
# ANVISA approved the proposal to carry out rapid tests (immunochromatographic assays) of antibodies to the new coronavirus (Sars-CoV-2) in pharmacies and drugstores. The measure is temporary and exceptional and aims to expand the supply and testing network, as well as reduce the high demand for public health services during the pandemic. However, it is important to note that the tests have no confirmatory purpose, serving only to assist in the diagnosis of Covid-19 (07/28/2020). Source: ANVISA
# 
# 4/27/2020
# 
# Update of the American CDC guidelines for patient assessment and testing for COVID-19 (04/27/2020). CDC source .
# Researchers at Yale University in order to improve the COVID-19 testing process demonstrated in a preprint that saliva is more sensitive than the use of nasopharyngeal swab to detect SARS-Cov-2 (22/04/2020). Source: MedRxiv.
# 
# https://translate.google.com.br/translate?hl=en&sl=pt&u=http://www.inpi.gov.br/menu-servicos/patente/tecnologias-para-covid-19/TESTES%2520PARA%2520DIAGNOSTICO&prev=search

# In[ ]:


fig = px.pie(df,
             values="negative",
             names="positive",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# 4/22/2020
# 
# Nature's article questions whether the use of rapid tests based on an individual's immune response can be used to help end isolation as it may give false results. The WHO and FDA say that only PCR-tests for the virus should be used to confirm the cases (21/04/2020). Source: Nature
# 
# Lancet article reviews the importance and use of serological tests (based on the individual's immune response) for rapid identification of cases and their use in control policies. The article points out that since PCR tests do not identify past infections, serological tests and data will be increasingly important to understand the past of pandemics and predict their future (21/04/2020). Source: The Lancet Infectious Diseases
# 
# 
# Anvisa publishes an article explaining the rapid antibody test (based on the individual's immune response) for the new coronavirus (SARS-CoV-2). The test can be used to support the assessment of the immune status of patients who have symptoms of COVID-19. (4/21/2020). Source: Anvisa
# https://translate.google.com.br/translate?hl=en&sl=pt&u=http://www.inpi.gov.br/menu-servicos/patente/tecnologias-para-covid-19/TESTES%2520PARA%2520DIAGNOSTICO&prev=search

# In[ ]:


fig = px.treemap(df, path=['positive'], values='negative',
                  color='negative', hover_data=['positive'],
                  color_continuous_scale='Rainbow')
fig.show()


# 4/21/2020
# 
# Article demonstrates a new rapid RT-LAMP test assay using reverse transcription loop-mediated isothermal amplification (RT-LAMP) for the specific detection of SARS-CoV-2. The test exhibited a rapid detection interval of 30 min, combined with colorimetric visualization. This test can detect SARS-CoV-2 specific viral RNAs without cross-reactivity to related coronaviruses, such as HCoV-229E, HCoV-NL63, HCoV-OC43 and MERS-CoV, in addition to infectious human influenza viruses (type B, H1N1pdm, H3N2, H5N1, H5N6, H5N8 and H7N9) and other viruses that cause respiratory diseases (RSVA, RSVB, ADV, PIV, MPV and HRV). The article demonstrates a high sensitivity and specificity, this isothermal amplification combined with a single-tube colorimetric detection method can contribute to public health responses and disease control, especially in areas with limited laboratory capacity (20/04/2020). Source: Emerging Microbes & Infections
# 
# 4/20/2020
# 
# In the next few days, Fiocruz's weekly test production will increase tenfold, from 20,000 to 200,000 kits. Fiocruz is the largest national producer, responsible for supplying the entire public health network. (20/04/2020) Source: Jornal da Manhã
# https://translate.google.com.br/translate?hl=en&sl=pt&u=http://www.inpi.gov.br/menu-servicos/patente/tecnologias-para-covid-19/TESTES%2520PARA%2520DIAGNOSTICO&prev=search

# In[ ]:


get_ipython().system('pip install chart_studio')


# 4/18/2020
# 
# ANVISA publishes questions and answers about the diagnostic tests carried out in Brazil, the answers refer to tests for the detection of diseases such as COVID-19, the types of samples used and how to record them. Source: ANVISA
# 
# Banco do Brasil and Bradesco will donate R $ 20 million to Fundação Oswaldo Cruz (Fiocruz) to help fight the new coronavirus. The money will be used in the production of quick diagnosis kits for the COVID-19, which will be destined to the Ministry of Health. All kits already have Brazilian technology developed by Fiocruz. (04/18/2020) Source: Globo.com
# 
# UFMG expands its ability to carry out tests to detect the coronavirus. Laboratory of the Faculty of Medicine should process 1,000 exams per day until the end of the month (04/17/2020). Source: UFMG
# https://translate.google.com.br/translate?hl=en&sl=pt&u=http://www.inpi.gov.br/menu-servicos/patente/tecnologias-para-covid-19/TESTES%2520PARA%2520DIAGNOSTICO&prev=search

# In[ ]:


pip install bubbly


# 4/17/2020
# 
# Characteristics and comparison between diagnostic tests for COVID-19 "drive-through" and "walk-through" used in South Korea (04/16/20). Source: Clincal end Experimental Emergency Medicine.
# 
# Unicamp uses national reagents and develops a cheaper coronavirus test. Partnerships were made with biotechnology companies and startups to develop an examination without dependence on international inputs, lacking in the market. (08/04/2020) Source: G1
# 
# Anvisa maintains in its portal a list of products for in vitro diagnostics for the detection of regulated COVID-19. This list is updated daily on the agency's page. The registrations granted under the (emergency) conditions of Art. 12 of RDC 348/2020 will be valid for 1 (one) year.
# Products registered based on Art. 11 of the same Resolution and those that meet all the requirements of RDC 36/2015 will have the standard registration validity of 10 (ten) years. (Daily update) Source: ANVISA
# https://translate.google.com.br/translate?hl=en&sl=pt&u=http://www.inpi.gov.br/menu-servicos/patente/tecnologias-para-covid-19/TESTES%2520PARA%2520DIAGNOSTICO&prev=search

# In[ ]:


from bubbly.bubbly import bubbleplot 
from plotly.offline import iplot
import chart_studio.plotly as py

figure = bubbleplot(dataset=df, x_column='investigation', y_column='negative', 
    bubble_column='date',size_column='positive', color_column='date', 
    x_title="Investigation", y_title="Resolved", title='Investigation vs Negative',
     scale_bubble=3, height=650)

iplot(figure, config={'scrollzoom': True})


# 4/14/2020
# 
# Bio-Manguinhos obtained from Anvisa registration for its two diagnostic kits for the new coronavirus: the TR DPP Covid-19 Kit, to simultaneously detect antibodies of the IgM classes (recent infection, from the 5th day after the onset of symptoms) and IgG (memory) independently and differentially, and the SARS-CoV2 Molecular Kit, capable of detecting viral infection from the first day of symptom onset even when the patient's viral load is low. (04/14/2020) Source: Bio-Manguinhos / Fiocuz
# 
# 4/13/2020
# 
# To increase the number of covid-19 diagnoses in Brazil, researchers from USP's Institute of Biomedical Sciences (ICB) developed different methods, on three fronts. One of them allows testing to identify the virus in equipment available in most laboratories in the country (04/13/2020). Source: USP Journal
# INCQS / Fiocruz performs previous analyzes on the diagnostic kits, which are composed of sets and reagents, calibrators and controls, class 3 (in case of outbreaks) and class 4, to compose the registration process with the National Surveillance Agency Sanitary (Anvisa). (04/13/2020) Source: INCQS / Fiocruz
# 
# 4/12/2020
# 
# Brazilian scientists, in a race against time, evaluate the effectiveness of diagnostic tests on the new coronavirus (12/04/2020). Source: G1
# https://translate.google.com.br/translate?hl=en&sl=pt&u=http://www.inpi.gov.br/menu-servicos/patente/tecnologias-para-covid-19/TESTES%2520PARA%2520DIAGNOSTICO&prev=search

# In[ ]:


plt.figure(figsize=(23,10))
plt.bar(df.date, df.investigation,label="investigation")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Investigation',fontsize=30)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()


plt.figure(figsize=(23,10))
plt.bar(df.date, df.negative,label="negative")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Negative',fontsize=30)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

plt.figure(figsize=(23,10))

plt.bar(df.date, df.positive,label="positive")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Positive',fontsize=30)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()


# #Codes from Sanket Parate https://www.kaggle.com/sanketparate5/data-visualization-with-pandas

# In[ ]:


ax = df.plot(figsize=(15,8), title='Testing Strategies')
ax.set_xlabel('Total,Negative, Investigation, Positive')
ax.set_ylabel('Count')


# 4/10/2020
# 
# The Federal Biomedical Agency of Russia (FMBA) has launched two new test systems, based on coronavirus detection chips. One of them allows to reduce the test time from 90 to 15-20 minutes (10/04/2020). Source: The General Digital-Partner
# 
# 04/09/2020
# 
# Researchers evaluate the accuracy of laboratory parameters in detecting COVID-19 patients with positive RT-PCR. The findings suggest that levels of lactate dehydrogenase (LDH), C-reactive protein (CRP), alanine aminotransferase (ALT) and neutrophil (NEU) can be used to predict the result of the COVID-19 test (04/04/2020) . Source: Arch Acad Emerg Med  
# 
# Editorial discusses the challenges of developing a diagnostic test for a new pathogen in the course of a pandemic, stressing the importance of communication between the laboratory professionals involved and the hospital administration.
# 
# 4/8/2020
# 
# ANVISA publishes periodic updates on products authorized for the diagnosis of COVID-19. Source: ANVISA
# Chinese researchers develop a fast, simple, sensitive and detection test for the new Coronavirus SARS-CoV-2. The RT-LAMP test (reverse transcription loop-mediated isothermal amplification) was compared to RT-PCR (04/04/2020) Source: Clinical Microbiology and Infection
# 
# 4/7/20202	
# The study portrays the current cellular evolution, epidemiology and diagnosis in response to the COVID-19 outbreak. In addition, it shows that studies exploring the genome and structure of viral proteins are essential to define prevention (07/04/2020). Source: Chemotherapy
# 
# 4/6/2020	
# The etiological diagnoses of COVID-19 are based essentially on virological aspects (such as RT-PCR molecular tests), the article points to the need to identify biomarkers (new or known), to guide doctors and laboratory professionals (06 / 04/2020). Source: Clinical Chemistry and Laboratory Medicine (CCLM)https://translate.google.com.br/translate?hl=en&sl=pt&u=http://www.inpi.gov.br/menu-servicos/patente/tecnologias-para-covid-19/TESTES%2520PARA%2520DIAGNOSTICO&prev=search

# In[ ]:


df.iloc[0]


# 04/04/2020	
# 
# ANBIOTEC informs that the startup iBench is offering products from several national suppliers, including the raw material for the COVID-19 tests, reagents and inputs for the diagnostic tests such as Taq and Master Mix produced by national biotechnology. The company Enzytec Biotecnologia, which works on the development of diagnostic kits and product registration with Anvisa, is supporting national industries for free in the first product registrations for detecting coronavirus in Brazil. The startup Pickcells developed an algorithm to identify pneumonia in imaging exams. Source: ANBIOTEC
# 
# Know the advantages and disadvantages of the different types of tests to diagnose coronavirus: RT-PCR, rapid tests and serological tests are arriving in Brazil and have different characteristics (04/04/2020). Source: O Globo
# 
# 04/03/2020	
# UFPB prepares two laboratories for testing COVID-19 and UFPB's partnership with Lacen / PB makes 40 COVID-19 tests per day possible (04/03/2020). Source: UFPB , UFPB
# 
# 4/2/2020	
# Laboratory of Molecular Virology at UFRJ carries out 280 tests per day and launches TESTAR-TESTAR-TESTAR campaign, to obtain resources to serve 10,000 people (4/2/2020). Source: G1
# https://translate.google.com.br/translate?hl=en&sl=pt&u=http://www.inpi.gov.br/menu-servicos/patente/tecnologias-para-covid-19/TESTES%2520PARA%2520DIAGNOSTICO&prev=search

# In[ ]:


df.plot.hist()


# 3/31/2020	
# 
# Retrospective study shows that ~ 15% of COVID-19 patients who become swab-negative for the virus remain with the same detectable in sputum or faeces (03/31/2020). Source: Annals of Internal Medicine
# 
# 3/29/2020	
# Researchers from the Universities of Campinas (Unicamp) and São Paulo (USP) joined efforts to develop a fast and low-cost test to diagnose cases of COVID-19 and, in addition, to identify patients at risk of progressing to insufficiency respiratory. Source: FAPESP
# 
# 03/27/2020	
# FDA approves Abbott's diagnostic test that works in 15 minutes (Abbott). Source: UOL
# 
# 3/26/2020	
# The United Kingdom orders 3.5 million tests to verify that the person has already had contact with the COVID-19 of the Dutch company Sensitest, which promises to sell on AMAZON. Source: NATURE
# 
# 3/25/2020	
# Hi Technologies startup from Paraná develops a quick test for COVID-19. Source: FINEP
# 
# 03/23/2020	
# Coppe and UFRJ develop a new test to detect SARS-CoV2 simpler and at a cost four times lower. Source: UFRJ
# https://translate.google.com.br/translate?hl=en&sl=pt&u=http://www.inpi.gov.br/menu-servicos/patente/tecnologias-para-covid-19/TESTES%2520PARA%2520DIAGNOSTICO&prev=search

# In[ ]:


df.plot(kind = 'hist', stacked = False, bins = 100)


# In[ ]:


df.plot(kind = 'hist', stacked = True, bins = 50, orientation = 'horizontal')


# In[ ]:


df.plot.scatter(x = 'investigation', y = 'positive', c = 'negative', s = 190)


# In[ ]:


df.plot.hexbin(x = 'investigation', y = 'positive', gridsize = 5, C = 'negative')


# In[ ]:


from pandas.plotting import scatter_matrix
scatter_matrix(df, figsize= (8,8), diagonal='kde', color = 'b')
plt.show()


# In[ ]:


df_grp = df.groupby(["investigation","positive"])[["id","negative","resolved", "deaths", "total"]].sum().reset_index()
df_grp.head()


# In[ ]:


df_grp.diff().hist(color = 'b', alpha = 0.1, figsize=(10,10))


# In[ ]:


from pandas.plotting import andrews_curves
andrews_curves(df_grp, 'positive')


# In[ ]:


df_grpt = df_grp.groupby(["investigation"])[["negative","positive"]].sum().reset_index()
df_grpt.head()


# In[ ]:


df_grpt.plot.pie(subplots = True, figsize = (25, 25))


# In[ ]:


#codes from Rodrigo Lima  @rodrigolima82
from IPython.display import Image
Image(url = 'data:image/jpeg;base64,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',width=400,height=400)


# cebm.net

# Kaggle Notebook Runner: Marília Prata  @mpwolke
