#!/usr/bin/env python
# coding: utf-8

# ## Table of Content
# 1. [Historical Introduction](#introduction)
# 2. [CoronaWhy](#coronawhy)
# 3. [Current challenges](#current_challenges)
# 4. [This competition is vague](#dataset_bad)
# 5. [Current progress](#current_progress)
# 6. [Current scope of work](#current_scope)
# 7. [Join Us](#join)
# 8. [Trello](#trello)
# 9. [Slack channels](#slack_channels)
# 10. [External data](#datasets)
# 11. [Shared Codebase (Git)](#codebase)
# 12. [First Results (IMPORTANT)](#results)
# 13. [Daily Calls](#calls)
# 14. [Mar 21, 2020 - Call Summary](#mar21.2020)
# 15. [Mar 23, 2020 - Call Summary](#mar23.2020)
# 16. [Mar 24, 2020 - Call Summary](#mar24.2020)
# 17. [Mar 25, 2020 - Call Summary](#mar25.2020)
# 18. [Mar 26, 2020 - Call Summary](#mar26.2020)
# 18. [Mar 27, 2020 - Call Summary](#mar27.2020)
# 18. [Mar 28, 2020 - Call Summary](#mar28.2020)
# 18. [Mar 29, 2020 - Call Summary](#mar29.2020)
# 18. [Mar 30, 2020 - Call Summary](#mar30.2020)
# 18. [Mar 31, 2020 - Call Summary](#mar31.2020)
# 18. [Apr 1, 2020 - Call Summary](#apr1.2020)
# 18. [Apr 2, 2020 - Call Summary](#apr2.2020)
# 18. [Apr 7, 2020 - Call Summary](#apr7.2020)
# 18. [Apr 8, 2020 - Call Summary](#apr8.2020)
# 18. [Apr 9, 2020 - Call Summary](#apr9.2020)
# 18. [Apr 10, 2020 - Call Summary](#apr10.2020)
# 18. [Apr 11, 2020 - Call Summary](#apr11.2020)
# 18. [Apr 12, 2020 - Call Summary](#apr12.2020)
# 18. [Apr 13, 2020 - Call Summary](#apr13.2020)
# 19. [Task Selection Results (Important update from March 23, 2020](#task-selection-results)

# READ ABOUT US IN WALL STREET JOURNAL:
# [![WSJ](https://uploads-ssl.webflow.com/5e729ef45e85eb79fe4418d8/5e8dd174ce1cd72c2e7332fe_t0xZiT6vw5lGmcAWpHj0EXan6IzEtx92hB37DYkh.png)
# ](http://)
# https://www.wsj.com/articles/machine-learning-experts-delve-into-47-000-papers-on-coronavirus-family-11586338201?shareToken=st1ebd0a3a0e2e491b915d62fb96914cef&reflink=share_mobilewebshare

# ### WATCH OUR INTERVIEW ON FOX CHANNEL:

# In[ ]:


from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/SXOE6GqiHyQ?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# 
# 
# ## 1. Historical Introduction (as of Mar 19,2020)  <a id="introduction"></a>
# My name is Artur Kiulian, at this point it does not matter who I am and what I do other than that we all need to do everything we can to help our society battle this crisis.
# I am creating this notebook as a response to the lack of self-organization and collaboration that I am experiecing while trying to organize an effective collaboration.
# Here is the context:
# https://www.linkedin.com/posts/artur-kiulian_whitehouse-coronavirus-activity-6646040949627715584-x_k4/
#     
# My intent is to organize a global team of multidisciplinary talent that is willing to help.
# 
# ## 2. CoronaWhy  <a id="coronawhy"></a>
# [CoronaWhy](httsp://www.coronawhy.org) is an international group of 200+ volunteers whose mission is to improve global coordination and analysis of all available data pertinent to the COVID-19 outbreak and ensure all findings reach those who need them. 
# 
# The team organically formed around Kaggle’s CORD-19 challenge, whose 10 key tasks seek to find answers to 10 of the most pressing issues surrounding the pandemic. 
# 
# In the few days since Artur Kiulian started the CoronaWhy the group has grown rapidly. With 250+ volunteers from a wide range of countries, disciplines, availability and levels of expertise we’re in the process of bootstrapping our organizational procedures, protocols, communications channels and research methodologies. 
# 
# 
# 
# ## 2. Current challenges <a id="current_challenges"></a>
# - Understanding main challenge that is posted on kaggle, specific tasks and goals
# - Establishing efficient and effective communication and collaboration
# - Establishing diverse multidisciplinary team to tackle the ongoing corona crisis
# 
# ### 2.1 The competition is vague, isn't it? <a id="dataset_bad"></a>
# 
# Tasks are very abstract but they make sense if you dig into it, the assumption is that we can create a knowledge base that can help people who have have an idea what to do to move faster, that's it. Obviously we are not genetics or protein folding experts, but we can help those people with specific domain knowledge to make progress faster by adding them to parse the dataset faster.
# 
# And yes, if you think that the whole challenge is confusing, you are not alone - and that's expected.	
# 
# This is NOT a challenge to fit the curve and predict some output and we are not trying to transform it into that at this point. And maybe we will find specific classic ML prediction tasks, like for risk factor modeling for example. But we are not there yet. My guess is we are a week away from it so let's try to reach that phase together.
# 
# If you currently looking for pure ML challenge, you can look into doing Covid-19 Forecasting challenge. And check "#pure-ml-tasks" channel on our slack	
# 
# 
# ## 3. Current progress <a id="current_progress"></a>
# It’s very hard to take the current challenge and transform it into traditional ML problem to be solved by community
# 
# ![current issues](https://drive.google.com/uc?id=1KDEHn7U2ue8631V1usYbYGhiiOv6VIZl)
# 
# But we can’t take all of the tasks and try to figure that out all at once, that’s why we have to figure out a way to score current scope of work against the feasibility of what we can do as a team. That’s why I came up with a 5 feature scoring structure to score current tasks:
# 
# ### Impact
# - 1 - small impact
# - 2 - moderate impact
# - 3 - really relevant 
# - 4 - large impact
# - 5 - extremely impactful right now
# 
# ### Data presence
# - 1 - no meaningful datasets
# - 2 - some datasets available
# - 3 - some relevant datasets 
# - 4 - pretty good relevant datasets
# - 5 - lots of context specific data
# 
# ### Similar solutions
# - 1 - nothing similar have been explored with ML techniques before
# - 2 - some similar research exists
# - 3 - some similar problems have been solved
# - 4 - very good coverage of similar problems and solutions
# - 5 - very similar solutions have been well researched and solved
# 
# ### Specificity
# - 1 - very abstract
# - 2 - somewhat defined
# - 3 - well defined
# - 4 - very clearly defined 
# - 5 - obvious task
# 
# ### Simplicity
# - 1 - impossible to solve
# - 2 - super complex
# - 3 - complex
# - 4 - somewhat easy
# - 5 - super easy
# 
# Essentially this gives us a meaningful compound metric to take and analyze against specific tasks. And after we get averages we can take top 3 of the tasks and highlight them.
# 
# 
# ![current issues](https://drive.google.com/uc?id=1xP5I2wyI9KYEBOHstp5RBMSVwkIYN-qJ)
# 
# Example of my (Artur’s) scoring and picking top 3 tasks above.
# 
# The goal is to crowdsource subjective reasoning about the tasks and then jump into solving them.
# 
# Current scoring tabs here (we do understand it won't work when 100 people will join us, if someone has a better idea how to streamline this process pls let us know)
# https://docs.google.com/spreadsheets/d/1AhDif1UUVFJAQjYM8UcbKSPXUFe0hRXfFnMkcdw3YmI/edit?usp=sharing
# 
# 
# We are currently trying to hit 20 people to full score the tasks, video instruction added here:
# https://trello.com/c/IttahQGQ/1-task-scoring-sheet
# 
# 
# ## 4. Current scope of work <a id="current_scope"></a>
# 
# 
# ### Step 1
# Score and identify tasks worth pursuing through five features that have 1-5 score (see notes of each column)
# 
# 
# ### Step 2
# pick top 3 tasks and discuss possible ways to solve them
# 
# 
# ### Step 3
# basic similarity between tasks and current dataset
# 
# Step 3 is a result of Artur & Anton deciding to try and lay out some knowledge foundation for people jumping into this, since there is NONE right now. Even a basic bag of words between task and dataset would help. Here is a problem description document (seems like there is a new notebook solving this, investigating it now)
# https://docs.google.com/document/d/1M1Cbnbcj6uQkZDXzNP7dS5CUgbhvey8jSUiWU4BVTJU/edit?usp=sharing
# 
# Adrii & Platon are trying to provide some results on this particular task.
# 
# 
# UPDATE: We've successfully hit 20 people and formulated the final score. 
# #### Click here: 
# [Task Selection Results (Important update from March 23, 2020](#task-selection-results)
# 
# 
# ## 5. Join Us <a id="join"></a>
# 
# Current scope of synchronous communication is becoming hard to handle, that’s why Artur committed to creating slack channel to streamline async communication.
# 
# Slack has been created, please fill out "join" form on our website:
# https://www.coronawhy.org/
# 
# Thank you.
# 
# 
# ## 6. Trello board - Managing Work <a id="trello"></a>
# Created initial structure for trello board
# https://trello.com/b/y4odX7yZ/covid-19-global-team
# 
# Each task is going through the "Rough idea" -> "Brainstorming" -> "Formalized problem" -> "To Do" -> "Doing" -> "Done"
# 
# Each task that goes into "To Do" has a capability to create separate slack channel by "+createchannel" comment which is a Zapier integration:
# 
# ![zapier task](https://drive.google.com/uc?id=1KPoD97KLz8UNhsexILZz7qI-ZMBZX0oW)
# 
# I think the key problem we are going to face is commitment and expectations management, obviously this is not a full time job or 9-5 job, so just need to be explicit about what everyone is doing and what everyone will plan to do so everyone is on the same page, this will look weird at first but I think saying smth like
# "I will do X piece for this task and plan to do it by Y" and we can use trello tasks with due dates and members that go into checklist of the main task, I've done it like that on other non-profit projects that require collaboration.
# 
# 
# 
# **Example of communication from #basic-similarity-between-tasks-and-current-dataset**
# 
# ***
# 
# 
# *Artur Kiulian  3:40 PM*
# 
# hey guys
# 
# here's what Platon sent me on this task
# 
# https://drive.google.com/file/d/1Z03bkuWuIbKkc5Q4hGhKdw5MX8LqZSgY/view?usp=sharing
# 
# Here is the notebook
# 
# https://drive.google.com/drive/folders/1QTvtDxU9KrmcaahKMSAA5_BEC1-L1Gfj?usp=sharing
# 
# Here is the data result
# 
# covid_data/covid_w2v_searchable.csv is similarity scoring for all docs for all tasks
# 
# @Adrià if you have time to explore - pls check
# 
# ***
# 
# 
# *Adrià 3:41 PM*
# 
# Yes, im going to explore. Thanks!
# 
# ***
# 
# 
# **This ideally goes into the checklist subtask of the main Problem**
# 
# https://trello.com/c/31s4htxV/2-basic-similarity-between-tasks-and-current-dataset
# 
# ![subtask example](https://drive.google.com/uc?id=1ZX9q_T8sFG7ecG-U7LYAFpiOOE6Y2SC_)
# 
# https://trello.com/c/hQb3flS1/9-notebook-for-initial-basic-similarity-metric-between-tasks
# 
# 
# ![subtask card example](https://drive.google.com/uc?id=1Dvbxqd8Vc1fJBFNIVVNQxTQq16zy1rsh)
# 
# 
# 
# ## 7. Slack channels taxonomy <a id="slack_channels"></a>
# 
# 
# ![visual guide](https://trello-attachments.s3.amazonaws.com/5e7fc67ce5890d050d1a9715/1199x885/a248f477fba7e6a033f51917d4b07f7b/visual-guide.jpg)
# 
# 
# 
# 
# ## 7. Shared Codebase <a id="codebase"></a>
# 
# We've created public github account, need to ideate how to manage codebase for specific tasks
# 
# https://github.com/orgs/CoronaWhy	
# 
# ## 8. External Data <a id="datasets"></a>
# 
# We've only started to assemble all the data, pls join the discussion on #datasets channel
# Here's a spreadsheet to track:
# https://docs.google.com/spreadsheets/d/13vO8jZ4mrYD1U86U8r1qolY2HV552D7e5Fmko3c6Vrg/edit?usp=sharing
# 
# 
# 
# 
# 
# ## 9. First Results (IMPORTANT) <a id="results"></a>
# 
# - We've got our first technical task is half-done (hooray! and thanks to Platon)
# - We've got amazing interactive visualization (thanks to Mike Honey)
# 
# ![BI results](https://drive.google.com/uc?id=1mqCRMbwNeVW9jjUfdl3BprpuEX3AlNjr)
# 
# Link: https://app.powerbi.com/view?r=eyJrIjoiYWRlZGMwYzEtMGUyNC00YWE5LWI2NzMtYzU1Y2M0YzNkZjY3IiwidCI6ImRjMWYwNGY1LWMxZTUtNDQyOS1hODEyLTU3OTNiZTQ1YmY5ZCIsImMiOjEwfQ%3D%3D
# 
# **currently working on revisualizing the latest results**
# 
# Now it's the time to explore the results and understand what to do next, at this point we have top 4 tasks to focus on so we should take those and explore top papers/top authors based on similarity and proceed to the subtask exploration.
# 
# 
# 

# 
# ## 8. Calls/Daily standups  <a id="calls"></a>
# 
# ![call convo](https://drive.google.com/uc?id=1jL1gdlzWpLa8OAsVEV0bA9-maGIiPU-v)
# 
# if anyone has ideas about this section and how to streamline multizone management for call planning - pls let us know.
# 
# 
# ***
# 

# Had a first successful video call on Mar 21, here's a summary:
# 
# #### Mar 21, 2020 - Call Summary <a id="mar21.2020"></a>
# 
# Tasks discussed:
# - figure out the missing title data from visualization
# https://trello.com/c/VhClMvu3/12-figure-out-the-missing-title-data-from-visualization
# 
# - add the similarity metric results to the current visualization https://trello.com/c/wmBgLn1B/18-add-the-similarity-metric-results-to-the-current-visualization
# 
# - hit 20 people mark on the scoring of the tasks in the scoring sheet https://trello.com/c/jhdA5pH5/16-hit-20-people-mark-on-the-scoring-of-the-tasks-in-the-scoring-sheet
# 
# - figure out what datasets medical experts are looking for https://trello.com/c/sPJZ0gAV/14-figure-out-what-datasets-medical-experts-are-looking-for
# 
# - figure out the list of datasets to manually enrich with the help of ScaleAI https://trello.com/c/RwZ2ZIO0/13-figure-out-the-list-of-datasets-to-manually-enrich-with-the-help-of-scaleai
# 
# - identify list of experts that would benefit the group and overall momentum (non-technical talent) https://trello.com/c/jzi8An3h/15-identify-list-of-experts-that-would-benefit-the-group-and-overall-momentum-non-technical-talent
# 
# - template letter to send to the most relevant authors based on specific tasks https://trello.com/c/LZYD6sUT/20-template-letter-to-send-to-the-most-relevant-authors-based-on-specific-tasks
# 
# - list of authors with emails and expertise https://trello.com/c/VwKFGXZI/19-list-of-authors-with-emails-and-expertise
# 
# - figure out 24/7 pairing of the responsible people to the tasks https://trello.com/c/7oLAITtz/17-figure-out-24-7-pairing-of-the-responsible-people-to-the-tasks
# 
# - figure out the best way to approach quantitative research https://trello.com/c/pkLuOUj4/21-figure-out-the-best-way-to-approach-quantitative-research
# 
# 

# In[ ]:


#### Mar 21, 2020 - Call Summary 
from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/pjHmqIXdoG0?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Mar 23, 2020 - Call Summary  <a id="mar23.2020"></a>
# 
# Tasks to do:
# 
# - figure out stack of languages, technologies to be used in the github pipeline (build, test, deploy)
# https://trello.com/c/XKKXVL6J/48-figure-out-stack-of-languages-technologies-to-be-used-in-the-github-pipeline-build-test-deploy
# 
# - figure out how to integrate non-technical talent to help us (what you need to activate people outside of kaggle)
# https://trello.com/c/e4fIlL9Y/49-figure-out-how-to-integrate-non-technical-talent-to-help-us-what-you-need-to-activate-people-outside-of-kaggle
# 
# - figure out shared channel between two slacks
# https://trello.com/c/7R8jMSNS/50-figure-out-shared-channel-between-two-slacks
# 
# - getting people from AI2/original kaggle competition to slack
# https://trello.com/c/QPdw90uP/51-getting-people-from-ai2-original-kaggle-competition-to-slack
# 

# In[ ]:


#### Mar 23, 2020 - Call Summary 
from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/fV1nDII3Jho?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# 

# ### Task Selection (Updated Mar 23, 2020) <a id="task-selection-results"></a>
# 
# ![scores-table](https://drive.google.com/uc?id=17mqbyA6ewOCY0WRXrBm07pP-s8Eivmv0)
# 
# [context can be found here](https://docs.google.com/document/d/1cMDCfr-7gz0ylvdmVklJ7FqZHmz0dnklTN-zPJvD6Dg/edit?usp=sharing)
# 
# This scoring serves as a strong indicator of which tasks to focus on. Combined with discussion in our daily video calls, these scores inform our imminent decision on which tasks to focus on first. Even though we’ve first thought in terms of focusing on top 3, it’s obvious that top 4 tasks stick out the most and geography one sounds like the one easier to solve in general.
# 
# #### Help us understand how geography affects virality.
# 
# #### What do we know about COVID-19 risk factors?
# 
# #### What is known about transmission, incubation, and environmental stability?
# 
# #### What do we know about vaccines and therapeutics?
# 
# With that decision made, our next steps are to select Principal Investigators for each task, and through discussion with our volunteer pool from core teams to work with each Principal Investigator.
# 
# Those teams can report on their resource needs such as computation, additional data, support. 
# 
# At that stage we can mobilize and further organize CoronaWhy’s team to best support these and any subsequent research projects undertaken.
# 

# #  Focus / Impact <a id="task_focus"></a>
# 
# So before go deep into geeking out on diagrams, lets try and understand what are the foundational pieces that be helpful in further ideation of the formalized problem. We've decided to introduce small impact/large impact dichotomy to simplify things and further define most immediate focus.
# 
# ### **Small impact**
# 
# (our most immediate focus)
# * Retrieving relevant bits of information that could help researchers scan through database
# * Retrieving relevant relationships between risk factors and existing research on viral diseases/COVID-19
# * Retrieve relevant measures/recommendations for each of the stages of disease
# 
# ### **Large impact**
# 
# (this is out of scope of current Kaggle challenge but we as CoronaWhy.org are planning to tackle these if things will be progressing in the same way after this challenge)
# * Being able to identify core relationships between each of the stages of disease and main risk factors
# * Being able to predict probability of results for each stage of disease
# * Identifying main groups of risk
# * Identifying underlying causes for co-occurence of diseases and complications
# * Producing models to invent new forms of measures/recommendations for each stage of disease
# 

# #### Mar 24, 2020 - Call Summary  <a id="mar24.2020"></a>
# 
# 
# **Agenda**
# 
# (20 min) We will dedicate a bit more time to discuss the next steps, pls read documents before the call:
# 
# - Reviewing final task scoring
# https://docs.google.com/document/d/1cMDCfr-7gz0ylvdmVklJ7FqZHmz0dnklTN-zPJvD6Dg/edit
# 
# - Figuring out next steps 
# https://trello.com/c/hhxRZf7d/57-explore-and-discuss-subtasks
# 
# - Discussing "Selecting Principal Investigators and Teams"
# https://docs.google.com/document/d/1G3iCxDcq0vGK_RmJPfKPiFQya43sqTjo9rTJHeYgwtU/edit?usp=sharing
# 
# 
# (10 min) And to discuss blockers and very important past tasks that we are struggling to accomplish, if you know how to help PLEASE HELP:
# 
# - Github research pipeline
# https://trello.com/c/QuJN0m4i/24-github-research-pipeline
# 
# - identify list of experts that would benefit the group and overall momentum (non-technical talent)
# https://trello.com/c/jzi8An3h/15-identify-list-of-experts-that-would-benefit-the-group-and-overall-momentum-non-technical-talent
# 
# - list of authors with emails and expertise
# https://trello.com/c/VwKFGXZI/19-list-of-authors-with-emails-and-expertise
# 
# - template letter to send to the most relevant authors based on specific tasks
# https://trello.com/c/LZYD6sUT/20-template-letter-to-send-to-the-most-relevant-authors-based-on-specific-tasks
# 
# 
# **Action items to do**
# 
# - start getting a collection of different outputs (documents, notebooks, results) 
# https://trello.com/c/mUebA55Z/73-start-getting-a-collection-of-different-outputs-documents-notebooks-results
# 
# - setting up individual channels and process for team assembly 
# https://trello.com/c/JuXSgz15/74-setting-up-individual-channels-and-process-for-team-assembly
# 
# - setting up a process for technical vs informational layers (trello vs github for specific tasks)
# https://trello.com/c/CJAtrQU0/75-setting-up-a-process-for-technical-vs-informational-layers-trello-vs-github-for-specific-tasks
# 
# - general checkin process
# https://trello.com/c/jGOTYAHp/76-general-checkin-process
# 
# - delivery piece workgroup (visualization, presenting data)
# https://trello.com/c/9L5MO8JU/77-delivery-piece-workgroup-visualization-presenting-data
# 

# In[ ]:


#### Mar 24, 2020 - Call Summary 
from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/Oh1D4VaNjCE?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Mar 25, 2020 - Call Summary  <a id="mar25.2020"></a>
# 
# **Agenda**
# 
# (20 min) Discuss the task kickoff process, we now have 4 channels paired with 4 trello boards for these tasks.
# 
# I took a stab at how to approach task kickoff from the perspective of Lead Researcher/PI responsible for the task, sorry if I sound too tired, that’s primarily because I am haha, as a result created some initial thought process to kickoff each of the tasks, took Risk Factors as an example task to model on. Recommended to review:
# 
# https://www.loom.com/share/78d87335e2e0400aa31f77c2ee8876ca
# 
# 
# Slack Channels + Boards:
# 
# #task-geo - https://trello.com/b/e4BDCjqj/task-geography
# #task-risk - https://trello.com/b/3ObaWsDL/task-risk-factors
# #task-ties - https://trello.com/b/5LUjJJ4q/task-ties
# #task-vt - https://trello.com/b/iHrEiwZh/task-vaccines-and-therapeutics
# 
# 
# (15 min)  We critically need domain experts to help with tasks
# 
# Ben Jones is finishing up email enrichment and need other pieces:
# https://trello.com/c/6zsSoMJr/33-domain-expert-communications
# 
# We need PMs, Communicators etc, need to finalize the list and responsibilities:
# https://trello.com/c/jzi8An3h/15-identify-list-of-experts-that-would-benefit-the-group-and-overall-momentum-non-technical-talent
# 
# 
# (10 min) Organizing task specific team calls throughout the day, assuming we get some meaningful number of members for each task after tomorrow
# 
# https://trello.com/c/jGOTYAHp/76-general-checkin-process
# 
# 
# **Cool things**
# 
# We've got shared channel with AI2 established, feel free to join #b_coronawhy-allen-institute and ask questions directly to organizers of Kaggle competition:
# https://trello.com/c/QPdw90uP/51-getting-people-from-ai2-original-kaggle-competition-to-slack
# 
# We've finally added similarity metric to public powerBI visualization, good to explore:
# https://trello.com/c/wmBgLn1B/18-add-the-similarity-metric-results-to-the-current-visualization
# 
# 
# **Action items**
# 
# Action items
# Identify leader/PM for each channel 
# https://trello.com/c/RvdAhPWM/82-identify-leader-pm-for-each-channel
# Organize task specific calls 
# https://trello.com/c/yibvatDM/83-organize-task-specific-calls
# Organizing workshop group with AI2
# https://trello.com/c/2Fkof75f/84-organizing-workshop-group-with-ai2
# Spreading call to action document
# https://trello.com/c/1wiPkIUi/85-call-to-action
# Keeping main notebook at the top (pls upvote and comment) :pray:
# https://trello.com/c/zr6e5rLw/5-main-notebook

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/LuzdR-7pl08?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Mar 26, 2020 - Call Summary  <a id="mar26.2020"></a>
# 
# 
# **Agenda**
# 
# - (5 min) Discuss current blockers and how to make progress within individual tasks
# - (5 min) NLP tasks as the first piece of any pipeline when it comes to existing kaggle datasets
# - (10 min) Discuss medical expert integration and dedicate responsible person
# - (10 min) Q&A and further feedback integration on what we can do better as a group
# 
# **Action items**
# 
# - set up reporting process for individual team/task progress
# https://trello.com/c/oLzmmfbH/94-set-up-reporting-process-for-individual-team-task-progress
# 
# - kickoff sync calls for individual teams
# https://trello.com/c/O9abbhj7/96-kickoff-sync-calls-for-individual-teams
# 
# - ideate best way to integrate results into the new kaggle summary page
# https://trello.com/c/TYoOt8dv/95-ideate-best-way-to-integrate-results-into-the-new-kaggle-summary-page
# 
# - integrate Natalie into medical expert the process
# https://trello.com/c/CQRcCoys/97-integrate-natalie-into-medical-expert-the-process

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/6K64BXMQDSs?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Mar 27, 2020 - Call Summary  <a id="mar27.2020"></a>
# 
# **Agenda**
# (5 min) Discuss current blockers and how to make progress within individual tasks
# (5 min) Individual team reports
# (5 min) Preparing output for the kaggle summary page on the progress so far
# (10 min) Discuss medical expert integration progress
# (5 min) Q&A and further feedback integration on what we can do better as a group
# 
# **Action items**
# - figure out visual diagram for onboarding people
# https://trello.com/c/JPSQpU39/99-visual-guide-to-what-to-do-based-on-personas
# 
# - get update on TIES task since Christine wasn’t able to join us
# https://trello.com/c/tUYYbWKd/100-get-update-on-ties-task-since-christine-wasnt-able-to-join-us
# 
# - consolidate channels
# https://trello.com/c/kD1F5T09/102-consolidate-channels

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/7t-pIX1c9Gs?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Mar 28, 2020 - Call Summary  <a id="mar28.2020"></a>
# 
# **Agenda**
# (5 min) Discuss current blockers and how to make progress within individual tasks
# (5 min) NLP needs for each of the tasks (similar patterns, modular pieces like NER)
# (5 min) Discuss medical expert integration and update from Natalie/Steve?
# (5 min) Discuss visual onboarding for new people
# https://trello.com/c/JPSQpU39/99-visual-guide-to-what-to-do-based-on-personas
# 
# **Action items**
# - visual onboarding (need to have something done today) and this will hopefully solve the information overload problem too
# https://trello.com/c/JPSQpU39/99-visual-guide-to-what-to-do-based-on-personas
# 
# - figuring out shared modular NLP needs document 
# https://trello.com/c/D3R39WL1/105-figuring-out-shared-modular-nlp-needs-document
# 
# - figuring out connector human resources
# https://trello.com/c/kDWYX1oZ/106-figuring-out-connector-human-resources
# 
# - figuring out second in command for each task
# https://trello.com/c/KayDcDBV/107-figuring-out-second-in-command-for-each-task
# 
# - figuring out what’s up with TIES task (either reformulate or find leader)
# https://trello.com/c/obbO9HG3/108-figuring-out-whats-up-with-ties-task

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/7mCEW21LU_o?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Mar 29, 2020 <a id="mar29.2020"></a>
# 
# **Agenda**
# * (5 min) Discuss current blockers and how to make progress within individual tasks
# 
# **Team reporting**
# - high level progress (quick summary)
# - time to results (how soon can you show existing progress externally)
# - blockers (what you need help with)
# 
# 
# 
# * (1 min) Task: Risk Factors (Mayya)
# * (1 min) Task: Geo (Daniel)
# * (1 min) Task: Transmission (Christine?)
# * (1 min) Task: Vaccines (Dan Sosa)
# 
# * (5 min) Discuss medical expert integration and update from Natalie/Steve?
# * (5 min) Current organizational challenges/resource needs

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/xhowWO60-9Y?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Mar 30, 2020 <a id="mar30.2020"></a>
# 
# **Agenda**
# * (5 min) Discuss current blockers and how to make progress within individual tasks
# * (5 min) Individual team reports
# * (5 min) Preparing output for the kaggle summary page on the progress so far
# * (10 min) Discuss medical expert integration progress
# * (5 min) Data-viz update and Team Best-practices update (see below)
# 
# **Action items**
# * talk to these people for these ongoing actions
# * task-transmission blockers: help with NLP and in general (lead: Christine Chen)
# * protocol requests
# * * github permissions for teams (manuel, michael, anton)
# * * datasets, improve documentation (mayya)
# * * slack hygeine, careful with \@channel and \@leader (mark_k, daniel, tina)

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/T9v0l7yTOnA?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Mar 31, 2020 <a id="mar31.2020"></a>
# 
# **Agenda**
# 
# * (5 min) Saving Brandon Mission
# https://drive.google.com/uc?id=1_reCmfPpU0OfiasKWzaMr0mVDxQPAZ1l
# 
# * (5 min) Discuss current blockers and how to make progress within individual tasks
# 
# * (1 min) Task: Risk Factors (Mayya)
# * (1 min) Task: Geo (Daniel)
# * (1 min) Task: Transmission (Christine)
# * (1 min) Task: Vaccines (Dan Sosa)
# 
# * (5 min) Discuss medical expert integration and update from Natalie/Steve?
# * (5 min) Current organizational challenges/resource needs
# 
# * (10 min) Q&A and further feedback integration on what we can do better as a group
# 
# 
# **Action items**
# - watch this video and listen for your favorite people.
# - drop your 4th - n priorities.
# - join #NLP-stack if any part of your talent or workflow pertains NLP, and take something off @Brandon Eychaner’s plate.
# - help triage the main trello board.
# - stop asking team leads to share a document on your favorite content-management list. do it for them.
# - maintain your own documents. label archived, updated, etc.
# - if your favorite action item isn’t here, tell @Artur Kiulian or @Mark_k and [sign up to be a note-taker](https://trello.com/c/qHSxPmit) for an upcoming meeting so it doesn’t happen again.
# - @Mark_k is officially dropping this ball. that means he’s not doing action items anymore.

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/1hWwVQqrAEQ?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Apr 1, 2020 <a id="apr1.2020"></a>
# 
# **Agenda**
# 
# * (5 min) update on #savingbrandon mission
# * (5 min) FAQ on CoronaWhy CORD-19 dataset
# https://trello.com/c/N2U0YBL3/126-coronawhy-cord-19-dataset
# 
# * (5 min) Discuss current blockers and how to make progress within individual tasks
# 
# * (1 min) Task: Risk Factors (Mayya)
# * (1 min) Task: Geo (Daniel)
# * (1 min) Task: Transmission (Christine)
# * (1 min) Task: Vaccines (Dan Sosa)
# 
# * (5 min) Discuss medical expert integration and update from Natalie/Steve?
# * (5 min) Current organizational challenges/resource needs
# 
# 
# 
# 

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/qxOI_xBq890?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Apr 2, 2020 <a id="apr2.2020"></a>
# 
# **Agenda**
# 
# * (5 min) New trello board structure intro/slack groups (Daniel)
# https://trello.com/b/y4odX7yZ/coronawhy-main
# 
# * (1 min) Quite note for team roster changes (no emails exposed from now on)
# 
# * (2 min) Potential for a non-Kaggle task that can be added to #task-vt scope of work via independent team, Artur had two calls about it and overall feedback from everyone makes sense to kick it off, need a team lead that is not Artur
# https://trello.com/c/jbonaiUt/143-side-effects-of-the-proposed-treatments
# 
# * (5 min) Discuss current blockers and how to make progress within individual tasks (Admin team)
# 
# * (1 min) Task: Risk Factors (Mayya)
# * (1 min) Task: Geo (Daniel)
# * (1 min) Task: Transmission (Christine)
# * (1 min) Task: Vaccines (Dan Sosa)
# 
# * (5 min) Discuss medical expert integration into specific tasks and update from Natalie/Steve?
# * (5 min) Current organizational challenges/resource needs
# 
# * (5 min) Q&A and further feedback integration on what we can do better as a group
# 
# * Action items
# - NEED HELP FROM SOMEONE TO TRANSFORM NOTES INTO ACTION ITEMS FOR CORRESPONDING TEAM @communications_team @group_pm

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/FSoqy5f6qQI?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Apr 3, 2020 <a id="apr3.2020"></a>
# 
# **Agenda**
# 
# **Timestamps**
# 
# * 00:53 - Amazing progress of Vaccines research team - You can check out which and how specific drugs are being covered in the WhiteHouse CORD-19 dataset.
# 
# * 05:05 - Latest update on Risk Factors, Geo, Transmission and Vaccines teams progress.
# 
# * 11:42 - Insights about achievements on NLP tasks.
# 
# * 12:48 - Our ideas on how to organize/manage medical experts integration into specific tasks.
# 
# * 16:00 - Successful presentation of CoronaWhy progress to internal Deloitte division with potential allocation of human resources

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/fCjVfHZfWys?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Apr 7, 2020 <a id="apr7.2020"></a>
# 
# Summary:
# * 0:07 - Youtube Annotating and Recording Calls
# * 1:57 - Why are we here Video Promo
# * 4:00 - Onboarding Newcomers and Coordinators (flowchart in progress)
# * 8:41 - Communications Update (outreach, presentation materials)
# * 11:03 - Team Report: Risk Factors (@Mayya)
# * 13:49 - Team Report: Task Geo (@Daniel Robert-Nicoud)
# * 14:55 - Team Report: Transmission (@Christine Chen)
# * 16:57 - Team Report: Vaccines/Therapeutics (@Dan Sosa)
# * 17:43 - IMPORTANT: An interesting approach on analyzing papers (could be very useful for other tasks as well)
# * 21:02 - The need for annotators with a medical background
# * 23:31 - Summary and Action Items
# 
# **Action Items**
# * Need help for “Why are we here” Video Promo
# https://trello.com/c/JLufinRP/163-need-help-for-why-are-we-here-video-promo
# * Finish up and upload the Onboarding Flowchart
# https://trello.com/c/17ihhNFc/164-finish-and-upload-onboarding-flowchart
# * Find Connectors to pass people from onboarding to team coordinator
# https://trello.com/c/l6hLJ5vJ/165-find-connectors-to-lead-newcomers-to-team-coordinators
# * Make a directory for presentations to other organizations
# https://trello.com/c/QVu1uzEW/166-make-a-directory-for-presentations-to-other-organizations
# * For each team: come up with a worst case scenario for Kaggle submission (find the bottlenecks)
# https://trello.com/c/8VHWJMhr/167-draft-of-worst-case-scenario-for-kaggle-submission-by-team
# * Dedicated NLP person for #task-ties
# https://trello.com/c/V9k0YpqO/170-find-dedicated-nlp-person-for-task-ties
# * NLP person with BERT experience needed for #task-vt
# https://trello.com/c/1KLu1uFi/168-find-nlp-person-with-bert-experience-for-task-vt
# * Paper annotators for drug correlation
# https://trello.com/c/kNFtbHG1/169-gather-paper-annotators-for-drug-correlation
# 
# 

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/e7RgYGQf_Lw?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Apr 8, 2020 <a id="apr8.2020"></a>
# 
# 
# **Summary**
# 
# * 0:00 - Marketing needs and Google Cloud credits
# * 1:34 - Onboarding coordinators vs. team advisors
# * 3:22 - HR Challenges and Team Needs
# * 4:09 - Process of Onboarding People for Tasks
# * 8:40 - Video Call Onboarding instead of just text-based
# * 9:39 - Team Report: Risk Factors (@Mayya)
# * 11:34 - Team Report: Task Geo (@Daniel Robert-Nicoud)
# * 13:15 - Team Report: Task Transmission (@Christine Chen)
# * 15:18 - Update on NLP Stack
# * 16:52 - Team Report: Vaccines/Therapeutics (@Dan Sosa)
# * 18:27 - SPECIAL: Dan Sosa’s presentation for onboarding to tasks
# * 20:04 - Trello (and in general, CoronaWhy) Guidelines
# * 24:09 - Introduction channel vs. general channel
# * 25:00 - Slack Guidelines: here, channel, and everyone
# * 27:10 - Action Points
# 
# **Action items**
# 
# * Onboarding orientation process (@Augaly S. Kiedi)
# https://trello.com/c/aBOH5uAi/173-create-an-onboarding-orientation-process
# * Onboarding new members for #task-vt (@Daniel Lindenberger)
# https://trello.com/c/C9esAGJz/174-onboard-new-members-for-task-vt
# * Move Dan’s onboarding presentation to a Google directory (@Dan Sosa)
# https://trello.com/c/OWpPxxyJ/175-move-onboarding-presentation-to-google-directory
# 

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/8jq92pOlLJk?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Apr 9, 2020 <a id="apr9.2020"></a>
# 
# **Summary**
# 
# * 0:00 - Challenges with Onboarding
# * 5:44 - Onboarding coordinators vs. team coordinators and types of people who we try to onboard
# * 9:22 - More different approaches that we could chase, but don’t have the bandwidth yet
# * 10:28 - HR and Team Needs
# * 12:21 - Team Report: Risk Factors (@Mayya)
# * 14:37 - Team Report: Task Geo (@Daniel Robert-Nicoud)
# * 17:34 - Team Report: Task Transmission (@Christine Chen)
# * 19:18 - Team Report: Vaccines/Therapeutics (@Shannon C-W on behalf of @Dan Sosa)
# * 20:46 - What will we do post-Kaggle?
# * 21:38 - Main Trello Maintenance and ToDos
# * 23:32 - Updates on Calendar and Meeting Scheduling
# * 25:57 - Summary and Action Items
# 
# **Action Items**
# * Find a Python developer to help maintain GitHub repo and validate PR’s for task-geo
# https://trello.com/c/TEQDpQ8q/177-find-python-developer-to-maintain-github-repo
# * Find Domain Experts and Epidemiologists (medical experts) for task-risk
# https://trello.com/c/FyeIL6SR/178-find-domain-experts-epidemiologists-for-task-vt
# * Find a coordinator to maintain to-do Trello Board tasks
# https://trello.com/c/bDZf4MFt/179-find-someone-to-maintain-tasks-on-trello

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/2-ZytfPxnds?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Apr 10, 2020 <a id="apr10.2020"></a>
# 
# **Summary**
# 
# * 0:43 - Establish workings of our organizations (our vision)
# * 3:13 - Onboarding coordinator and team coordinator progress
# * 7:41 - HR and Team Needs
# * 9:31 - Emergence of the new task datasets
# * 11:38 - Podcast Ideation
# * 15:13 - Team Report: Risk Factors (@Mayya)
# * 17:14 - Team Report: Task Geo (@Manuel Alvarez on behalf of @Daniel Robert-Nicoud)
# * 20:28 - Team Report: Task Transmission (@Christine Chen)
# * 22:40 - Team Report: Vaccines/Therapeutics (@Dan Sosa)
# * 24:09 - Creating a Slack channel for team leader needs
# * 25:06 - Slack Channel and Direct Message Organization (@Tyler)
# * 31:34 - Summary and Action Items
# 
# **Action Items**
# - Create slack channel for team needs (I’m creating one called #team-needs, will start adding people shortly)
# https://trello.com/c/7n0idxv8/182-create-slack-channel-for-needs-of-team-leaders
# - Update Slack guidelines with Tyler’s suggestions
# https://trello.com/c/XAnSMzuJ/184-update-slack-guidelines-with-tyler-suggestions

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/8KVC9ORmfss?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Apr 11, 2020 <a id="apr11.2020"></a>
# 
# **Summary**
# 
# * 0:50 - Quick Wins and Sponsorships
# * 2:12 - Brainstorming on our mission and core values
# * 3:02 - Onboarding Coordinators and Team Coordinator Progress
# * 5:56 - CoronaWhy and Work
# * 7:39 - Need for medical community coordination
# * 10:00 - HR and Team Needs
# * 11:10 - Collisions between different Onboarding Processes
# * 16:57 - Team Report: Risk Factors (@Mayya)
# * 18:39 - Team Report: Task Geo (@Daniel Robert-Nicoud)
# * 20:05 - Team Report: Task Transmission (@Christine Chen)
# * 21:46 - Team Report: Vaccines/Therapeutics (@Dan Sosa)
# * 23:01 - When we can see the first results
# 
# 
# **Action Items**
# - Create a task for submissions from all 4 tasks
# https://trello.com/c/zCzVeXcH/186-create-a-submissions-trello-for-all-tasks

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/1xEYr2j3a2U?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Apr 12, 2020 <a id="apr12.2020"></a>
# 
# **Summary**
# 
# * 0:38 - Focusing on First Kaggle Submission
# * 3:48 - Task Geo structured as a Horizontal Team rather than a separate task
# * 5:48 - Onboarding Coordinators and Team Coordinators (esp with the medical community)
# * 10:38 - HR and Team Needs
# * 14:17 - Communications Update
# * 15:51 - Crowdsourcing the effort to establish core values
# * 18:51 - NEW! Team Report: Task Datasets (@Brandon Eychaner)
# * 23:18 - Team Report: Risk Factor (@Mayya)
# * 24:32 - Team Report: Task Geo (@Daniel Robert-Nicoud)
# * 26:16 - Team Report: Task Transmission (@Christine Chen)
# * 27:14 - Team Report: Vaccines/Therapeutics (@Dan Sosa)
# * 28:15 - Q/A: Location of the Team Roster/Database
# * 30:08 - Q/A: Editing Discrepancies in Google Calendar
# * 31:29 - Central Document Directory Structure
# 
# **Action Items**
# - Google Doc to Promote Webinar about Ideation of Post-Kaggle work (@Tyler, @Augaly S. Kiedi)
# https://trello.com/c/g8YOZ5Tu/187-promo-flyer-for-webinar-on-post-kaggle-discussion
# - Push a list of the “skills matrix” for all teams
# https://trello.com/b/y4odX7yZ/coronawhy-main
# - Feedback for Tyler’s core values draft
# https://trello.com/c/mX3NKkaG/188-feedback-for-tylers-core-values-draft
# 

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/yDc0IRCxsSY?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Apr 13, 2020 <a id="apr13.2020"></a>
# 
# **Summary**
# 
# * 0:39 - Focus on Kaggle Submission
# * 1:18 - Onboarding/Team/Medical Coordinators
# * 5:28 - HR Challenges and Team Needs
# * 8:26 - Communications Update: Podcast/Website
# * 9:15 - A new understanding of Core Values
# * 11:00 - Team Report: Risk Factors (@Mayya)
# * 13:18 - Team Report: Task Geo (@Daniel Robert-Nicoud)
# * 15:58 - Team Report: Task Transmission (@Christine Chen)
# * 16:49 - Team Report: Vaccines/Therapeutics (@Dan Sosa)
# * 17:15 - Team Report: Datasets (needs a coordinator)
# * 20:40 - Call with Professor Stibe (Smart Cities)
# * 24:51 - Calendar Updates
# * 25:51 - Onboarding Updates
# * 27:03 - Access to Final Kaggle Notebook Submission
# * 27:41 - Action Items/Summary
# 
# **General Action Items**
# * Make an org chart (@Tyler)
# https://trello.com/c/VHwXsFpB/189-make-a-coronawhy-org-chart
# * Grant #task-risk the permissions to Final Kaggle Notebook (@Artur Kiulian)
# https://trello.com/c/nYqMEe48/190-grant-permissions-to-final-kaggle-notebook

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/GXEC9_4XfFQ?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Apr 15, 2020 <a id="apr15.2020"></a>
# video recording: https://www.youtube.com/watch?v=I0gHkIeP2Hw
# (you can watch at 2x speed)

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/I0gHkIeP2Hw?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# #### Call Summary - Apr 16, 2020 <a id="apr16.2020"></a>
# 
# video recording: https://www.youtube.com/watch?v=8iKFz7BIHB8
# (you can watch at 2x speed)

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/8iKFz7BIHB8?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# ### general - Apr 18, 2020 - daily call
# 
# **Summary:**
# * 3:29 - Progress with the search engine
# * 8:02 - Many different/diverse arenas of study
# * 8:48 - Presenting our results to the medical community (through webinar)
# * 11:00 - What is our final product?
# * 15:00 - Our work compared to the creation of Google (a science experiment)
# * 18:19 - Selective search based on the audience
# * 20:35 - A Human input driven system
# * 22:08 - How to crowdsource our need for human input
# * 26:11 - We need a solid structure (outline)
# * 29:52 - Picture CoronaWhy as a guild
# * 34:51 - The benefits of CoronaWhy gamification
# * 37:00 - Guild Setup: Getting Skills Information from People
# * 40:57 - The Social Aspect of CoronaWhy
# * 45:03 - Interview with AI2
# * 46:29 - Potential to impact policy
# * 50:54 - Participatory Economics Channel
# 
# Action Items:
# - Work on interview document with AI2

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/019mZXdcLow?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# ### #general - April 20, 2020 - Daily Stand-up
# 
# **Summary:**
# * 0:00 - The study of the CoronaWhy ant colony
# * 3:00 - Direction: Post-Kaggle submission
# * 7:20 - The product that we are building (AI literature review)
# * 8:38 - Understand the ontology of the actual data in CORD-19
# * 12:05 - The importance of data enrichment and preprocessing
# * 13:28 - A Round 1 Assessment
# * 14:09 - More people can make a better infrastructure
# * 15:52 - The Kaggle CORD-19 dataset
# * 18:48 - Where and how do people engage?
# * 21:12 - Can we publish a scientific paper of our research?
# * 25:23 - How do the vertical teams proceed?
# * 28:01 - The Search process: Integrating vertical teams with the NLP search engine team
# * 31:28 - Structuring of the teams to prepare for Round 2
# * 33:13 - Andrea’s anonymous roster (matching skills with tasks)
# * 36:36 - Team Reporting: Vaccines/Therapeutics (@Dan Sosa)
# * 37:38 - Team Reporting: Task Transmission (@Christine Chen)
# * 40:00 - Team Reporting: Risk Factors (@Mayya)
# * 43:56 - Team Reporting: Task Geo (@Manuel Alvarez)
# * 46:16 - A tighter schedule for the general call
# 
# 

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/ZYGKapsIyMw?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# ### #general - April 23, 2020 - Daily Stand-up
# 
# **Summary:**
# * 2:32 - Zooming into teams so that everyone can understand their functionality
# * 5:39 - Daily call is just a glimpse
# * 7:34 - Our webinar and inviting medical experts
# * 8:36 - The Converger’s call
# * 9:26 - Team Reporting: Risk Factors (@Mayya)
# * 12:07 - Software Infrastructure Channel
# * 15:19 - Team Reporting: Task Transmission (@Christine Chen)
# * 18:14 - Team Reporting: Task Geo (@Manuel Alvarez)
# * 20:18 - Task Geo’s team structure and next steps
# * 23:00 - Splitting teams into fractals (ex. VT and future Geo)
# * 28:34 - What is task-geo’s common theme? Data Infrastructure
# * 32:48 - Team Reporting: Vaccines/Therapeutics (@Shannon Cahill-Weisser on behalf of @Dan Sosa)
# * 33:57 - We need medical experts
# 

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/oIT7lMDiVe0?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# ### #general - April 24, 2020 - Daily Stand-up
# 
# **Summary:**
# 

# In[ ]:



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/8xRaWGJGSEw?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# # NEXT CALL
# 
# 10am PST
