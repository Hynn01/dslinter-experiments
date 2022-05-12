#!/usr/bin/env python
# coding: utf-8

# I have exported the animation methods into a Python file.
# 
# Here we generate the animation given an episode ID. Animating takes between 2 to 5 minutes.
# 
# You can generate also the animation given the replay json file, or a simulated match between two agents.
# 
# See source [data source](https://www.kaggle.com/code/huikang/kore-2022-match-analysis) for code and examples. See [discussion](https://www.kaggle.com/competitions/kore-2022/discussion/320987) for explanations and future plans.
# 
# Feel free to attach this animation at the end of your notebooks.

# In[ ]:


get_ipython().system('cp ../input/kore-2022-match-analysis/kore_analysis.py .')


# In[ ]:


from kore_analysis import KoreMatch, load_from_simulated_game, load_from_replay_json, load_from_episode_id


# In[ ]:


env = load_from_episode_id(36519532)


# In[ ]:


kore_match = KoreMatch(env.steps)


# In[ ]:


kore_match.animate()


# In[ ]:


kore_match.html_animation

