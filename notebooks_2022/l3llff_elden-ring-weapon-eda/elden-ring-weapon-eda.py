#!/usr/bin/env python
# coding: utf-8

# ![](https://static0.srcdn.com/wordpress/wp-content/uploads/2022/02/Elden-Ring-Berserk-Easter-Egg-Guts-Greatsword.jpg)
# 
# 
# <center><h1>⚔️ Elden Ring Weapons EDA ⚔️</h1></center>
# 
# # Introduction
# 
# **ELDEN RING** was created under the guidance of FromSoftware’s Hidetaka Miyazaki, incorporating the expansive worldview through the eyes of George R. R. Martin, author of the fantasy novel series, "A Song of Ice and Fire," while applying FromSoftware’s expertise in developing pulse pounding, action based titles, and with the game’s vast and explorative world filled with excitement and adventure, customers worldwide have been showering the game with praise for its unprecedented and overwhelming gameplay experience. Simultaneous release of the game in fourteen languages, coupled with the worldwide network tests prior to release, raised expectations to bring about sales of more than 12M units worldwide.
# 
# > **💡 Goal**: In the game world you can find up to 300 unique weapons. The aim of this notebook is too analyse weapons stats and probably find the best one. 
# 

# In[ ]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Color palette
my_colors = ["#1a4441", "#2b5553", "#e6bf7b", "#c8a36c", "#936747"]
sns.palplot(sns.color_palette(my_colors))

# Set Style
sns.set_style("white")
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

df = pd.read_csv('../input/-elden-ring-weapons/elden_ring_weapon.csv')
df['Phy'] = df['Phy'].replace('-', '0').astype('int')


# #### Helper Functions

# In[ ]:


def show_values_on_bars(axs, h_v="v", space=0.4):
    '''Plots the value at the end of the a seaborn barplot.
    axs: the ax of the plot
    h_v: weather or not the barplot is vertical/ horizontal'''
    
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, format(value, ','), ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, format(value, ','), ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


# ### 1. Weapon Types ⚔️

# In[ ]:


# Plot
fig, ax = plt.subplots(figsize=(25, 13))
plt.title('⚔️ Most Frequent Weapon Types ⚔️', size=22, weight="bold")

sns.barplot(data=df['Type'].value_counts().reset_index(), x='Type', y='index', palette=my_colors[1:3]);

show_values_on_bars(axs=ax, h_v="h", space=0.1)
plt.ylabel("Weapon Type", size = 16, weight="bold")
plt.xlabel("")
plt.xticks([])
plt.yticks(size=16)
plt.tick_params(size=16)


# Here're some examples of the most common weapon types.
# 
# |Straight Sword|Greatsword|Glintstone Staff|Halberd|Spear|
# |--:--|--:--|--:--|--:--|--:--|
# |<img src="https://i.ibb.co/h8kM4gF/ER-Icon-weapon-Broadsword.png" width="400">|<img src="https://i.ibb.co/PtFPpgr/ER-Icon-weapon-Dark-Moon-Greatsword.png" width="400">|<img src="https://i.ibb.co/MSfDCmn/ER-Icon-weapon-Azur-s-Glintstone-Staff.png" width="400">|<img src="https://i.ibb.co/b1x222V/ER-Icon-weapon-Golden-Halberd.png" width="400">|<img src="https://i.ibb.co/tJb6HX1/ER-Icon-weapon-Short-Spear.png" width="400">|
# 
# |Dagger|Hammer|Colossal Weapon|Curved Sword|Warhammer|
# |--:--|--:--|--:--|--:--|--:--|
# |<img src="https://i.ibb.co/TvRs9jf/ER-Icon-weapon-Reduvia.png" width="400">|<img src="https://i.ibb.co/4Mhbmvc/ER-Icon-weapon-Morning-Star.png" width="400">|<img src="https://i.ibb.co/P9DP1h0/ER-Icon-weapon-Prelate-27s-Inferno-Crozier.png" width="400">|<img src="https://i.ibb.co/GpfcwQd/ER-Icon-weapon-Magma-Blade.png" width="400">|<img src="https://i.ibb.co/mGRVVHm/ER-Icon-weapon-Cranial-Vessel-Candlestand.png" width="400">|
# 
# |Axe|Greataxe|Colossal Sword|Curved Greatsword|Fist|
# |--:--|--:--|--:--|--:--|--:--|
# |<img src="https://i.ibb.co/FWb5RtP/ER-Icon-weapon-Icerind-Hatchet.png" width="400">|<img src="https://i.ibb.co/RvrJmPp/ER-Icon-weapon-Axe-of-Godrick.png" width="400">|<img src="https://i.ibb.co/fkk6cFP/ER-Icon-weapon-Grafted-Blade-Greatsword.png" width="400">|<img src="https://i.ibb.co/QX6vsxR/ER-Icon-weapon-Bloodhound-2527s-Fang.png" width="400">|<img src="https://i.ibb.co/GV6N2rv/ER-Icon-weapon-Spiked-Caestus.png" width="400">|
# 

# ### 2. Weapons Weight 🗿
# 
# The distribution of weapons weight looks like exponential one. There're tons of quite light weapons in the game.

# In[ ]:


fig, ax = plt.subplots(figsize=(25, 13))
sns.histplot(data=df, x='Wgt', color=my_colors[0]);

plt.ylabel("Count", size = 16, weight="bold")
plt.xlabel("Weight", size = 16, weight="bold")
plt.yticks(size=16)
plt.tick_params(size=16)


# Let's see how the heaviest weapons looks like:

# In[ ]:


df[df['Wgt'] == df['Wgt'].max()]


# ![](https://i2.wp.com/assets.gamepur.com/wp-content/uploads/2022/04/22193303/elden_ring_giant_crusher-1.jpg)

# ### 3. Damage 🗡
# Let's have a look at weapons physical damage distribution.

# In[ ]:


fig, ax = plt.subplots(figsize=(25, 13))
sns.histplot(data=df, x='Phy', color=my_colors[3]);

plt.ylabel("Count", size = 16, weight="bold")
plt.xlabel("Physical Damage", size = 16, weight="bold")
plt.yticks(size=16)
plt.tick_params(size=16)


# It looks like we have two outliers there. Let's look closer...

# In[ ]:


df[df['Phy'] == df['Phy'].max()]


# Well, that's a cannon :)

# ![](https://i1.wp.com/static0.gamerantimages.com/wordpress/wp-content/uploads/2022/04/elden-ring-jar-cannon.jpg?resize=1140,570)

# **Notebook in progress**...

# **P.S. thank you kindly for the incredible notebooks** @andradaolteanu 
