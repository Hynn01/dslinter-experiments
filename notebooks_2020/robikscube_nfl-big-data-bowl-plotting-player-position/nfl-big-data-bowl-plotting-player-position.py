#!/usr/bin/env python
# coding: utf-8

# # NFL Big Data Bowl
# ## Plotting the field
# 
# Using some code that I developed for last year's NFL challenege, this notebook shows how to plot player positions during a play on the football field.
# 
# Using matplotlib we can call the `create_football_field` function to create a figure with the football field drawn out. You can then overlay any information from the training data to help visualize how players positions look on the field.
# 
# The design is loosely based off of the 1991 video game [Techo Super Bowl](https://en.wikipedia.org/wiki/Tecmo_Super_Bowl). A game which I spent many hours playing in my next door neighbor's basement growing up (I wasn't allowed to own a video game console so we had to play at his house).
# ![](https://upload.wikimedia.org/wikipedia/en/d/d7/Tecmosuperbowl.png)
# 

# In[ ]:


import pandas as pd
import numpy as np
import os
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as patches
pd.set_option('max_columns', 100)

train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
train2021 = pd.read_csv('../input/nfl-big-data-bowl-2021/plays.csv')


# ## Function to Create The Football Field

# In[ ]:


def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
    return fig, ax

create_football_field()
plt.show()


# # Adding Players For a Play

# In[ ]:


fig, ax = create_football_field()
train.query("PlayId == 20170907000118 and Team == 'away'")     .plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=30, legend='Away')
train.query("PlayId == 20170907000118 and Team == 'home'")     .plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=30, legend='Home')
plt.title('Play # 20170907000118')
plt.legend()
plt.show()


# # Highlight the line of scrimmage

# In[ ]:


playid = 20181230154157
train.query("PlayId == @playid").head()


# In[ ]:


yl = train.query("PlayId == @playid")['YardLine'].tolist()[0]
fig, ax = create_football_field(highlight_line=True,
                                highlight_line_number=yl+54)
train.query("PlayId == @playid and Team == 'away'")     .plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=30, legend='Away')
train.query("PlayId == @playid and Team == 'home'")     .plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=30, legend='Home')
plt.title(f'Play # {playid}')
plt.legend()
plt.show()


# # On 2021 Datset
# - Note in the 2021 dataset some things have changed.
#     - The `X` and `Y` columns are now lowercase `x` and `y`
#     - Data is provided for the entire play, not just one time within a play.
#     - PlayId is not unique across games, only within the game. So for unique ids you need to use a combination of gameId and playId

# In[ ]:


train2021 = pd.read_csv('../input/nfl-big-data-bowl-2021/week11.csv')

example_play_home = train2021.query('gameId == 2018111900 and playId == 5577 and team == "home"')
example_play_away = train2021.query('gameId == 2018111900 and playId == 5577 and team == "away"')

fig, ax = create_football_field()
example_play_home.query('event == "ball_snap"').plot(x='x', y='y', kind='scatter', ax=ax, color='orange', s=30, legend='Away')
example_play_away.query('event == "ball_snap"').plot(x='x', y='y', kind='scatter', ax=ax, color='blue', s=30, legend='Home')
plt.title('Game #2018111900 Play #5577 at Ball Snap')
plt.legend()
plt.show()


# In[ ]:


fig, ax = create_football_field()
example_play_home.plot(x='x', y='y', kind='scatter', ax=ax, color='orange', s=30, legend='Away')
example_play_away.plot(x='x', y='y', kind='scatter', ax=ax, color='blue', s=30, legend='Home')
plt.title('Game #2018111900 Play #5577 at Ball Snap')
plt.legend()
plt.show()

