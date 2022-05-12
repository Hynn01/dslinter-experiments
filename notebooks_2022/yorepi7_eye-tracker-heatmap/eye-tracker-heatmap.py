#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from matplotlib import animation

def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

fig = plt.figure()

def init():
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    img, extent = myplot(x, y, 50)
    plt.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
    
def animate(i):
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    img, extent = myplot(x, y, 50)
    plt.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
anim = animation.FuncAnimation(fig, animate, init_func = init, frames=60, 
                               interval=30, repeat = False)

anim.save('animation.mp4', fps=1.0, dpi=200)
plt.close()

