#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def draw(ps):
    n = ps.shape[0]
    X, Y = [ps[n-1][0]], [ps[n-1][1]]
    for i in range(n):
        X.append(ps[i][0])
        Y.append(ps[i][1])
    plt.plot(X, Y)
    plt.show()

def work(ps, z0 = 0.5):
    n = ps.shape[0]
    z1 = (1-z0)/(2*np.cos(np.pi*2/n))
    nps = np.zeros([n, 2])
    for i in range(n):
        nps[i] = ps[i]*z0+(ps[(i+1)%n]+ps[(i-1)%n])*z1
    return nps

n = 200
ps = np.random.rand(n, 2)
ps = ps - np.mean(ps, axis=0)

for j in range(20):
    for i in range(100):
        ps = work(ps,0.2)
    draw(ps)

