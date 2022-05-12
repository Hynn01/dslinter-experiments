#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore", UserWarning)

csv_files = []
for dirname, _, filenames in os.walk('../input/clustering-exercises'):
    for filename in sorted(filenames):
        csv_files.append(os.path.join(dirname, filename))


# # Utils

# In[ ]:


def invert_rgb(h):
    rgb = [h[1:3], h[3:5], h[5:]]
    rgb = list(map(lambda x: int(255 - int(x, 16)), rgb))
    return '#' + ''.join(map('{:02x}'.format, rgb)).upper()


# In[ ]:


import matplotlib as mpl

def color_gradient(c1, c2, mix_rate=0.0):
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix_rate)*c1 + mix_rate*c2)


# In[ ]:


def create_n_colors(n):
    a = []
    for i in range(n):
        r = i / n
        c = ''
        if r < 1 / 3:
            c = color_gradient('red', 'blue', r * 3)
        elif r < 2 / 3:
            c = color_gradient('blue', 'green', (r - 1/3) * 3)
        else:
            c = color_gradient('green', 'red',  (r - 2/3) * 3)
        a.append(c)
    return a


# In[ ]:


def plot_df(df, colors, **kwargs):
    cmap = create_n_colors(len(set(colors)))
    c = list(map(lambda x: cmap[x], colors))
    return sns.scatterplot(data=df, x='x', y='y', c=c, **kwargs)


# In[ ]:


def sec(start_time, end_time):
    diff = end_time - start_time
    s, ms = diff.seconds, diff.microseconds
    return f'{s}.{str(ms)[:2]}'


# In[ ]:


import math
from datetime import datetime

def benchmark(files, suptitle, callback, figsize=(5, 5), **kwargs):
    COLS = 6
    ROWS = math.ceil(len(files) / COLS)
    ROWS += int(ROWS == 1)
    fig, axes = plt.subplots(ROWS, COLS, figsize=figsize)
    plt.suptitle(suptitle, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(**kwargs)
    mean_times = []
    for i, csv in enumerate(files):
        r, c = i // COLS, i % COLS
        ts = datetime.now()
        items, tm = callback(pd.read_csv(csv), ax=axes[r][c])
        if type(tm) == type(None):
            tm = sec(ts, datetime.now())
        else:
            tm = sec(tm[0], tm[1])
        title = axes[r][c].get_title()
        axes[r][c].set_title(title+'\n'+f'{items} items ({tm}sec)')
        mean_times.append(items / float(tm))
    print(suptitle, 'process average {:.2f} item(s)/sec'.format(np.mean(mean_times)))
    for ax in axes.flatten():
        ax.axis('off')
    fig.show()


# # Preview a sample

# In[ ]:


df = pd.read_csv(csv_files[0])


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, n_cluster in enumerate([2, 3, 4, 5]):
    kmeans = cluster.KMeans(n_clusters=n_cluster).fit(df[['x', 'y']])
    axes[i//2][i%2].set_title(f'KMeans(n_clusters={n_cluster})')
    plot_df(df, kmeans.labels_, ax=axes[i//2][i%2])
fig.show()


# # Labels on Dataset

# In[ ]:


def label_answer(df, ax):
    labels = df['color']
    answer = len(set(labels))
    plot_df(df, labels, ax=ax, s=5)
    info = f'cluster: {answer}'
    ax.set_title(info)
    return len(df), None

benchmark(csv_files, 'Answer Labels', label_answer, figsize=(14, 16), wspace=0.2, hspace=0.4, top=0.9)


# # AffinityPropagation
# 
# This takes too looooong time, so removed from this kernel.

# In[ ]:


# %%time

# def affinity(df, ax):
# #     df = df.sample(frac=0.5)
#     answer = len(df['color'].unique())
#     labels = cluster.AffinityPropagation().fit_predict(df[['x', 'y']])
#     label_count = len(set(labels))
#     plot_df(df, labels, ax=ax, s=5)
#     info = f'cluster: {label_count}'
#     ax.set_title(info)
#     return len(df), None

# benchmark(csv_files, 'Affinity Propagation', affinity, figsize=(14, 16), wspace=0.2, hspace=0.4, top=0.9)


# # K-Means

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef kmeans(df, ax):\n#     df = df.sample(frac=0.5)\n    answer = len(df['color'].unique())\n    labels = cluster.KMeans(n_clusters=answer).fit_predict(df[['x', 'y']])\n    label_count = len(set(labels))\n    plot_df(df, labels, ax=ax, s=5)\n    info = f'cluster: {label_count}'\n    ax.set_title(info)\n    return len(df), None\n\nbenchmark(csv_files, 'K Means', kmeans, figsize=(14, 16), wspace=0.2, hspace=0.4, top=0.9)")


# # DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef dbscan(df, ax):\n#     df = df.sample(frac=0.5)\n    answer = len(df['color'].unique())\n    l, r = [1e-6, 1e2]\n    while r - l > 1e-6:\n        eps = (l + r) / 2\n        start = datetime.now()\n        labels = cluster.DBSCAN(eps=eps, min_samples=2, leaf_size=30).fit_predict(df[['x', 'y']])\n        end = datetime.now()\n        label_count = len(set(labels))\n        if label_count < answer:\n            r = eps - 1e-6\n        else:\n            l = eps + 1e-6\n    plot_df(df, labels, ax=ax, s=5)\n    info = f'cluster: {label_count}\\n' + \\\n           'eps: {:.1f}'.format(eps)\n    ax.set_title(info)\n    return len(df), (start, end)\n    \n\nbenchmark(csv_files, 'DBSCAN', dbscan, figsize=(14, 16), wspace=0.2, hspace=0.5, top=0.9)")


# # Mean Shift

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef meanshift(df, ax):\n#     df = df.sample(frac=0.5)\n    answer = len(df['color'].unique())\n    bandwidth = []\n    for c in df.color.unique():\n        ebw = cluster.estimate_bandwidth(df.loc[df.color == c, ['x', 'y']])\n        bandwidth.append(ebw)\n    bw = np.array(bandwidth).mean()\n    labels = cluster.MeanShift(bandwidth=bw).fit_predict(df[['x', 'y']])\n    label_count = len(set(labels))\n    plot_df(df, labels, ax=ax, s=5)\n    info = f'cluster: {label_count}\\n' + \\\n           'bandwidth={:.1f}'.format(bw)\n    ax.set_title(info)\n    return len(df), None\n\nbenchmark(csv_files, 'Mean Shift', meanshift, figsize=(14, 16), wspace=0.2, hspace=0.5, top=0.9)")


# # Spectral Clustering

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef spectral(df, ax):\n#     if len(df) > 3000:\n#         df = df.sample(frac=0.5)\n    answer = len(df['color'].unique())\n    labels = cluster.SpectralClustering(n_clusters=answer, affinity='nearest_neighbors', n_init=10).fit_predict(df[['x', 'y']])\n    label_count = len(set(labels))\n    plot_df(df, labels, ax=ax, s=5)\n    info = f'cluster: {label_count}\\n' + \\\n           f'answer: {answer}'\n    ax.set_title(info)\n    return len(df), None\n\nbenchmark(csv_files, 'Spectral Clustering (discretize)', spectral, figsize=(14, 16), wspace=0.2, hspace=0.3, top=0.9)")


# # Gaussian Mixture

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn import mixture\n\ndef gaumix(df, ax):\n#     df = df.sample(frac=0.5)\n    answer = len(df['color'].unique())\n    labels = mixture.GaussianMixture(n_components=answer, max_iter=300, covariance_type='full').fit_predict(df[['x', 'y']])\n    label_count = len(set(labels))\n    plot_df(df, labels, ax=ax, s=5)\n    info = f'cluster: {label_count}'\n    ax.set_title(info)\n    return len(df), None\n\nbenchmark(csv_files, 'Gaussian Mixture', gaumix, figsize=(14, 16), wspace=0.2, hspace=0.4, top=0.9)")


# # Agglomerative Clustering

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef agglomerative(df, ax):\n#     df = df.sample(frac=0.5)\n    answer = len(df['color'].unique())\n    labels = cluster.AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto', \\\n                                             linkage='ward', n_clusters=5) \\\n                    .fit_predict(df[['x', 'y']])\n    label_count = len(set(labels))\n    plot_df(df, labels, ax=ax, s=5)\n    info = f'cluster: {label_count}'\n    ax.set_title(info)\n    return len(df), None\n\nbenchmark(csv_files, 'Agglomerative Clustering', agglomerative, figsize=(14, 16), wspace=0.2, hspace=0.4, top=0.9)")


# # OPTICS

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef optics(df, ax):\n#     df = df.sample(frac=0.5)\n    answer = len(df['color'].unique())\n    clust = cluster.OPTICS(min_samples=20, xi=0.05, metric='euclidean') \\\n                   .fit(df[['x', 'y']])\n    reachability = clust.reachability_[clust.ordering_]\n    labels = clust.labels_[clust.ordering_]\n    label_count = len(set(labels))\n    df['label'] = labels\n    df_found = df.loc[df[df.label != -1].index, ['x', 'y', 'label']]\n    plot_df(df_found, df_found['label'], ax=ax, s=5)\n    df_notfound = df.loc[df[df.label == -1].index, ['x', 'y', 'label']]\n    ax.plot(df_notfound['x'], df_notfound['y'], 'k+', alpha=0.1, markersize=3)\n    info = f'cluster: {label_count}'\n    ax.set_title(info)\n    return len(df), None\n\nbenchmark(csv_files, 'OPTICS', optics, figsize=(14, 16), wspace=0.2, hspace=0.5, top=0.9)")


# # BIRCH

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef birch(df, ax):\n#     df = df.sample(frac=0.5)\n    answer = len(df['color'].unique())\n    labels = cluster.Birch(branching_factor=200, threshold=1, n_clusters=answer) \\\n                    .fit_predict(df[['x', 'y']])\n    label_count = len(set(labels))\n    plot_df(df, labels, ax=ax, s=5)\n    info = f'cluster: {label_count}'\n    ax.set_title(info)\n    return len(df), None\n\nbenchmark(csv_files, 'BIRCH', birch, figsize=(14, 16), wspace=0.2, hspace=0.3, top=0.9)")

