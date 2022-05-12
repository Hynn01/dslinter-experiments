#!/usr/bin/env python
# coding: utf-8

# # Koreye 2022 integration
# 
# This notebook shows how to use [Koreye 2022](https://jmerle.github.io/koreye-2022/) with Kaggle notebooks.
# 
# It's pretty easy: simply copy-and-paste the following cell into your notebook and call `render_env(env, ...)` instead of `env.render(mode="ipython", ...)`. The `render_env` function shows a button to open the episode in Koreye 2022 along with the official visualizer. Additionally you can pass `open_koreye=True` to open the episode in Koreye 2022 immediately (only while editing the notebook).

# In[ ]:


import json
from IPython import get_ipython
from IPython.display import display, HTML

def render_env(env, open_koreye=False, show_koreye_button=True, show_official_viewer=True, *args, **kwargs):
    """Renders an env and includes a button to open the episode in Koreye 2022.

    :param env: the env to render
    :param open_koreye: whether the episode should be opened in Koreye 2022 immediately (only while editing the notebook)
    :param show_koreye_button: whether a button to open the episode in Koreye 2022 should be shown
    :param show_official_viewer: whether the official episode viewer should be shown
    :param *args: args passed on to env.render(mode="ipython", ...) when show_official_viewer is True
    :param **kwargs: kwargs passed on to env.render(mode="ipython", ...) when show_official_viewer is True
    """
    data_var = None
    html = ""

    if open_koreye or show_koreye_button:
        data_var = f"koreye2022Data{get_ipython().execution_count}"
        html += f"""
<script>
function openKoreye2022(data) {{
    const tab = window.open('https://jmerle.github.io/koreye-2022/kaggle', '_blank');
    for (const ms of [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 10000]) {{
        setTimeout(() => tab.postMessage(data, 'https://jmerle.github.io'), ms);
    }}
}}

{data_var} = {{
    episode: {env.render(mode="json")},
    logs: {json.dumps(env.logs)},
}};
</script>
        """

    if open_koreye:
        html += f"""
<script>
if (window.location.host.endsWith('.jupyter-proxy.kaggle.net')) {{
    openKoreye2022({data_var});
}}
</script>
        """

    if show_koreye_button:
        html += f"""
<style>
.koreye-2022-button {{
    border-radius: 18px;
    cursor: pointer;
    font-family: Inter;
    font-style: normal;
    font-weight: 500;
    font-size: 14px;
    line-height: 20px;
    height: 28px;
    padding: 0 16px;
    transition: all 0.3s ease;
    width: fit-content;
    box-sizing: content-box;
}}
</style>

<button onclick="openKoreye2022({data_var})" class="koreye-2022-button">Open in Koreye 2022</button>
        """

    if html != "":
        display(HTML(html))

    if show_official_viewer:
        env.render(mode="ipython", *args, **kwargs)


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install --upgrade kaggle-environments')


# In[ ]:


from kaggle_environments import make
env = make("kore_fleets")
env.run(["balanced", "miner"]);


# In[ ]:


render_env(env, width=800, height=600)

