#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go
import warnings
from IPython.core.display import display, HTML, Javascript

warnings.filterwarnings("ignore")


# Some common utils and helpers that I'll be using throughout the notebook

tiers = {
    0: "Novice",
    1: "Contributor",
    2: "Expert",
    3: "Master",
    4: "Grandmaster",
    5: "Kaggle Team"
}

palette = ["#573a31","#9f8e78","#508084","#2f494b","#20322e","#191410"]

annotation_text_color = "#333333"

# copy-pasted helper to make annotating in plotly a bit easier
def annotation_helper(fig, texts, x, y, line_spacing, align="left", bgcolor="rgba(0,0,0,0)", borderpad=0, ref="axes", xref="x", yref="y", width=100, debug = False):
    
    is_line_spacing_list = isinstance(line_spacing, list)
    total_spacing = 0
    
    for index, text in enumerate(texts):
        if is_line_spacing_list and index!= len(line_spacing):
            current_line_spacing = line_spacing[index]
        elif not is_line_spacing_list:
            current_line_spacing = line_spacing
        
        fig.add_annotation(dict(
            x= x,
            y= y - total_spacing,
            width = width,
            showarrow=False,
            text= text,
            align= align,
            borderpad=4 if debug == False else 0, # doesn't work with new background box implementation :S
            xref= "paper" if ref=="paper" else xref,
            yref= "paper" if ref=="paper" else yref,
            
            bordercolor= "#222",
            borderwidth= 2 if debug == True else 0 # shows the actual borders of the annotation box
        ))
        
        total_spacing  += current_line_spacing
    
    if bgcolor != "rgba(0,0,0,0)":
        fig.add_shape(type="rect",
            xref= "paper" if ref=="paper" else xref,
            yref= "paper" if ref=="paper" else yref,
            xanchor = x, xsizemode = "pixel", 
            x0=-width/2, x1= +width/2, y0=y + line_spacing[-1], y1=y -total_spacing,
            fillcolor= bgcolor,
            line = dict(width=0))
        
styling = """
<style>
.kaggle-tier-percs, hg-table{
}

table.hg-table tr th{
    text-transform: uppercase;
    font-weight: 600;
    text-align: left;
    background: #222;
    color: #eee;
    font-family: Helvetica;
    font-size: 16px;
    padding: 10px 15px;
}

table.hg-table tr td{
    text-align: left;
    font-family: Helvetica;
    font-size: 16px;
    padding: 15px 15px;
    border: none;
    border-bottom: 1px solid #999;
}



.kaggle-tier-percs .td__9290{ background-image: linear-gradient(to right,#508084, #508084 92.90%, transparent 92.90%, transparent);}
.kaggle-tier-percs .td__0572{ background-image: linear-gradient(to right,#508084, #508084 5.72%, transparent 5.72%, transparent);}
.kaggle-tier-percs .td__0107{ background-image: linear-gradient(to right,#508084, #508084 1.07%, transparent 1.07%, transparent);}
.kaggle-tier-percs .td__0018{ background-image: linear-gradient(to right,#508084, #508084 0.18%, transparent 0.18%, transparent);}



.featured-nb-percs tr th{
    line-height: 20px;
    font-size: 15px;
}

.featured-nb-percs tr th .small-text{
    text-transform: none;
    font-weight: 400;
    font-size: 10px;
}

.featured-nb-percs .td--0126{ background-image: linear-gradient(to right,#508084, #508084 calc(100%*(01.26/35)), transparent calc(100%*(01.26/35)), transparent);}
.featured-nb-percs .td--0174{ background-image: linear-gradient(to right,#508084, #508084 calc(100%*(01.74/35)), transparent calc(100%*(01.74/35)), transparent);}
.featured-nb-percs .td--0710{ background-image: linear-gradient(to right,#508084, #508084 calc(100%*(07.10/35)), transparent calc(100%*(07.10/35)), transparent);}
.featured-nb-percs .td--2140{ background-image: linear-gradient(to right,#508084, #508084 calc(100%*(21.40/35)), transparent calc(100%*(21.40/35)), transparent);}
.featured-nb-percs .td--3210{ background-image: linear-gradient(to right,#508084, #508084 calc(100%*(32.10/35)), transparent calc(100%*(32.10/35)), transparent);}
</style>
</style>
"""


# In[ ]:


# Reading in our data, and a basic join
gems = pd.read_csv("/kaggle/input/notebooks-of-the-week-hidden-gems/kaggle_hidden_gems.csv")
users = pd.read_csv("/kaggle/input/meta-kaggle/Users.csv")

merged = pd.merge(gems, users, left_on="author_kaggle", right_on="UserName")

# reading notebook data
nbs = pd.read_csv("/kaggle/input/meta-kaggle/Kernels.csv")


# # Hidden Gems - What makes them shine?
# As someone that has followed Martin's series for a while now, I'm always surprised at how he manages to find such *gems* of notebooks. The series has always brought me absolute delights of notebooks when all other recommendation systems have failed me. To me, the series and this dataset, celebrates the efforts of extremely talented individuals whose works haven't quite received the recognition they deserve.
# 
# In this notebook I want to delve deeper into the authors that produce these featured notebooks and highlight certain key characteristics that set them and their notebooks apart. And having just started working on this today ( yay procrastination!), I'll be taking you along on this journey!
# 
# <hr>
# 
# ## Who are these gems?
# When looking at the featured authors in our dataset, the broadest classification we can make would be on the basis of their ranks in Kaggle's progression system. This gives us a sense of how this group of users differs from the general Kaggle userbase at large.

# In[ ]:


# Dataprep for Kaggle rank stacked bar chart

merged_unique_users = merged.drop_duplicates("author_kaggle")
merged_unique_users["PerformanceTier"] = merged_unique_users["PerformanceTier"].map(tiers)

# removing Kaggle staff as it isn't a part of the progression system
merged_unique_users = merged_unique_users[merged_unique_users["PerformanceTier"] != "Kaggle Team"] 
tier_stacked_bar_data = merged_unique_users["PerformanceTier"].value_counts() / merged_unique_users.shape[0] # transform to percentage of total

# -------------
# Plot creation for Kaggle rank stacked bar chart

fig = go.Figure()

labels = list(tiers.values())

for tier in list(tiers.keys())[:-1]:
    tier_name = tiers[tier]
    
    trace = go.Bar(x=[0], 
                   y=[tier_stacked_bar_data[tier_name]],
                   name=tier_name, 
                   marker = dict( color = palette[tier]),
                   width = 1.0,
                   texttemplate = " <span style='color: #fff'>%{y:.2p}</span> ",
                   insidetextanchor="start",
                   hoverinfo = "none",
                  )
    
    fig.add_trace(trace)
    
layout = dict(
    showlegend = False,
    legend = dict(
        orientation="h",
        traceorder="reversed",
        yanchor="top",
        y=1.12,
        font=dict(family="Helvetica", size=14, color="rgba(0,0,0,100)"),
        bgcolor = 'rgba(255,255,255,100)',
        xanchor="left",
        x= -0.06,
    ),
    barmode = "stack",
    margin = dict(t=100, b=0, l=0, pad=6),
    plot_bgcolor= '#fff',
    xaxis = dict(showticklabels = False, range=[-1,4]),
    yaxis = dict(categoryorder='array', categoryarray = labels[::-1], tickfont=dict(color="#fff") ),
    height = 600,
    width = 600
)

fig.update_layout(layout)

text = [
    "<span style='color:%s; font-family:Tahoma; font-size:14px'><b style='color:%s'>Kaggle Grandmasters</b></span>" % (annotation_text_color, palette[4]),
    "<span style='color:%s; font-family:Tahoma; font-size:14px'>Almost 1 in 7 authors were Kaggle GMs.</span>" % (annotation_text_color),
    "<span style='color:%s; font-family:Tahoma; font-size:14px'>A group which otherwise comprises about</span>" % (annotation_text_color),
    "<span style='color:%s; font-family:Tahoma; font-size:14px'>0.004%% of the Kaggle userbase.</span>" % (annotation_text_color)
]
annotation_helper(fig, text, 2.3, 0.98, line_spacing = [0.05,0.05,0.05], width= 300 )

text = [ "<span style='color:%s; font-family:Tahoma; font-size:14px'><b style='color:%s'>Kaggle Masters</b></span>" % (annotation_text_color, palette[3]) ]
annotation_helper(fig, text, 2.3, 0.68, line_spacing = [0.05], width= 300 )

text = [ "<span style='color:%s; font-family:Tahoma; font-size:14px'><b style='color:%s'>Kaggle Experts</b></span>" % (annotation_text_color, palette[2]) ]
annotation_helper(fig, text, 2.3, 0.39, line_spacing = [0.05], width= 300 )

text = [ "<span style='color:%s; font-family:Tahoma; font-size:14px'><b style='color:%s'>Kaggle Contributors</b></span>" % (annotation_text_color, palette[1]) ]
annotation_helper(fig, text, 2.3, 0.14, line_spacing = [0.05], width= 300 )

text = [ "<span style='color:%s; font-family:Tahoma; font-size:14px'><b style='color:%s'>Kaggle Novices</b></span>" % (annotation_text_color, palette[0]) ,
"<span style='color:%s; font-family:Tahoma; font-size:14px'>98%% of all Kaggle users are novices.</span>" % (annotation_text_color)]
annotation_helper(fig, text, 2.3, 0.075, line_spacing = [0.05], width= 300 )

text = [
    "<span style='font-size:24px; font-family: Georgia;'>Kaggle Ranks of Authors</span>", 
    "<span style='font-size:15px; font-family:Tahoma'>The chart shows where the 225 Hidden Gems authors currently</span>",
    "<span style='font-size:15px; font-family:Tahoma'>stand in Kaggle's progression system.</span>" 
]

annotation_helper(fig, text, 1.17, 1.14, [0.065,0.04,0.052],ref="paper", width=500)

fig.show()


# In the above plot we see how <b>almost 9 out of 10 featured authors are non-novices</b>. This is a stark difference when comparing with the general userbase of Kaggle which consists of 98% being Kaggle Novices!
# 
# We also see better representation of the higher ranks - of all <b>non-novice users</b> the breakdown of Kaggle userbase is as follows:
# <table class="kaggle-tier-percs hg-table">
#     <tr> <th> Rank </th> <th> Percentage </th> </tr>
#     <tr> <td> Kaggle Contributor </td> <td class="td__9290"> 92.9% </td> </tr>
#     <tr> <td> Kaggle Expert </td> <td class="td__0572"> 5.72% </td> </tr>
#     <tr> <td> Kaggle Master </td> <td class="td__0107"> 1.07% </td> </tr>
#     <tr> <td> Kaggle Grandmaster </td> <td class="td__0018"> <b>0.18%</b> </td> </tr>
# </table>
# 
# <hr>
# 
# The quality of the notebooks featured in the series are often the product of years of experience and/or a knack for creating insightful works with data. When looking at how long each of these Kaggle tiers have been around, we get the following chart:

# In[ ]:


# Dataprep for Kaggle experience bar chart
merged["RegisterDate"] = pd.to_datetime(merged["RegisterDate"])
basedate = pd.Timestamp('2022-04-26') # hard-coding the date since it should look the same when rerunning at a later date
merged["DaysSinceRegisterDate"] = (basedate - merged['RegisterDate']).dt.days

registration_time_by_tier = merged.groupby("PerformanceTier")["DaysSinceRegisterDate"].mean() / 365
registration_time_by_tier.index = registration_time_by_tier.index.map(tiers)
registration_time_by_tier = registration_time_by_tier.drop("Kaggle Team")


# Plot creation for Kaggle experience bar chart
fig = go.Figure()
layout = dict(
    margin = dict(t=100, l=0, b=0, pad=6),
    plot_bgcolor= '#fff',
    xaxis = dict(dtick=2, showticklabels=False),
    height = 450,
    width = 600
)

fig.update_layout(layout)

trace = go.Bar(
        x = registration_time_by_tier.values ,
        y = ["<span style='color:%s; font-family:Tahoma; font-size:18px'>%s</span> " % ("#666", tier) for tier in registration_time_by_tier.index],
        width = 0.9,
        marker = dict( color= palette),
        texttemplate = " <span style='color: #fff; font-size: 16px'> %{x:.2f}</span> ",
        textposition = ["inside"],
        insidetextanchor="start",
        orientation = "h",
        hoverinfo = "none",
        showlegend=False,
    )

fig.add_trace(trace) 

text = [ "<span style='color:%s; font-family:Tahoma; font-size:14px'>Average years since registration</span>" % (annotation_text_color) ]
annotation_helper(fig, text, 2.2, 4.67, line_spacing = [0.05], width= 300 )

text = [
    "<span style='font-size:24px; font-family: Georgia;'>How long have they been on Kaggle?</span>", 
    "<span style='font-size:15px; font-family:Tahoma'>While the higher ranks dominate the hidden gems, we also</span>",
    "<span style='font-size:15px; font-family:Tahoma'>see how they have spent the <b>longest time on the platform<b></span>" 
]
annotation_helper(fig, text, 1.25, 1.26, [0.09,0.057],ref="paper", width=500)

fig.show()


# Note that the above only considers <b>featured authors from the competition dataset</b>. The same numbers could look quite different when considering the Kaggle userbase as a whole.
# 
# Perhaps the greater experience in the higher ranks leads to them being featured more often, in spite of being fewer in number.
# 
# ## Consistent output matters
# Along with being around for longer, Hidden gems authors, especially those at the higher ranks, consistently post notebooks for various competitions and often tend to have something in the works. Not only does this help them improve the skills which help them stand out, but this also gives them a larger collection of high-quality notebooks which may be picked for Hidden Gems episodes.

# In[ ]:


hg_author_ids = merged["Id"].unique().tolist() # List of hidden gems authors
hg_notebooks = nbs[nbs["AuthorUserId"].isin(hg_author_ids)]

# Getting a list of all notebooks by all of the featured authors
hg_notebooks = pd.merge(hg_notebooks, merged[["Id","PerformanceTier", "author_name"]], left_on="AuthorUserId", right_on="Id", how="left")

hg_notebooks_author_count = hg_notebooks.groupby(["AuthorUserId","PerformanceTier"])["CurrentUrlSlug"].count().reset_index()
hg_notebooks_tier_median = hg_notebooks_author_count.groupby("PerformanceTier")["CurrentUrlSlug"].median()
hg_notebooks_tier_median = hg_notebooks_tier_median.drop(5)

fig = go.Figure()

layout = dict(
    showlegend = False,
    margin = dict(t=110, l=0, b=0, pad=6),
    plot_bgcolor= '#fff',
    xaxis = dict(dtick=2, showticklabels=False),
    height = 450,
    width = 600
)

fig.update_layout(layout)

trace = go.Bar(
        x = hg_notebooks_tier_median.values ,
        y = ["<span style='color:%s; font-family:Tahoma; font-size:18px'>%s</span> " % ("#666", tiers[tier]) for tier in hg_notebooks_tier_median.index],
        width = 0.9,
        marker = dict( color= palette),
        texttemplate = ["<span style='color: #222; font-size: 16px'> %{x}</span>"]*2 +["<span style='color: #fff; font-size: 16px'>  %{x}</span>"]*3,
        textposition = ["outside"]*2 + ["inside"]*3,
        insidetextanchor="start",
        orientation = "h",
        hoverinfo = "none",
        showlegend=False,
    )

fig.add_trace(trace) 

text = [ "<span style='color:%s; font-family:Tahoma; font-size:14px'><b>median</b> notebooks created</span>" % (annotation_text_color) ]
annotation_helper(fig, text, 17.3, 4.7, line_spacing = [0.05], width= 300 )

text = [
    "<span style='font-size:24px; font-family: Georgia;'>How many notebooks do they create?</span>", 
    "<span style='font-size:15px; font-family:Tahoma'>We see how those in higher ranks have produced <b>far</b></span>",
    "<span style='font-size:15px; font-family:Tahoma'><b>more notebooks</b>, increasing their chances of having</span>",
    "<span style='font-size:15px; font-family:Tahoma'>one of them featured as a Hidden Gem.</span>" 
]
annotation_helper(fig, text, 1.24, 1.33, [0.1,0.06,0.06],ref="paper", width=500)

fig.show()


# When looking at their featured notebooks in context with all of the work they have published on Kaggle, we realise how little of their work is actually does actually make it in the series. In a way the series also celebrates all of the work that goes unnoticed behind the scenes.
# 
# <br>
# 
# <table class="hg-table featured-nb-percs">
#     <tr> 
#         <th> Rank<br> &nbsp </th> <th> Featured<br>Notebooks</th> <th> Total<br>notebooks </th><th> Ratio <br><span class="small-text">(entire block = 35%)</span></th></tr>
#     <tr> <td> Novice </td> <td> 26 </td> <td> 81 </td> <td class="td--3210"> 32.10% </td> </tr>
#     <tr> <td> Contributor </td> <td> 55 </td> <td> 257 </td> <td class="td--2140"> 21.40% </td> </tr>
#     <tr> <td> Expert </td> <td> 84 </td> <td> 1182 </td> <td class="td--0710"> 7.10% </td> </tr>
#     <tr> <td> Master </td> <td> 76 </td> <td> 4357 </td> <td class="td--0174"> 1.74% </td> </tr>
#     <tr> <td> Grandmaster </td> <td> 53 </td> <td> 4197 </td> <td class="td--0126"> 1.26% </td> </tr>
# </table>
# 
# <br>
# The featured novices and contributors have the highest ratios. A reason for this could be due to individuals that have prior experience in data analysis. Perhaps they had a unique take on how they approached a problem, or maybe they used a lesser known dataviz tool? Maybe they showcased a niche area of analysis like exploring graph related insights? 
# 
# 
# Either way this welcomes new Kagglers with fresh perspectives who would have <b>otherwise taken a much longer time to make a mark</b> on the community. These were notebooks would have almost certainly gone unnoticed had it not been for the shout-out from Hidden Gems! Perhaps, this was just the motivation they needed to continue to produce high quality work on the platform.

# In[ ]:


HTML(styling)






















