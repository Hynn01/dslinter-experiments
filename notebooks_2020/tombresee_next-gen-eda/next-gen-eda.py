#!/usr/bin/env python
# coding: utf-8

# ![title](https://www.desipio.com/wp-content/uploads/2019/06/walter-payton-leap-2-ah.jpg)
# <br>&ensp; *Walter Payton (34) and the need for z-coordinate data ...*

# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\n     \n\n    \ndiv.h2 {\n    background-color: #159957;\n    background-image: linear-gradient(120deg, #155799, #159957);\n    text-align: left;\n    color: white;              \n    padding:9px;\n    padding-right: 100px; \n    font-size: 20px; \n    max-width: 1500px; \n    margin: auto; \n    margin-top: 40px; \n}\n                                     \n                                      \nbody {\n  font-size: 12px;\n}    \n     \n                                    \n                                      \ndiv.h3 {\n    color: #159957; \n    font-size: 18px; \n    margin-top: 20px; \n    margin-bottom:4px;\n}\n   \n                                      \ndiv.h4 {\n    color: #159957;\n    font-size: 15px; \n    margin-top: 20px; \n    margin-bottom: 8px;\n}\n   \n                                      \nspan.note {\n    font-size: 5; \n    color: gray; \n    font-style: italic;\n}\n  \n                                      \nhr {\n    display: block; \n    color: gray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}\n  \n                                      \nhr.light {\n    display: block; \n    color: lightgray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}   \n    \n                                      \ntable.dataframe th \n{\n    border: 1px darkgray solid;\n    color: black;\n      <table align="left">\n    ...\n  </table>\n    background-color: white;\n}\n    \n                                      \ntable.dataframe td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 11px;\n    text-align: center;\n} \n   \n            \n                                      \ntable.rules th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 11px;\n    align: left;\n}\n       \n                                      \ntable.rules td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 13px;\n    text-align: center;\n} \n   \n                                      \n                                      \ntable.rules tr.best\n{\n    color: green;\n}    \n    \n                                      \n.output { \n    align-items: left; \n}\n        \n                                      \n.output_png {\n    display: table-cell;\n    text-align: left;\n    margin:auto;\n}                                          \n                                                                    \n                                      \n                                      \n</style>  ')


# In[ ]:


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Reference: 
#      - I really liked the way JohnM's punt kaggle submission had the headers, extremely aesthetically pleasing
#        and aids viewing - borrowing his div.h header concept (so much nicer looking than using conventional
#        ## headers etc), and adding a 'cayman' color theme to it, as a nod to R ...  
#        Isn't it nice looking ?  ->  https://jasonlong.github.io/cayman-theme/
#      - I would strongly suggest we follow JohnM's push into professoinal looking css-based headers, we can't 
#        keep using old-fashioned markdown for headers, its so limited... just my personal opinion
#
# -%%HTML
# <style type="text/css">
#
# div.h2 {
#     background-color: steelblue; 
#     color: white; 
#     padding: 8px; 
#     padding-right: 300px; 
#     font-size: 20px; 
#     max-width: 1500px; 
#     margin: auto; 
#     margin-top: 50px;
# }
# etc
# etc
# --- end reference ---


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UNCOMMENT THIS OUT WHEN YOU ARE READY TO OFFICIALLY SUBMIT ! 
# from kaggle.competitions import nflrush
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#T2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.patches as patches
import seaborn as sns  #I will mainly be using seaborn and bokeh
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UNCOMMENT THIS OUT WHEN YOU ARE READY TO OFFICIALLY SUBMIT ! 
# from kaggle.competitions import nflrush
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#I wont be using plotly, and being honest it has its strong points, but
#I actually prefer bokeh now 
# import plotly as py
# import plotly.express as px
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# from plotly.offline import download_plotlyjs
# from plotly.offline import init_notebook_mode
# from plotly.offline import plot,iplot
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#init_notebook_mode(connected=True)  # remove  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import warnings
warnings.filterwarnings('ignore')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#import sparklines
import colorcet as cc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from IPython.display import HTML
from IPython.display import Image
from IPython.display import display
from IPython.core.display import display
from IPython.core.display import HTML
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from PIL import Image
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import scipy 
from scipy import constants
import math
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ styles ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import colorcet as cc
plt.style.use('seaborn') 
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
##%config InlineBackend.figure_format = 'retina'   < - keep in case 
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
#USE THIS in some form:
# th_props = [('font-size', '13px'), ('background-color', 'white'), ('color', '#666666')]
# td_props = [('font-size', '15px'), ('background-color', 'white')]
#styles = [dict(selector="td", props=td_props), dict(selector="th", props=th_props)]
# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###?sns.set_context('paper')  #Everything is smaller, use ? 
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
##This helps set size of all fontssns.set(font_scale=1.5)
#~~~~~~~~~~~~~~~~~~~~~~~~~ B O K E H ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.io import show
from bokeh.io import push_notebook
from bokeh.io import output_notebook
from bokeh.io import output_file
from bokeh.io import curdoc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.plotting import show                  
from bokeh.plotting import figure                  
from bokeh.plotting import output_notebook 
from bokeh.plotting import output_file
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.models import ColumnDataSource
from bokeh.models import Circle
from bokeh.models import Grid 
from bokeh.models import LinearAxis
from bokeh.models import Plot
from bokeh.models import Slider
from bokeh.models import CategoricalColorMapper
from bokeh.models import FactorRange
from bokeh.models.tools import HoverTool
from bokeh.models import FixedTicker
from bokeh.models import PrintfTickFormatter
from bokeh.models.glyphs import HBar
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.core.properties import value
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.palettes import Blues4
from bokeh.palettes import Spectral5
from bokeh.palettes import Blues8
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.layouts import row
from bokeh.layouts import column
from bokeh.layouts import gridplot
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.sampledata.perceptions import probly
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.transform import factor_cmap
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ M L  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
import gc, pickle, tqdm, os, datetime
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 1. kaggle import raw data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
gold = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
dontbreak = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
from kaggle.competitions import nflrush
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2. laptop import raw data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# df = pd.read_csv('input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
# gold = pd.read_csv('input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
# dontbreak = pd.read_csv('input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
killed_columns=['xyz','etc']
def drop_these_columns(your_df,your_list):
    #KILL KOLUMNS
    your_df.drop(your_list,axis=1,inplace=True)
    return(your_df)
YRS = dontbreak[dontbreak.NflId==dontbreak.NflIdRusher].copy()
YR1 = YRS[YRS.Season==2017]
YR2 = YRS[YRS.Season==2018]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# df_play.drop('Yards', axis=1)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##base = df[df["NflIdRusher == NflId"]]
##killed_kolumns = ["GameId","PlayId","Team","Yards","TimeHandoff","TimeSnap"]
#ingplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python#
#NOTES:
#          sns.despine(bottom=True, left=True)
#  ax.set_title("Rankings Given by Wine Magazine", fontsize=20)

# df04 = tf.groupby('PossessionTeam')['Yards'].agg(sum).sort_values(ascending=False)
# df04 = pd.DataFrame(df04)
# df04['Team'] = df04.index
# df04
#
#
#Some Links:
# Source:  http://www.ncaa.org/about/resources/research/estimated-probability-competing-professional-athletics
#
#
# >>> df = pd.DataFrame(np.random.randn(10, 4))
# >>> df.style.set_table_styles(
# ...     [{'selector': 'tr:hover',
# ...       'props': [('background-color', 'yellow')]}]
# ... )
# sns.despine(left=True)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   KEEP: 
#
#      FULL WIDTH SCREEN
#          display(HTML("<style>.container { width:99% !important; }</style>"))   
#
## Set CSS properties for th elements in dataframe
# th_props = [
#   ('font-size', '11px'),
#   ('text-align', 'center'),
#   ('font-weight', 'bold'),
#   ('color', '#6d6d6d'),
#   ('background-color', '#f7f7f9')
#   ]
# # Set CSS properties for td elements in dataframe
# td_props = [
#   ('font-size', '11px')
#   ]
# # Set table styles
# styles = [
#   dict(selector="th", props=th_props),
#   dict(selector="td", props=td_props)
#   ]
# (df.style
#     .applymap(color_negative_red, subset=['total_amt_usd_diff','total_amt_usd_pct_diff'])
#     .format({'total_amt_usd_pct_diff': "{:.2%}"})
#     .set_table_styles(styles))
#
#   df.style.set_properties(**{'text-align': 'right'})
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# value_counts=> dfvc1
# ref = pd.DataFrame({'AlphaCol':dfvc1.index, 'Count':dfvc1.values}).sort_values("AlphaCol")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GOLD:
# cm = sns.light_palette("green", as_cmap=True)
# s = df.style.background_gradient(cmap=cm)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (monthly_sales
#  .style
#  .format(format_dict)
#  .hide_index()
#  .highlight_max(color='lightgreen')
#  .highlight_min(color='#cd4f39'))
# # USE: 
# dfStyler = df.style.set_properties(**{'text-align': 'left'})
# dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

# filter = df['COUNTRY_FLYING_MISSION'].isin(('USA','GREAT BRITAIN'))
# df = df[filter]


# <br>
# <a id='bkground'></a>
# <div class="h2"><i>NG-EDA</i></div>
# <div class="h3">Next-Generation Exploratory Data Analysis:</div>
# <div class="h3"><i>NFL Run Data</i></div>
# <br>

# <div class="h3"><i>Author: Tom Bresee (Frisco, TX, USA)</i></div>
# &ensp; &ensp;   - All work is my own  
# &ensp; &ensp;   - My plan is to keep working on this even after the competition ends, I feel like I'm just scratching the surface here, but am slowly starting to get more and more insight the more I analyze the data...  
# &ensp; &ensp; &ensp;  - **Update 12-2-2019**: &nbsp; Fixed up spelling and grammer mistakes/typos
# 

# <div class="h2"><i>Goal:</i></div>
# * Explore the NFL Next Gen Stats data provided for this NFL Big Data Bowl challenge
# * Determine the impact various features have on the ebb and flow of the run game
# * Develop a meaningful machine learning model to predict yards gained in an NFL game run play 

# <div class="h3"><i>Approach:</i></div>
# * Emphasis on _extremely_ clear visualization
# * New and innovative approaches to examining our data
# * Get beyond surface analysis and probe for true insights into this sport 
# * Practical data science
# * Personal Opinion:
#   * I prefer Seaborn with a sprinkle of Bokeh thrown in as I learn more about it (I think Bokeh > Plotly)
#   * Check out JohnM's excellent past NFL punt kernel: https://www.kaggle.com/jpmiller/nfl-punt-analytics (some really good Bokeh/Holoviews visualization going on there...)

# <div class="h3"><i>The NFL:</i></div>
# * The 2019 NFL season is the **100th** season of the National Football League (NFL), that's pretty impressive. 
# * The NFL should be applauded for its development of 'Next Gen Stats', which will provide teams with highly granular data to analyze trends and player performance, and also enhancing the game experience for fans.  It will be very interesting to see how this can also help teams come up with even more enhanced play-calling approaches.  As we approach the year 2020, data becomes critical to understand and analyze...

# <div class="h3"><i>Respect:</i></div>
# * All attempts have been made to focus on analyzing the NFL (National Football League) Next Gen Stats data for patterns, trends, and insights into NFL rushing plays, and NOT discuss individual players and their individual set performance.  NFL players have honed their skills for years to be elite-level athletes, and they deserve our respect, because they have earned it.  It is my personal opinion that it is not fair to call out the lower performing athletes when it comes to speed or acceleration or set parameters - mainly because saying an NFL player is in the bottom tier of performance for such a competitive sport is like making fun of a guy that goes to the Olympics and gets 6th place in the world, it is simply nonsense.  I believe it is important to a certain extent to anonymize the data when discussing individual performance... 
# * I will however describe rankings by TEAM in certain categorical analysis. 
# * As data scientists, our job is to objectively analyze the datapoints, without bias.  
# <br>
# 
# **2018: &nbsp; To level-set just how elite NFL players are:**
# * *Data Sources:* 
#   * [2019 NCAA Probablility of Competing Beyond High School](http://www.ncaa.org/about/resources/research/football-probability-competing-beyond-high-school)
#   * [2017-18 High School Athletics Participation Survey](https://members.nfhs.org/participation_statistics)
# * Number of college-level football teams in the United States: &nbsp;  774 
# * Number of Division I college-level football teams in the United States: &nbsp; 130
# * Number of college-level football players: &nbsp;  73,557
# * Number of college-level football players that are NFL draft eligible: &nbsp; 16,346
# * Number of high school level football players:&nbsp;  <span style="color:red">1,036,842</span>  
#   *That is not a typo. There are over 1 million high schoolers playing football right now.*
# * Football is the **most** popular sport in America (a country with a population of over 327 million)
# * IF you are a statistical anomaly physically, you **may** be able to get into the NFL via the path of attending a Division II college (maybe), but most likely it will be via Division I.  
# * Probability of getting into a Division I college football program from high school: &nbsp; 2.8%
# * Probability of getting into the NFL from college football program: &nbsp; 1.6% 
# * <span style="color:red">Total number of college players drafted into the NFL last year:&nbsp; <b>256</b></span>
# 

# Examining the raw data: 
# 
# Taking the raw data we were given of the 2017 and 2018 seasons, and examining each *unique* NFL athlete, I have compiled their college attended previous to their entry into the NFL, and ranked in order of the density of graduates that are in the NFL.  You see some colleges absolutely <u>dominate</u>.  Which means its EVEN harder to get into the NFL than the statistics shown above, because it looks like your best shot is by attending certain certain dominant colleges, which in turn have less and less room on their roster, lowering the player's chance of getting into the NFL.  Research also seems to indicate that approximately 85% of the NFL players were former **Division I college** students.  
# 
# Does it make more sense now why **101,821** people attended the LSU vs Alabama game (November 9th) ?  
# 
# Better yet, the front runner Heisman Trophy candidate [Joe Burrow](https://twitter.com/LSUfootball/status/1194754755123277824) ?  He attends LSU...
# (UPDATE:  Joe did go on to win the Heisman in an overwhelming fashion)
# 
# Plot is interactive, click on a college and you will see its NFL player count and ranking...

# In[ ]:


from bokeh.transform import factor_cmap
from bokeh.palettes import Blues8
from bokeh.palettes import Blues, Spectral6, Viridis, Viridis256, GnBu, Viridis256
from bokeh.palettes import Category20b,Category20c,Plasma,Inferno,Category20
from bokeh.palettes import cividis, inferno, grey
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SUPERHBAR:  i started learning bokeh two days ago, so this quality sucks 
# To draw horizontal bars by specifying a (center) y-coordinate, height, and left and right
# endpoints, use the hbar() glyph function:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
my_data = df[['PlayerCollegeName','NflId', 'DisplayName']].drop_duplicates().copy()

college_attended = my_data["PlayerCollegeName"].value_counts()

df_cc = pd.DataFrame({'CollegeName':college_attended.index, 'Count':college_attended.values}).sort_values("Count", ascending = False)

#df_cc.Count.astype('int', inplace=True)

df_cc = df_cc[df_cc.CollegeName != 'Louisiana State']

df_cc.at[42,'Count']=51

#df_cc[df_cc.CollegeName=='LSU']['Count']

#THIS IS UNBELIEVABLE.  SOMEONE COUNTED LSU AND LOUSISIANA STATE AS DIFF COLLEGES ! ! ! ! THATS A BIG 
#MISTAKE.  LSU HAS A MASSIVE NUMBER OF PLAYERS CURRENTLY IN THE NFL, and so consolidating the values...

df_cc.sort_values('Count',ascending=False, inplace=True)

#pd.set_option('display.max_rows', 500)
df_cc.index = df_cc.index + 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mysource = ColumnDataSource(df_cc)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
p = figure(
  y_range=df_cc.CollegeName[:50],    # I need to enter the SAME thing here as y points, i find that odd 
  # wait:  can i set this as the range, but not below ? ? ? 
  # i think caegorical just list in a list the categories here 
  title = '\nNFL Player Count by College Attended\n',
  x_axis_label ='# of NFL players that attended the college prior\n',
  plot_width=600,
  plot_height=700,
  tools="hover",       # or tools="" 
  toolbar_location=None,   
  #background_fill_color="#efe8e2")
  #min_border=0))
)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TEMP KILL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# p.hbar(
#     y='CollegeName',  # center of your y coordinate launcher, 40 points as def above ... 
#     left=0, # or left=20, etc
#     right='Count',    # right is 40 points... 
#     height=0.8,
#     alpha=.6,
#     #color='orange',    #color=Spectral3  #color=Blues8,   
#     #background_fill_color="#efe8e2", 
#     #     fill_color=Blues8,
#     #     fill_alpha=0.4, 
#     source = mysource,
#     fill_alpha=0.9,
#     line_color='blue'   # line_coolor='red'
# ) 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
p.hbar(
    y='CollegeName',  # center of your y coordinate launcher, 40 points as def above ... 
    left=0, # or left=20, etc
    right='Count',    # right is 40 points... 
    height=0.8,
    alpha=.6,
    #color='orange',    #color=Spectral3  #color=Blues8,   
    #background_fill_color="#efe8e2", 
    #     fill_color=Blues8,
    #     fill_alpha=0.4, 
    
    fill_color=factor_cmap(
        'CollegeName',
        palette=grey(50), #inferno(50),  #cividis(50),  #d3['Category20b'][4],  #Category20b(2),  #[2],   #Category20b,   #Viridis256,    #GnBu[8], #,#Spectral6,             #viridis(50),  #[3], #Spectral6,  #|Blues[2],
        factors=df_cc.CollegeName[:50].tolist()     #'CollegeName'  #but i think i need this: car_list
    ),

    source = mysource,
    fill_alpha=0.9,
    #line_color='blue'  
) 










#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TITLE: 
# p.title.text = 'Current frame:'
# p.title.text_color = TEXT_COLOR
# p.title.text_font = TEXT_FONT
p.title.text_font_size = '11pt'
# p.title.text_font_style = 'normal'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AXES: 
# p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
# p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
# p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
# p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
# p.xaxis.axis_line_color = None    # or 'red'
# p.yaxis.axis_line_color = GRID_COLOR 
#
# X-TICKS:
# p.xaxis[0].ticker = FixedTicker(ticks=[0, 1])
# p.xaxis.major_tick_line_color = GRID_COLOR
# p.xaxis.major_label_text_font_size = '7pt'
# p.xaxis.major_label_text_font = TEXT_FONT
# p.xaxis.major_label_text_color = None   #TEXT_COLOR
#
# Y-TICKS:
# p.yaxis[0].ticker = FixedTicker(ticks=np.arange(1, len(labels) + 1, 1).tolist())
# p.yaxis.major_label_text_font_size = '0pt'
p.yaxis.major_tick_line_color = None
p.axis.minor_tick_line_color = None  # turn off y-axis minor ticks

# p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GRID:
# p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None   
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # LEGENDgend.location = 'top_left'
# p.legend.orientation='vertical'
# p.legend.location='top_right'
# p.legend.label_text_font_size='10px'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### NOTES here> 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HOVER:
#     hover.names = ['bars']
#     hover.tooltips = [
#         ('Event', '@label'),
#         ('Probability', '@pretty_value')]
#
hover = HoverTool()
#p.select(HoverTool).tooltips = [("x1","@x1"), ("x2","@x2")]
#
# hover.tooltips = [
#         ('Event', '@label')
#         #('Probability', '@pretty_value'),
#     ]
# hover.tooltips = [
#     ("Total:", "@Count")
#     #("x1", "@x1"),
#     #("Totals", "@TONS_HE High Explosive / @TONS_IC Incendiary / @TONS_FRAG Fragmentation")
#     ]
###########################hover.mode = 'vline'
#????curdoc().add_root(p)
# hover.tooltips = """
#     <div>
#         <br>
#         <h4>@CollegeName:</h4>
#         <div><strong>Count: &ensp; </strong>@Count</div>
#     </div>
# """
hover.tooltips = [
    ("College Name:", "@CollegeName"),
    ("Ranking by Count", "$index"),
    ("Number of gradutes that entered the NFL:", "@Count"),
]
#<div><strong>HP: </strong>@Horsepower</div>       
p.add_tools(hover)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
output_notebook(hide_banner=True)
show(p); 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hover.tooltips = [
#     ("index", "$index"),
#     ("(x,y)", "($x, $y)"),
#     ("radius", "@radius"),
#     ("fill color", "$color[hex, swatch]:fill_color"),
#     ("foo", "@foo"),
#     ("bar", "@bar"),
# ]
#
#

# IF YOU WANT ALL BLUES, KEEP THIS: 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # SUPERHBAR:  i started learning bokeh two days ago, so this quality sucks 
# # To draw horizontal bars by specifying a (center) y-coordinate, height, and left and right
# # endpoints, use the hbar() glyph function:
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# my_data = df[['PlayerCollegeName','NflId', 'DisplayName']].drop_duplicates().copy()

# college_attended = my_data["PlayerCollegeName"].value_counts()

# df_cc = pd.DataFrame({'CollegeName':college_attended.index, 'Count':college_attended.values}).sort_values("Count", ascending = False)

# #df_cc.Count.astype('int', inplace=True)

# df_cc = df_cc[df_cc.CollegeName != 'Louisiana State']

# df_cc.at[42,'Count']=51

# #df_cc[df_cc.CollegeName=='LSU']['Count']

# #THIS IS UNBELIEVABLE.  SOMEONE COUNTED LSU AND LOUSISIANA STATE AS DIFF COLLEGES ! ! ! ! THATS A BIG 
# #MISTAKE.  LSU HAS A MASSIVE NUMBER OF PLAYERS CURRENTLY IN THE NFL, and so consolidating the values...

# df_cc.sort_values('Count',ascending=False, inplace=True)

# #pd.set_option('display.max_rows', 500)
# df_cc.index = df_cc.index + 1

# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# mysource = ColumnDataSource(df_cc)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# p = figure(
#   y_range=df_cc.CollegeName[:50],    # I need to enter the SAME thing here as y points, i find that odd 
#   # wait:  can i set this as the range, but not below ? ? ? 
#   # i think caegorical just list in a list the categories here 
#   title = '\nNFL Player Count by College Attended\n',
#   x_axis_label ='# of NFL players that attended the college prior\n',
#   plot_width=600,
#   plot_height=700,
#   tools="hover",       # or tools="" 
#   toolbar_location=None,   
#   #background_fill_color="#efe8e2")
#   #min_border=0))
# )
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# p.hbar(
#     y='CollegeName',  # center of your y coordinate launcher, 40 points as def above ... 
#     left=0, # or left=20, etc
#     right='Count',    # right is 40 points... 
#     height=0.8,
#     alpha=.6,
#     #color='orange',    #color=Spectral3  #color=Blues8,   
#     #background_fill_color="#efe8e2", 
#     #     fill_color=Blues8,
#     #     fill_alpha=0.4, 
#     source = mysource, 
#     line_color='blue'   # line_coolor='red'
# ) 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # TITLE: 
# # p.title.text = 'Current frame:'
# # p.title.text_color = TEXT_COLOR
# # p.title.text_font = TEXT_FONT
# p.title.text_font_size = '11pt'
# # p.title.text_font_style = 'normal'
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # AXES: 
# # p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
# # p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
# # p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
# # p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
# # p.xaxis.axis_line_color = None    # or 'red'
# # p.yaxis.axis_line_color = GRID_COLOR 
# #
# # X-TICKS:
# # p.xaxis[0].ticker = FixedTicker(ticks=[0, 1])
# # p.xaxis.major_tick_line_color = GRID_COLOR
# # p.xaxis.major_label_text_font_size = '7pt'
# # p.xaxis.major_label_text_font = TEXT_FONT
# # p.xaxis.major_label_text_color = None   #TEXT_COLOR
# #
# # Y-TICKS:
# # p.yaxis[0].ticker = FixedTicker(ticks=np.arange(1, len(labels) + 1, 1).tolist())
# # p.yaxis.major_label_text_font_size = '0pt'
# p.yaxis.major_tick_line_color = None
# p.axis.minor_tick_line_color = None  # turn off y-axis minor ticks

# # p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# # p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
# p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # GRID:
# # p.xgrid.grid_line_color = None
# p.ygrid.grid_line_color = None   
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # # LEGENDgend.location = 'top_left'
# # p.legend.orientation='vertical'
# # p.legend.location='top_right'
# # p.legend.label_text_font_size='10px'
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ### NOTES here> 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # HOVER:
# #     hover.names = ['bars']
# #     hover.tooltips = [
# #         ('Event', '@label'),
# #         ('Probability', '@pretty_value')]
# #
# hover = HoverTool()
# #p.select(HoverTool).tooltips = [("x1","@x1"), ("x2","@x2")]
# #
# # hover.tooltips = [
# #         ('Event', '@label')
# #         #('Probability', '@pretty_value'),
# #     ]
# # hover.tooltips = [
# #     ("Total:", "@Count")
# #     #("x1", "@x1"),
# #     #("Totals", "@TONS_HE High Explosive / @TONS_IC Incendiary / @TONS_FRAG Fragmentation")
# #     ]
# ###########################hover.mode = 'vline'
# #????curdoc().add_root(p)
# # hover.tooltips = """
# #     <div>
# #         <br>
# #         <h4>@CollegeName:</h4>
# #         <div><strong>Count: &ensp; </strong>@Count</div>
# #     </div>
# # """
# hover.tooltips = [
#     ("College Name:", "@CollegeName"),
#     ("Ranking by Count", "$index"),
#     ("Number of gradutes that entered the NFL:", "@Count"),
# ]
# #<div><strong>HP: </strong>@Horsepower</div>       
# p.add_tools(hover)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# output_notebook(hide_banner=True)
# show(p); 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # hover.tooltips = [
# #     ("index", "$index"),
# #     ("(x,y)", "($x, $y)"),
# #     ("radius", "@radius"),
# #     ("fill color", "$color[hex, swatch]:fill_color"),
# #     ("foo", "@foo"),
# #     ("bar", "@bar"),
# # ]


# > It gets worse:
#   * Even if you enter the NFL draft, it also depends on what **position** you play.  They may not need certain players depending on how the season before had gone and other external factors. 
# 
# The 256 players that entered the NFL in the year 2018:  &nbsp; *Source: Wikipedia*
# <img src="https://github.com/tombresee/Temp/raw/master/ENTER/draft.png" width="1300x">
# <br>

# <div class="h3"><i>Running Back Categories:</i></div>
# * <u>NFL Running Backs are premiere athletes capable of speed and movement that 99.99% of humans aren't capable of.  In my estimation, they fall into one of three categories:</u>
#   * **Elusive** - outmaneuvering the defense, in a finesse manner - finding running lanes and bursting through
#   * **Pure** - unique combinations of speed and power, capable of outrunning and just outplaying the defense 
#   * **Bruiser** - this running back is going to run through you to get to the goaline, and even if you slow him down, it will take at least two of you to stop him
#   <br>
# * Perhaps a video example would help:
#   * [Elusive - Barry Sanders](https://www.youtube.com/watch?v=PBhn1wMyzV4)
#   * [Pure - Walter Payton](https://www.youtube.com/watch?v=uQz7LWdOYc8) and [another highlight video](https://www.youtube.com/watch?v=b9O19IxOGNc)
#   * [Bruiser - Jim Brown](https://www.youtube.com/watch?v=9cqsIedJew4) and [another highlight video](https://www.youtube.com/watch?v=b6cCXNBeVfc)
#     *  'Make sure when anyone tackles you he remembers how much it hurts.' - Jim Brown
# * Interested in checking out some other running backs ?  Some of the best to ever play the game:
#   * Emmitt Smith, Eric Dickerson, Earl Campbell, Bo Jackson, Tony Dorsett, Marshall Faulk, Adrian Peterson

# <div class="h3"><i>Jim Brown:</i></div>
# * Yes, he deserves his own section alone.  Why ?  Because he was one of if not the most dominating running back ever. 
# * He played from 1957 through 1965 (i.e. 54 years ago), and in an era where each decade emerges stronger/faster/quicker athletes, his resume **still** shines, that alone is pretty impressive
# * His first season he scored 17 touchdowns. The SAME number of touchdowns he scored in his second to last season as a player...    
# * NFL Rookie of the Year
# * Pro Bowl invitee <u><b>every</b></u> season he was in the league.  Yes, nine straight Pro Bowls. 
# * 3x NFL MVP
# * NFL Champion (and twice runner-up)
# * His 12,312 rushing yards and 15,459 combined net yards put him in a then-class by himself
# * He. never. missed. a. game.  during. his. career. Ever.   
#   * (there should be a NFL stat for extreme durability)
# * His last game ?  He scores three touchdowns in his final Pro Bowl game
# * <u><b>He led the league in rushing yards in eight out of his NFL nine seasons</b></u>
#   * Compare that to more modern atheletes, no one comes even close to that accomplishment alone.  This is still an NFL record. 
#   * <u>The only player in the history of the NFL to average over 100 rushing yards per game for his ENTIRE career</u>
# *  <span style="color:red">**Update**: &nbsp; Announced as the first member -- unanimously selected -- of the *NFL 100th Anniversary All-Time Team* (a distinguished list of the 100 greatest players in the NFL league's 100-year history)</span>
# * It is my opinion that the number #32 should be universally retired from **all** NFL teams.  If he were in the military and his accomplishments were shown on his chest with metals, he wouldn't be able to walk under the weight of all of them.  

# <div class="h3"><i>Some Background:</i></div> 
# 
# * Just like soccer and cricket, American football features two opposing squads of eleven (11) players each
# 
# * A regulation football field is 100 yards (~91m) long by 53 yards (~49m) wide
# 
# * A soccer field is slightly larger, ranging from 100 to 130 yards long and 50 to 100 yards wide
#  
# * Soccer players in general form a single team/unit, while football players are assigned to offense, defense or special teams 
# 
# * Football fields feature markings every 10 yards and hash marks for single yards, while soccer fields mark out a kickoff circle, a midfield line and two penalty areas
# 
# * Typically the quarterback (QB) in football is roughly equivalent to the central midfielder in soccer  
# 
# * Jersey numbers are *important* to football players.  They try to keep the same number as they transition from college to the NFL.  
# 
# * When you actually go out to a football field, the first thing you notice is that it is wider than it looks on TV.  There is more room to run than you actually think...
# 
# * ~ Approximate Rugby mapping: Think of the NFL offensive line as a combination of the Loosehead Prop, Hooker, Tighthead Prop and two Second Rows, where the NFL center is a 'Hooker', the NFL left and right tackles are 'Props', and the NFL left and right guards are "2nd Rows'.  The NFL line works together to push, just like the rugby front row and second row (props/hooker/2ndRow) interconnect/interlock.  A very good mapping is the NFL quarterback (QB) to the rugby 'Scrum Half'; the rugby scrumhalf will decide to kick the ball or pass it out to his runners, and the NFL QB will decide to throw the ball or hand it off to the runningback (RB) next to him.  Obviously there is no forward 'passes' in rugby, but the way the scrumhalf kinda is the brains and runs the show and makes split second command decisions is similar to how an NFL QB behaves.  And on the defensive front think of flankers as roughly NFL linebackers, where the flankers would be trying tackle the scrumhalf (like NFL LBs try to hit the QB).  On defense, think of defensive line as equivalent to the offensive line mapping we just discussed, but the combination of the 8-man and the two flankers is roughly similar to how a three linebacker setup would be in the NFL.  In rugby, the flankers and the 8-man aren't 'tied' to all the shoving too much (like offensive linemen are 'in the trenches', and thus they can break off and make tackles and be more 'mobile' than linemen.   Think of an NFL fullback as similar to a rugby fly-half, he is a banger, used to contact, and is not small, and can in fact run but is used for more dense defense smashes.  Think if you were to merge the rugby Inside Center and Outside Center into one unit, <u>it would be the NFL runningback (RB)</u>.  They both see holes and attack. So to make the rugby mapping complete, think of the front and second row getting the ball back to the scrumhalf (QB), and the scrumhalf deciding to skip the pass to the fly half and send it directly to a Inside/Outside Center (runningback), where in the NFL they dont really throw the ball to their runningback, they hand it off, so really think of the scrumhalf physically handing off the ball to the Center, and that is what this whole exercise we are looking at is about... In rugby the Right Wing is similar to an NFL wide receiver, they are blindingly fast, and both probably make sure their hair looks good before they show up for the game... 
# 
# * A really good link to start seeing the big picture of what is available via NG Stats with regard to rushing (running), is here: &nbsp;  https://nextgenstats.nfl.com/stats/rushing
# 
# * At a high level, we have `linemen`, that basically block for a runningback, and then the runningback can run in any direction that will gain him yards, see image I have created below: 
# 
# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/upload2.jpg" width="700px"> 
# 

# <div class="h3"><i>Football Player Jersey Numbers:</i></div>
# Jersey numbers are considered important/lucky/sentimental to NFL players.  This is a quick breakdown of which jersey number is mapped to which football position. 
# 
# * 1 - 19: &nbsp; &nbsp;  Quarterbacks, punters, and kickers
# * 20 - 49: &nbsp; Running backs and defensive backs 
# * 50 - 59: &nbsp; Centers
# * 60 - 79: &nbsp; Defensive linemen and offensive linemen 
# * 80 - 89: &nbsp; Receivers and tight ends 
# * 90 - 99: &nbsp; Defensive linemen and linebackers

# <div class="h3"><i>Acryonyms:</i></div>
# Many times you will see the positions via acronyms.  RB is a term you will see alot in this analysis since it is usually the player that runs the ball (although it should be noted that pretty much any player if they want can run the ball:
# * RB: &nbsp; &nbsp;  Runningback
# * QB: &nbsp; &nbsp;  Quarterback
# * FB: &nbsp; &nbsp;  Fullback
# * WR: &nbsp;         Wide Receiver 
# * SS: &nbsp; &nbsp;  Strong Safety
# * CB: &nbsp; &nbsp;  Cornerback
# * DE: &nbsp; &nbsp;  Defensive End
# * CB: &nbsp; &nbsp;  Cornerback
# * T: &nbsp; &nbsp; &nbsp;  Tackle
# * C: &nbsp; &nbsp;  &nbsp; Center
# 
# 
# 
# 

# <div class="h3"><i>Some Helpful Conversions:</i></div>
# <table class="datatable" align="left"><tr><th align="left">Yard [yd]</th><th>Meter [m]</th><tr><td align="right">1 yd</td><td>0.9144 m</td></tr><tr><td align="right">2 yd</td><td>1.8288 m</td></tr><tr><td align="right">3 yd</td><td>2.7432 m</td></tr><tr><td align="right">5 yd</td><td>4.572 m</td></tr><tr><td align="right">10 yd</td><td>9.144 m</td></tr><tr><td align="right">20 yd</td><td>18.288 m</td></tr><tr><td align="right">50 yd</td><td>45.72 m</td></tr><tr><td align="left">100 yd</td><td>91.44 m</td></tr></table><br><br>
# 

# <div class="h3"><i>The NFL Teams:</i></div>
# <img src="https://github.com/tombresee/Temp/raw/master/ENTER/teams.png" width="500px">
# 
# *NFL Team Name Abbreviations:*    
# 
# 
# ARI: Arizona Cardinals  
# ATL: Atlanta Falcons     
# BAL: Baltimore Ravens  
# BUF: Buffalo Bills    
# CAR: Carolina Panthers  
# CHI: Chicago Bears   
# CIN: Cincinnati Bengals   
# CLE: Cleveland Browns  
# DAL: Dallas Cowboys    
# DEN: Denver Broncos   
# DET: Detroit Lions   
# GB: Green Bay Packers        
# HOU: Houston Texans   
# IND: Indianapolis Colts        
# JAX: Jacksonville Jaguars    
# KC: Kansas City Chiefs   
# MIA: Miami Dolphins   
# MIN: Minnesota Vikings    
# NE: New England Patriots  
# NO: New Orleans Saints        
# NYG: New York Giants    
# NYJ: New York Jets  
# OAK: Oakland Raiders        
# PHI: Philadelphia Eagles  
# PIT: Pittsburgh Steelers  
# SD: San Diego Chargers  
# SEA: Seattle Seahawks        
# SF: San Francisco 49ers  
# STL: Saint Louis Rams  
# TB: Tampa Bay Buccaneers  
# TEN Tennessee Titans  
# WAS: Washington Redskins        

# <div class="h3"><i>NFL Teams by Conference:</i></div>
# <img src="https://github.com/tombresee/Temp/raw/master/ENTER/teamsall.png" width="1000px">
# 

# <div class="h3"><i>Data Features:</i></div>
# I always find that it helps to break out the data features.  See below. 
# 
#    
# <style type="text/css">.tg-sort-header::-moz-selection{background:0 0}.tg-sort-header::selection{background:0 0}.tg-sort-header{cursor:pointer}.tg-sort-header:after{content:'';float:right;margin-top:7px;border-width:0 5px 5px;border-style:solid;border-color:#404040 transparent;visibility:hidden}.tg-sort-header:hover:after{visibility:visible}.tg-sort-asc:after,.tg-sort-asc:hover:after,.tg-sort-desc:after{visibility:visible;opacity:.4}.tg-sort-desc:after{border-bottom:none;border-width:5px 5px 0}</style><table id="tg-ld6dq" style="border-collapse:collapse;align=left;border-spacing:0;border-color:#aaa;margin:0px left;table-layout: fixed; width: 603px" class="tg"><colgroup><col style="width: 163px"><col style="width: 440px"></colgroup>
# 
# 
# <tr><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:bold;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:inherit;color:#9a0000;background-color:#ffffff;text-align:left;vertical-align:top">Feature</th><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:bold;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:inherit;color:#9a0000;background-color:#ffffff;text-align:left;vertical-align:top">Description</th></tr>
# 
# 
# <tr><td style="font-family:Arial, sans-serif;font-size:12px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:inherit;color:#000000;background-color:#ffffff;font-weight:bold;font-style:italic;text-align:left;vertical-align:top" colspan="2"></td></tr><td style="font-family:Arial, sans-serif;font-size:12px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:left;vertical-align:top">Season</td><td style="font-family:Arial, sans-serif;font-size:12px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:left;vertical-align:top">year of the season</td></tr>
# 
# 
# <tr><td style="font-family:Arial, sans-serif;font-size:12px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:left;vertical-align:top">Week</td><td style="font-family:Arial, sans-serif;font-size:12px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:left;vertical-align:top">week into the season</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:12px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:left;vertical-align:top">HomeTeamAbbr</td><td style="font-family:Arial, sans-serif;font-size:12px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:left;vertical-align:top">home team abbreviation (home vs visitor)</td></tr>
# 
# 
#     
#     
#     
#     
#     
#    

# **Helpful:** The below is a quick way of referring to the pertinent column without having to actually type out the full name.  Just use `df.iloc[:,<the number index of column below>]`. i.e. listing out df.Team you could use `df.iloc[:,2]`.  This helps when trying to list out numerous columns...where you could then enter `df.iloc[:,[2,4,6,7,10]]`.  I find it faster...

# In[ ]:


refer = pd.DataFrame(df.columns)
refer.columns=['Mapper']
refer.index.name='Ref:'
refer.style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])])


# Lets take a look at a single sample datapoint (presented in vertical form for clarity):

# In[ ]:


df.head(1).T. style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])])


# Every game has a unique id, every play has a unique play id, and all single play related data is included such as game venue information, player information, and even the offense and defense formations at the time of the snap.  This is really detailed information and its amazing they have this much data available on a play by play basis, the sky is the limit in terms of what you could do from an analysis perspective.  I especially think that the score over the course of time may impact the manner and style of run plays. 

# <div class="h3"><i>Summary of our dataset:</i></div>
# <p style="margin-top: 50px">It is always important to look at our entire dataset and examine the descriptive statistics:</p>
# 
# &ensp; **Number of football teams in the NFL:** &ensp; 32  
# &ensp; **Number of unique NFL players in our dataset:** &ensp; 2,231  
# &ensp; **Number of 2017 Season players:** &ensp; 1,788  
# &ensp; **Number of 2018 Season players:** &ensp; 1,783   
# &ensp; **Number of players playing both yrs:** &ensp; 1,340    
# &ensp; **Number of players allowed per team:** &ensp; 53    
# &ensp; **Number of games a team plays in a NFL season:** &ensp; 16      
# &ensp; **Number of weeks in a NFL season:** &ensp; 17   
# &ensp; **Total unique NFL games played per season:** &ensp; 256  
# &ensp; **Number of NFL seasons in the dataset:** &ensp; 2  
# &ensp; **Dataset NFL season years:** &ensp; 2017 and 2018 Seasons    
# &ensp; **Dataset total number of unique NFL games:** &ensp; 512  
# &ensp; **Number of unique run plays in our dataset:** &ensp; 23,171  
# &ensp; **Number of 2017 Season run plays:** &ensp; 11,900  
# &ensp; **Number of 2018 Season run plays:** &ensp; 11,271  
# &ensp; **Number of unique NFL jersey numbers:** &ensp; 99  
# &ensp; **Number of players on roster that never played:** &ensp; 11  
# &ensp; **Size of a typical NFL field (in acres):** &ensp; 1.32

# In[ ]:


#--- raw counts for above ---
# len(df_train.Season.unique())
# len(df_train.NflId.unique())
# len(df_train.PlayId.unique())
# len(df_train[df_train.Season==2017].PlayId.unique())
# len(df_train[df_train.Season==2018].PlayId.unique())

def create_field(linenumbers=True,figsize=(10,5)):

    field = patches.Rectangle((0, 0), 100, 53.3, linewidth=5.5,
                             edgecolor='black', facecolor='grey', zorder=0)
    
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(field)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    plt.plot([50, 50], [0, 53.3], color='darkgrey')
    
    plt.xlim(-10, 120); plt.ylim(0, 58.3)
    plt.axis('off')
    
        
    return fig, ax
create_field(); 


# <div class="h2"><i>It's Gametime...</i></div>
# <br>

# <div class="h3"><i>Let's walk thru a theoretical NFL game:</i></div>
# * There are four quarters of play, and each quarter is 15:00 mins.   NFL games are divided into four 15-minute quarters, separated by a 12-minute break at halftime. 
# * **Your** team will be kicking off the ball to **my** team.  I catch it, run for a bit, and am tackled.  **Now I will begin what is known as a series.**
# * Down 1: &nbsp; I am trying to reach the far end goaline to score, but in the meantime, all I need to do is reach 10 total yards over the course of a max of 4 'downs' (think of 'downs' as 'turns'), and I can continue to retain possession of the ball.  I have my running back (RB) run the ball, and he is tackled for a 'yard gain' of 3 yards.  7 more yards to go to reach my goal of 10 !  
# * Down 2:  &nbsp; I run a sweep to the right, and just before the RB is tackled, he reaches 4 more yards.  I have run 7 yards at this point, 3 more yards to go ! 
# * Down 3: &nbsp;  I run right up the middle, and crunch out 2 more yards.  I have 1 more yard to go, and I MUST reach this in the next play or I will have to give **you** possession of the ball.  Note:  At this point, I do have the option to punt the ball, if I don't feel confident here, but the point of this is to illustrate how running works...   
# * Down 4: &nbsp;  I fake a run right up the middle, and I have the quarterback (QB) run just to the left (L), fake right (R), run left (L), and squeek out a yard and a half.  Because I was able to reach my goal of at least 10 yards within 4 plays, we RESET the 'downs', now I am back to 1st down ! (The Math:  3 + 4 + 2 + 1.5 = 10.5 > 10). I get to keep the ball and keep going... 
# * Down 1: &nbsp;  I see that the defense (you) is respecting my running ability and are putting MORE guys on the 'line' (line of scrimmage) and an addition guy (linebacker) in the middle to slow down my runs. I am going to adjust my strategy and try to avoid some of those guys and run outside more and not right towards the 'box'.  My RB gets the ball, but your defense swarms around him, and he didn't even make it to the line of scrimmage, he was tacked TWO yards back, so this means my yards gained was -2 !  So now I am reaching Down 2, and I have to reach 12 total yards now !  I have to make something happen here...
# * Down 2: &nbsp; I fake a pass, and I have my RB run a sweep to the (R), and he gets only 2 yards !  This is not good, they are killing us here, I am facing Down 3 and I still have 10 more yards to reach my goal !  
# * Down 3: &nbsp; RB runs up the middle, sees a hole (because he has 'vision', an important characteristic for running backs to have), and bam, he gets 9 total yards !  I have ONE more shot to reach my final 1 yard, here we go...
# * Down 4: &nbsp; The defense knows I'm going to run it and attempt to smash thru for one more yard, and they stack the box with 9 defenders ! ! !  This math is not looking so good for me.  My RB gives it everything he has, but is stopped in the backfield for a loss of 4 yards (yards gained = -4).  I didn't make my goal :(  I have to turn the ball back over to the defense.  
# * YOU now have the ball, and the first thing you notice is how much TIME I took up off the clock ! ! !  This series of running on my side actually took up almost 4 mins of the 15 mins in the quarter !  
# * Down 1: &nbsp; I am now on **defense** (you have the ball now, thus you are on **offense**), and I stack the box with 4 guys on the line, and 3 linebackers (LB).  I thus have 7 total defenders 'in the box'.  You run a sweep to the (L), and your running back is GOOD.  He runs 14 yards !  He made his goal of at least 10 yards, and the down is thus reset to 1.  Important to understand:  IF you make your 10 yard goal in less than four plays, the down is 'reset' to 1.  
# * Down 1: &nbsp;  You runs again, right up the middle, and I can't seem to stop you.  Your RB runs 8 yards, so he has 2 more yards to go to get the '1st down' yet again.   
# * Down 2: &nbsp;  Your team is running like crazy, I have to stop him.  I add **one more guy to the box** (on the line of scrimmage in this case, to get some *pressure* on your RB).  Your RB runs thru ALL of my defense, and has a massive run of 62 yards, all the way to the endzone, where now your team is awareded 6 total points, for scoring what is known as a **touchdown**.  It's going to be a long day for me...
# * I need to huddle up and come up with a strategy to beat you, this isn't looking good.  In my opinion, the best way to beat a really good team is to KEEP them from even touching the ball.  **Every minute I have the ball**, you **DON'T** have the ball.  So the key now is to keep running the ball to eat into the clock, and slowly and methodically continue to get first downs, and then score after that.  If I do this 'ball possession' just right, you will notice that there is less and less time left on the clock, and your premiere 'Walter Payton like' RB barely even gets to touch the ball...By now we can see the strategic advantage of having a solid running back, it allows one to retain possession of the ball, and sometimes depending on the matchup, its just hard to stop a RB.  Moreover, if the team I am facing is a fast-paced high scoring team, if I can manage ball control I can prevent them from scoring as much as they usually do.  A combination of good tackling on my side (good defense), and also proper usage of my RB in a balanced attack, can be a devastating combination.    
# * The game continues for four (4) quarters.  Adjusting is _very_ very common in football, and as the game transpires, offense will try different 'formations' (how they place their players like the RB on the field), and defense will try different 'formations' (put more big guys on the line of scrimmage, or place linebackers (LB) in different positions).  The game of football is a chess match (but played with real human pieces), and the winning team is a combination of more talent and better coaching (IN theory). 
# * Sometimes one defensive player can change an entire game:  Enter Lawrence Taylor.  
#   * Think you have what it takes to play running back ?  Watch this [video](https://www.youtube.com/watch?v=ePLg6eTqpZ0) (turn up the audio) and then let me know afterwards if you still believe you have what it takes to play RB... 
#   * Another [link](https://www.youtube.com/watch?v=jnNP7mWAP5k) (thanks for the link JohnM!) 
# * Some high level strategy:
#   * On offense, I should try to keep the defense 'off-guard', i.e. I should try to be unpredictable in my passes (throwing the ball) or runs.  IF I have a strong runningback, I should try to use him, and even also use him as a diversion sometimes (fake the handoff to him, and then do some other play, which can be quite effective).  
#   * Although the number of yards a runningback gets per play is important, what is more important is winning the game, and thus even if my runningback is not getting that many yards rushing, I still have the option of just running him more in the game.  This is sometimes why you will see that a runningback maybe didn't get a great average yards per carrier in a game, but the team won.  Sometimes my runningback is my best bet out there to win a game...
# * Some terms you will see:  
#   * Yds:  Another term for Yards (sometimes seen as YDS)
#   * Y/A:  Yards per Rush Attempt
#   * Y/G:  Rushing yards per Game

# <div class="h2"><i>It's time to analyze some NG Stats data !</i></div>
# <br>

# <div class="h4"><i>But before we get started...</i></div>
# * <u>I would like to propose a **new** term</u>

# Look at this below image for a moment: 

# <img src="https://wiki.mobilizingcs.org/_media/rstudio/wordbarchart.jpeg" width="1500px">

# How much meaning are you really getting out of this visualization ??    
# 
# This seems to happen a lot when data is spread over a large range. The highest values 'drown' out the ability to see the relationships in the heights of the smaller values...  
# 
# Data in the field of data science seems to include data distributions like this a lot, perhaps we can fix this visualization issue.   You see plots like the above a fair amount. 

# In[ ]:



######################################################################### 
#                                                                       #
#   Creating an example visualization to illustrate the core problem    #
#                                                                       #
#########################################################################

# #Styling
# sns.set_context('paper')
# sns.set(font_scale=1)
# sns.set_style("white", {'grid.linestyle': '--'})
# plt.style.use('seaborn-white')

sns.set_style("white", {'grid.linestyle': '--'})


#Creating a synthetic dataset
synthetic_data   = [12,15,19,21,25,29,35,45,65,90,105,190,305,405,420,430,1700,2300,2450,2855,3105]
synthetic_points = ['U','T','S','R','Q','P','O','N','M','L','K','J','I','H','G','F','E','D','C','B','A']
     
#Creating core dataframe
mich24 = pd.DataFrame(synthetic_data,index=synthetic_points)
mich24.columns =['Count']
mich24 = mich24.sort_values(['Count'], ascending=False)
plt.figure(figsize=(12,7))

ax = sns.barplot(mich24.index, 
                 mich24.Count, 
                 color='gray', 
                 alpha=.6, 
                 linewidth=.1, 
                 edgecolor="red",
                 saturation=80)

ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.set(xlabel="\n\n\n", ylabel='Count\n')
ax.set_xticklabels(mich24.index, color = 'black', alpha=.8)

for item in ax.get_xticklabels(): 
    item.set_rotation(0)
    
for i, v in enumerate(mich24["Count"].iteritems()):        
    ax.text(i ,v[1], "{:,}".format(v[1]), color='gray', va ='bottom', rotation=0, ha='center')
    

ax.tick_params(axis='x', which='major', pad=9)    
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)  
#################################################plt.tight_layout()

plt.axvline(4.5, 0,0.95, linewidth=1.4, color="#00274C", label="= Proposed 'Charbonnet Cut'", linestyle="--")

plt.legend(loc='center', fontsize=13)

#  plt.text(3+0.2, 4.5, "An annotation", horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.text(0, -425, "\nThis is a synthetic dataset I created to illustrate a core problem seen when plotting histograms/boxplots with highly variable data", fontsize=11)

#Remove unnecessary chart junk   
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 
# #sns.despine()

plt.title('\n\n\n\nCreating a splitting point can lead to better visualization, if we also plot the second tier/level data...''\n\n',fontsize=12, loc="left")    

plt.text(6.2,700,"|--- This region contains a lack of **visual** insight, we should split data based on Charbonnet Cut ---|", fontsize=10)

plt.show();


# There is nothing wrong with keeping the original plot, but a subplot should be created for the area 'east' of the Cut, to see the RELATIONSHIP between the data points.   
# 
# I will now plot the datapoints east of the cut in its own subplot, for visualization clarity...

# In[ ]:



sns.set_style("white", {'grid.linestyle': '--'})

#Creating a synthetic dataset
synthetic_data   = [12,15,19,21,25,29,35,45,65,90,105,190,305,405,420,430]
synthetic_points = ['U','T','S','R','Q','P','O','N','M','L','K','J','I','H','G','F']
     
#Creating core dataframe
mich24 = pd.DataFrame(synthetic_data,index=synthetic_points)
mich24.columns =['Count']
mich24 = mich24.sort_values(['Count'], ascending=False)
plt.figure(figsize=(12,7))

ax = sns.barplot(mich24.index, 
                 mich24.Count, 
                 color='gray', 
                 alpha=.6, 
                 linewidth=.1, 
                 edgecolor="red",
                 saturation=80)

ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.set(xlabel="\n\n\n", ylabel='Count\n')
ax.set_xticklabels(mich24.index, color = 'black', alpha=.8)

for item in ax.get_xticklabels(): 
    item.set_rotation(0)
    
for i, v in enumerate(mich24["Count"].iteritems()):        
    ax.text(i ,v[1], "{:,}".format(v[1]), color='gray', va ='bottom', rotation=0, ha='center')
    

ax.tick_params(axis='x', which='major', pad=9)    
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)  
plt.tight_layout()

plt.legend(loc='center', fontsize=13)

#  plt.text(3+0.2, 4.5, "An annotation", horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.text(0, -65, "\nNow we can see the relationship in the heights of the 'second tier' (east of the Charbonnet Cut) data...", fontsize=12, color="#00274C")

#Remove unnecessary chart junk   
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 
# #sns.despine()

plt.title('2nd Tier data has been plotted, and now we can see the relationships without data being drowned out...\n\n',fontsize=12, loc="left")    
plt.show();


# **Concept Reference**  -  You will see [this guy](https://mgoblue.com/roster.aspx?rp_id=19098) in the NFL some day
#   * Formulated during the viewing of [this](https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/michvsnd.png) game (there was some beer involved)

# **Mark II Concept: (creating and in process)**
# 
# ````
# Maybe something like this:
# 
# 
# seaborn.barplot(x=None, y=None, 
#   !
#   #--- create these option ---
#   charbonnet=(bool Y|N) 
#   charbonnet.method=(manual,auto)  
#   #---------------------------
#   !
#   !
#   !
#   hue=None, data=None,                 
#   order=None, hue_order=None, estimator=<function mean>, 
#   ci=95, n_boot=1000, units=None, orient=None, 
#   color=None, palette=None, etc etc.) 
#                     
# 
# Notes:
#  Based on algorithm *automatic* split of data into 'tier 1' 
#  and 'tier 2' values, seaborn then plots Gridspec main barplot
#  and also secondary tier 2 barplot to relay to viewer the 
#  relationship in magnitude between values. Option for manual 
#  cut value (x-val) would be supported as well. 
# ````

# <div class="h3"><i>Side Note:</i></div>
# * I did considerable analysis to see if the windspeed would have any factor in the outcome of the running play, and found **no real evidence based on the data that it made any type of difference**
# * I would posit that windspeed and direction would have a considerable impact on passing plays though (go outside and play catch with someone when its windy, a football is not a baseball, even if you put zip on it a football will drift a bit under high wind conditions
# * I wish I had more temp and humidity data, my guess is that it would make a difference in yards gained as defensive linemen started getting fatigued, but then again the offensive line would get fatigued as well possibly the same amount, hard to say without the data 
# * Personal Opinion:  IF one is introduced in doing a serious analysis of this data, and diving beyond the surface, I think its important to constantly keep your eye out for the types of distributions seen (age/height/weight may be gaussian for instance, but there are MANY features/attributes that are inherently skewed, even the actual yards gained).  Where am I going with this ?  You cannot apply the same advanced statistics principles on skewed data that you can on gaussian distributions.  But either way, a general guideline I follow here is that when a feature is skewed in some form, it is IMPORTANT to realize that a better measure of central tendency of the data is median over mean (average)...
# * <u>I am guilty of initial bias:</u>
#   * I was **convinced** that turf versus grass would make a difference in the running back performance, and yet I found no real evidence that running backs perform better on one surface versus the other.  I have no included this analysis but will probably post at some point.  
#   * I will wager a guess:  Based on my experience playing rugby, a *muddy* field seems to slow down even the fastest runners, and offers an advantage to the 'forwards' (think linemen), but the quality and conditions of NFL playing fields never really result in a truly muddy field, and thus to a certain extent it is likely that a well-kept grass field and a turf field are both going to allow the runner to run as fast as he desires... 
#   * I read online extensively on this 'turf vs grass' debate, and there seems to be the understanding that neither field type really offers a substantial quantifiable advantage...so for now we will drop this line of discussion...
# 

# <br>

# <div class="h4"><i>Let's now begin:  Initial Examination of overall running (rushing) yards per play:</i></div>
# * Let's take a look at the most important feature, the yards, which we will need to be able to predict going forward after our machine learning model is created
#     

# In[ ]:


##sns.palplot(sns.color_palette("RdBu_r", 7))

tf = df.query("NflIdRusher == NflId")

sns.set_style("white", {'grid.linestyle': '--'})

fig, ax = plt.subplots(figsize=(11,8))
ax.set_xlim(-10,26)

###ccc['colors'] = ['red' if x <= 0 else 'green' for x in ccc['Yards']]
###specialcolors = ['red' if x <= 0 else 'green' for x in tf.Yards]
c = [ 'r' if i < 0 else 'b' for i in tf.Yards]


sns.distplot(tf.Yards, kde=False, color='b', bins=100, 
            hist_kws={"linewidth": .9, 'edgecolor':'black'})

#########################ax.set_xlim(80,100)

## Remove the x-tick labels:  plt.xticks([])
plt.yticks([])
## This method also hides the tick marks

plt.title('\nCombined Data: Overall distribution of yards gained during an individual running play\n',fontsize=12)
plt.xlabel('\nYards (yd) Gained $\Rightarrow$\n', fontsize=9)
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
plt.axvline(x=0, color='blue', linestyle="--", linewidth=.5)
plt.text(-4.55, 2914, r'Line of Scrimmage >', {'color': 'blue', 'fontsize': 9})
plt.tight_layout()
plt.show();


# <div class="alert alert-block alert-info">
# <b>Histogram Hit:</b>  
# 
# Its important to understand that a histogram is a great starting point to examine the data's distribution, but there is some data that is smoothed out by doing this (as it is inherently a binning process).  It is understood that when examining extremely large datasets it is important to start somewhere, and the histogram in general has many many positives, but I'm creating the term `Histogram Hit` so it is understood it is somewhat of a smoothing process AND depending on the bin size chosen can steer the visualization in many different directions.  I am aiming to create an equation to quantity the actual 'hit' one takes when creating histogrames, but for now I'll introduce the term and come back to this at some later point.  Please understand that a histogram is an A-, its great, it is a quick way of visualizing data, but it is not flawless...</div>

# <div class="h4"><i>Visualizing yards gained and lost:</i></div>
# * I will create a simple visualization encompassing the gains and losses in a random set of games
# * I will grab three random games from our 2018 dataset, and plot the overall **run yards gained and lost** (assuming we combine both of the team's data to just get an idea of the ebb and flow of the plays, so if you see 40 plays, that means between both teams they ran a combined total of 40 plays for instance. 
# * The order is **chronological**, i.e. play 0 is the first play of the chosen game, all the way to lets say play 41, which is the 40th run of the game, I don't differentiate between teams, we are just trying to get an idea of what we are dealing with here... 
# * Zero-yard gains represented as missing bar, so look for 'skips' in the bars, that is a yardage = 0 scenario

# In[ ]:


plt.style.use('dark_background')

#aaa is our temp df
aaa = gold
aaa['IsRunner'] = aaa.NflId == aaa.NflIdRusher
#bbb is now the unique run play runners in the year 2018 only
bbb = aaa[aaa.IsRunner & (aaa.Season == 2018)]
#ccc is now a specific actual game
ccc=bbb[bbb.GameId==2018121601] # we grab random game #1 
ccc = ccc[['Yards']][:]
ccc['colors'] = ['red' if x <= 0 else 'green' for x in ccc['Yards']]
##ccc.sort_values('Yards', inplace=True)
ccc.reset_index(inplace=True)
plt.figure(figsize=(9,14))
plt.hlines(y=ccc.index, xmin=0, xmax=ccc.Yards, color=ccc.colors, alpha=0.8, linewidth=9)
plt.gca().set(ylabel='$Play$\n', xlabel='\n$Yards$')
plt.yticks(fontsize=6)
plt.title('\nPositive and Negative Yards for random NFL game #1 - (2018 Season)\n', fontdict={'size':10})
plt.grid(linestyle='--', alpha=0.5)
sns.despine(top=True, right=True, left=True, bottom=True)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show();


# In[ ]:


ccc


# In[ ]:


plt.style.use('dark_background')

ccc=bbb[bbb.GameId==2018121500]
ccc = ccc[['Yards']][:]
ccc['colors'] = ['red' if x <= 0 else 'green' for x in ccc['Yards']]
##ccc.sort_values('Yards', inplace=True)
ccc.reset_index(inplace=True)
plt.figure(figsize=(9,12), dpi= 300)
plt.hlines(y=ccc.index, xmin=0, xmax=ccc.Yards, color=ccc.colors, alpha=0.8, linewidth=9)
plt.gca().set(ylabel='$Play$\n', xlabel='\n$Yards$')
plt.yticks(fontsize=6)
plt.title('\nPositive and Negative Yards for random NFL game #2 - (2018 Season)\n', fontdict={'size':10})
plt.grid(linestyle='--', alpha=0.5)
sns.despine(top=True, right=True, left=True, bottom=True)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show();


# In[ ]:


plt.style.use('dark_background')

ccc=bbb[bbb.GameId==2018121501]
ccc = ccc[['Yards']][:]
ccc['colors'] = ['red' if x <= 0 else 'green' for x in ccc['Yards']]
##ccc.sort_values('Yards', inplace=True)
ccc.reset_index(inplace=True)
plt.figure(figsize=(9,12), dpi= 300)
plt.hlines(y=ccc.index, xmin=0, xmax=ccc.Yards, color=ccc.colors, alpha=0.8, linewidth=9)
plt.gca().set(ylabel='$Play$\n', xlabel='\n$Yards$')
plt.yticks(fontsize=6)
plt.title('\nPositive and Negative Yards for random NFL game #3 - (2018 Season)\n', fontdict={'size':10})
plt.grid(linestyle='--', alpha=0.5)
sns.despine(top=True, right=True, left=True, bottom=True)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show();


# **What insights can we gain from the above plots ?**
# * Game 1:
#   * A **lot** of run plays (49 total plays combined from the two teams), with one run for over 25 yards (big gain), and a fair amount of longish runs.  Only two runs produced negative yards (one hurt, it was for a loss of 5 yards), so looks like the Offense on both teams is doing pretty good, with only two of the run play for zero-yards gained.  Good game to watch to see trends in running, we have a lot of sample points here.  And we can also analyze WHY there were so many long run plays, i.e. what defensive formation was being run in this game ?  This game was most likely a battle between two teams with strong runningbacks. 
# * Game 2:
#   * Longest run was for 14 yards, but very interesting:  **10** runs were zero-yard gain, **6** were negative gain, and in general I see very little yardage gained even when it was positive.  There is some **excellent**  defense going on in this game, it must have been a hardcore defensive battle...
# * Game 3:
#   * One very long run for 40 yards !  Let's think about this for a second:  If the offense runs lets say 20 run plays in a game, and most of them were for 2-4 yards, you can quickly see how devasting one run of 40 yards can be.  A good NFL rusher can gain 100 total yards in a game, but if one of those alone was for 40 yards, that is a BIG deal.  Here we see how guys like Walter Payton would have carved up defenses, with his long gains.  A nightmare to defend against.  And you can also see how a running back is worth his weight in gold, as he can change the outcome of a game with yardage gains.  
#   * As a side note:  A good rule of thumb is that a pretty good running back is averaging over 4 yards per carry, and below that is 'ok'
# * I would also say that the more I watch NFL games, the more I realize there is an entire element of 'calculated risk' at play

# <div class="h4"><i>2018 - Top 10 Longest Rushes:</i></div>
# * Let's take a look at the top ten most spectacular rushes:
# 

# In[ ]:


cm = sns.light_palette("green", as_cmap=True)
tom = df.query("NflIdRusher == NflId")
tom = tom[tom.Season==2018] 
# tom.columns
# Index(['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation',
#        'Dir', 'NflId', 'DisplayName', 'JerseyNumber', 'Season', 'YardLine',
#        'Quarter', 'GameClock', 'PossessionTeam', 'Down', 'Distance',
#        'FieldPosition', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay',
#        'NflIdRusher', 'OffenseFormation', 'OffensePersonnel',
#        'DefendersInTheBox', 'DefensePersonnel', 'PlayDirection', 'TimeHandoff',
#        'TimeSnap', 'Yards', 'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate',
#        'PlayerCollegeName', 'Position', 'HomeTeamAbbr', 'VisitorTeamAbbr',
#        'Week', 'Stadium', 'Location', 'StadiumType', 'Turf', 'GameWeather',
#        'Temperature', 'Humidity', 'WindSpeed', 'WindDirection'],
#       dtype='object')
# define this dict that will help normalize the data
fixup = {"ARZ":"ARI","BLT":"BAL","CLV":"CLE","HST":"HOU"}
tom.PossessionTeam.replace(fixup, inplace=True)

#tom.groupby('PossessionTeam')['Yards'].agg(max).sort_values(ascending=False)[:10].values.tolist()
# [99, 97, 92, 90, 78, 77, 75, 71, 70, 67]
#tom.groupby('PossessionTeam')['Yards'].agg(max).sort_values(ascending=False)[:10].values.tolist()
#tom.groupby('PossessionTeam')['Yards'].agg(max).sort_values()
#.agg(max).sort_values(ascending=False)[:10].values.tolist()
tom.groupby(['PossessionTeam'], as_index=False)['Yards'].agg(max).set_index('PossessionTeam').sort_values('Yards', ascending=False)[:10].style.set_caption('TOP TEN LONGEST RUNS:').background_gradient(cmap=cm)


# <u>The first was NOT a kickoff return, it was a **handoff**</u>.  &nbsp; Now look at HOW far back he is in the endzone !   
# The guy ran probably 107 yards on that play (but gets credit for 99 yards from line of scrimmage at their own 1 yard line; that day he ran for a total of a staggering 238 yards).  This tied the NFL record for longest rush, set 35 years earlier.   
# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/unbelievable.png" width="700px">
# 
# Derrick Henry from the Tennessee Titans is a [*beast*](https://twitter.com/NFL/status/1070863698791550976?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1070863698791550976&ref_url=https%3A%2F%2Fwww.theguardian.com%2Fsport%2F2018%2Fdec%2F06%2Fderrick-henry-touchdown-titans-jaguars-nfl-99-yards).  He is so money you could hold him sideways, swipe him across an ATM machine, and money would just keep streaming out for hours...
# The best part:  He celebrated by striking the Heisman pose, which is perfectly fine, since he won it in 2015.  
# 
# 1. Derrick Henry - Tennessee Titans
# 1. Lamar Miller	- Houston Texans 
# 1. Nick Chubb	- Cleveland Browns
# 1. Adrian Peterson	- Washington Redskins	
# 

# In[ ]:


# tf = df.query("NflIdRusher == NflId")


# sns.set_style("white", {'grid.linestyle': '--'})

# fig, ax = plt.subplots(figsize=(10,8))

# sns.distplot(tf.Yards, kde=False, color="b", 
#             hist_kws={"linewidth": .9, 'edgecolor':'steelblue'})

# #########################ax.set_xlim(80,100)

# ## Remove the x-tick labels:  plt.xticks([])
# plt.yticks([])
# ## This method also hides the tick marks


# plt.title('\nOverall distribution of yards gained during an individual running play\n',fontsize=12)
# plt.xlabel('\nYards (yd) Gained -->\n')
# sns.despine(top=True, right=True, left=True, bottom=True)
# plt.tight_layout()
# ##################plt.axvline(x=22, color='maroon', linestyle="--", linewidth=.5)

# ###############plt.text(22.8, 114, r'Median: 22 carries', {'color': 'maroon', 'fontsize': 9})


# # sns.set_style("white", {'grid.linestyle': '--'})

# # # sns.set_style("ticks", {'grid.linestyle': '--'})
# # ##sns.set(style="white", palette="muted", color_codes=True)
# # ##sns.set(style="white", palette="muted", color_codes=True)

# # ##t2 = tf.groupby(['GameId','Team'])['PlayId'].count()
# # ##t2 = pd.DataFrame(t2)

# # fig, ax = plt.subplots(figsize=(9,8))


# # sns.distplot(tf.Yards, kde=False, color="b", 
# #             hist_kws={"linewidth": .9, 'edgecolor':'lightgrey'}, bins=24)


# # # #~~~~~~~~~~~ ax.set ~~~~~~~~~~~~~~~~
# # # ##ax.set_xlim(0, 6)
# # # ##ax.set_ylim(0, 6)
# # # ax.set_title('Average yards gained as the season progresses (week by week)\n')
# # # ax.set(ylabel='Yards Gained\n')
# # # ax.set(xlabel='\nWeek Number (in the season)')
# # # ax.yaxis.grid(True)   # Show the horizontal gridlines
# # # ax.xaxis.grid(False)  # Hide x-axis gridlines 
# # # # ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# # # # ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
# # # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # ## Remove the x-tick labels:  plt.xticks([])
# # plt.yticks([])
# # ## This method also hides the tick marks
# # plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',
# #           fontsize=12, loc="left")
# # plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',fontsize=12, loc="left")
# # plt.xlabel('\nNumber of times the ball was run in the game\n')
# # sns.despine(top=True, right=True, left=True, bottom=True)
# # plt.tight_layout()
# plt.axvline(x=22, color='maroon', linestyle="--", linewidth=.5)

# plt.text(22.8, 114, r'Median: 22 carries', {'color': 'maroon', 'fontsize': 9})

# plt.tight_layout()
# plt.show();



# tf = df.query("NflIdRusher == NflId")


# sns.set_style("white", {'grid.linestyle': '--'})

# fig, ax = plt.subplots(figsize=(10,8))
# ax.set_xlim(-10,26)


# sns.distplot(tf.Yards, kde=False, color="b", bins=100,
#             hist_kws={"linewidth": .9, 'edgecolor':'grey'})

# #########################ax.set_xlim(80,100)

# ## Remove the x-tick labels:  plt.xticks([])
# plt.yticks([])
# ## This method also hides the tick marks


# plt.title('\nOverall distribution of yards gained during an individual running play\n',fontsize=12)
# plt.xlabel('\nYards (yd) Gained -->\n')
# sns.despine(top=True, right=True, left=True, bottom=True)
# plt.tight_layout()
# ##################plt.axvline(x=22, color='maroon', linestyle="--", linewidth=.5)

# ###############plt.text(22.8, 114, r'Median: 22 carries', {'color': 'maroon', 'fontsize': 9})
# plt.tight_layout()
# plt.show();

# # tf.Yards.describe()


# <div class="h4"><i>Yards vs Down:</i></div>
# * Plotting the distribution of yards by series 'Down'.  Note that I have configured many of my plots to show granular ultra-precise 1-yard increments ! 
# * In this case, I have removed a few outliers so as to see the general trend of the data 
# * I am doing this for a very specific reason, you will see in a sec    &ensp; - *Tom Bresee*
#     

# * >Note: &nbsp; I am creating a new term I will call 'unit grid'. When creating plots where the y or x axis, depending on the plot, is in a relatively short magnitiude range (lets say approximately 20 units or below), I find it helps to actually use the grid lines to expand to the plot on a somewhat granular basis.  When that is needed or helpful, spacing the grid at 'unit' levels shall be known as 'unit grid', i.e. the grids are every 1 unit on the scale.  I think it helps the viewer quickly quantify actual values, to the point where it approaches the information transfer of a barplot...

# In[ ]:



tf = df.query("NflIdRusher == NflId")
sns.set_style("ticks", {'grid.linestyle': '--'})
######sns.set_style("ticks", {"xtick.major.size":1,"ytick.major.size":1})
flierprops = dict(markerfacecolor='0.75', 
                  markersize=1,
                  linestyle='none')

fig, ax = plt.subplots(figsize=(9,7))

ax.set_ylim(-7, 14)
ax.set_title('Yards Gained by Down\n', fontsize=12)

sns.boxplot(x='Down',
            y='Yards',
            data=tf,
            ax=ax,
            showfliers=False, 
            #color='blue'
            )
            #flierprops=flierprops)
    
#Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 

ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))


# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
    
##ax.legend(frameon=False)

# ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')

ax.set(xlabel='')
ax.set_xticklabels(['1st Down', '2nd Down', '3rd Down', '4th Down'])
plt.tight_layout(); plt.show();


# * First and second down we see almost an identical distribution of runs (i.e. In the first two quarters, the teams run about the same distribution of runs in the quarters)
#   * What i find suprising is that in the first two quarters, examining the plots we see that 25% of the plays generated **less** than 1 yard total gained. Running the ball has risk, it does not always end with yards gained.   
# * In the third down, we see a slight drop in the number of yards gained, and the median yards gained has dropped a solid yard.  In a game of inches, this is a big deal. 
# * Fourth down performance is relatively poor, but then again, most of the time a team does NOT run the ball on 4th down, due to the risk. 

# <div class="alert alert-block alert-warning">
# <b>Warning:</b> It is important to list the sample size for each of the histograms, because one may draw the erroneous conclusion that the number of times the ball was run was the 'same' for each of the downs, when in fact it wasn't...</div>

# In[ ]:


# #value swirl
# YDS_by_down = tf.groupby("Down")['Yards'].size()
# total_run_plays = YDS_by_down.sum()
# df_ydsbydown = pd.DataFrame( {'Down':YDS_by_down.index, 'Count':YDS_by_down.values}).sort_values('Count', ascending=False)

# s = df.style.background_gradient(cmap=cm)
# # zebra = df_ydsbydown.style.set_caption('Top 10 Percentage of plays by Personnel (top 10):').background_gradient(cmap=cm)
# # display(HTML(df_ydsbydown.to_html(index=False)))

# #s = df.style.background_gradient(cmap=cm)
# df_ydsbydown.style.set_caption('Count of plays by Personnel:').background_gradient(cmap=cm)

# temp101 = pd.DataFrame(tf.DefensePersonnel.value_counts())
# temp101.index.name = 'Down'
# temp101.columns=['Play Count']
# temp101.reset_index('')
# cm = sns.light_palette("green", as_cmap=True)
# #s = df.style.background_gradient(cmap=cm)
# temp101.style.set_caption('Count of plays by Personnel:').background_gradient(cmap=cm)



# VANILLA SWIRL
YDS_by_down = tf.groupby("Down")['Yards'].size()
total_run_plays = YDS_by_down.sum()
df_ydsbydown = pd.DataFrame( {'Down':YDS_by_down.index, 'Count':YDS_by_down.values}).sort_values('Count', ascending=False)
df_ydsbydown.set_index('Down', drop=True, inplace=True)
# s = df.style.background_gradient(cmap=cm)
# df_ydsbydown.style.set_caption('Play count per Down:').background_gradient(cmap=cm)
df_ydsbydown['Percentage']=round(df_ydsbydown.Count/total_run_plays*100, 2)
cm = sns.light_palette("green", as_cmap=True)
df_ydsbydown.style.set_caption('PLAY COUNT PER DOWN:').background_gradient(cmap=cm)


# print(df_ydsbydown)
# s = df.style.background_gradient(cmap=cm)
# # zebra = df_ydsbydown.style.set_caption('Top 10 Percentage of plays by Personnel (top 10):').background_gradient(cmap=cm)
# # display(HTML(df_ydsbydown.to_html(index=False)))
# #s = df.style.background_gradient(cmap=cm)
# df_ydsbydown.style.set_caption('Count of plays by Personnel:').background_gradient(cmap=cm)

# temp107 = pd.DataFrame(round(tf.DefensePersonnel.value_counts(normalize=True) * 100,2)).head(10)
# temp107.index.name = 'DefensePersonnel'
# temp107.columns=['Play Percentage']
# cm = sns.light_palette("green", as_cmap=True)
# #s = df.style.background_gradient(cmap=cm)
# temp107.style.set_caption('Top 10 Percentage of plays by Personnel (top 10):').background_gradient(cmap=cm)
# sns.boxplot(x='Down',
#             y='Yards',
#             data=tf,
#             ax=ax,
#             showfliers=False, 
#             #color='blue'
#             )


# ![](http://)> **INSIGHT**: &nbsp; For a sample size of all NFL rush plays over the course of <u>two entire years</u>, only **7.83%** of the runs were on 3rd down, and less than **1%** were rushes on 4th down. 

# <div class="h4"><i>Yards vs Quarter of the Game:</i></div>
# * Plotting the distribution of yards by game quarter, where 5Q symbolizes overtime...

# In[ ]:


tf = df.query("NflIdRusher == NflId")
flierprops = dict(markerfacecolor='0.75', 
                  markersize=1,
                  linestyle='none')
fig, ax = plt.subplots(figsize=(9,7))
ax.set_ylim(-7, 17)
ax.set_title('Yards Gained by Game Quarter\n\n', fontsize=12)

sns.boxplot(x='Quarter',
            y='Yards',
            data=tf,
            ax=ax,
            showfliers=False , 
            #color='blue'
            )
            #flierprops=flierprops)
    
#Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 

ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
# ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
ax.set(xlabel='')
ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])
plt.tight_layout(); plt.show();


# <div class="h4"><i>Yards Gained vs Box Defender Count:</i></div>
# * Plotting the distribution of yards gained vs number of defenders in the box.  We will call this the defensive 'density' count...
# * A helpful reference image I created is shown below. 

# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/thebox.png" width="400px">

# * Where it gets interesting:
#   * If you look thru the data, you see that there are defensive fronts such as 4:3 or 3:4, but that does NOT necessarily mean there are '7 in the box'.  IF the defense stacks an additional defender on the line of scrimmage, etc, it would count as an extra person in the box, i.e. be careful:  you CANNOT say if you have a 4:3 that you have 7 in the box, it MAY be more.  We will have to trust (since there is no physical way of proving or disproving) that the 'DefendersIntheBox' statitics in our dataset is accurate. 
#   * I was hoping to breakout a 4:3 vs a 3:4, but that is NOT possible just by looking at defender in the box category.  Thus it is imporant to understand that a true defensive analysis would revolve around a) finding out the defensive personnel formation of the play (such as '3 DL, 4 LB, 4 DB') AND examining the defender in the box count. One can breakout 4:3 vs 3:4 defenses in general, but to couple it with defender in the box count one must understand that one does not pre-suppose the other  

# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/box2.png" width="400px">

# * One of course would see how the more 'dense' the box is, the more difficult it would be for a rusher to gain considerable yards
# * And now you know why the NFL tracks this stat:
#   * **8+ Defenders in the Box (8+D%):** &nbsp;  'On every play, Next Gen Stats calculates how many defenders are stacked in the box at snap. Using that logic, DIB% calculates how often does a rusher see 8 or more defenders in the box against them.'
#   * And thus we can insert a fairness factor, where rushers should be judged by how often they had a larger defensive 'density' employed against them...

# In[ ]:


dff = tf[tf.DefendersInTheBox>2]
dff.DefendersInTheBox = dff.DefendersInTheBox.astype('int')

flierprops = dict(markerfacecolor='0.75', 
                  markersize=1,
                  linestyle='none')

fig, ax = plt.subplots(figsize=(9,7))
ax.set_ylim(-7, 23)
ax.set_title('Yards Gained vs number of Defenders in the box\n\n', fontsize=12)
sns.boxplot(x='DefendersInTheBox',
            y='Yards',
            data=dff,
            ax=ax,
            showfliers=False , 
            #color='blue'
            )
            #flierprops=flierprops)
#Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 

ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
# ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
ax.set(xlabel="\nNumber of defensive players in the 'Box'\n\n")
# ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])

# plt.title('My subtitle',fontsize=16)
# plt.suptitle('My title',fontsize=24, x=0, y=1,ha="left")
# plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)

plt.tight_layout()
plt.show();


# * When there are nine defensive players in the box, 25% of the runs gained LESS than 0 yards, and half the runs were for LESS than 2 yards. 
# * It is very rare for defenses to line up with four or less players, but when they do, the Offense seems to gain a fair amount of yards.

# * <div class="h4"><i>Yards Gained vs Defensive Personnel 'Groupings':</i></div>
# * Plotting the distribution of yards gained vs Defensive Formation...
# * First lets start by looking at the combined 2017/2018 dataset formations by play count (i.e. how many times in the two year season data that particular Defensive Schema were run)
# * We will then look exclusively look at 2018 stats

# In[ ]:



temp101 = pd.DataFrame(tf.DefensePersonnel.value_counts())
temp101.index.name = 'DefensePersonnel'
temp101.columns=['Play Count']

cm = sns.light_palette("green", as_cmap=True)

#s = df.style.background_gradient(cmap=cm)

temp101.style.set_caption('Count of plays by Personnel:').background_gradient(cmap=cm)


# tf.DefensePersonnel.value_counts()
# 4 DL, 2 LB, 5 DB          6358
# 4 DL, 3 LB, 4 DB          6205
# 3 DL, 4 LB, 4 DB          3656
# 2 DL, 4 LB, 5 DB          2588
# 3 DL, 3 LB, 5 DB          2222
# 2 DL, 3 LB, 6 DB           529
# 4 DL, 1 LB, 6 DB           418
# 4 DL, 4 LB, 3 DB           237
# 3 DL, 2 LB, 6 DB           193
# 5 DL, 2 LB, 4 DB           161
# 5 DL, 3 LB, 3 DB           108
# 1 DL, 4 LB, 6 DB            65
# 3 DL, 5 LB, 3 DB            64
# 6 DL, 4 LB, 1 DB            56
# 5 DL, 4 LB, 2 DB            53
# 6 DL, 3 LB, 2 DB            47
# 5 DL, 1 LB, 5 DB            41
# 6 DL, 2 LB, 3 DB            32
# 1 DL, 5 LB, 5 DB            31
# 2 DL, 5 LB, 4 DB            22
# 1 DL, 3 LB, 7 DB            13
# 2 DL, 2 LB, 7 DB            13
# 3 DL, 1 LB, 7 DB            12
# 5 DL, 5 LB, 1 DB             7
# 5 DL, 3 LB, 2 DB, 1 OL       7
# 0 DL, 5 LB, 6 DB             6
# 4 DL, 5 LB, 2 DB             5
# 0 DL, 4 LB, 7 DB             4
# 4 DL, 0 LB, 7 DB             3
# 2 DL, 4 LB, 4 DB, 1 OL       3
# 5 DL, 4 LB, 1 DB, 1 OL       3
# 4 DL, 6 LB, 1 DB             2
# 0 DL, 6 LB, 5 DB             2
# 6 DL, 1 LB, 4 DB             1
# 3 DL, 4 LB, 3 DB, 1 OL       1
# 4 DL, 5 LB, 1 DB, 1 OL       1
# 1 DL, 2 LB, 8 DB             1
# 7 DL, 2 LB, 2 DB             1


# tf.DefensePersonnel.sort_values().unique()
# array(['0 DL, 4 LB, 7 DB', '0 DL, 5 LB, 6 DB', '0 DL, 6 LB, 5 DB',
#        '1 DL, 2 LB, 8 DB', '1 DL, 3 LB, 7 DB', '1 DL, 4 LB, 6 DB',
#        '1 DL, 5 LB, 5 DB', '2 DL, 2 LB, 7 DB', '2 DL, 3 LB, 6 DB',
#        '2 DL, 4 LB, 4 DB, 1 OL', '2 DL, 4 LB, 5 DB', '2 DL, 5 LB, 4 DB',
#        '3 DL, 1 LB, 7 DB', '3 DL, 2 LB, 6 DB', '3 DL, 3 LB, 5 DB',
#        '3 DL, 4 LB, 3 DB, 1 OL', '3 DL, 4 LB, 4 DB', '3 DL, 5 LB, 3 DB',
#        '4 DL, 0 LB, 7 DB', '4 DL, 1 LB, 6 DB', '4 DL, 2 LB, 5 DB',
#        '4 DL, 3 LB, 4 DB', '4 DL, 4 LB, 3 DB', '4 DL, 5 LB, 1 DB, 1 OL',
#        '4 DL, 5 LB, 2 DB', '4 DL, 6 LB, 1 DB', '5 DL, 1 LB, 5 DB',
#        '5 DL, 2 LB, 4 DB', '5 DL, 3 LB, 2 DB, 1 OL', '5 DL, 3 LB, 3 DB',
#        '5 DL, 4 LB, 1 DB, 1 OL', '5 DL, 4 LB, 2 DB', '5 DL, 5 LB, 1 DB',
#        '6 DL, 1 LB, 4 DB', '6 DL, 2 LB, 3 DB', '6 DL, 3 LB, 2 DB',
#        '6 DL, 4 LB, 1 DB', '7 DL, 2 LB, 2 DB'], dtype=object)


# Let's examine the same values, but broken out by percentage, i.e. what percentage of the time did the run play go against a particular DefensePersonnel Schema, and lets grab the top 10, since after that there is an extremely small percentage of plays incorporating that style: 

# In[ ]:


temp107 = pd.DataFrame(round(tf.DefensePersonnel.value_counts(normalize=True) * 100,2)).head(10)
temp107.index.name = 'DefensePersonnel'
temp107.columns=['Percentage']
cm = sns.light_palette("green", as_cmap=True)

#s = df.style.background_gradient(cmap=cm)
temp107.style.set_caption('Top 10 Percentage of plays by Defensive Personnel Grouping:').background_gradient(cmap=cm)


# **The Top Five Formations:**
# 
# 1. 4 DL, 2 LB, 5 DB
#   * 4 linemen, 2 linebackers, and 5 defensive backs (6 in the 'box')
# 2. 4 DL, 3 LB, 4 DB	
#   * your conventional *4:3* type defense (7 in the 'box')
# 3. 3 DL, 4 LB, 4 DB
#   * your conventional *3:4* type defense  (7 in the 'box')
# 4. 2 DL, 4 LB, 5 DB	
#   * four linebackers, with only 2 guys on the line  (6 in the 'box')
# 5. 3 DL, 3 LB, 5 DB
#   * a type of formation build to stop the pass (6 in the 'box')
#  

# In[ ]:



sns.set_style("ticks", {'grid.linestyle': '--'})

pers = tf
dff = pers 

flierprops = dict(markerfacecolor='0.2', 
                  markersize=1,
                  linestyle='none')

fig, ax = plt.subplots(figsize=(9,12))
ax.set_ylim(-7, 22)
ax.set_title('\nAverage yards gained by Defensive Personnel Schema\n', fontsize=12)



# sns.boxplot(y='DefensePersonnel',
#             x='Yards',
#             data=dff,
#             ax=ax,
#             showfliers=False , 
#             #color='blue'
#             )

sns.boxplot(y=dff['DefensePersonnel'].sort_values(ascending=False),
            x=dff['Yards'],
            ax=ax,
            showfliers=False ,
            linewidth=.8
            #color='blue'
            )


            #flierprops=flierprops)
#Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.yaxis.grid(False)   # Show the horizontal gridlines
ax.xaxis.grid(True)  # Hide x-axis gridlines 

ax.xaxis.set_major_locator(plt.MultipleLocator(1))
#ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
# ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
ax.set(xlabel="\nYards Gained\n")

# plt.title('My subtitle',fontsize=16)
# plt.suptitle('My title',fontsize=24, x=0, y=1,ha="left")
# plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)

# ax.spines['top'].set_linewidth(0)  
# ax.spines['left'].set_linewidth(.3)  
# ax.spines['right'].set_linewidth(0)  
# ax.spines['bottom'].set_linewidth(.3) 
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_ticks_position('none') 

# ax.fill_between(t, upper_bound, X, facecolor='blue', alpha=0.5)
# plt.axhspan(9,10)  #horizontal shading
# plt.axvspan(9,10)  #horizontal shading

#ax.text(15,78, "#1", ha='center')

ax.text(15,17.3, '#1',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=12)

ax.text(15,16.3, '#2',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)

ax.text(15,21.3, '#3',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)

ax.text(15,24.3, '#5',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)

ax.text(15,27.3, '#4',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)

ax.text(9,2, '6 guys on the line',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)


ax.text(0,.2, 'line of scrimmage >',
        verticalalignment='bottom', horizontalalignment='right',
        color='blue', fontsize=9)


#-----more control-----#
ax.grid(linestyle='--', 
        linewidth='0.3', 
        color='lightgray', 
        alpha=0.8,
        axis='x'
       )

plt.axvline(0, 0,1, linewidth=.4, color="blue", linestyle="--")

plt.tight_layout()
plt.show();


# **Very very interesting...**  
# * This analysis is important to understanding how defense is set up, its critical to understanding how to predict run productivity
# * The `#1` used defensive scheme actually resulted in slightly longer yardage plays, but also slightly higher yards lost for the offense
# * You can see how the first and second scheme is relatively good at containing the run, and as you get lower on the y-axis, you are giving up higher and higher yards
# * The most common is a 4-2-5, which is a good coverage against the pass (you have 5 DB). And since roughly 35-40% of NFL plays are runs, and the other percentage are pass, you can see why this is common. 
# * With a median yardage gain allowed of 4 yards, its pretty good against the run, AND you can see that in some cases you can get losses of up to -6 yards. 75% of the runs against this defense are held to 6 yards or less. 
# * The next most popular is your typical 4-3 defense, where you can see it holds runs to a bit shorter yardage, obviously since you have an extra linebacker involved in the tackling. But what i find interesting is that the third most common defense (3-4) has almost PRECISELY characteristics based on the data, look at the boxplots.  The 3-4 is run less than the 4-3 and the 4-2, but it seems to hold up pretty well against the run. The 3-4 in a flexible defense, and provides some great advantages when it comes to rushing the quarterback and defending against the pass. The 3-4 can be confusing for opposing quarterbacks, who find that wherever they look downfield there is a linebacker.  IF one could argue that the 3-4 is a better defense than the 4-3 in terms of rushing the QB, AND it holds up relatively well against the run (as it appears it does), then it would appear more teams SHOULD be running the 3-4 !  
# * The `#2` and `#3` most occurring run defense resulted in almost precisely the same running yards allowed distribution
# * Having 6 men on the line may appear to be a great idea against the run (and it does seems to 'squash' the run), you see that although it lowers the potential yards a runner could get, it offers no real ability to gain you negative yards on run plays, and its penetration ability are limited.  It does work well against runs, BUT if the play is a pass, you are devasted as you have very few DB to stop the pass. 
# * When there are nine defensive players in the box, 25% of the runs gained LESS than 0 yards, and half the runs were for LESS than 2 yards. 
# * It is very rare for defenses to line up with four or less players, but when they do, the Offense seems to gain a fair amount of yards.

# In[ ]:



# pers = tf
# dff = pers 

# # sns.boxplot(y=dff['DefensePersonnel'].sort_values(ascending=False),
#             x=dff['Yards'],
#             ax=ax,
#             showfliers=False ,
#             linewidth=.8
#             #color='blue'
#             )

# ax.yaxis.grid(False)   # Show the horizontal gridlines
# ax.xaxis.grid(True)  # Hide x-axis gridlines 

# ax.xaxis.set_major_locator(plt.MultipleLocator(1))
# #ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# # Add transparency to colors
# for patch in ax.artists:
#   r, g, b, a = patch.get_facecolor()
#   patch.set_facecolor((r, g, b, .3))
    
# # ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
# ax.set(xlabel="\nYards Gained\n")

# sns.despine(top=True, right=True, left=True, bottom=True)
# ax.xaxis.set_ticks_position('none') 

# # ax.fill_between(t, upper_bound, X, facecolor='blue', alpha=0.5)
# # plt.axhspan(9,10)  #horizontal shading
# # plt.axvspan(9,10)  #horizontal shading

# #ax.text(15,78, "#1", ha='center')

# #-----more control-----#
# ax.grid(linestyle='--', 
#         linewidth='0.3', 
#         color='lightgray', 
#         alpha=0.8,
#         axis='x'
#        )

# plt.axvline(0, 0,1, linewidth=.4, color="blue", linestyle="--")
# plt.tight_layout()
# plt.show();




# from bokeh.plotting import figure, output_file, show

# p = figure(plot_width=800, plot_height=700,
#            title = '\nYards by Distance\n',
#            x_axis_label ='Distance to Go\n',
#            y_axis_label ='Yards\n')

# p.circle(dff.Distance,
#          dff['Yards'],
#          size=5, 
#          color="navy", 
#          alpha=0.5)


# show(p); 






# from numpy import linspace
# from scipy.stats.kde import gaussian_kde
# from bokeh.io import output_file, show
# from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter
# from bokeh.plotting import figure
# from bokeh.sampledata.perceptions import probly
# import colorcet as cc


# def ridge(category, data, scale=20):
#     return list(zip([category]*len(data), scale*data))

# cats = list(reversed(probly.keys()))

# palette = [cc.rainbow[i*15] for i in range(17)]

# x = linspace(-20,110, 500)

# source = ColumnDataSource(data=dict(x=x))

# p = figure(y_range=cats, plot_width=700, x_range=(-5, 105), toolbar_location=None)

# for i, cat in enumerate(reversed(cats)):
#     pdf = gaussian_kde(probly[cat])
#     y = ridge(cat, pdf(x))
#     source.add(y, cat)
#     p.patch('x', cat, color=palette[i], alpha=0.6, line_color="black", source=source)

# p.outline_line_color = None
# p.background_fill_color = "#efefef"

# p.xaxis.ticker = FixedTicker(ticks=list(range(0, 101, 10)))
# p.xaxis.formatter = PrintfTickFormatter(format="%d%%")

# p.ygrid.grid_line_color = None
# p.xgrid.grid_line_color = "#dddddd"
# p.xgrid.ticker = p.xaxis[0].ticker

# p.axis.minor_tick_line_color = None
# p.axis.major_tick_line_color = None
# p.axis.axis_line_color = None

# p.y_range.range_padding = 0.12

# show(p); 






# In[ ]:



pers = tf
dff = pers 


# sns.boxplot(y=dff['DefensePersonnel'].sort_values(ascending=False),
#             x=dff['Yards'],
#             ax=ax,
#             showfliers=False ,
#             linewidth=.8
#             #color='blue'
#             )

# ax.yaxis.grid(False)   # Show the horizontal gridlines
# ax.xaxis.grid(True)  # Hide x-axis gridlines 

# ax.xaxis.set_major_locator(plt.MultipleLocator(1))
# #ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# # Add transparency to colors
# for patch in ax.artists:
#   r, g, b, a = patch.get_facecolor()
#   patch.set_facecolor((r, g, b, .3))
    
# # ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
# ax.set(xlabel="\nYards Gained\n")

# sns.despine(top=True, right=True, left=True, bottom=True)
# ax.xaxis.set_ticks_position('none') 

# # ax.fill_between(t, upper_bound, X, facecolor='blue', alpha=0.5)
# # plt.axhspan(9,10)  #horizontal shading
# # plt.axvspan(9,10)  #horizontal shading

# #ax.text(15,78, "#1", ha='center')

# #-----more control-----#
# ax.grid(linestyle='--', 
#         linewidth='0.3', 
#         color='lightgray', 
#         alpha=0.8,
#         axis='x'
#        )

# plt.axvline(0, 0,1, linewidth=.4, color="blue", linestyle="--")
# plt.tight_layout()
# plt.show();



from bokeh.plotting import figure, output_file, show

p = figure(plot_width=800, plot_height=700,
           title = '\nYards by Rusher Weight\n',
           x_axis_label ='Rusher Weight (in lbs)\n',
           y_axis_label ='Yards\n')

p.circle(dff.PlayerWeight,
         dff['Yards'],
         size=5, 
         color="navy", 
         alpha=0.5)


show(p); 


# <div class="h4"><i>Yards gained average vs season week number:</i></div>
# * Plotting average yards gained per play, on a week by week basis as the season transpires   

# In[ ]:



#MDK

# # sns.set(style="white", palette="muted", color_codes=True)
# sns.set_style("ticks", {'grid.linestyle': '--'})


# t = tf[['Week', "Yards"]].groupby('Week').mean().sort_values(by = "Yards")
# ###sns.set_style("white", {'grid.linestyle': '--'})
# fig, ax = plt.subplots(figsize=(8,7))

# sns.barplot(x=t.index,
#             y=t.Yards,
#             ax=ax, 
#             linewidth=.5, 
#             facecolor=(1, 1, 1, 0),
#             errcolor=".2", 
#             edgecolor=".2")

        
# # for patch in ax.artists:
# #     r, g, b, a = patch.get_facecolor()
# #     patch.set_facecolor((r, g, b, .1))
    
# #~~~~~~~~~~~ ax.set ~~~~~~~~~~~~~~~~
# ##ax.set_xlim(0, 6)
# ax.set_ylim(2, 6)
# ax.set_title('Average yards gained per play as the season progresses (week by week)\n', fontsize=12)
# ax.set(ylabel='Yards Gained\n')
# ax.set(xlabel='\nWeek Number (in the season)')
# ax.yaxis.grid(True)   # Show the horizontal gridlines
# ax.xaxis.grid(False)  # Hide x-axis gridlines 
# ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

# plt.tight_layout()
# plt.show();


# In[ ]:



# clrs = ['grey' if (x < max(values)) else 'green' for x in values ]
# sns.barplot(x=labels, y=values, palette=clrs) # color=clrs)
# #Rotate x-labels 
# plt.xticks(rotation=40)

sns.set(style="white", palette="muted", color_codes=True)

#sns.set_style("ticks", {'grid.linestyle': '--'})
# # this may not work right
# sns.set_style({'grid.linestyle': '--'}, )

t = tf[['Week', "Yards"]].groupby('Week').mean().sort_values(by = "Yards")
###sns.set_style("white", {'grid.linestyle': '--'})

fig, ax = plt.subplots(figsize=(9,6))

specific_colors=['grey']*17
specific_colors[8]='#ffbf00'
specific_colors[5]='#169016'

#print(specific_colors)

#sns.set_color_codes('pastel')             
sns.barplot(x=t.index,
            y=t.Yards,
            ax=ax, 
            linewidth=.2, 
            #color='red'
            #facecolor='#888888',
            #facecolor=(1, 1, 1, 0),
            #facecolor='specific_colors',
            #errcolor=".2",
            edgecolor="black",
            palette=specific_colors)

#~~~~~~~~~~~ ax.set ~~~~~~~~~~~~~~~~
##ax.set_xlim(0, 6)
ax.set_ylim(0, 5.5)
ax.set_title('\nOverall Average yards gained per play as the season progresses (week by week)\n\n', fontsize=11)
# ax.set(ylabel='Yards Gained\n', rotation='horizontal')
ax.set(xlabel='\nWeek Number (in the season)')
ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 
ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

#-----more control-----#
ax.grid(linestyle='--', 
        linewidth='0.7', 
        color='lightgray', 
        alpha=0.9,
        axis='y'
       )

# Don't allow the axis to be on top of your data
# ax.set_axisbelow(True)

for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))
ax.spines['top'].set_linewidth(0)  
ax.spines['left'].set_linewidth(.3)  
ax.spines['right'].set_linewidth(0)  
ax.spines['bottom'].set_linewidth(.3) 

plt.ylabel("YDS\n", fontsize=11, rotation=90)

plt.tight_layout()
plt.show()



#----------------------------------------------------------------------------




#https://seaborn.pydata.org/generated/seaborn.lineplot.html
t = tf[['Week', "Yards"]].groupby('Week').mean().sort_values(by = "Yards")
t['WeekInSeason']= t.index
t.reset_index(drop=True, inplace=True)
starter= t.loc[0,'Yards']
t['gain']=t.Yards/starter
t['gainpct']=round(100*(t.gain-1), 3)


fig, ax = plt.subplots(figsize=(9.5,5))


sns.lineplot(x="WeekInSeason", y="gainpct", data=t, 
            color='grey', 
            ax=ax,
            markers=True, marker='o', 
            #palette=specific_colors, 
            dashes=True) 

ax.set_title('\nPercent Gain in the average running yards gained per play (week by week)\n\n', fontsize=11)

# ax.xaxis.set_major_locator(plt.MultipleLocator(13))
# ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.set(ylabel='Gain in average YDS per carry (in %)\n')

ax.set(xlabel='\nWeek Number (in the season)')
ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 


ax.spines['top'].set_linewidth(0)  
ax.spines['left'].set_linewidth(.3)  
ax.spines['right'].set_linewidth(0)  
ax.spines['bottom'].set_linewidth(.3); 

plt.tight_layout()
plt.show(); 


# * **First Plot:** 
#   * Barplot of average yards gained per game per week    
#   
# 
# * **Second Plot:**
#   * Our baseline is week 1, and then from there, we compare where the run game is at week by week to that baseline, i.e. if in week 5 the average yards/carry is 10% more compared to week 1 game, then we graph that value in the plot...
#   * Progress:
#     * It appears in the first month of the season, there is a strong climb in runner performance.  By week 6, the runners are peaking in terms of productivity.  Potential fatigue factor kicks in two months into the season, then strong push for the second half of the season as the teams are getting stronger and stronger, making a run towards the playoffs.  

# <div class="h4"><i>Number of run plays called per NFL game per team:</i></div>
# * Histogram plot of the total number of run plays called per game per season.  
# * This takes into consideration every game played, where each team takes turns calling run plays (and contains both 2017 and 2018 data)
# 

# In[ ]:


sns.set_style("white", {'grid.linestyle': '--'})

# sns.set_style("ticks", {'grid.linestyle': '--'})
##sns.set(style="white", palette="muted", color_codes=True)
##sns.set(style="white", palette="muted", color_codes=True)

t2 = tf.groupby(['GameId','Team'])['PlayId'].count()
t2 = pd.DataFrame(t2)
fig, ax = plt.subplots(figsize=(9,7))

sns.distplot(t2.PlayId, kde=False, color="b", 
            hist_kws={"linewidth": .9, 'edgecolor':'black'}, bins=24)


# #~~~~~~~~~~~ ax.set ~~~~~~~~~~~~~~~~
# ##ax.set_xlim(0, 6)
# ##ax.set_ylim(0, 6)
# ax.set_title('Average yards gained as the season progresses (week by week)\n')
# ax.set(ylabel='Yards Gained\n')
# ax.set(xlabel='\nWeek Number (in the season)')
# ax.yaxis.grid(True)   # Show the horizontal gridlines
# ax.xaxis.grid(False)  # Hide x-axis gridlines 
# # ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# # ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Remove the x-tick labels:  plt.xticks([])
plt.yticks([])
## This method also hides the tick marks
plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',
          fontsize=12, loc="left")
plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',fontsize=12, loc="left")
plt.xlabel('\nNumber of times a team ran the ball in the game\n', fontsize=9)
sns.despine(top=True, right=True, left=True, bottom=True)
plt.axvline(x=22, color='maroon', linestyle="--", linewidth=.5)

plt.text(22.8, 114, r'Median: 22 carries', {'color': 'maroon', 'fontsize': 9})
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))


plt.tight_layout()
plt.show();


# * The median number of times there is a run play per team in a game is 22, i.e. if a single running back was used, he would be running roughly 22 plays per game, but there is a fairly wide variation here from 10 up to about 40 plays in a game.  30 is considered a fair number of plays for a running back, beyond 40 is considered *extreme* for a single player...
# * Distribution appears to be bi-modal, where there is a peak at 20 and a peak at about 28 carries.  One could argue this could even be the difference between teams that run the ball a fair amount (as part of their offensive strategy), and those that choose to prefer the pass with a balance of some running plays to keep the defense off guard...
# * This does bring up the fact that to play in the NFL, as a premier running back you will be getting the ball many times, and **durability** becomes a major factor as the season goes on ! 
# * Also, remember that the RB does not run every run play, sometimes there are substitutes made

# Diving Deeper:  
# 
# The below plot is an exact count visualization of the number of run plays that occurred in a game, specifically in the entire 2018 season 
# * By using the swarmplot, we see the precise distribution - and this gives a better representation of the distribution of values (where 1:1 viz, i.e. one dot is one game that had a specific count of run plays)
# * We also can **quickly** see the second, third, and fourth most run play count in a random game

# In[ ]:


#number_plays_2018 = bbb.groupby(['GameId'], as_index=False).agg({'PlayId': 'nunique'})
number_plays_2018_perteam = bbb.groupby(['GameId', 'Team'], as_index=False).agg({'PlayId': 'nunique'})

sns.set_style("white", {'grid.linestyle': '--'})
fig, ax = plt.subplots(figsize=(7,7))

#Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

#ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(True)  # Hide x-axis gridlines 

ax.xaxis.set_major_locator(plt.MultipleLocator(5))
#ax.yaxis.set_minor_locator(plt.MultipleLocator(5))


sns.swarmplot(number_plays_2018_perteam.PlayId, color="b", ax=ax)
sns.despine(top=True, right=True, left=True, bottom=True)

plt.ylabel('The Number of Teams that ran the x-axis play count value\n', fontsize=10)

plt.xlabel('\nTotal Run Plays by a Team in an entire game', fontsize=10)
plt.title('\n2018 Season: Number of Run Plays Distribution by Team\n',fontsize=12, loc="left")


# - - - - - - - - 
plt.tight_layout()
plt.show();


# Add transparency to colors
# for patch in ax.artists:
#   r, g, b, a = patch.get_facecolor()
#   patch.set_facecolor((r, g, b, .3))


# * What we find interesting is the peaks are not **that** pronouced though, i.e. there are many teams that will run the ball 17, 18, 19, 20, up to 22 times in a single game, and also a fair amount of teams that will run the ball 24, 25, 26, up to 27 times in a single game...
# * It should be noted that this is an intriguing factor:
#   * In a single game, there are not a tremendous number of run plays either way, meaning our sample size per team per game of run plays is somewhat limited, so deriving a predictive model will contain many factors with a number of samples that is relatively small, offering a challenge...

# Continuing Analysis:
# 

# <br>

# <div class="h4"><i>Total Rushing Yards per NFL Team:</i></div>
# * The following shows the total run yards per team, over the course of two individual seasons.  
# * I specifically use the total over two years to show the effect the running game can have on a team's performance.  I will eventually plot the yards on a per game average basiss, but but the point here is to show the vast amount of offensive yards that the top teams had over the others.  
# * The New England Patriots won the 2018 season superbowl (against the LA Rams).  I believe the running offense was a major factor in that. 
# * **Note:** I include a new plotting term called **`icicles`** to enhance the visualization of barplots.  Using `icicles`, one can not clutter the plot excessively but still relay x-axis values superimposed onto the chart.  Thus it is not necessary to cover the entire plot with a grid, but rather only the section that specifically needs it and where it is pertinent.  
#   * *This term does not currently exist in mainstream visualization, I'm creating it.*

# In[ ]:


plt.style.use('dark_background')


df04 = tf.groupby('PossessionTeam')['Yards'].agg(sum).sort_values(ascending=True)
df04 = pd.DataFrame(df04)
df04['group'] = df04.index

my_range=range(1,33)

fig, ax = plt.subplots(figsize=(9,9))

# Create a color if the group is "B"
##my_color=np.where(df04['group']=='NE', 'orange', 'skyblue')

##my_color=np.where(df04[  ('group'=='NE') | ('group'=='NO')  ], 'orange', 'skyblue')

my_color=np.where( (df04.group == 'NE') | (df04.group == 'NO') | (df04.group == 'LA') , 'orange', 'skyblue')

##movies[(movies.duration >= 200) | (movies.genre == 'Drama')]
##df04[(df04.group == 'NE') | (df04.group == 'NO') ]
##(movies.duration >= 200) & (movies.genre == 'Drama')

my_size=np.where(df04['group']=='B', 70, 30)
 
plt.hlines(y=my_range, xmin=0, xmax=df04['Yards'], color=my_color, alpha=0.4)

plt.scatter(df04.Yards, my_range, color=my_color, s=my_size, alpha=1)
 
# Add title and exis names
plt.yticks(my_range, df04.group)
plt.title("\nTotal Rushing Yards per Team \n(over the course of two NFL seasons)\n\n", loc='left', fontsize=12)
plt.xlabel('\n Total Rushing Yards', fontsize=10)
plt.ylabel('')
##############plt.ylabel('NFL\nTeam\n')

ax.spines['top'].set_linewidth(.3)  
ax.spines['left'].set_linewidth(.3)  
ax.spines['right'].set_linewidth(.3)  
ax.spines['bottom'].set_linewidth(.3)  


plt.text(0, 33.3, r'Top Three:  LA Rams, New England Patriots, and New Orleans Saints absolutely dominating the rushing game...', {'color': 'white', 'fontsize': 8.5})
sns.despine(top=True, right=True, left=True, bottom=True)

plt.text(4005, 2, '<-- I call these icicles', {'color': 'white', 'fontsize': 8})

plt.axvline(x=3500, color='lightgrey', ymin = .01, ymax=.82, linestyle="--", linewidth=.4)
plt.axvline(x=4000, color='lightgrey', ymin = .01, ymax=.9, linestyle="--", linewidth=.4)
plt.axvline(x=3000, color='lightgrey', ymin = .01, ymax=.43, linestyle="--", linewidth=.4)
plt.axvline(x=2500, color='lightgrey', ymin = .01, ymax=.07, linestyle="--", linewidth=.4)

plt.tight_layout()
plt.show();


# <div class="h4"><i>Correlation between PlayerWeight and JerseyNumber:</i></div>
# *  If you run a correlation between generalized player weight and jersey number and position, you see high correlation, but why ? 
# *  We accidentally have a good feature to use, which is on the surface jersey number shouldn't matter in any of this, **but** if we change JerseyNumber into a categorical bin (1-19), (20-29), we see that it can be quite helpful.  Because only certain positions are actually allowed to wear certain ranges of jersey numbers.  Thus during our modelling we will in fact include jersey number into bins.  
# 

# <div class="h4"><i>Analyzing Player Speeds:</i></div>
# > The very fastest players on the team are **not** always the ones that run the ball alot, in fact, WR (wide receivers) are generally quite fast, but rushing the ball in the NFL is not a track meet.  It could be if track introduced 300lb guys every 3 yards that were trying to level you. Let's examine the speeds we observed. 

# In[ ]:


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sns.set_style("white", {'grid.linestyle': '--'})
# sns.set_style("ticks", {'grid.linestyle': '--'})
##sns.set(style="white", palette="muted", color_codes=True)
##sns.set(style="white", palette="muted", color_codes=True)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
speed = bbb.groupby(['DisplayName'])['S'].agg('max').sort_values(ascending=True)
speed = pd.DataFrame(speed)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots(figsize=(9,7))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sns.distplot(speed, kde=False, color="m", 
 hist_kws={"linewidth": .9, 'edgecolor':'lightgrey'}, bins=38)
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_yticklabels([])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.title('\nDistribution of running speed for all players in the 2017/2018 seasons (yds/s)\n',
           fontsize=12, loc="center")
ax.set(xlabel="\nRunner Speed (yds/sec)\n")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.tight_layout()
plt.show();
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# # #~~~~~~~~~~~ ax.set ~~~~~~~~~~~~~~~~
# # ##ax.set_xlim(0, 6)
# # ##ax.set_ylim(0, 6)
# # ax.set_title('Average yards gained as the season progresses (week by week)\n')
# # ax.set(ylabel='Yards Gained\n')
# # ax.set(xlabel='\nWeek Number (in the season)')
# # ax.yaxis.grid(True)   # Show the horizontal gridlines
# # ax.xaxis.grid(False)  # Hide x-axis gridlines 
# # # ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# # # ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
# # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ## Remove the x-tick labels:  plt.xticks([])
# plt.yticks([])
# ## This method also hides the tick marks
# plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',
#           fontsize=12, loc="left")
# plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',fontsize=12, loc="left")
# plt.xlabel('\nNumber of times the ball was run in the game\n')
# plt.tight_layout()
# plt.axvline(x=22, color='maroon', linestyle="--", linewidth=.5)

# plt.text(22.8, 114, r'Median: 22 carries', {'color': 'maroon', 'fontsize': 9})
# plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))


# plt.tight_layout()
# plt.show()


# ttt = bbb[bbb.Position.isin(['RB','WR','HB'])]

# ttt['kg']=ttt["PlayerWeight"] * 0.45359237
# ttt['Force_Newtons']=ttt['kg'] * ttt['A'] * 0.9144
# tips = ttt[['Force_Newtons', 'Yards']]

# sns.scatterplot(x="Force_Newtons", y="Yards", data=tips, s=1, ax=ax, color='r', markers='o', edgecolor='r')
# #sns.lmplot(x="Force_Newtons", y="Yards", data=tips)

# plt.title('Correlation between Yards Gained and Player Kinetic Force',fontsize=12)
# plt.suptitle('Kinetic Force',fontsize=13, x=0, y=1,ha="left")
# ##plt.text(x=4.7, y=14.7, s='Sepal Length vs Width', fontsize=10, weight='bold')

# ax.set(xlabel="\nPlayer Kinetic Force\n")
# ax.set(ylabel="Yards Gained\n")

# sns.despine(top=True, right=True, left=True, bottom=True)


# # dff = tf[tf.DefendersInTheBox>2]
# # dff.DefendersInTheBox = dff.DefendersInTheBox.astype('int')

# # flierprops = dict(markerfacecolor='0.75', 
# #                   markersize=1,
# #                   linestyle='none')

# # fig, ax = plt.subplots(figsize=(9,7))
# # ax.set_ylim(-7, 23)
# # ax.set_title('Yards Gained vs number of Defenders in the box\n\n', fontsize=12)
# # sns.boxplot(x='DefendersInTheBox',
# #             y='Yards',
# #             data=dff,
# #             ax=ax,
# #             showfliers=False , 
# #             #color='blue'
# #             )
# #             #flierprops=flierprops)
# # #Completely hide tick markers...
# # ax.yaxis.set_major_locator(plt.NullLocator())
# # ax.xaxis.set_major_formatter(plt.NullFormatter())

# # ax.yaxis.grid(False)   # Show the horizontal gridlines
# # ax.xaxis.grid(False)  # Hide x-axis gridlines 

# # ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# # # Add transparency to colors
# # for patch in ax.artists:
# #   r, g, b, a = patch.get_facecolor()
# #   patch.set_facecolor((r, g, b, .3))
    
# # # ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
# # ax.set(xlabel="\nNumber of defensive players in the 'Box'\n\n")
# # # ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])

# # # plt.title('My subtitle',fontsize=16)
# # # plt.suptitle('My title',fontsize=24, x=0, y=1,ha="left")
# # # plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# # # plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)

# # plt.tight_layout()
# # plt.show();


# # PLOT THE RAW RUNNER WEIGHT ? 


# In[ ]:


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sns.set_style("white", {'grid.linestyle': '--'})
# sns.set_style("ticks", {'grid.linestyle': '--'})
##sns.set(style="white", palette="muted", color_codes=True)
##sns.set(style="white", palette="muted", color_codes=True)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
speed = bbb.groupby(['DisplayName'])['S'].agg('max').sort_values(ascending=True)
speed = pd.DataFrame(speed)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots(figsize=(9,7))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sns.distplot(speed*2.04545, kde=False, color="orange", 
 hist_kws={"linewidth": .8, 'edgecolor':'black'}, bins=38)
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_yticklabels([])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.title('\nDistribution of running speed for all players in the 2017/2018 seasons (mph)\n',
           fontsize=12, loc="center")
ax.set(xlabel="\nRunner Speed (miles per hour mph)\n")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.tight_layout()
plt.show();
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# # #~~~~~~~~~~~ ax.set ~~~~~~~~~~~~~~~~
# # ##ax.set_xlim(0, 6)
# # ##ax.set_ylim(0, 6)
# # ax.set_title('Average yards gained as the season progresses (week by week)\n')
# # ax.set(ylabel='Yards Gained\n')
# # ax.set(xlabel='\nWeek Number (in the season)')
# # ax.yaxis.grid(True)   # Show the horizontal gridlines
# # ax.xaxis.grid(False)  # Hide x-axis gridlines 
# # # ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# # # ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
# # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ## Remove the x-tick labels:  plt.xticks([])
# plt.yticks([])
# ## This method also hides the tick marks
# plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',
#           fontsize=12, loc="left")
# plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',fontsize=12, loc="left")
# plt.xlabel('\nNumber of times the ball was run in the game\n')
# plt.tight_layout()
# plt.axvline(x=22, color='maroon', linestyle="--", linewidth=.5)

# plt.text(22.8, 114, r'Median: 22 carries', {'color': 'maroon', 'fontsize': 9})
# plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))


# plt.tight_layout()
# plt.show()


# ttt = bbb[bbb.Position.isin(['RB','WR','HB'])]

# ttt['kg']=ttt["PlayerWeight"] * 0.45359237
# ttt['Force_Newtons']=ttt['kg'] * ttt['A'] * 0.9144
# tips = ttt[['Force_Newtons', 'Yards']]

# sns.scatterplot(x="Force_Newtons", y="Yards", data=tips, s=1, ax=ax, color='r', markers='o', edgecolor='r')
# #sns.lmplot(x="Force_Newtons", y="Yards", data=tips)

# plt.title('Correlation between Yards Gained and Player Kinetic Force',fontsize=12)
# plt.suptitle('Kinetic Force',fontsize=13, x=0, y=1,ha="left")
# ##plt.text(x=4.7, y=14.7, s='Sepal Length vs Width', fontsize=10, weight='bold')

# ax.set(xlabel="\nPlayer Kinetic Force\n")
# ax.set(ylabel="Yards Gained\n")

# sns.despine(top=True, right=True, left=True, bottom=True)


# # dff = tf[tf.DefendersInTheBox>2]
# # dff.DefendersInTheBox = dff.DefendersInTheBox.astype('int')

# # flierprops = dict(markerfacecolor='0.75', 
# #                   markersize=1,
# #                   linestyle='none')

# # fig, ax = plt.subplots(figsize=(9,7))
# # ax.set_ylim(-7, 23)
# # ax.set_title('Yards Gained vs number of Defenders in the box\n\n', fontsize=12)
# # sns.boxplot(x='DefendersInTheBox',
# #             y='Yards',
# #             data=dff,
# #             ax=ax,
# #             showfliers=False , 
# #             #color='blue'
# #             )
# #             #flierprops=flierprops)
# # #Completely hide tick markers...
# # ax.yaxis.set_major_locator(plt.NullLocator())
# # ax.xaxis.set_major_formatter(plt.NullFormatter())

# # ax.yaxis.grid(False)   # Show the horizontal gridlines
# # ax.xaxis.grid(False)  # Hide x-axis gridlines 

# # ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# # # Add transparency to colors
# # for patch in ax.artists:
# #   r, g, b, a = patch.get_facecolor()
# #   patch.set_facecolor((r, g, b, .3))
    
# # # ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
# # ax.set(xlabel="\nNumber of defensive players in the 'Box'\n\n")
# # # ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])

# # # plt.title('My subtitle',fontsize=16)
# # # plt.suptitle('My title',fontsize=24, x=0, y=1,ha="left")
# # # plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# # # plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)

# # plt.tight_layout()
# # plt.show();


# # PLOT THE RAW RUNNER WEIGHT ? 


# * A good rule of thumb:  double the yards/sec value and you will get mph.  Why ?  Because 3,600 is almost 3,520. (i.e. 3 feet per yard, 1,760 yards per mile, 3,600 seconds per hour, and doubleing 1,760 gets you 3,520, which is very close to 3,600)
# * Sometimes the runner is not moving at the time of the handoff to him, and other times, he is running as fast as he can FROM a standstill.  The question then becomes how quickly can a running back accelerate from zero to full speed. The data we have been given is the speed at the time of the handoff.  
#   * Per competition director: &nbsp; *"in this contest, you're only receiving this tracking information at the moment a handoff is made"*
#   * Meaning:  The actual speed data we have been given really represents the runners 'starting velocity', i.e. it is not a runningback at 'cruising speed' (top-end speed). Think of this speed as the speed of a jet as it is about to take off the ground, it is still getting started.  And even THEN some of the values in mph are very impressive.  18mps is fast.  Side Note:  In the next competition I think it would be helpful to add in the 'top-end' velocity a running back hits during a game, thus an apples to apples comparison would be possible...
# * One of the biggest misconceptions that emerges from the NFL Combine each year is the importance of 40-yard dash times, when in reality the 10-yard split is a **very** important indicator of how well the runningback may do in the NFL.  Explosive speed trumps raw speed...

# <div class="h4"><i>Offense Formation Strategies:</i></div>
# * We will now examine which offense formation strategies appeared to result in the best yardage gained

# In[ ]:


# bbb.OffenseFormation.value_counts()
# SINGLEBACK    4920
# SHOTGUN       3626
# I_FORM        2058
# PISTOL         340
# JUMBO          244
# WILDCAT         66
# EMPTY           15

my_x = bbb.groupby('OffenseFormation')['Yards'].mean().sort_values(ascending=False).values
my_y = bbb.groupby('OffenseFormation')['Yards'].mean().index

## original !  deleting for sec sns.set(style="white", palette="muted", color_codes=True)
#sns.set(style="white", palette="muted", color_codes=True)
# sns.set_style("ticks", {'grid.linestyle': '--'})
sns.set(style="white", palette="muted", color_codes=True)

#sns.set_style("ticks", {'grid.linestyle': '--'})
# # this may not work right
# sns.set_style({'grid.linestyle': '--'}, )

##t = tf[['Week', "Yards"]].groupby('Week').mean().sort_values(by = "Yards")
###sns.set_style("white", {'grid.linestyle': '--'})


fig, ax = plt.subplots(figsize=(9,7))

sns.barplot(x=my_y,
            y=my_x,
            ax=ax, 
            linewidth=.2, 
            edgecolor="black")

    
#~~~~~~~~~~~ ax.set ~~~~~~~~~~~~~~~~
##ax.set_xlim(0, 6)
##ax.set_ylim(2, 6)
ax.set_title('\n2018: Avg YDS gained per playOffense Formations\n', fontsize=12)
# ax.set(ylabel='Yards Gained\n', rotation='horizontal')
ax.set(xlabel='\nOffense Formations')
ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 
ax.yaxis.set_major_locator(plt.MultipleLocator(1))
#ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

#  I am the author of all of this work:  Tom Bresee (this is my notebook)

#-----more control-----#
ax.grid(linestyle='--', 
        linewidth='0.9', 
        color='lightgray', 
        alpha=0.9,
        axis='y'
       )

# ax.set_axisbelow(True)

plt.ylabel("Avg YDS gained per Play\n", rotation=90)
sns.despine(top=True, right=True, left=True, bottom=True)


# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
plt.tight_layout()
plt.show();


# * I would say there was a relatively large difference in the yards gained based on offensive scheme
# * **Wildcat** - doesn't appear to be that effective.  But it should be noted that it is not run very often in the NFL.  But when it is, it performs pretty poorly.
# * **Shotgun** - performs surprisingly low compared to the other offensive schemes.  One could argue it is a kinda pass play, but more offense* **Empty** - is the clear winner. 'Empty' simply means there is no back in the backfield.  All five eligible receivers are assembled at the line of scrimmage in some fashion.
# * The **I-formation** is one of the more tried and true offensive formations seen in football, and you will likely see it used in short-yardage running siturations.  The I-formation places the runningback 6 - 8 yards behind the line of scrimmage with the quarterback under the center and a fullback splitting them in a three-point stance;  which also means that it is highly likely the defense can see where the runningback is going, but then again, he will probably have a fair amount of speed by the time he hits the line of scrimmage.  

# <hr>

# <div class="h2"><i>Particle (Player) Physics</i></div>
# <br>
# <div class="h4"><i>Towards a deeper understanding of the game of football in relation to Newtonian Physics</i></div>
# 

# * I'm going to start drinking and the answer is just going to come to me, here we go

# <div class="h4"><i>Core Assumptions:</i></div>
# * list out...

# It should be noted that both the Force `F` and the acceleration `a` are both technically vectors, and that the mass `m` is a 'scalar'.  

# $$\mathbf{\vec F} = m\;\mathbf{\vec a}$$

# $$\mathbf{\vec a} = \frac{\mathbf{\vec F}}{m}$$

# In standard mathematics, vectors either have the arrow over the vector, OR the letter is just bolded, we will stick with the bold technique, but you get the point...

# $$\mathbf{F} = m\;\mathbf{a}$$

# i.e. in our case: 

# $$\mathbf{F}_{player} = m\;\mathbf{a}_{player}$$

# The weight of an object is the force of gravity on the object and may be defined as the mass times the acceleration of gravity (commonly referred to as the scalar value $g$ )

# We will focus on the 2018 Season exclusively for this analysis. 

# <div class="h4"><i>Weight Distribution:</i></div>
# * Let's dive into examining player weight information 
# 

# In[ ]:



aaa = gold
aaa['IsRunner'] = aaa.NflId == aaa.NflIdRusher
bbb = aaa[aaa.IsRunner & (aaa.Season == 2018)]


fig, ax = plt.subplots(figsize=(10,6))
ax.set_xlim(150,380)
ax.set_title('2018 Season: Player Weight distribution (Runners vs Non-Runners)\n\n', fontsize=12)

sns.kdeplot(bbb.PlayerWeight, shade=True, color="orange", ax=ax)
sns.kdeplot(aaa[~aaa.IsRunner & (aaa.Season == 2018)].PlayerWeight, shade=True, color='blue', ax=ax)

ax.xaxis.set_major_locator(plt.MultipleLocator(10))
ax.xaxis.set_minor_locator(plt.MultipleLocator(5))

# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
    
####plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)
sns.despine(top=True, right=True, left=True, bottom=True)

# Turn off tick labels
ax.set_yticklabels([])
#ax.set_xticklabels([])


ax.set(xlabel="\nPlayer Weight\n\n")
plt.legend(title='Category:  Ran the ball, or did not ever run the ball', loc='upper right', labels=['Runners', 'Non-Runners'])
plt.tight_layout()
plt.show();


# **Read very carefully:**
# * I specifically am *not* looking at the unique player distribution here and plotting their weights.  That is not what I am doing here.  I am gathering up the total number of running plays of all of the combined teams over the entire 2018 seasons, and I am creating a distribution of the weight of the runner who made the play (in orange), and also during those SAME plays, gathering up the weight distribution of those who did NOT run the ball.  I believe this thus gives us a very good idea of the weights of the players that were **on** the field during the season (broken out by rushing player versus non-rushing player), and starts to paint a picture of being able to predict the expected yards gained during running plays. 
#   * I care about **who** is on the playing field here, that is the key for future prediction models. 
#   * As long as the weights of the players are updated throughout the season, this also is an extremely granular way of determining kinetic energy on the field as well.
#   * I guess my real point is this - if they aren't on the field, or aren't on the field much, do I really care what their weight is when i figure out my model ? 
# * Thus, of those that ran the ball in the 2018 NFL season, they had an average weight of **217 lbs**, and a median weight of 220 lbs. 
# * Non-Runners had a pretty wide distribution, obviously depends on position they played...
#   * There is a pronounced peak at 310lbs, which is our linemen...

# <div class="h4"><i>All Sources Metadata:</i></div>
# * Let's take a look at the weight distribution now for every player in the 2018 NFL season.  This time we will just examine all players who played in the season and were on the roster, to get a ballpark on some differences in weight vs position:

# In[ ]:


#
#Creating a playah profile, as a reference df:
#
#
player_profile=aaa.loc[:,['DisplayName','Position','NflId' 'PlayerBirthDate', 'PlayerWeight', 'PlayerCollegeName']].drop_duplicates()
player_profile_2018=aaa[aaa.Season==2018]
player_profile_2018 = player_profile_2018.loc[: ,['DisplayName','Position','NflId' 'PlayerBirthDate', 'PlayerWeight', 'PlayerCollegeName'] ].drop_duplicates()
#
#
# len(player_profile)
# len(player_profile_2018)
#
#
#
player_profile_2018["kg"] = player_profile_2018["PlayerWeight"] * 0.45359237
#
#
##player_profile_2018.PlayerCollegeName.value_counts()
#
#
z = player_profile_2018.groupby('Position')['PlayerWeight'].agg(['min', 'median', 'mean', 'max']).round(1).sort_values(by=['median'], 
                                                                                                                   ascending=False)
z['Avg Mass (kg)'] = (z['mean'] * 0.45359237).round(1)
z


# * I like this view (in order of descending median weight, by position).  You immediately see that all of the linemen are just over 300lbs. And they make up a LARGE distribution of the players on the field, i.e. there are some BIG BOYS on that field.  
# * I find it suprising that FB (fullbacks) are as heavy as they are.  I would imagine one could argue that two things are pretty critical to determining the performance of a running back: 
#   * How big are the offensive linemen ???  (ideally we knew how strong they were as well, but no information contained about that)
#   * How big is the fullback ?  A fullback with some size would really help blocking for the running back and I believe would be directly proportional to the success of the runningback.  
#   * Look at how large the OTs (offensive tackles) are.  One would imagine a run off the OT being a smart play, IF the defensive linebacker at that area was smaller as well...

# In[ ]:



# ####dfv = gold.loc[:,['NflId', 'DisplayName', 'PlayerBirthDate', 'PlayerWeight', 'PlayerHeight']].drop_duplicates()
# #
# #
# #
# #Plot
# sns.distplot(bbb, kde=False, color="b", 
#             hist_kws={"linewidth": .9, 'edgecolor':'lightgrey'}, bins=24)
# #
# #
# #
# #
# fig, ax = plt.subplots(figsize=(12, 8))
# #
# ax.set_xlim(150,380)
# ax.set_title('2018 Season: Player Weight distribution (Runners vs Non-Runners)\n\n', fontsize=12)


# sns.kdeplot(bbb.PlayerWeight, shade=True, color="orange", ax=ax)
# sns.kdeplot(aaa[~aaa.IsRunner & (aaa.Season == 2018)].PlayerWeight, shade=True, color='blue', ax=ax)

# ax.xaxis.set_major_locator(plt.MultipleLocator(10))
# ax.xaxis.set_minor_locator(plt.MultipleLocator(5))

# # Add transparency to colors
# for patch in ax.artists:
#   r, g, b, a = patch.get_facecolor()
#   patch.set_facecolor((r, g, b, .3))
    
    
# ####plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# # plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)
# sns.despine(top=True, right=True, left=True, bottom=True)

# # Turn off tick labels
# ax.set_yticklabels([])
# #ax.set_xticklabels([])


# ax.set(xlabel="\nPlayer Weight\n\n")
# plt.legend(title='Category:  Ran the ball, or did not ever run the ball', loc='upper right', labels=['Runners', 'Non-Runners'])
# plt.tight_layout()
# plt.show();


# <div class="h4"><i>Collisions:</i></div>
# * Collisions between dynamic bodies can either be elastic or inelastic
# * We will defined an `angle of attack`, similar to airfoil design.  This angle of attack will be the angle at which contact is made from the defender onto the offensive runningback.  We will then able able to also break out momentum and force into components with one common reference frame
#   * Assumption is that runner is running from left to right m
# * Every runningback should PREFER inelastic collisions !  
#   * Why ? **Because it creates the separation that is their advantage**
#   * How ? Ideally with an alpha less than 45 degrees - this way they bounce off and keep going somewhat in the x-axis direction, but ideally it is not an inelastic collision with alpha of 0 or near it, that is going to completely stop runningback momentum
#   * Effectively a runningback wants to maneuver, and when maeuvering is no longer much of an option, to 'bounce' off the tackler
# 

# <img src="https://github.com/tombresee/Temp/raw/master/ENTER/contact.png" width="800px"> 

# Why is it so important to analyze the governing dynamics ? 
#   * Check out this *2016 NFL Combine Report* on Dallas Cowboy runningback Ezekiel Elliott
#   * Source: `http://www.nfl.com/player/ezekielelliott/2555224/combine`
# <br>
# <img src="https://github.com/tombresee/Temp/raw/master/ENTER/elliott.png" width="800px">
# * i.e. the key to running effectively I believe is contained within the specific verbage above, these are some key factors for producing actual yards... 
# 

# <br>

# <div class="h4"><i>How to stop a truck:</i></div>
# * Americans are terrible with the metric system.  It's unexplainable. 
# * Think a lb is about two kg.  In other words, if someone told you they weighed 100lbs, they are about 50kg (although technically 45.359 kg)
#   * Think of it like this:  When I have pounds, take about 45% of that value, and now you have kilograms (kg)
# 

# * Lets convert all the runner's weight values to kilograms
# * THEN we will convert the runner's weight (which is now in kg) to MASS, which is NOT the same thing as weight.  I 'weigh' 200 lbs on the planet Earth, while I 'weigh' 33.07 lbs on the moon.  Thus the need for a value 'mass' that does not change depending on the planet you are on... 
#   * Technically a body's mass is its weight (in kg) divided by the gravitational constant $g$, which is 9.8 m/s^2, where that $g$ is really the `standard acceleration of gravity`
#   * IF you want to get tricky, do what I did:  `from scipy import constants`, and then use `scipy.constants.value(g)` as your python value, instead of always entering in numbers manually, its safer and smarter 

# In[ ]:


# bbb.Position.value_counts()
# RB    10476
# WR      372
# HB      327
# QB       40
# FB       35
# TE       16
# CB        3
# DE        1
# DT        1
#---------------------------
#10476/11271=93%
#---------------------------

from scipy import constants
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10,10))

#Specifically only using runners like RB, HB, and WR...
ttt = bbb[bbb.Position.isin(['RB','WR','HB'])]


# the kg column will be the true 'mass' of the body
# convert weight to kg and then divide by g to get the true mass 
ttt['kg']=ttt["PlayerWeight"] * 0.45359237 / scipy.constants.g


# the momentum is just mass (in kg) X speed in m/s (so convert from yards/sec to mps)
ttt['True Momentum']=ttt['kg'] * ttt['S'] * 0.9144 
tips = ttt[['True Momentum', 'Yards']]

sns.scatterplot(x="True Momentum", y="Yards", data=tips, s=4, ax=ax, color='cyan', markers='.', edgecolors='cyan', alpha=.8)
##sns.lmplot(x="Force_Newtons", y="Yards", data=tips, facecolor='cyan', edgecolors='cyan')

plt.title('Correlation between Yards Gained and Player Momentum\n',fontsize=11)
plt.suptitle('Kinetic Momentum',fontsize=10, x=0, y=1,ha="left")
##plt.text(x=4.7, y=14.7, s='Sepal Length vs Width', fontsize=10, weight='bold')

ax.set(xlabel="Player Kinetic Momentum $\Rightarrow$\n")
ax.set(ylabel="Yards Gained  $\Rightarrow$\n")

sns.despine(top=True, right=True, left=True, bottom=True)


# dff = tf[tf.DefendersInTheBox>2]
# dff.DefendersInTheBox = dff.DefendersInTheBox.astype('int')

# flierprops = dict(markerfacecolor='0.75', 
#                   markersize=1,
#                   linestyle='none')

# fig, ax = plt.subplots(figsize=(9,7))
# ax.set_ylim(-7, 23)
# ax.set_title('Yards Gained vs number of Defenders in the box\n\n', fontsize=12)
# sns.boxplot(x='DefendersInTheBox',
#             y='Yards',
#             data=dff,
#             ax=ax,
#             showfliers=False , 
#             #color='blue'
#             )
#             #flierprops=flierprops)
# #Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

# ax.yaxis.grid(False)   # Show the horizontal gridlines
# ax.xaxis.grid(False)  # Hide x-axis gridlines 

# ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# # Add transparency to colors
# for patch in ax.artists:
#   r, g, b, a = patch.get_facecolor()
#   patch.set_facecolor((r, g, b, .3))
    
# # ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
# ax.set(xlabel="\nNumber of defensive players in the 'Box'\n\n")
# # ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])

# # plt.title('My subtitle',fontsize=16)
# # plt.suptitle('My title',fontsize=24, x=0, y=1,ha="left")
# # plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# # plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)

# plt.tight_layout()
# plt.show();

plt.xticks([])
#plt.yticks([])


# sns.relplot(x="True Momentum", y="Yards", data=tips, s=4, ax=ax, color='cyan', markers='.', edgecolors='cyan', alpha=.4)
# ##sns.lmplot(x="Force_Newtons", y="Yards", data=tips, facecolor='cyan', edgecolors='cyan')

plt.tight_layout()
plt.show();


# In[ ]:


plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10,10))

#Specifically only using runners like RB, HB, and WR...
ttt = bbb[bbb.Position.isin(['RB','WR','HB'])]

# the kg column will be the true 'mass' of the body
# convert weight to kg and then divide by g to get the true mass 
ttt['kg']=ttt["PlayerWeight"] * 0.45359237 / scipy.constants.g


# the momentum is just mass (in kg) X speed in m/s (so convert from yards/sec to mps)
ttt['True Momentum']=ttt['kg'] * ttt['S'] * 0.9144 
tips = ttt[['True Momentum', 'Yards']]

sns.scatterplot(x="True Momentum", y="Yards", data=tips, s=4, ax=ax, color='cyan', markers='.', edgecolors='cyan', alpha=.8)
##sns.lmplot(x="Force_Newtons", y="Yards", data=tips, facecolor='cyan', edgecolors='cyan')

plt.title('Correlation between Yards Gained beyond 6 and Player Momentum\n',fontsize=11)
plt.suptitle('Kinetic Momentum',fontsize=10, x=0, y=1,ha="left")
##plt.text(x=4.7, y=14.7, s='Sepal Length vs Width', fontsize=10, weight='bold')

ax.set(xlabel="Player Kinetic Momentum $\Rightarrow$\n")
ax.set(ylabel="Yards Gained  $\Rightarrow$\n")

sns.despine(top=True, right=True, left=True, bottom=True)


# dff = tf[tf.DefendersInTheBox>2]
# dff.DefendersInTheBox = dff.DefendersInTheBox.astype('int')

# flierprops = dict(markerfacecolor='0.75', 
#                   markersize=1,
#                   linestyle='none')

# fig, ax = plt.subplots(figsize=(9,7))
ax.set_ylim(6,100)
# ax.set_title('Yards Gained vs number of Defenders in the box\n\n', fontsize=12)
# sns.boxplot(x='DefendersInTheBox',
#             y='Yards',
#             data=dff,
#             ax=ax,
#             showfliers=False , 
#             #color='blue'
#             )
#             #flierprops=flierprops)
# #Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

# ax.yaxis.grid(False)   # Show the horizontal gridlines
# ax.xaxis.grid(False)  # Hide x-axis gridlines 

# ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# # Add transparency to colors
# for patch in ax.artists:
#   r, g, b, a = patch.get_facecolor()
#   patch.set_facecolor((r, g, b, .3))
    
# # ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
# ax.set(xlabel="\nNumber of defensive players in the 'Box'\n\n")
# # ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])

# # plt.title('My subtitle',fontsize=16)
# # plt.suptitle('My title',fontsize=24, x=0, y=1,ha="left")
# # plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# # plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)

# plt.tight_layout()
# plt.show();

plt.xticks([])
#plt.yticks([])

plt.tight_layout()
plt.show();


# > **INSIGHT**: &nbsp; There appears to be a moderate connection between high momentum and yards gained, i.e. beyond a certain momentum threshold, a marked increase in the number of yards beyond 6 is seen...

# * There is something here but I can't put my finger on it.  Inherent Cauchy distribution ?  Some version of Lorentz distribution ?  I think drinking might help here. 
# * I think if we were able to dive deeper we would see this is something like a fire distribution
#    * I'm making the term up, but something to the effect of I make a fire and it has an intensity centerfied, but rising embers, or else there is nothing really here and its actually two sep distinct distributions superimposed
#    * But it seems to mimic this [click](https://www.shutterstock.com/video/clip-4927760-large-fire-burning-night-smoke-sparks-rising)

# In[ ]:



plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10,10))

ttt = bbb[bbb.Position.isin(['RB','WR','HB'])]


# true mass in kg 
ttt['kg']=ttt["PlayerWeight"] * 0.45359237 / scipy.constants.g

# F = ma 
ttt['Force_Newtons']=ttt['kg'] * ttt['A'] * 0.9144
tips = ttt[['Force_Newtons', 'Yards']]

sns.scatterplot(x="Force_Newtons", y="Yards", data=tips, s=1, ax=ax, color='r', markers='o', edgecolor='r')
#sns.lmplot(x="Force_Newtons", y="Yards", data=tips)

plt.title('\nCorrelation between Yards Gained and Player Kinetic Force\n',fontsize=11)
plt.suptitle('Kinetic Force',fontsize=10, x=0, y=1,ha="left")
##plt.text(x=4.7, y=14.7, s='Sepal Length vs Width', fontsize=10, weight='bold')

ax.set(xlabel="Player Kinetic Force$\Rightarrow$\n")
ax.set(ylabel="Yards Gained $\Rightarrow$\n")
sns.despine(top=True, right=True, left=True, bottom=True)



# dff = tf[tf.DefendersInTheBox>2]
# dff.DefendersInTheBox = dff.DefendersInTheBox.astype('int')

# flierprops = dict(markerfacecolor='0.75', 
#                   markersize=1,
#                   linestyle='none')

# fig, ax = plt.subplots(figsize=(9,7))
# ax.set_ylim(-7, 23)
# ax.set_title('Yards Gained vs number of Defenders in the box\n\n', fontsize=12)
# sns.boxplot(x='DefendersInTheBox',
#             y='Yards',
#             data=dff,
#             ax=ax,
#             showfliers=False , 
#             #color='blue'
#             )
#             #flierprops=flierprops)
# #Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

# ax.yaxis.grid(False)   # Show the horizontal gridlines
# ax.xaxis.grid(False)  # Hide x-axis gridlines 

# ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# # Add transparency to colors
# for patch in ax.artists:
#   r, g, b, a = patch.get_facecolor()
#   patch.set_facecolor((r, g, b, .3))
    
# # ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
# ax.set(xlabel="\nNumber of defensive players in the 'Box'\n\n")
# # ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])

# # plt.title('My subtitle',fontsize=16)
# # plt.suptitle('My title',fontsize=24, x=0, y=1,ha="left")
# # plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# # plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)

# plt.tight_layout()
# plt.show();
plt.xticks([])

plt.tight_layout()
plt.show();


# <div class="h4"><i>Kinetic Energy:</i></div>
# * An interesting thing occurs when we look at Kinetic Energy

# ```python
# 
# def kinetic_energy(player_mass, player_velocity):
#     #--- v in m/s, and mass in kg 
#     KE = (1/2) * player_mass * (player_velocity ** 2)
#     #--- output the amount of kinetic energy the player has
#     return KE
# 
# kinetic_energy(player_mass, player_velocity)```

# Kinetic Energy is proportional to the **square** of a player's velocity.  Which means for instance if you double the player's velocity, you quadruple his KE, and if you triple the player's velocity, you have a 9x increase in his kinetic energy, meaning it is exponentially proportional to the velocity of the player, while still linearly proportional to the player's mass (i.e. if you double the player's mass, you double his KE, and if you triple a player's mass, you triple his KE)

# <br>

# <div class="h2"><i>Towards Understanding and Gauging Human Performance:</i></div>

# > <div class="h3"><i>Chapter 1: &nbsp;  Power</i></div>

# Inevitably, a major factor in longer yardage in a league as competive as the NFL is human 'power'.  After contact is made, the yardage gained is going to be continent to a large extent on player momentum and power.  It is important to understand that momentum is the instantaneous 'mass' x 'velocity' at the point of contact, and is a physics definition, but I would define power as the ability to push **after** contact (or right before), i.e. directly linked to the overall strength and effort of the individual runner.  It is hard to really defined this term, but I consider power strongly linked if not almost identical to propulsion, i.e. the action of driving or pushing forward, where the 'engine' strength of the runner propels him forward.  The following video I believe is the key to understanding this concept of runner power/propulsion:  

# In[ ]:


from IPython.display import YouTubeVideo
# The Hill
# Video credit: Beastly67.
YouTubeVideo('dqmxWZ8Rbwc')


# * At 1:46 into this video, you also see an interesting thing:  the extreme angle at which he runs with regard to his direction, i.e. If one were to superimpose that angle onto running on a level surface, you see that the x-component of his posture is very high, while the y-component is lower.  Effectively, one can posit that transitioning to a 'leaning-forward' run style during collision/contact will enable the runner to continue forward more effectively, while also lowering his cross-sectional tackling 'target' profile. 
# * So what is the OPPOSITE of this ? 
#   * A posture style built inherently for speed - one example of this: Michael Johnson the track star.  
#   * Check out this [video](https://www.youtube.com/watch?v=JQ9cBQANjiw) and especially 00:54 - 00:59 seconds into the video you see how **straight** his posture is, its almost *precisely* vertical.  Built for NO contact (of course, due to the nature of track the sport)

# *Where am I going with this ?*
# * I believe we should begin methodology to gauge the power that runningbacks have (via an innovative approach), and that eventually we will begin to define also the term 'propulsion'
# * **How do you measure running power/propulsion ?**
#   * <span style="color:red">3, 5, 9, and 12-yard split times of runners while wearing precisely 20% of their body weight, in a suit similar to a tshirt+football_pants combination, spread uniformly about their body, with 50% of the weight above their center of gravity (while standing), and the other 50% spread among their lower half (below their center of gravity).  This suit MUST ensure that the lower 50% of the weight reaches to the knee, to simulate weight across the upper legs.  This portion is **critical**. </span>
# >   * These performance statistics should be incorporated in the NFL combine for all 'runners' or those expected to 'rush' with the ball.  This allows the ability to track much more than just 40yard dash speed components. 
# * There is no current real existing definition of football runner <u>propulsion</u> and in my opinion it is needed - simply because we need a way of comparing the fast 'track-like' runner versus the fast 'steamroller-like' runner.  Yards attained after collision MUST incorporate the amount of raw momentum that was transferred to the runner upon collision, i.e. if two defensive players hit a runner, and he was able to reach 5 more yards after that, then the 5 yards should also be recorded in **context** (the total x-axis momentum of the defensive players combined was XYZ)...
#   

# *We should thus create a new matrix to quantify a runner's power/horsepower/propulsion*
# 
# *  This power matrix $P_{pl}$ should reflect the nuances of power with respect to NFL rushing
# *  It should be defined as the below:
# $$P_{pl} =\begin{bmatrix} p_{11}&p_{12}&p_{13}&p_{14}  \\ p_{21}&p_{22}&p_{23}&p_{24}  \\ p_{31}&p_{32}&p_{33}&p_{34} \\ p_{41}&p_{42}&p_{43}&p_{44}  \end{bmatrix}$$
# * The values:
#   * $p_{11}$:
#     * Combine-defined bench press
#   * $p_{12}$:
#     * Max Impulse via stiff-arm motion
#   * $p_{13}$:
#     * upper body 3 (shoulder strength)
#   * $p_{14}$:
#     * upper body 4 (torque strength)
#   * $p_{21}$ through $p_{24}$:
#     * Combine-defined vertical jump, broad jump, squat strength, sitting squat strength
#   * $p_{31}$ through $p_{34}$:
#     * The 10, 20, 30, and 40 yard split-times (in seconds) while running along a grade of `x` degrees (to be known as the 'Hill Test'). 
#   * $p_{41}$ through $p_{44}$:
#     * The 3, 5, 9, and 12 yard split-times (in seconds) while weighted as defined above, i.e. 20% of body weight. 

# <div class="h3"><i>Chapter 2: &nbsp;  Speed</i></div>

# Inevitably, a major factor in longer yardage in a league as competive as the NFL is human 'velocity', which is probably the more accurate term than 'speed'.  Also, acceleration is a key factor for runningbacks. 

# *Where am I going with this ?*
# * I believe we should begin methodology to gauge the motion_related factors that runningbacks have.
# 

# *We should thus create a new matrix to quantify a runner's motion performance*
# 
# *  This power matrix $S_{pl}$ should reflect the nuances of dynamic motion with respect to NFL rushing
# *  It should be defined as the below:
# $$S_{pl} =\begin{bmatrix} s_{11}&s_{12}&s_{13}&s_{14}  \\ s_{21}&s_{22}&s_{23}&s_{24}  \\ s_{31}&s_{32}&s_{33}&s_{34} \\ s_{41}&s_{42}&s_{43}&s_{44}  \end{bmatrix}$$

# <div class="h3"><i>Chapter 3:&nbsp;  CoS</i></div>

# Never ever underestimate the power of a grudge.

# <div class="h3"><i>Chapter 4: &nbsp; The Statistics</i></div>
# 

# *A Few Thoughts:*
# * I believe there is something to be said for creating new statistics specifically to gauge individual linemen (both O and D) performance
# * One *cannot* deny the close link between an outstanding offensive line and a running back's performance
# * The probability of a running back being stopped is obviously directly proporational to being hit quickly, slowed down, or stopped
# * I would posit that the success of a running back play is more tied to his offensive line performance and less the defensive line performance
#   * Many times a defensive linemen is a 'NF' (non-factor) in a play due to simple positioning (he is literally too far away from the running back to provide a tackle, or lacks the speed, or a combination of the two)
#   * Many times the linebacker is the one the running back has to worry about (linebackers can blitz, and inherently usually have considerable speed)
# * A true ultra-granular machine learning model would take into consideration the exact performance stats of every player on the field, which will be possible by 2022 I would think.  Once **Extreme-Generation Stats** are available, it comes down to major factors like the summation of the conditional probabilities of the defensive players making a tackle, or being involved in the actual stoppage of the runner.  
#  * And thus in the below diagram, you can see that the play (in this example) really comes down to the summation of:  the probability that the two key blocks are made on the RHS (right hand side), GIVEN the performance characteristic of the O-Linemen + the probability that the Linebackers can make a play GIVEN the performance characteristics of whomever is blocking him, etc.  There are defensive players that are simply non-factors, there are defensive players that are not involved due to the performance of the offense, and there are defensive players that are statistically given a good shot of making the tackle.  IF one were able to gather up exact stats, one could literally match up perfectly in a game plan against certain opponents.  
#  

# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/upload.jpg" width="800px">
# 

# <br>

# <div class="h2"><i>Machine Learning Model</i></div>

# In[ ]:


#
#
#  YR2.count()  -  all the value counts for each col 
#
#  df['Churn'].value_counts(normalize=True)
#
#
# len(df.NflId.unique())                    1788
# len(df[df.Season==2017].NflId.unique())   1783
# len(df[df.Season==2018].NflId.unique())   2231
# delta between these numbers:              448
#
#
#   len(df[df.Season==2017.DisplayName.unique())   .  1780        1788 
#   len(df[df.Season==2018].DisplayName.unique())  .  1782        1783
#   len(YRS.DisplayName.unique())                  .  2230.448delta
#
#
#    509762 total rows of data
#    509762/22 = 23,171 rows of true runner data 
#    23,171 split between 2018 and 2017 for about 11k each 
#    1696 + practice players = total allowed in the NFL per season
#    1856 if you allow practice players 
#    Remove these columns uniformly
#  killed_columns=['xyz','etc']
#
#
# YRS = dontbreak[dontbreak.NflId==dontbreak.NflIdRusher].copy()
# YR1 = YRS[YRS.Season==2017]
# YR2 = YRS[YRS.Season==2018]
#
#len(YRS)==len(YR1)+len(YR2)
#
#
# df_play.drop('Yards', axis=1)
#
#
#---------------------------------
def drop_these_columns(your_df,your_list):
    #KILL KOLUMNS
    your_df.drop(your_list,axis=1,inplace=True)
#---------------------------------
#
#
#
#
allcolumns = """
['GameId',
 'PlayId',
 'Team',
 'X',
 'Y',
 'S',
 'A',
 'Dis',
 'Orientation',
 'Dir',
 'NflId',
 'DisplayName',
 'JerseyNumber',
 'Season',
 'YardLine',
 'Quarter',
 'GameClock',
 'PossessionTeam',
 'Down',
 'Distance',
 'FieldPosition',
 'HomeScoreBeforePlay',
 'VisitorScoreBeforePlay',
 'NflIdRusher',
 'OffenseFormation',
 'OffensePersonnel',
 'DefendersInTheBox',
 'DefensePersonnel',
 'PlayDirection',
 'TimeHandoff',
 'TimeSnap',
 'Yards',
 'PlayerHeight',
 'PlayerWeight',
 'PlayerBirthDate',
 'PlayerCollegeName',
 'Position',
 'HomeTeamAbbr',
 'VisitorTeamAbbr',
 'Week',
 'Stadium',
 'Location',
 'StadiumType',
 'Turf',
 'GameWeather',
 'Temperature',
 'Humidity',
 'WindSpeed',
 'WindDirection',
 'IsRunner']
"""
#
# def common_elements(list1, list2): 
#     return [element for element in list1 if element in list2]
#
# z1 = df[df.Season==2017].NflId.unique()
# z2 = df[df.Season==2018].NflId.unique()
# len(common_elements(z1,z2))
#
#
# len(YR1.columns)
# len(YR2.columns)
#---------------------------------
kill=['WindSpeed','WindDirection','StadiumType','Temperature','GameWeather']
drop_these_columns(YR1,kill)
drop_these_columns(YR2, kill)
drop_these_columns(YRS,kill)
#---------------------------------
# len(YR1.columns)
# len(YR2.columns)
#
#
#YR2.PlayDirection.value_counts()
#
#------------------------------------------------------------------------------------------
#  EVENTUALLY REMOVE THE YARDS, THE NFLID AND THE NFLIDRUSHER AS THEY WILL NOT BE NEEDED...
#     df.drop(['Yards'], axis=1, inplace=True)
#     df.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)
#
#
#
# to strip a particular columns   -    df[0] = df[0].str.strip()
#YRS.count()
#YRS.nunique()
# for i in YRS.columns:
#     print(YRS[i].value_counts())
#YRS.nunique()
#
#
#

# #  Create master missing count and percentage chart for YRS: 
# missing_values_count = YRS.isnull().sum()
# missing=pd.DataFrame(missing_values_count[missing_values_count != 0])
# missing.columns=['MissingCount']
# missing['% Missing'] = round(missing.MissingCount/23171*100,2)
# missing = missing.sort_values(by='% Missing', ascending=False)
# missing.index.name='Both Years Combined'
# missing


# #  Create master missing count and percentage chart for YRS: 
# missing_values_count17 = YR1.isnull().sum()
# missing17=pd.DataFrame(missing_values_count17[missing_values_count17 != 0])
# missing17.columns=['MissingCount']
# missing17['% Missing'] = round(missing17.MissingCount/23171*100,2)
# missing17 = missing17.sort_values(by='% Missing', ascending=False)
# missing17.index.name='2017'
# missing17

# #  Create master missing count and percentage chart for YRS: 
# missing_values_count18 = YR2.isnull().sum()
# missing18=pd.DataFrame(missing_values_count18[missing_values_count18 != 0])
# missing18.columns=['MissingCount']
# missing18['% Missing'] = round(missing18.MissingCount/23171*100,2)
# missing18 = missing18.sort_values(by='% Missing', ascending=False)
# missing18.index.name='2018'
# missing18

# def showmeaplay():
#     display(df.iloc[0:22:,0:20])
#     print("")
#     display(df.iloc[0:22:,21:33])
#     print("")
#     display(df.iloc[0:22:,34:49])
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     showmeaplay()


# *I'm going to have to spin up a second jupyter notebook to cover my machine learning model here, I still plan on adding a signficant amount of stuff above for EDA-related visuals...*  
# 
# *I will include the link to the second notebook, but as it stands I still want to do more EDA here, so this notebook will be dedicated to EDA exclusively, but you see the Feature Engineering perspective as well here*

# <div class="h3"><i>UPDATE: Latest News</i></div>

# The NFL is revealing the 100 greatest players and 10 greatest coaches in NFL history.  
# 
# NFL 100 All-Time Team running backs revealed as of two days ago: 
# * Jim Brown, Walter Payton, Barry Sanders, Earl Campbell, Emmitt Smith, Earl "Dutch" Clark,  Eric Dickerson, Lenny Moore, Marion Motley, Gale Sayers, Steve Van Buren, O.J. Simpson

# <br>

# <div class="h3" style="text-align: center">Final Observations and Recommendations:</div>
# 

# 1. &nbsp; I believe as part of the NFL Next Gen stats that the runner's <span style="color:blue">Median Yards per Carry</span> **should** be included, and perhaps could be designated as the term 'MED'.  In the world of statistics, we include median as well as mean (average) on a daily basis, and yet it is lacking in common usage for NFL football players.  Inclusion of median values in general prevents skewing performance views of players. If the term 'AVG' is currently being used for Average Yards per Carry, then the term 'MED' would seem to make a lot of sense to include...
# 1. &nbsp; Currently it appears that NFL runner's stats include 'Number of times they ran 20+ yards' and 'Number of times they ran 40+ yards', but based on the distribution of runs we have analyzed, those longer yardage values are quite rare, and it may help to include an additional stat of <span style="color:blue">'Number of times they ran 10+ yards'</span> 
# 1. &nbsp; Football in my eyes is very similar to an 'Operations Research' type problem combined with a zero-sum gain mathematical model. A balance of resources. 
# 1. &nbsp; The probability of the defensive end (DE) or defensive tackle (DT) picking up the ball on a fumble and **running** with it are extremely rare, only happened three times total in two seasons...
# 1. &nbsp; The vast majority of runs were via the running back (RB), followed by the wide receiver (WR), and then the half back (HB)
# 1. &nbsp; * Player 'Heart' factor (otherwise known as the $D\alpha\kappa$-factor) is real 
# 1. &nbsp; **Some data that would help in the future to create an even better predictive model:**
#   * **z-coordinate** data 
#     * player center of gravity, player shoulder pad height off the ground
#     * runner angle of attack lean
#     * real-time player speeds/acceleration/power
# 

# <br><br>

# <div class="h4"><i><u>About Me:</u></i></div>
# * **Name:  Tom Bresee**
# * Location:  Frisco, TX, USA
# * Master's student in Applied Data Science at the University of Michigan 
# * Background:  Bachelors in Applied Physics, Masters in Electrical Engineering (Communication Systems)
# * [My Linkedin Profile](https://www.linkedin.com/in/tombresee/)
# * Strong Tufte proponent
# * Played rugby in college (Texas) under Coach Bob Macnab
# * Played high school football at Seattle's [O'Dea High School](https://www.odea.org/athletics/fall-sports/football/)
#   * I wore jersey #52 and I was ok but not like amazing
#   * But what is amazing is the [history](https://www.odea.org/athletics/fall-sports/football/) of this powerhouse Seattle football program, currently ranked **#1** in the state of Washington (after beating the No. 11 team in the country as ranked by MaxPreps.com):
#   <img src="https://github.com/tombresee/Temp/raw/master/ENTER/odearesp.png" width="1100px">
#   <img src="https://github.com/tombresee/Temp/raw/master/ENTER/stateapp.png" width="600px">
# * State tournament update as of November 27th (still hanging in there):  https://www.maxpreps.com/high-schools/odea-fighting-irish-(seattle,wa)/football/home.htm)
#   * Lets hope its a state finals showdown against the rich kids of Eastside...
# * Update:  O'Dea High School (12-0) reaches State Finals, to be played Saturday 12/7/19 @ 12:00p
# 
# * This is my first ever Kaggle submission, let's see how it goes...**upvotes are much appreciated and help to keep me motivated !** 
# 

# <div class="h5" style="text-align: center"><i>Status: &nbsp; Still in progress, stay tuned for more updates</i></div>
# <div class="h5" style="text-align: center"><i><b>Original Upload:</b>  Monday November 4th, 2019</i></div>
# 

# <br><br><br><br>

# #### **Appendix A:**  &ensp; One Random Entire Play of Data (22 datarows of data, where each player on both offense and defense is shown)
# * The rows are spread over three sections for clarity
# * I find this 'per play' view helps to see the big picture of what we have to work with...

# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/new1.png" width="1000" height="800">

# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/new2.png" width="1000" height="800">

# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/new3.png" width="1000" height="800">

# #### **Appendix B:**  &ensp; Player Profile Generator Function

# Seperate Function:  Enter the name of the player into this function, it will output their specifics, use it as you see fit...

# In[ ]:



def find(player):
    plyr = dontbreak.loc[:,['DisplayName',
                                      'Position',
                                      'JerseyNumber',
                                      'PlayerWeight',
                                      'PlayerHeight',
                                      'PlayerBirthDate',
                                      'NflId',
                                      'PlayerCollegeName']].drop_duplicates()
    output = plyr[plyr.DisplayName==str(player)]
    output.index.name='Player Profile'
    #output.reset_index()
    #print(output.columns)
    #display(HTML(output.T.to_html(index=False)))
    return output.T
    # ADD:  PLAYER AGE

person_I_want_to_show = 'Tom Brady'  # enter whatever name you want here ... 
find(person_I_want_to_show).style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'right')])])


# In[ ]:



# # N = 9
# # y = np.linspace(-2, 2, N)
# # x = y**2
# # source = ColumnDataSource(dict(y=y, right=x,))

# # p  = Plot(title=None, plot_width=300, plot_height=300,
# #      min_border=0, toolbar_location=None)

# # glyph = HBar(y="y", right="right", left=0, height=0.5, fill_color="#b3de69")

# # p.add_glyph(source, glyph)

# # xaxis = LinearAxis()

# # p.add_layout(xaxis, 'below')

# # yaxis = LinearAxis()
# # p.add_layout(yaxis, 'left')

# # p.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
# # p.add_layout(Grid(dimension=1, ticker=yaxis.ticker))
# # curdoc().add_root(p)
# # output_notebook(hide_banner=True)
# # show(p); 

# my_data = df[['PlayerCollegeName','NflId', 'DisplayName']].drop_duplicates().copy()

# college_attended = my_data["PlayerCollegeName"].value_counts()

# df_cc = pd.DataFrame({'CollegeName':college_attended.index, 'Count':college_attended.values}).sort_values("Count", ascending = False)

# #df_cc.Count.astype('int', inplace=True)

# df_cc = df_cc[df_cc.CollegeName != 'Louisiana State']

# df_cc.at[42,'Count']=51
# #df_cc[df_cc.CollegeName=='LSU']['Count']

# df_cc.sort_values('Count', )


# pd.set_option('display.max_rows', 500)

# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# p = figure(
#   y_range=df_cc.CollegeName[:40],          
#   title = '\nNumber of players that attended Colleges Attended - Player Count\n',
#   x_axis_label ='# of NFL players that attended the college prior\n',
#   plot_width=500,
#   plot_height=700,
#   tools="", toolbar_location=None)
#   #min_border=0
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# p.hbar(
#     y=df_cc.CollegeName[:40],              
#     right=df_cc.Count[:40],   
#     left=0,
#     height=0.4,
#     color='orange',
#     fill_alpha=0.4
# )
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
# p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
# p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
# p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
# #p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# #p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ###readmore:  https://medium.com/@deallen7/visualizing-data-with-pythons-bokeh-package-310315d830bb
# output_notebook(hide_banner=True)
# #show(p)


# In[ ]:


# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # SUPERHBAR:  i started learning bokeh two days ago, so this sucks 
# # To draw horizontal bars by specifying a (center) y-coordinate, height, and left and right
# # endpoints, use the hbar() glyph function:
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# my_data = df[['PlayerCollegeName','NflId', 'DisplayName']].drop_duplicates().copy()

# college_attended = my_data["PlayerCollegeName"].value_counts()

# df_cc = pd.DataFrame({'CollegeName':college_attended.index, 'Count':college_attended.values}).sort_values("Count", ascending = False)

# #df_cc.Count.astype('int', inplace=True)

# df_cc = df_cc[df_cc.CollegeName != 'Louisiana State']

# df_cc.at[42,'Count']=51

# #df_cc[df_cc.CollegeName=='LSU']['Count']

# df_cc.sort_values('Count',ascending=False, inplace=True)

# #pd.set_option('display.max_rows', 500)
# df_cc.index = df_cc.index + 1


# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# mysource = ColumnDataSource(df_cc)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# p = figure(
#   y_range=df_cc.CollegeName[:50],    # I need to enter the SAME thing here as y points, i find that odd 
#   # wait:  can i set this as the range, but not below ? ? ? 
#   # i think caegorical just list in a list the categories here 
#   title = '\nNFL Player Count by College Attended\n',
#   x_axis_label ='# of NFL players that attended the college prior\n',
#   plot_width=700,
#   plot_height=800,
#   tools="hover",       # or tools="" 
#   toolbar_location=None,   
#   #background_fill_color="#efe8e2")
#   #min_border=0))
# )
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# p.hbar(
#     y='CollegeName',  # center of your y coordinate launcher, 40 points as def above ... 
#     left=0, # or left=20, etc
#     right='Count',    # right is 40 points... 
#     height=0.8,
#     alpha=.6,
#     #color='orange',    #color=Spectral3  #color=Blues8,   
#     #background_fill_color="#efe8e2", 
#     #     fill_color=Blues8,
#     #     fill_alpha=0.4, 
#     source = mysource, 
#     line_color='blue'   # line_coolor='red'
# ) 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # TITLE: 
# # p.title.text = 'Current frame:'
# # p.title.text_color = TEXT_COLOR
# # p.title.text_font = TEXT_FONT
# p.title.text_font_size = '11pt'
# # p.title.text_font_style = 'normal'
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # AXES: 
# # p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
# # p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
# # p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
# # p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
# # p.xaxis.axis_line_color = None    # or 'red'
# # p.yaxis.axis_line_color = GRID_COLOR 
# #
# # X-TICKS:
# # p.xaxis[0].ticker = FixedTicker(ticks=[0, 1])
# # p.xaxis.major_tick_line_color = GRID_COLOR
# # p.xaxis.major_label_text_font_size = '7pt'
# # p.xaxis.major_label_text_font = TEXT_FONT
# # p.xaxis.major_label_text_color = None   #TEXT_COLOR
# #
# # Y-TICKS:
# # p.yaxis[0].ticker = FixedTicker(ticks=np.arange(1, len(labels) + 1, 1).tolist())
# # p.yaxis.major_label_text_font_size = '0pt'
# p.yaxis.major_tick_line_color = None
# p.axis.minor_tick_line_color = None  # turn off y-axis minor ticks

# # p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# # p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
# p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # GRID:
# # p.xgrid.grid_line_color = None
# p.ygrid.grid_line_color = None   
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # # LEGENDgend.location = 'top_left'
# # p.legend.orientation='vertical'
# # p.legend.location='top_right'
# # p.legend.label_text_font_size='10px'
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ### NOTES here> 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # HOVER:
# #     hover.names = ['bars']
# #     hover.tooltips = [
# #         ('Event', '@label'),
# #         ('Probability', '@pretty_value')]
# #
# hover = HoverTool()
# #p.select(HoverTool).tooltips = [("x1","@x1"), ("x2","@x2")]
# #
# # hover.tooltips = [
# #         ('Event', '@label')
# #         #('Probability', '@pretty_value'),
# #     ]
# # hover.tooltips = [
# #     ("Total:", "@Count")
# #     #("x1", "@x1"),
# #     #("Totals", "@TONS_HE High Explosive / @TONS_IC Incendiary / @TONS_FRAG Fragmentation")
# #     ]
# ###########################hover.mode = 'vline'
# #????curdoc().add_root(p)
# # hover.tooltips = """
# #     <div>
# #         <br>
# #         <h4>@CollegeName:</h4>
# #         <div><strong>Count: &ensp; </strong>@Count</div>
# #     </div>
# # """
# hover.tooltips = [
#     ("College Name:", "@CollegeName"),
#     ("Ranking by Count", "$index"),
#     ("Number of gradutes that entered the NFL:", "@Count"),
# ]
# #<div><strong>HP: </strong>@Horsepower</div>       
# p.add_tools(hover)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# output_notebook(hide_banner=True)
# show(p); 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # hover.tooltips = [
# #     ("index", "$index"),
# #     ("(x,y)", "($x, $y)"),
# #     ("radius", "@radius"),
# #     ("fill color", "$color[hex, swatch]:fill_color"),
# #     ("foo", "@foo"),
# #     ("bar", "@bar"),
# # ]




# ### older:
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #HBAR
# #    need y and 'right' (i.e. x) values
# #To draw horizontal bars by specifying a (center) y-coordinate, height, and left and right endpoints, use the hbar() glyph function:
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# my_data = df[['PlayerCollegeName','NflId', 'DisplayName']].drop_duplicates().copy()

# college_attended = my_data["PlayerCollegeName"].value_counts()

# df_cc = pd.DataFrame({'CollegeName':college_attended.index, 'Count':college_attended.values}).sort_values("Count", ascending = False)

# df_cc = df_cc[df_cc.CollegeName != 'Louisiana State']

# df_cc.at[42,'Count']=51

# #df_cc[df_cc.CollegeName=='LSU']['Count']

# df_cc.sort_values('Count', ascending=False)

# pd.set_option('display.max_rows', 500)


# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# mysource = ColumnDataSource(df_cc)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# p = figure(
#   y_range=df_cc.CollegeName[:50],    # I need to enter the SAME thing here as y points, i find that odd        
#   title = '\nNumber of players that attended Colleges Attended - Player Count\n',
#   x_axis_label ='# of NFL players that attended the college prior\n',
#   plot_width=500,
#   plot_height=700,
#   tools="", toolbar_location=None)
#   #min_border=0
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# p.hbar(
#     y=df_cc.CollegeName[:40],  # center of your y coordinate launcher, 40 points... 
#     left=0, # or left=20, etc
#     right=df_cc.Count[:40],    # right is 40 points... 
#     height=0.4,
#     color='orange',
#     fill_alpha=0.4,)
#     #source = mysource)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Axes: 
# p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
# p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
# p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
# p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
# #p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# #p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# # Grid: 
# p.xgrid.grid_line_color = None
# p.ygrid.grid_line_color = None   
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ### NOTES here: 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# output_notebook(hide_banner=True)
# show(p); 
# #~~~~~~

