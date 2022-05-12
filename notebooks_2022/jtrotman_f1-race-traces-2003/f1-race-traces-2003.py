#!/usr/bin/env python
# coding: utf-8

# # Formula One Race Traces — 2003
# 
# This notebook shows race traces for each F1 race in 2003, using the [Formula 1 Race Data][2] data set, which is sourced from [ergast.com](http://ergast.com/mrd/),
# and the [Formula 1 Race Events dataset](https://www.kaggle.com/jtrotman/formula-1-race-events) using data from Wikipedia.
# 
# A race trace is a way to visualise the progress of an entire Grand Prix, they show gaps between cars and general field spread, as well as the relative pace of each car throughout the race (as line gradient).
# The traces are calculated from cumulative race lap times, adjusted by the median lap time at that point in the race.
# You can think of the horizontal zero line as a *virtual reference car* doing the average lap time of the field, lines above are faster and lower are slower.
# 
# - Line **colours** for each driver are based on general team colours.
# - **Fastest lap** is marked with a ★
# - **Safety car** affected laps are highlighted in yellow - gaps between cars shrink.
# - **[Virtual safety cars][1]** were introduced in 2015 and force all cars to slow - this causes the lines to spread out temporarily as the leaders have completed more of the first affected lap at racing speeds (some now highlighted in orange).
# - Laps where a car went via the **pit lane** are marked with a &#9679; (for years with pit-stop data: 2012 onwards).
# - **Over-takes** are visible where the lines cross, often due to pit-stops.
# - Truncated lines indicate **retirement** from the race.
# 
# A **shadow** at the bottom shows which cars have have been lapped by the lead car, this helps to see some effects:
#  - Leaders will often make a pit-stop to *avoid* slower traffic ahead.
#  - The backmarkers must allow leading cars through - the edge of the shadow shows *why* their lines suddenly dip.
#  - Leaders may ease off to let a lapped car through at the end of a race.
# 
# *Note:* good safety car data is hard to find! The data scraped from Wikipedia appears to have a few mistakes; the 
# virtual safety car highlights are my estimates, using the lap-time data.
# 
# *Tip:* to see better resolution, &lt;*Right click*&gt; &rarr; *Open Image in New Tab*.
# 
# 
# ## Revisions
# 
# <pre>
# V2: moved legends to bottom to allow more width
# V3: added constructors championship plot and driver/team table links
# V4: add origin to championship traces & link for 2021
# V5: add fastest lap marker
# V6: added laps per position and top drivers chart
# V7: safety car, red flag & lapped cars highlighting
# </pre>
# 
# 
# ## All F1 Race Traces
# 
# There is a notebook for every year that has lap-time data:
# 
# [1996](https://www.kaggle.com/code/jtrotman/f1-race-traces-1996), 
# [1997](https://www.kaggle.com/code/jtrotman/f1-race-traces-1997), 
# [1998](https://www.kaggle.com/code/jtrotman/f1-race-traces-1998), 
# [1999](https://www.kaggle.com/code/jtrotman/f1-race-traces-1999), 
# [2000](https://www.kaggle.com/code/jtrotman/f1-race-traces-2000), 
# [2001](https://www.kaggle.com/code/jtrotman/f1-race-traces-2001), 
# [2002](https://www.kaggle.com/code/jtrotman/f1-race-traces-2002), 
# [2003](https://www.kaggle.com/code/jtrotman/f1-race-traces-2003), 
# [2004](https://www.kaggle.com/code/jtrotman/f1-race-traces-2004), 
# [2005](https://www.kaggle.com/code/jtrotman/f1-race-traces-2005), 
# [2006](https://www.kaggle.com/code/jtrotman/f1-race-traces-2006), 
# [2007](https://www.kaggle.com/code/jtrotman/f1-race-traces-2007), 
# [2008](https://www.kaggle.com/code/jtrotman/f1-race-traces-2008), 
# [2009](https://www.kaggle.com/code/jtrotman/f1-race-traces-2009), 
# [2010](https://www.kaggle.com/code/jtrotman/f1-race-traces-2010), 
# [2011](https://www.kaggle.com/code/jtrotman/f1-race-traces-2011), 
# [2012](https://www.kaggle.com/code/jtrotman/f1-race-traces-2012), 
# [2013](https://www.kaggle.com/code/jtrotman/f1-race-traces-2013), 
# [2014](https://www.kaggle.com/code/jtrotman/f1-race-traces-2014), 
# [2015](https://www.kaggle.com/code/jtrotman/f1-race-traces-2015), 
# [2016](https://www.kaggle.com/code/jtrotman/f1-race-traces-2016), 
# [2017](https://www.kaggle.com/code/jtrotman/f1-race-traces-2017), 
# [2018](https://www.kaggle.com/code/jtrotman/f1-race-traces-2018), 
# [2019](https://www.kaggle.com/code/jtrotman/f1-race-traces-2019), 
# [2020](https://www.kaggle.com/code/jtrotman/f1-race-traces-2020), 
# [2021](https://www.kaggle.com/code/jtrotman/f1-race-traces-2021),
# [2022](https://www.kaggle.com/code/jtrotman/f1-race-traces-2022).
# 
#  [1]: https://en.wikipedia.org/wiki/Safety_car#Virtual_safety_car_(VSC)
#  [2]: https://www.kaggle.com/cjgdev/formula-1-race-data-19502017
#  [3]: https://www.kaggle.com/jtrotman/formula-1-race-events
# 

# In[ ]:


YEAR = 2003
DRIVER_LS = {2:1,4:0,8:0,11:2,14:1,15:1,17:0,18:0,21:0,22:1,23:1,30:0,31:0,35:1,42:1,43:0,44:1,47:2,48:2,49:0,50:1,51:0,52:1,53:2}
DRIVER_C = {2:"#003B76",4:"#B4B400",8:"#7F7F7F",11:"#9E004F",14:"#7F7F7F",15:"#B4B400",17:"#008400",18:"#9E004F",21:"#7FFE00",22:"#FF0000",23:"#007FFE",30:"#FF0000",31:"#007FFE",35:"#9E004F",42:"#008400",43:"#E300E3",44:"#E300E3",47:"#7FFE00",48:"#007FFE",49:"#003B76",50:"#C46200",51:"#C46200",52:"#7FFE00",53:"#C46200"}
TEAM_C = {1:"#7F7F7F",3:"#007FFE",4:"#B4B400",6:"#FF0000",7:"#E300E3",15:"#003B76",16:"#9E004F",17:"#7FFE00",18:"#C46200",19:"#008400"}
LINESTYLES = ['-', '-.', '--', ':', '-', '-']


# # 2003 Formula One World Championship
# 
# For background see [Wikipedia](https://en.wikipedia.org/wiki/2003_Formula_One_World_Championship); here's an excerpt:
# 
# The 2003 FIA Formula One World Championship was the 57th season of FIA Formula One motor racing. It began on 9 March 2003 and ended on 12 October after sixteen races. World Championship titles were awarded for both drivers and constructors with Michael Schumacher winning the former and Ferrari awarded the latter.
# 
# The 2003 season saw the introduction of new regulations intended to increase F1's excitement and to help alleviate the financial difficulties of the smaller teams.  One-lap qualifying was introduced as a way for smaller teams to get more television exposure. Optional Friday testing at Grand Prix events was introduced in exchange for fewer miles on stand-alone test days. This was intended to give smaller teams a cheaper alternative to these test days, which were to be banned in 2004. Only one type of wet weather tyre was allowed to be used in wet weather races. The points system for both the Constructors' and Drivers' titles was changed from 10–6–4–3–2–1 for the first six finishers at each round to 10–8–6–5–4–3–2–1 for the first eight finishers in an attempt to make the title contests closer.
# 
# While Ferrari's Michael Schumacher had won the 2002 championship by 67 points from his teammate Rubens Barrichello, the 2003 season was much closer. For a great part of the 2003 season, several drivers from several teams had mathematical chances of winning the world championship. Eight different drivers won a Grand Prix, amongst them three first time winners. Kimi Räikkönen, driving for McLaren-Mercedes, and Juan Pablo Montoya, driving for BMW Williams, both had a chance of claiming the 2003 championship until late in the season, with Räikkönen still mathematically in contention at the final race, the Japanese Grand Prix. Räikkönen lost the championship to Schumacher by two points, although he won only one race to Schumacher's six. It was Schumacher's sixth World Drivers' title overall, breaking Juan Manuel Fangio's 46-year-old record of five World Drivers' titles. Ferrari's defence of the Constructors' title was challenged throughout the year by Williams and McLaren, one of the few seasons where there were three front-running teams but in the end, Ferrari emerged victorious once again and the team clinched their fifth consecutive World Constructors' title since 1999.
# 
# Notable races include the chaotic Brazilian Grand Prix which was hampered by monsoon conditions, and the British Grand Prix where the track was invaded by the now-defrocked priest Neil Horan, who ran onto the Hangar straight, running towards the 250 km/h train of cars, wearing a green kilt and waving religious banners.
# 
# After failing to complete the 2002 season due to financial difficulties, the Arrows team had their application for admission to the 2003 championship rejected by the FIA prior to the season start date. No reason was publicly given by the FIA and Arrows subsequently folded after 25 years in Formula One since 1978.
# 
# 2003 also saw a major leap forward in Formula One safety, with the HANS device being made a mandatory requirement for drivers to wear at all races beginning from the Australian Grand Prix onwards. However, this was not without controversy, as many drivers voiced their complaints about the device, including Barrichello, Jacques Villeneuve, Justin Wilson and Nick Heidfeld.

# In[ ]:


import json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import urllib
from collections import Counter

def read_csv(name, **kwargs):
    df = pd.read_csv(f'../input/formula-1-race-data-19502017/{name}', **kwargs)
    return df

def races_subset(df, races_index):
    df = df[df.raceId.isin(races_index)].copy()
    df['round'] = df.raceId.map(races['round'])
    df['round'] -= df['round'].min()
    return df.set_index('round').sort_index()

def add_lap_0(df):
    copy = df.T
    copy.insert(0, 0, 0)
    return copy.T

IMG_ATTRS = 'style="display: inline-block;" width=16 height=16'
YT_IMG = f'<img {IMG_ATTRS} src="https://youtube.com/favicon.ico">'
WK_IMG = f'<img {IMG_ATTRS} src="https://wikipedia.org/favicon.ico">'
GM_IMG = f'<img {IMG_ATTRS} src="https://maps.google.com/favicon.ico">'

# Read data
circuits = read_csv('circuits.csv', encoding='ISO-8859-1', index_col=0)
constructorResults = read_csv('constructorResults.csv', index_col=0)
constructors = read_csv('constructors.csv', index_col=0)
constructorStandings = read_csv('constructorStandings.csv', index_col=0)
drivers = read_csv('drivers.csv', encoding='ISO-8859-1', index_col=0)
driverStandings = read_csv('driverStandings.csv', index_col=0)
lapTimes = read_csv('lapTimes.csv')
pitStops = read_csv('pitStops.csv')
qualifying = read_csv('qualifying.csv', index_col=0)
races = read_csv('races.csv', index_col=0)
results = read_csv('results.csv', index_col=0)
seasons = read_csv('seasons.csv', index_col=0)
status = read_csv('status.csv', index_col=0)

# Additional dataset:
# https://www.kaggle.com/jtrotman/formula-1-race-events
safety_cars = pd.read_csv('../input/formula-1-race-events/safety_cars.csv')
red_flags = pd.read_csv('../input/formula-1-race-events/red_flags.csv')
with open('../input/formula-1-race-events/virtual_safety_car_estimates.json') as f:
    virtual_safety_cars = json.load(f)

# For compatibility with 2018 onwards version
races['raceKey'] = races.index.values

def url_extract(s):
    return (s.str.split('/') 
             .str[-1].fillna('') 
             .apply(urllib.parse.unquote) 
             .str.replace('_', ' ', regex=False) 
             .str.replace('\s*\(.*\)', '', regex=True))

# Fix circuit data
idx = circuits.url.str.contains('%').fillna(False)
circuits.loc[idx, 'name'] = url_extract(circuits[idx].url)
circuits.location.replace({ 'MontmelÌ_':'Montmeló',
                            'SÌ£o Paulo':'São Paulo',
                            'NÌ_rburg':'Nürburg'}, inplace=True)

# Fix driver data
idx = drivers.url.str.contains('%').fillna(False)
t = url_extract(drivers.url)
drivers.loc[idx, 'forename'] = t[idx].str.split(' ').str[0]
drivers.loc[idx, 'surname'] = t[idx].str.split(' ').str[1:].str.join(' ')

# Fix Montoya (exception not fixed by above code)
drivers.loc[31, 'forename'] = 'Juan Pablo'
drivers.loc[31, 'surname'] = 'Montoya'

idx = drivers.surname.str.contains('Schumacher').fillna(False)
drivers['display'] = drivers.surname
drivers.loc[idx, 'display'] = drivers.loc[idx, 'forename'].str[0] + ". " + drivers.loc[idx, 'surname']

# For display in HTML tables
drivers['Driver'] = drivers['forename'] + " " + drivers['surname']
drivers['Driver'] = drivers.apply(lambda r: '<a href="{url}">{Driver}</a>'.format(**r), 1)
constructors['label'] = constructors['name']
constructors['name'] = constructors.apply(lambda r: '<a href="{url}">{name}</a>'.format(**r), 1)

# Join fields
results['status'] = results.statusId.map(status.status)
results['Team'] = results.constructorId.map(constructors.name)
results['score'] = results.points>0
results['podium'] = results.position<=3

# Cut data to one year
races = races.query('year==@YEAR').sort_values('round').copy()
results = results[results.raceId.isin(races.index)].copy()
lapTimes = lapTimes[lapTimes.raceId.isin(races.index)].copy()
driverStandings = races_subset(driverStandings, races.index)
constructorStandings = races_subset(constructorStandings, races.index)

def t2s(t):
    return f'{t.hours:.0f}:{t.minutes:02.0f}:{t.seconds:02.0f}.{t.milliseconds:03.0f}'

# Original 'time' field is corrupt
results['time'] = ('+' + results.time.str.replace('+', '', regex=False))
results['Time'] = pd.to_timedelta(results.milliseconds*1e6)

lapTimes = lapTimes.merge(results[['raceId', 'driverId', 'positionOrder']], on=['raceId', 'driverId'])
lapTimes['seconds'] = lapTimes.pop('milliseconds') / 1000

def formatter(v):
    if type(v) is str:
        return v
    if pd.isna(v) or v <= 0:
        return ''
    if v == int(v):
        return f'{v:.0f}'
    return f'{v:.1f}'

def table_html(table, caption):
    return (f'<h3>{caption}</h3>' +
            table.style.format(formatter).to_html())

# Processing for Drivers & Constructors championship tables
def format_standings(df, key):
    df = df.sort_values('position')
    gb = results.groupby(key)
    df['Position'] = df.positionText
    df['scores'] = gb.score.sum()
    df['podiums'] = gb.podium.sum()
    return df

# Drivers championship table
def drivers_standings(df):
    index = 'driverId'
    df = df.set_index(index)
    df = df.join(drivers)
    df = format_standings(df, index)
    df['Team'] = results.groupby(index).Team.last()
    use = ['Position', 'Driver',  'Team', 'points', 'wins', 'podiums', 'scores', 'nationality' ]
    df = df[use].set_index('Position')
    df.columns = df.columns.str.capitalize()
    return df

# Constructors championship table
def constructors_standings(df):
    index = 'constructorId'
    df = df.set_index(index)
    df = df.join(constructors)
    df = format_standings(df, index)
    
    # add drivers for team
    tmp = results.join(drivers.drop(labels="number", axis=1), on='driverId')
    df = df.join(tmp.groupby(index).Driver.unique().str.join(', ').to_frame('Drivers'))

    use = ['Position', 'name', 'points', 'wins', 'podiums', 'scores', 'nationality', 'Drivers' ]
    df = df[use].set_index('Position')
    df.columns = df.columns.str.capitalize()
    return df

# Race results table
def format_results(df):
    df['Team'] = df.constructorId.map(constructors.name)
    df['Position'] = df.positionOrder
    df = df.sort_values('Position').set_index('Position').copy()
    df.loc[[1], 'time'] = df.loc[[1], 'Time'].dt.components.apply(t2s, axis=1)
    use = ['Driver', 'Team', 'number', 'grid', 'points', 'laps', 'time', 'status' ]
    df = df[use]
    df.columns = df.columns.str.capitalize()
    return df


# In[ ]:


plt.rc("figure", figsize=(16, 12))
plt.rc("font", size=(14))
plt.rc("axes", xmargin=0.01)

display(HTML(
    f'<h1 id="drivers">Formula One Drivers\' World Championship &mdash; {YEAR}</h1>'
))

# Championship position traces
champ = driverStandings.groupby("driverId").position.last().to_frame("Pos")
champ = champ.join(drivers)
order = np.argsort(champ.Pos)

color = [DRIVER_C[d] for d in champ.index[order]]
style = [LINESTYLES[DRIVER_LS[d]] for d in champ.index[order]]
labels = champ.Pos.astype(str) + ". " + champ.display

chart = driverStandings.pivot("raceId", "driverId", "points")
# driverStandings may have a subset of races (i.e. season in progress) so reindex races
chart.index = races.reindex(chart.index).name.str.replace("Grand Prix", "GP").rename("Race")
chart.columns = labels

# Add origin
row = chart.iloc[0]
chart = pd.concat(((row * 0).to_frame("").T, chart))

chart.iloc[:, order].plot(title=f"F1 Drivers\' World Championship — {YEAR}", color=color, style=style)
plt.xticks(range(chart.shape[0]), chart.index, rotation=45)
plt.grid(axis="x", linestyle="--")
plt.ylabel("Points")
legend_opts = dict(bbox_to_anchor=(1.02, 0, 0.2, 1),
                   loc="upper right",
                   ncol=1,
                   shadow=True,
                   edgecolor="black",
                   mode="expand",
                   borderaxespad=0.)
plt.legend(**legend_opts)
plt.tight_layout()
plt.show()

display(HTML(table_html(drivers_standings(driverStandings.loc[driverStandings.index.max()]), "Results:")))


# In[ ]:


display(HTML(
    f'<h1 id="constructors">Formula One Constructors\' World Championship &mdash; {YEAR}</h1>'
))

# Championship position traces
champ = constructorStandings.groupby("constructorId").position.last().to_frame("Pos")
champ = champ.join(constructors)
order = np.argsort(champ.Pos)

color = [TEAM_C[c] for c in champ.index[order]]
labels = champ.Pos.astype(str) + ". " + champ.label

chart = constructorStandings.pivot("raceId", "constructorId", "points")
# constructorStandings may have a subset of races (i.e. season in progress) so reindex races
chart.index = races.reindex(chart.index).name.str.replace("Grand Prix", "GP").rename("Race")
chart.columns = labels

# Add origin
row = chart.iloc[0]
chart = pd.concat(((row * 0).to_frame("").T, chart))

chart.iloc[:, order].plot(title=f"F1 Constructors\' World Championship — {YEAR}", color=color)
plt.xticks(range(chart.shape[0]), chart.index, rotation=45)
plt.grid(axis="x", linestyle="--")
plt.ylabel("Points")
plt.legend(**legend_opts)
plt.tight_layout()
plt.show()

display(HTML(table_html(constructors_standings(constructorStandings.loc[constructorStandings.index.max()]), "Results:")))


# In[ ]:


# Show race traces
NEGATIVE_CUTOFF = -180
fastest_laps = Counter()
fontdict = {"fontstyle":"italic", "fontsize":14}

def display_header(race):

    circuit = circuits.loc[race.circuitId]
    qstr = race["name"].replace(" ", "+")
    map_url = "https://www.google.com/maps/search/{lat}+{lng}".format(**circuit)
    vid_url = f"https://www.youtube.com/results?search_query=f1+{YEAR}+{qstr}"

    lines = [
        '<h1 id="race{round}">R{round} — {name}</h1>'.format(**race),
        '<p><b>{date}</b> — '.format(img=WK_IMG, **race),
        '<b>Circuit:</b> <a href="{url}">{name}</a>, {location}, {country}'.format(**circuit),
        '<br><a href="{url}">{img} Wikipedia race report</a>'.format(img=WK_IMG, **race),
        f'<br><a href="{map_url}">{GM_IMG} Map Search</a>',
        f'<br><a href="{vid_url}">{YT_IMG} YouTube Search</a>',
    ]
    
    display(HTML("\n".join(lines)))


for race_key, times in lapTimes.groupby(lapTimes.raceId.map(races["raceKey"])):

    race = races.query("raceKey==@race_key").squeeze()
    fullname = str(race["year"]) + " " + race["name"]
    title = "Round {round} — F1 {name} — {year}".format(**race)
    
    res = results.query("raceId==@race.name").set_index("driverId")
    res = res.join(drivers.drop(labels="number", axis=1))

    display_header(race)

    # Lap time data: One row per lap, one column per driver, values are lap time in seconds
    chart = times.pivot_table("seconds", "lap", "driverId")

    sc = safety_cars.query("Race==@fullname")[["Deployed", "Retreated"]]
    sc[["Deployed", "Retreated"]] -= 1
    sc = sc.fillna(len(chart)).astype(int)
    vsc = virtual_safety_cars.get(fullname, [])
    flags = red_flags.query("Race==@fullname")[["Lap"]]
    flags = flags.astype(int)

    annotation = ""
    if len(sc):
        lst = ", ".join(sc.Deployed.astype(str) + "-" + sc.Retreated.astype(str))
        annotation += f" Safety Car Laps: [{lst}]"
    if len(vsc):
        annotation += f" Virtual Safety Car Laps: {vsc}"
    if len(flags):
        lst = ", ".join(flags.Lap.astype(str))
        annotation += f" Red Flag: [{lst}]"

    # Re-order columns by race finish position for the legend
    labels = res.loc[chart.columns].apply(lambda r: "{positionOrder:2.0f}. {display}".format(**r), 1)
    order = np.argsort(labels)
    show = chart.iloc[:, order]

    basis = chart.median(1).cumsum() # reference laptime series
    frontier = chart.cumsum().min(1) # running best cumulative time

    # A red flag stoppage will create very long lap-times.
    # If this is late in the race & the cars are on different laps
    # (i.e. some are 1,2 or 3 laps down) the median time may be low for all those laps.
    # This means the overall median does not adjust the cumulative times enough,
    # this code corrects for that... e.g. Monaco 2011
    if any((frontier-basis)>100):
        adjust = ((frontier-basis)>100) * (frontier-basis).max()
        basis = (chart.median(1) + adjust.diff().fillna(0)).cumsum()
    
    # Subtract reference from cumulative lap times
    show = -show.cumsum().subtract(basis, axis=0)

    # Fix large outliers (again, due to red flags), e.g. Australia 2016
    show[show>1000] = np.nan

    # Pitstops
    stops = pitStops.query("raceId==@race.name")
    if len(stops):
        # Brazil 2014 has pitstop times for Kevin Magnussen but no laptimes!
        stops = stops[stops.driverId.isin(chart.columns)]
        # Find x,y points for pitstops
        # (pd.DataFrame.lookup could do this with 1 line but it's deprecated)
        col_ix = list(map(show.columns.get_loc, stops.driverId))
        row_ix = list(map(show.index.get_loc, stops.lap))
        stops_y = show.to_numpy()[row_ix, col_ix]

    fastest_lap = times.iloc[np.argmin(np.asarray(times.seconds))]
    fastest_lap_y = show.loc[fastest_lap.lap, fastest_lap.driverId]

    color = [DRIVER_C[d] for d in show.columns]
    style = [LINESTYLES[DRIVER_LS[d]] for d in show.columns]
    show.columns = labels.values[order]

    # Main Plot
    show.plot(title=title, style=style, color=color)
    plt.scatter(fastest_lap.lap,
                fastest_lap_y,
                s=200,
                marker='*',
                c=DRIVER_C[fastest_lap.driverId],
                alpha=.5)

    if len(stops):
        plt.scatter(stops.lap,
                    stops_y,
                    s=20,
                    marker='o',
                    c=list(map(DRIVER_C.get, stops.driverId)),
                    alpha=.5)

    top = show.max(1).max()
    bottom = max(NEGATIVE_CUTOFF, show.min(1).min())
    span = (top-bottom)
    ymin = bottom
    ymax = top+(span/50)

    # Add the shadow of where the lead car is compared to previous lap
    leader_line = add_lap_0(show).max(1)
    leader_times = add_lap_0(chart).cumsum().min(1).diff().shift(-1)
    plt.fill_between(leader_times.index,
                     (leader_line-leader_times).clip(NEGATIVE_CUTOFF),
                     -1000,
                     color='k',
                     alpha=.1)
    
    # Highlight safety cars
    for idx, row in sc.iterrows():
        plt.axvspan(row.Deployed, row.Retreated, color='#ffff00', alpha=.2);

    # Highlight virtual safety cars
    for lap in vsc:
        plt.axvspan(lap, lap+1, color='#ff9933', alpha=.2);

    # Highlight red flags
    for idx, row in flags.iterrows():
        plt.vlines(row.Lap, ymin, ymax, color='r', ls=':')

    # Finishing touches
    xticks = np.arange(0, len(chart)+1, 2)
    if len(chart) % 2:  # odd number of laps: nudge last tick to show it
        xticks[-1] += 1

    plt.xlabel("Lap", loc="right")
    plt.ylabel("Time Delta (s)")
    plt.xticks(xticks, xticks)
    plt.ylim(ymin, ymax)
    plt.grid(linestyle="--")
    plt.annotate(annotation, (10, -40), xycoords="axes pixels", **fontdict);
    plt.legend(bbox_to_anchor=(0, -0.2, 1, 1),
               loc=(0, 0),
               ncol=6,
               shadow=True,
               edgecolor="black",
               mode="expand",
               borderaxespad=0.)
    plt.tight_layout()
    plt.show()

    fastest_laps[fastest_lap.driverId.squeeze()] += 1
    fastest_lap = fastest_lap.to_frame('').T.join(drivers, on='driverId')
    fastest_lap.columns = fastest_lap.columns.str.capitalize()
    
    lines = [
        table_html(fastest_lap[['Lap', 'Position', 'Time', 'Driver']], "Fastest Lap"),
        table_html(format_results(res), "Results")
    ]
    display(HTML("\n".join(lines)))


# # Laps Per Position

# In[ ]:


champ = driverStandings.groupby("driverId").position.last().to_frame("Pos")
champ = champ.sort_values("Pos").join(drivers)
labels = champ.Driver + " (" + champ.Pos.astype(str) + ")"
grid = lapTimes.groupby(["driverId", "position"]).size().unstack()
grid = grid.reindex(champ.index)
grid = grid.fillna(0).astype(int)
grid.index = grid.index.map(labels)
grid.style.background_gradient(axis=1, cmap="Blues")


# # Top F1 Drivers 2003

# In[ ]:


results["win"] = (results["position"] == 1)
results["pole"] = (results["grid"] == 1)
results["top10"] = (results["grid"] <= 10)
results["dnf"] = results.position.isnull()

gb = results.groupby("driverId")

table = pd.DataFrame({
    "Laps": gb.laps.sum(),
    "Points": gb.points.sum(),
    "Wins": gb.win.sum(),
    "Podiums": gb.podium.sum(),
    "Scores": gb.score.sum(),
    "Poles": gb.pole.sum(),
    "Top 10 Starts": gb.top10.sum(),
    "Fastest Laps": fastest_laps,
    "Finishes": gb.position.count(),
    "DNFs": gb.dnf.sum()
}).fillna(0)

labels = champ.display + " (" + champ.Pos.astype(str) + ")"
data = table.reindex(champ.head(10).index)[::-1]
data.index = data.index.map(labels)


# In[ ]:


facecolor = "#F1F1F1"
fig = plt.figure(figsize=(20, 7), facecolor=facecolor)
colors = plt.get_cmap("tab20").colors

for i, col in enumerate(data, 1):
    ax = fig.add_subplot(1, table.shape[1], i, facecolor=facecolor)
    ax.barh(data.index, data[col], color=colors[i])
    ax.set_title(col, fontsize=15, fontweight="medium")
    ax.tick_params(left=False, bottom=False, right=False, top=False, labelleft=(i<=1))
    ax.xaxis.set_ticks([])

    for _, spine in ax.spines.items():
        spine.set_visible(False)

    for ind, val in data[col].iteritems():
        ax.text(0, ind, formatter(val), fontweight="bold")

plt.suptitle(f"Top F1 Drivers {YEAR}", fontsize=22, x=.12, y=1, fontweight="bold");


# # More F1 Race Traces
# 
# [1996](https://www.kaggle.com/code/jtrotman/f1-race-traces-1996), 
# [1997](https://www.kaggle.com/code/jtrotman/f1-race-traces-1997), 
# [1998](https://www.kaggle.com/code/jtrotman/f1-race-traces-1998), 
# [1999](https://www.kaggle.com/code/jtrotman/f1-race-traces-1999), 
# [2000](https://www.kaggle.com/code/jtrotman/f1-race-traces-2000), 
# [2001](https://www.kaggle.com/code/jtrotman/f1-race-traces-2001), 
# [2002](https://www.kaggle.com/code/jtrotman/f1-race-traces-2002), 
# [2003](https://www.kaggle.com/code/jtrotman/f1-race-traces-2003), 
# [2004](https://www.kaggle.com/code/jtrotman/f1-race-traces-2004), 
# [2005](https://www.kaggle.com/code/jtrotman/f1-race-traces-2005), 
# [2006](https://www.kaggle.com/code/jtrotman/f1-race-traces-2006), 
# [2007](https://www.kaggle.com/code/jtrotman/f1-race-traces-2007), 
# [2008](https://www.kaggle.com/code/jtrotman/f1-race-traces-2008), 
# [2009](https://www.kaggle.com/code/jtrotman/f1-race-traces-2009), 
# [2010](https://www.kaggle.com/code/jtrotman/f1-race-traces-2010), 
# [2011](https://www.kaggle.com/code/jtrotman/f1-race-traces-2011), 
# [2012](https://www.kaggle.com/code/jtrotman/f1-race-traces-2012), 
# [2013](https://www.kaggle.com/code/jtrotman/f1-race-traces-2013), 
# [2014](https://www.kaggle.com/code/jtrotman/f1-race-traces-2014), 
# [2015](https://www.kaggle.com/code/jtrotman/f1-race-traces-2015), 
# [2016](https://www.kaggle.com/code/jtrotman/f1-race-traces-2016), 
# [2017](https://www.kaggle.com/code/jtrotman/f1-race-traces-2017), 
# [2018](https://www.kaggle.com/code/jtrotman/f1-race-traces-2018), 
# [2019](https://www.kaggle.com/code/jtrotman/f1-race-traces-2019), 
# [2020](https://www.kaggle.com/code/jtrotman/f1-race-traces-2020), 
# [2021](https://www.kaggle.com/code/jtrotman/f1-race-traces-2021),
# [2022](https://www.kaggle.com/code/jtrotman/f1-race-traces-2022).
# 
# 
# ## See Also
# 
# This [notebook shows the same idea for one MotoGP race](https://www.kaggle.com/jtrotman/motogp-race-traces-from-pdf), and explores several ways of adjusting the plots to highlight new details.
# 

# *This notebook uses material from the Wikipedia article <a href="https://en.wikipedia.org/wiki/2003_Formula_One_World_Championship">"2003 Formula One World Championship"</a>, which is released under the <a href="https://creativecommons.org/licenses/by-sa/3.0/">Creative Commons Attribution-Share-Alike License 3.0</a>.*
