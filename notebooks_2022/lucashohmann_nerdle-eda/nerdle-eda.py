#!/usr/bin/env python
# coding: utf-8

# # **EDA and Pre Processing to simplify further Analysis and Simulations**
# 
# ### Steps:
# 
# + Split each equation into separate columns
# + Count the amount of unique elements per equation
# + Count how many times does an element appear in each equation
# + Count the minimum and maximum times an element can appear in all equations
# + Count how many times does each element appear in each position across all equations
# + Correlate elements
# + Correlate elements across positions

# In[ ]:


import plotly.graph_objects as go
import plotly.subplots as sp
from tqdm import tqdm
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 10)


# ## Defining constants

# In[ ]:


NUMBER_OF_ELEMENTS = 8
ELEMENTS = list("1234568790+-*/=")


# # Loading the data
# 
# *(And setting the index)*

# In[ ]:


df_raw = pd.read_csv("../input/nerdle-valid-equations/equations_standard.csv")
df_raw.index = df_raw["equation"]

df_raw


# ### Spliting the equations

# In[ ]:


df_positions = pd.DataFrame()

# Extract each position
for i in range(NUMBER_OF_ELEMENTS):
    df_positions[f"p{i}"] = df_raw["equation"].str[i].convert_dtypes()


df_positions


# ### Counting the number of unique elements per equation

# In[ ]:


df_unique = pd.DataFrame()

# Compute the number of unique elements
df_unique["count"] = df_raw["equation"].apply(lambda x: len(set(x))).convert_dtypes()


df_unique


# Checking the distribution of equations by number of unique elements

# In[ ]:


# Counting unique elements distribution
df_temp = df_unique["count"].value_counts()

# Creating figure
fig = go.Figure()

# Adding trace
fig.add_trace(
    go.Pie(
        labels=df_temp.index,
        values=df_temp,
    )
)

# Updating the layout
fig.update_layout(
    title="Unique elements distribution",
    height=600,
    width=600,
)

# Showing the figure
fig.show()


# Checking the mean element *"uniqueness"*

# In[ ]:


df_unique["count"].mean()


# ### Counting the amount of each element in each equation

# In[ ]:


df_elements = pd.DataFrame()

# Compute the element count in each equation
for element in ELEMENTS:
    df_elements[element] = df_raw["equation"].apply(lambda x: x.count(element)).convert_dtypes()


df_elements


# ## Joining all DataFrames

# In[ ]:


# Concatenating the DataFrames into a multi-indexed single one
df = pd.concat([df_positions, df_unique, df_elements], axis=1, keys=['positions', 'unique', 'elements'])

df.head()


# # Counting element occurances

# In[ ]:


df_elements_occurances = pd.DataFrame()

# Counting the values of each element
for element in ELEMENTS:
    df_elements_occurances = pd.concat([df_elements_occurances, df["elements"][element].value_counts()], axis=1)

# Transposing the DataFrame
df_elements_occurances = df_elements_occurances.transpose()

# Fillin the missing values and converting "dtypes"
df_elements_occurances = df_elements_occurances.fillna(0)
df_elements_occurances = df_elements_occurances.convert_dtypes()


df_elements_occurances


# In[ ]:


# Plotting bars

fig = go.Figure()

for i in range(min(*df_elements_occurances.columns), max(*df_elements_occurances.columns) + 1):
    fig.add_trace(go.Bar(x=df_elements_occurances.index, y=df_elements_occurances[i], name=i))

fig.update_layout(
    title="Elements occurances",
    barmode='stack',
    height=500,
    width=1000,
)

fig.show()


# # Counting the 'min' and 'max' times an element can be found in a word

# In[ ]:


df_elements_minmax = pd.DataFrame()

# Computing the minimum and maximum number of occurances of each element
for element in ELEMENTS:
    df_elements_minmax.loc["min", element]= df[("elements", element)].min()
    df_elements_minmax.loc["max", element]= df[("elements", element)].max()

# Converting 'dtypes'
df_elements_minmax = df_elements_minmax.convert_dtypes()


df_elements_minmax.style.background_gradient(axis=1, cmap="plasma")


# # Counting element frequency

# In[ ]:


# Computing element frequency
df_freq_count = df["positions"].apply(pd.Series.value_counts)

# Computing the "sum"
df_freq_count["sum"] = df_freq_count.sum(axis=1)

# Inserting missing elements
for element in ELEMENTS:
    if element not in df_freq_count.index:
        df_freq_count.loc[element] = 0

# Computing the "total"
totals = []
for element in df_freq_count.index:
    totals.append(df["elements"][element].value_counts().drop(0, errors="ignore").sum())

df_freq_count["total"] = totals

# Filling the missing values, converting the "dtypes" and sorting
df_freq_count = df_freq_count.fillna(0)
df_freq_count = df_freq_count.convert_dtypes()
df_freq_count = df_freq_count.sort_index()


df_freq_count


# In[ ]:


# Ploting Heatmap

# Sorting data
df_freq_count = df_freq_count.sort_index()

# Creating figure
fig = sp.make_subplots(
    rows=1, 
    cols=2,
    column_widths=[0.85, 0.15],
)

# Creating "Positions" traces
fig.add_trace(
    go.Heatmap(
        x=[f"p{i}" for i in range(NUMBER_OF_ELEMENTS)],
        y=df_freq_count.index,
        z=df_freq_count / len(df),
        coloraxis=f"coloraxis1",
        name="Positions",
    ),
    row=1,
    col=1,
)

# Creating "Total Unique" trace
fig.add_trace(
    go.Heatmap(
        x=["Total"],
        y=df_freq_count.index,
        z=df_freq_count[["total"]] / len(df),
        coloraxis=f"coloraxis2",
        name="Total",
    ),
    row=1,
    col=2,
)

# Updating layout
fig.update_layout(
    title="Element Frequency",
    height=600,
    width=600,
    coloraxis1=dict(
        showscale=False,
        colorscale="Portland",
    ),
    coloraxis2=dict(
        showscale=False,
        colorscale="Portland",
    ),
)

# Showing figure
fig.show()


# In[ ]:


# Ploting Bars

# Creating figure
fig = sp.make_subplots(
    rows=NUMBER_OF_ELEMENTS + 1, 
    cols=1,
)

# Creating a trace for each position
for i, position in enumerate(df["positions"].columns):

    # Sorting the data
    df_temp = df_freq_count.sort_values(by=position, ascending=False)

    # Creating the trace
    fig.add_trace(
        go.Bar(
            x=df_temp.index,
            y=df_temp[position] / len(df),
            name=position,
        ),
        row=i+1,
        col=1,
    )

# Creating a trace for the "Total"
df_freq_count = df_freq_count.sort_values(by="total", ascending=False)

fig.add_trace(
    go.Bar(
        x=df_freq_count.index,
        y=df_freq_count["total"] / len(df),
        name="Total",
    ),
    row=NUMBER_OF_ELEMENTS + 1,
    col=1,
)

titles_positions = {f"yaxis{i + 1}_title": position for i, position in enumerate(df["positions"].columns)}
title_total = {f"yaxis{NUMBER_OF_ELEMENTS + 1}_title": "Total"}

# Updating the layout
fig.update_layout(
    title="Element frequency",
    height=800,
    width=1000,
    showlegend=False,
    **titles_positions,
)

fig.update_layout(
    **title_total,
)

# Showing the figure
fig.show()


# # Correlating elements

# In[ ]:


df_elements_corr = df["elements"].corr()
df_elements_corr


# In[ ]:


# Ploting Heatmap

# Masking the diagonal
mask = np.triu(np.ones_like(df_elements_corr, dtype=bool))
df_temp = df_elements_corr.mask(mask)

# Creating figure
fig = go.Figure()

# Creating trace
fig.add_trace(
    go.Heatmap(
        x=df_temp.columns,
        y=df_temp.columns,
        z=df_temp,
        colorscale="Portland",
    )
)

# Updating layout
fig.update_layout(
    title="Elements correlation",
    height=600,
    width=600,
)

# Showing figure
fig.show()


# # Correlating positions

# In[ ]:


df_positions_corr = df_freq_count[df["positions"].columns].corr()
df_positions_corr


# In[ ]:


# Plotting Heatmap

# Masking the diagonal
mask = np.triu(np.ones_like(df_positions_corr, dtype=bool))
df_temp = df_positions_corr.mask(mask)

# Creating figure
fig = go.Figure()

# Creating trace
fig.add_trace(
    go.Heatmap(
        x=df_temp.columns,
        y=df_temp.columns,
        z=df_temp,
        colorscale="Portland",
    )
)

# Updating layout
fig.update_layout(
    title="Positions correlation",
    height=600,
    width=600,
)

# Showing figure
fig.show()


# # Correlating each element to another element based on their positions
# 
# *(This method is extremely slow, but it's an easy way to get all elements in the table (including the ones not present in a given position))*

# In[ ]:


def correlate_positions(df, position_0, position_1, pbar):
    """ Correlating the frequencies of elements in two given positions """

    # Initializing a temporary dataframe
    df_positions_corr = pd.DataFrame([[0 for _, _ in enumerate(ELEMENTS)] for _, _ in enumerate(ELEMENTS)], columns=ELEMENTS, index=ELEMENTS)

    # Iterating over the first element
    for i, e in enumerate(ELEMENTS):

        # Iterating over the second element
        for j, _ in enumerate(ELEMENTS):
            counter = 0

            # Update the progress bar
            pbar.set_postfix_str(f"Positions: {position_0},{position_1} - Elements: {ELEMENTS[i]},{ELEMENTS[j]}")

            # Iterating over all the equations in the dataframe
            for row in df.itertuples():

                # If the element in the positions are the same, increment the counter
                if row[position_0 + 1] == ELEMENTS[i] and row[position_1 + 1] == ELEMENTS[j]:
                    counter += 1
                
                # Update the progress bar
                pbar.update(1)

            # Storing the counter value
            df_positions_corr[ELEMENTS[i]].loc[ELEMENTS[j]] = counter
    
    # Returning the dataframe
    return df_positions_corr


def correlate_elements(df):
    """ Correlating the frequencies of elements across all positions """

    # Creating the progress bar and an empty DataFrame
    pbar = tqdm(total=int(((NUMBER_OF_ELEMENTS ** 2) - NUMBER_OF_ELEMENTS) / 2) * len(df) * len(ELEMENTS) ** 2)
    df_elements_corr = pd.DataFrame()

    # Iterating over the first position
    for i in range(NUMBER_OF_ELEMENTS):
        df_i = pd.DataFrame()

        # Iterating over the second position
        for j in range(i + 1, NUMBER_OF_ELEMENTS):
            
            # Computing the correlation between the two positions
            df_j = correlate_positions(df, i, j, pbar)

            # Concatenating the dataframes
            df_j = pd.concat([df_j], keys=[f"p{j}"], names=[f'position', 'element'], axis=0)
            df_j = pd.concat([df_j], keys=[f"p{i}"], names=[f'position', 'element'], axis=1)
            df_i = pd.concat([df_i, df_j], axis=0)

        # Concatenating the dataframes again
        df_elements_corr = pd.concat([df_elements_corr, df_i], axis=1)

    # Converting dtypes and closing the progress bar
    df_elements_corr = df_elements_corr.convert_dtypes()
    pbar.close()

    # Returning the dataframe
    return df_elements_corr


df_elements_pos_corr = correlate_elements(df["positions"])


# In[ ]:


# Ploting Heatmap

# Creating figure
fig = sp.make_subplots(
    rows=NUMBER_OF_ELEMENTS - 1, 
    cols=NUMBER_OF_ELEMENTS - 1,
    shared_xaxes=True,
    shared_yaxes=True,
    vertical_spacing=0.01,
    horizontal_spacing=0.01,
)

# Creating traces
for i in range(NUMBER_OF_ELEMENTS):
    for j in range(i + 1, NUMBER_OF_ELEMENTS):

        df_temp = df_elements_pos_corr[f"p{i}"].loc[f"p{j}"]

        if df_temp is None:
            continue

        fig.add_trace(
            go.Heatmap(
                x=df_temp.columns,
                y=df_temp.index,
                z=df_temp / len(df),
                name=f"{i}-{j}",
                visible=True,
                coloraxis=f"coloraxis{i + j + 1}",
            ),
            row=j, 
            col=i + 1,
        )

        fig.add_trace(
            go.Heatmap(
                x=df_temp.columns,
                y=df_temp.index,
                z=df_temp / len(df),
                name=f"{i}-{j}",
                visible=False,
                coloraxis=f"coloraxis1",
            ),
            row=j, 
            col=i + 1,
        )

# Updating layout
coloraxis = dict(colorscale="Portland", showscale=False)
coloraxes = {f"coloraxis{i + 1}": coloraxis for i in range(int(((NUMBER_OF_ELEMENTS ** 2) - NUMBER_OF_ELEMENTS) / 2) + 1)}

fig.update_layout(
    title="Element correlation per position",
    height=1000,
    width=1000,
    **coloraxes,
)

fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)

for i in range(NUMBER_OF_ELEMENTS - 1):
    fig.update_xaxes(title_text=f"p{i}", side="bottom", row=NUMBER_OF_ELEMENTS - 1, col= i + 1)
    fig.update_yaxes(title_text=f"p{NUMBER_OF_ELEMENTS - i - 1}", row=NUMBER_OF_ELEMENTS - i - 1, col=1)

# Adding dropdown
fig.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=[{"visible": [True, False]}],
                    label="Independent Scales",
                ),
                dict(
                    args=[{"visible": [False, True]}],
                    label="Single Scale",
                ),
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1,
            xanchor="right",
            y=1.1,
            yanchor="top"
        ),
    ]
)

# Showing figure
fig.show()

