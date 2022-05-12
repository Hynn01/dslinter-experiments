#!/usr/bin/env python
# coding: utf-8

# # Web Scraping: Location map of nursing homes in Spain data 2020.
# 
# The big challenge has been the web scraping of the 52 excels of the different provinces of Spain. Some residences are badly located due to the original latitude and longitude data.

# In[ ]:


import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from skimage import io
import requests 
import numpy as np
import matplotlib.pyplot as plt
import requests


# In[ ]:


get_ipython().system('pip install openpyxl')


# In[ ]:


def get_data(link):
    hdr = {'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Mobile Safari/537.36'}

    req = requests.get(link,headers=hdr)
    content = req.content
  
    return content

test_URL = 'http://envejecimiento.csic.es/recursos/residencias/por_provincia.html'
data = get_data(test_URL)
soup = BeautifulSoup(data, "html.parser")

links=[]
for link in soup.find_all('a'):
    links.append(link.get('href'))


# In[ ]:


l=links[16:(72-4)]
p=[]
for i in l:
    p.append("http://envejecimiento.csic.es/"+i)

z=[]
for i in p:
    g=pd.read_excel(i)
    z.append(g[3:])

dato=pd.concat(z)


# In[ ]:


dato.columns =['Denominación', 'Dirección', 'D2', 'D3', 'CP', 'Municipio', 'Telefono', 'Titularidad', 'Plazas', 'url', 'Latitud', 'Longitud', 'Actualizado','3','4']
dato['Latitud']=[float(str(i).replace(",", "")) for i in dato.Latitud]
dato['Latitud']=dato['Latitud']*0.000001
dato['Plazas'] = dato['Plazas'].fillna(0).astype(int)
dato['Longitud']=[float(str(i).replace(",", "")) for i in dato.Longitud]
dato['Longitud']=dato['Longitud']*0.000001


# In[ ]:


g=dato['Plazas'].mean()
M=dato['Plazas'].max()
m=dato['Plazas'].min()
dato['plazas']=dato['Denominación']+ '; ' + dato['Plazas'].apply(str) + ' plazas'
print(g)
print(M)
print(m)


# In[ ]:


import plotly.express as px
mapbox_access_token = 'pk.eyJ1IjoiZ3VpbGxlcm1vbmF2YXJyMCIsImEiOiJja3ZhbXRoZGUwNHBqMzFscWV2b3ZjY240In0.aIj8aOGKHbZx3cq_DFRGvA'
px.set_mapbox_access_token(mapbox_access_token)
df = dato
fig = px.scatter_mapbox(df, lat="Latitud", lon="Longitud",     color="Plazas", size="Plazas",
                  color_continuous_scale='aggrnyl', size_max=20, zoom=3, hover_name="plazas")
fig.show()


# In[ ]:


import plotly.graph_objects as go



fig = go.Figure()



fig.add_trace(go.Scattermapbox(
        lat=dato['Latitud'],
        lon=dato['Longitud'],
        mode='markers',
        name='Ubicación residencias de mayores en España',
        marker=go.scattermapbox.Marker(
            size=abs(round((dato['Plazas']+dato['Plazas'].mean())/dato['Plazas'].std()*5,0)),
            color=dato['Plazas'],
            colorscale='aggrnyl',
            opacity=1
        ),
        text=dato['plazas'],
        hoverinfo='text',legendgroup="group2"
    ))




fig.update_layout(
    title='Ubicaciones de las residencias',
    autosize=True,
    width=1500,
    height=800,
    
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=41.65518,
            lon=-4.72372
        ),
        pitch=0,
        zoom=3,
        style='basic'
    ),
)

fig.update_traces(showlegend=True)



fig.show()

