#!/usr/bin/env python
# coding: utf-8

# # 🇧🇷 Este projeto foi descontinuado em 08/06/2020
# # 🇺🇸 This project has been discontinued in June 8th, 2020
# 
# ---
# 
# # 🇧🇷 Panorama do COVID-19 no Brasil/ 🇺🇸 COVID-19 in Brazil: An Overview
# 
# 🇧🇷 O objetivo deste notebook é analisar os dados disponíveis na base de dados brasileira sobre o COVID-19 [disponível aqui](https://www.kaggle.com/unanimad/corona-virus-brazil/kernels) e, por meio de perguntas e respostas de alto-nível, colaborar para a análise da situação. O código produzido é uma maneira de atestar como as respostas foram obtidas, estando livre para consultas, revisão e sugestões de melhorias ou outras investigações por meio do painel de discussão.
# 
# **Disclaimer/Aviso Legal**: Essas informações devem servir aos interessados como uma primeira orientação. As informações gerais aqui contidas, no entanto, não fornecem qualquer garantia. Desse modo, está excluída a garantia ou responsabilidade de qualquer tipo, por exemplo, de precisão, confiabilidade, completude e atualidade das informações. 
# 
# ---
# 
# 🇺🇸 The objective of this notebook is to analyze the data available in the Brazilian COVID-19 database [available here](https://www.kaggle.com/unanimad/corona-virus-brazil/kernels) and, through questions and high-level answers, collaborate in the analysis of the situation. The code produced attests how the answers were obtained, open for queries, reviews, suggestions for improvements, or new investigations through the discussions panel.
# 
# **Disclaimer/Legal warning**: The information here contained should serve as first guidance. The general information here contained does not provide any guarantees, however. Therefore, it is excluded the guarantee or responsibility of any kind, such as precision, reliability, completeness, and currentness of the information.
# 
# 
# **Elloá B. Guedes**  
# ebgcosta@uea.edu.br  
# www.elloaguedes.com  
# 
# **Júlio B. Guedes**  
# julio.costa@ccc.ufcg.edu.br  
# [COVID-19 Timeline](https://juliobguedes.codes/covid)
# 
# ---
# 
# ## 🇧🇷 Confira nosso outro projeto de análise do COVID-19 em âmbito global
# 
# Siga o link: https://juliobguedes.codes/covid
# 
# ## 🇺🇸 Check out our project on COVID-19 worldwide timeline
# 
# Click here: https://juliobguedes.codes/covid

# # 🇧🇷 Confira minha participação no PyData Manaus
# 
# <iframe width="560" height="315" src="https://www.youtube.com/embed/VXDU3nzFTTw?controls=0&amp;start=334" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
# 
# 

# In[ ]:


#Última execução
import datetime
print(datetime.datetime.now())
today = datetime.datetime.now().strftime('%d/%m/%Y')


# In[ ]:


# imports
import numpy as np
import pandas as pd
import os
import numpy as np

# bokeh packages
from bokeh.io import output_file,show,output_notebook,push_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper
from bokeh.layouts import row,column,gridplot
from bokeh.models.widgets import Tabs,Panel
from bokeh.models import GeoJSONDataSource
output_notebook()

# plotly packages
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs import *

import json
import geopandas as gpd
import plotly.graph_objects as go
import unidecode


# In[ ]:


data = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19.csv')
data.head()


# In[ ]:


data.tail()


# # Data Analytics
# 
# - 🇧🇷 Análise de dados sobre os casos, óbitos, proporções, distribuição geográfica, etc.
# - Base de dados atualizada diariamente e oriunda daqui: https://www.kaggle.com/unanimad/corona-virus-brazil
# 
# ---
# 
# - 🇺🇸 Data analysis over the confirmed cases, deaths, proportions, geographic distribuition, etc
# - Database daily updated acquired here: https://www.kaggle.com/unanimad/corona-virus-brazil

# ### 🇧🇷 Pergunta: A qual período de tempo os dados se referem?
# 
# **Resposta**:  
# 
# ### 🇺🇸 To which period of time the data refers to?
# 
# **Answer**:  

# In[ ]:


print(min(data['date']))
print(max(data['date']))


# ### 🇧🇷 Pergunta: Qual a incidência diária de casos suspeitos, confirmados e mortes no período?
# 
# **Resposta**: Os gráficos a seguir mostram esta informação para casos suspeitos, confirmados e óbitos. Note que a escala de cada gráfico é diferente, mas que a ordem de crescimento em todos segue de maneira ascendente.
# 
# Em 20/03/2020, o Ministério da Saúde passa a declarar estado de transmissão comunitária do COVID-19 no País e, com isso, casos suspeitos deixam de ser contabilizados. Vide: [Estadão, 20/03/2020, 19h27min](https://saude.estadao.com.br/noticias/geral,ministerio-da-saude-declara-estado-de-transmissao-comunitaria-de-coronavirus-em-todo-o-pais,70003242077)
# 
# ### 🇺🇸 Question: What is the daily incidence of suspected, confirmed and death cases in this period?
# 
# **Asnwer**: The plots below show this information for suspected, confirmed and death cases. Note that the scale of each plot is different, but the growth rate is ascending in all of them.
# 
# In March 20th, 2020 the Brazilian Health Ministry declared the state of comunitary transmission of COVID-19 and suspected cases stop being accounted for. If you want to know more, please see the news at [Estadao, March 20th, 2020, 7:27PM](https://saude.estadao.com.br/noticias/geral,ministerio-da-saude-declara-estado-de-transmissao-comunitaria-de-coronavirus-em-todo-o-pais,70003242077).

# In[ ]:


## Síntese diária
df2 = data.groupby(['date'])['cases','deaths'].agg('sum')
df2.head()


# In[ ]:


### Atualizando com antiga versão do dataset
old = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19_old.csv')
old = old.groupby(['date'])['suspects'].agg('sum')

layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis = dict(
        tickmode = 'array',
        tickvals = old.index,
        ticktext = old.index
    ),
    xaxis_title="Data",
    yaxis_title = "Quantidade"
)
suspeitos = old.loc[:'2020-03-21']
fig = px.bar(title='Casos suspeitos -- Descontinuado a partir de 21/03/2020', x=suspeitos.index, y=suspeitos)
fig['layout'].update(layout)
fig.show()


# In[ ]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs import *

layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)

fig = make_subplots(rows=2, cols=1,subplot_titles=('Casos Confirmados até ' + today, 'Óbitos até '+ today))
fig.append_trace(go.Bar(name='Confirmados', x=df2.index, y=df2['cases']), row=1, col=1)
fig.append_trace(go.Bar(name='Óbitos', x=df2.index, y=df2['deaths']), row=2, col=1)

fig.update_xaxes(title_text="Data", row=1, col=1)
fig.update_yaxes(title_text="Quantidade", row=1, col=1)
fig.update_xaxes(title_text="Data", row=2, col=1)
fig.update_yaxes(title_text="Quantidade", row=2, col=1)

fig['layout'].update(layout)

fig.show()


# In[ ]:


import plotly.graph_objects as go

layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title="Visualização Conjunta de Casos e Óbitos até " + today,
)

fig = go.Figure(data=[
    go.Bar(name='Confirmados', x=df2.index, y=df2['cases']),
    go.Bar(name='Óbitos', x=df2.index, y=df2['deaths'])
])
fig.update_xaxes(title_text='Data')
fig.update_yaxes(title_text='Quantidade')
fig.update_layout(barmode='stack')
fig['layout'].update(layout)

fig.show()


# ### 🇧🇷 Pergunta: Qual a distribuição geográfica dos casos confirmados?
# ### 🇺🇸 Question: What is the confirmed cases geographic distribution?

# In[ ]:


# utils
def remove_accents(a):
    unaccented_string = unidecode.unidecode(a)
    return unaccented_string


# In[ ]:


#data.drop('hour',axis= 1, inplace=True)
atual = max(data['date'])
df3 = data.loc[data['date'] == max(data['date'])].groupby(['state'])['cases','deaths'].agg('sum')
df4 = pd.DataFrame({"name": df3.index, 'cases': df3['cases'], 'deaths':df3['deaths']})
df4.index = range(0,27)

brazil = gpd.read_file('/kaggle/input/brazil-states-geojson/brazil.geojson')

df4['name'] = df4['name'].apply(remove_accents)
df4 = df4.sort_values('name')
brazil['name'] = brazil['name'].apply(remove_accents)
brazil = brazil.sort_values('name')

pop_states = brazil.merge(df4, left_on = 'name', right_on = 'name')
geosource = GeoJSONDataSource(geojson = pop_states.to_json())
merged_json = json.loads(pop_states.to_json())
json_data = json.dumps(merged_json)
geosource = GeoJSONDataSource(geojson = json_data)


# In[ ]:


from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
from bokeh.palettes import brewer
from bokeh.palettes import magma,viridis,cividis
from bokeh.layouts import row

def myplot3(geosource,tema, complemento = '',jump = 1,high = 100):
    
    tipo = 'Óbitos'
    palette = magma(256)
    if tema.startswith('case'):
        tipo = 'Casos'
        palette = viridis(256)[:248]
    elif tema.startswith('letalidade'):
        tipo = 'Letalidade'
        palette = cividis(256)[:248]
    elif tema.startswith('leitospor100mil'):
        tipo = 'Leitos de UTI por 100 mil habitantes'
        palette = magma(256)
    elif tema.startswith('leitos'):
        tipo = 'Leitos de UTI'
        palette = viridis(256)[:248]
    elif tema.startswith('testesRapidos'):
        tipo = 'Testes Rápidos'
        palette = viridis(256)[:248]
    elif tema.startswith('testesRTPCR'):
        tipo = 'Testes RT-PCR'
        palette = magma(256)
        
        
    palette = palette[::-1]
    color_mapper = LinearColorMapper(palette = palette, low = 0, high = high)

    #Define custom tick labels for color bar.
    if (not tema.startswith('letalidade')):
        d = {}
        for i in range(0,int(high),jump):
            d[str(i)] = str(i)

            
        d[str(int(high) + 1)] = '>' + str(int(high) + 1)
                
        hover = HoverTool(tooltips = [ ('Estado','@name'),('Quantidade', '@{'+tema+'}{%d}')], formatters={'@{'+ tema +'}' : 'printf'})
    elif (tema.startswith('leitos') or tema.startswith('teste')):
        d = {}
        for i in np.arange(0, high+1, jump):
            d[str(round(i,2))] = str(round(i,2))
        d[str(high + 1)] = '>'+ str(high + 1)
        hover = HoverTool(tooltips = [ ('Estado','@name'),('Quantidade', '@{'+tema+'}{%d}')], formatters={'@{'+ tema +'}' : 'printf'})
    else:
        d = {}
        for i in np.arange(0, high+0.5, jump):
            d[str(round(i,2))] = str(round(i,2))
        d[str(round(high + 0.5,2))] = '>'+ str(round(high + 0.5,2))
        hover = HoverTool(tooltips = [ ('Estado','@name'),('Taxa', '@{'+tema+'}{%.2f%%}')], formatters={'@{'+ tema +'}' : 'printf'})
    
    
    tick_labels = d
    #Create color bar. 
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 300, height = 20,
    border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)



    #Create figure object.
    p = figure(title = tipo + complemento + ' em {0}'.format((datetime.datetime.now()).strftime('%d/%m/%Y')), plot_height = 430 , plot_width = 330, toolbar_location = None, tools =[hover])
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False


    p.patches('xs','ys', source = geosource,fill_color = {'field' :str(tema), 'transform' : color_mapper},
              line_color = 'black', line_width = 0.25, fill_alpha = 1)

    p.add_layout(color_bar, 'below')
    return p


# In[ ]:


show(row(myplot3(geosource = geosource,tema = 'cases',jump = 2000, high = max(df4['cases'])),
         myplot3(geosource = geosource,tema = 'deaths', jump = 1000, high = max(df4['deaths']))))


# ### 🇧🇷 Pergunta: Qual a incidência de casos e óbitos por Estado, a cada 100 mil habitantes?
# ### 🇺🇸 Question: What is the incidence of confirmed cases and deaths by State, every 100 thousand habitants?

# In[ ]:


populacao = pd.read_csv('/kaggle/input/dadosbrasil/populacao.csv',sep=";")
populacao['name'] = populacao['name'].apply(remove_accents)
populacao = populacao.sort_values('name')
populacao = populacao.merge(df4, left_on = 'name', right_on = 'name')
populacao['casespor100mil'] = (populacao['cases']/populacao['populacao'])*100000
populacao['deathspor100mil'] = (populacao['deaths']/populacao['populacao'])*100000
populacao['leitospor100mil'] = (populacao['leitos']/populacao['populacao'])*100000
populacao['letalidade'] = round(populacao['deaths']/populacao['cases'],3)*100

# Abertura do mapa
brazil = gpd.read_file('/kaggle/input/brazil-states-geojson/brazil.geojson')
## mesclagem das bases
brazil['name'] = brazil['name'].apply(remove_accents)
brazil = brazil.sort_values('name')
pop_states = brazil.merge(populacao, left_on = 'name', right_on = 'name')
# Input GeoJSON source that contains features for plotting
geosource = GeoJSONDataSource(geojson = pop_states.to_json())

import json

#Read data to json.
merged_json = json.loads(pop_states.to_json())
json_data = json.dumps(merged_json)


# In[ ]:


show(row(myplot3(geosource = geosource,tema = 'casespor100mil',jump = 10, high = max(populacao['casespor100mil']), complemento = ' por 100 mil habitantes '),
         myplot3(geosource = geosource,tema = 'deathspor100mil', jump = 2, high = max(populacao['deathspor100mil']), complemento = ' por 100 mil habitantes ')))


# ## 🇧🇷 Análise da Letalidade/ 🇺🇸 Lethality Analysis

# ### 🇧🇷 Pergunta: Qual a taxa de letalidade do Coronavírus no Brasil?
# 
# **Resposta**:
# 
# - Qual a taxa de letalidade no Brasil, obtida com os dados mais recentes disponíveis?
# - O cálculo da taxa de letalidade é dado por:
# 
# $$
# \textrm{Taxa de letalidade} =  \frac{\sum \textrm{óbitos}}{\sum \textrm{casos}}
# $$
# 
# ### 🇺🇸 Question: What is the lethality rate of Coronavirus in Brazil?
# 
# **Answer**:  
# - What is the lethality rate in Brazil, calculated with the most recent data?
# - The lethality rate is calculated by:
# 
# $$
# \textrm{Lethality rate} = \frac{\sum \textrm{deaths}}{\sum \textrm{cases}}
# $$
# 

# In[ ]:


letalidade = sum(df4['deaths'])/sum(df4['cases'])
print("Taxa de Letalidade em " + today + ": {0:6.3f}%".format(letalidade*100))


# ### 🇧🇷 Pergunta: Como foi a modificação da letalidade ao longo do tempo em âmbito nacional?
# 
# **Método**: Calcular a taxa de letalidade dia a dia  
# **Considerações**: Observa-se que a taxa de letalidade vem crescendo vertiginosamente!
# 
# ### 🇺🇸 Question: How the lethality changed with time, in national scope?
# 
# **Method**: Calculate the lethality rate each day.  
# **Considerations**: Its observed that the lethality rate grows vertiginously!  

# In[ ]:


df2['letalidade'] = df2['deaths']/df2['cases']
df2.fillna(0,inplace=True)
df2 = df2.reset_index()


# In[ ]:


layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title= "Letalidade ao Longo do Tempo",
    xaxis_title="Data",
    yaxis_title="Taxa de Letalidade",
    yaxis_tickformat = '.2%')

fig = go.Figure(data=[
    go.Scatter(x=df2['date'], y=df2['letalidade'])])
fig['layout'].update(layout)

fig.show()


# ### 🇧🇷 Pergunta: Qual a taxa de letalidade por Estado?
# ### 🇺🇸 Question: What is the lethality rate by State?
# 

# In[ ]:


show(myplot3(geosource = geosource,tema = 'letalidade', jump = 0.5, high = max(populacao['letalidade']), complemento = ''))


# # 🇧🇷 Relação com Indicadores Sócio-Economicos / 🇺🇸 Relation with Socio-Economic Indicators
# 
# - 🇧🇷 As análises a seguir contemplam aspectos de indicadores sócio-econômicos e recursos disponíveis para o combate ao COVID-19 em cada estado
# 
# - 🇺🇸  The following analisis contemplate socio-economic indicators and resources available to fight COVID-19 in each state
# 

# ### 🇧🇷 Pergunta: Quantos leitos de UTI do SUS há por Estado?
# 
# - Os dados obtidos neste sentido são de 2018 e não contemplam atualizações recentes, decorrente dos hospitais de campanha que estão sendo construídos, por exemplo.
# - Este dado auxilia a estimar os recursos para enfrentamento dos casos mais graves.
# - Dados de UTI de 2018 extraídos do site do Conselho Federal de Medicina. Conferir nos links úteis.  
# - Os leitos de UTI somam todos os tipos, desde adulto nos graus I a III, infantil I a III, neonatal I a III, queimados, coronariana e outros.
# 
# ### 🇺🇸 Question: How many ICU beds each state have?
# 
# - The data obtained in this matter are from 2018 and have no recent updates, from field hospitals being built, for instance.
# - This data helps estimating the resources to face the most serious cases.
# - The ICU data of 2018 were extracted from the Federal Council of Medicine. Check the useful links.
# - The ICU beds number is the sum of every type of ICU: Adult I to III, Infant I to III, neonatal I to III, burns, coronary, and others.

# In[ ]:


show(row(myplot3(geosource = geosource,tema = 'leitos', jump = 1, high = max(populacao['leitos']), complemento = ''),myplot3(geosource = geosource,tema = 'leitospor100mil', jump = 1, high = max(populacao['leitospor100mil']), complemento = '')))


# ### 🇧🇷 Pergunta: Os Estados com maior PIB também mais investem mais em leitos de UTI no SUS?
# 
# Vamos analisar esta pergunta por meio da correlação do PIB com o número de leitos existentes. Se a correlação for positiva e forte, há uma boa evidência de que isto, de fato, ocorra, pois viria a demonstrar um retorno para população por meio de um bom serviço de saúde.
# 
# A correlação resultou em 0.97, indicando fortes evidências a favor desta hipótese.
# 
# ### 🇺🇸 Question: The States with the highest GDP also invest more in ICU beds in Public Health?
# 
# **Clarification**: SUS is an abbreviation of Sistema Único de Saúde, which stands for Unique Health System. SUS is the primary public health care in Brazil.
# 
# Let us analyze this question through the correlation between GDP and the number of ICU beds. If it is a positive strong correlation, there is good evidence that this, in fact, occurs, since it demonstrates a return to the population by a good health system.

# In[ ]:


populacao['pib'] = [int(x.replace('.','')) for x in populacao['pib']]


# In[ ]:


populacao['pib'].corr(populacao['leitos'])


# In[ ]:


populacao['populacao'].corr(populacao['leitos'])


# ### 🇧🇷 Pergunta: O número de leitos de UTI tem relação com o PIB per capita?
# 
# - Seguindo a mesma estratégia anterior, apenas checando os fatores de desigualdade social.
# - Essa questão foi brilhantemente sugerida por [Claudio de Pizzo](http://https://www.kaggle.com/claudiodipizzo)  
# - **Conclusão**: Tal correlação parece mesmo fraca. Intrigante!
# 
# ### 🇺🇸 Question: Is the number of UCI units related to PIB per capita?
# 
# - Following the same strategy as before, just checking the factors of social inequality.
# - This question was  brilliantly suggested by [Claudio de Pizzo](http://https://www.kaggle.com/claudiodipizzo)
# - **Conclusion**: Such correlation seems to be weak. Intriguing!

# In[ ]:


newData = pd.read_csv('/kaggle/input/testdata/testData.csv')
newData['name'] = newData['name'].apply(remove_accents)
newData = newData.merge(populacao, left_on = 'name', right_on = 'name')
newData['pibpercapita'].corr(newData['leitos'])


# In[ ]:


layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title= "PIB per capita versus Leitos de UTI",
    xaxis_title="PIB per capita",
    yaxis_title="Leitos de UTI")

fig = go.Figure(data=[
    go.Scatter(x=newData['pibpercapita'], y=newData['leitos'],mode='markers')])
fig['layout'].update(layout)

fig.show()


# ### 🇧🇷 Pergunta: E quanto ao IDH-M?
# 
# O IDH-M data de 2017.
# 
# A correlação encontra-se na categoria moderada (de 0,5 a 0,7), mas próximo ao limiar inferior, sugindo fraqueza.  
# Assim, vê-se que a prepoderância de outros aspectos da qualidade de vida aferidos no IDH-M sejam mais relevantes que os leitos de UTI. De fato, quando examina-se a composição do IDH-M, esta hipótese é corroborada. Ver links úteis no final.
# 
# ### 🇺🇸 Question: How about the cities' GDP?
# 
# The data for this analysis is from 2017.
# 
# The correlation is moderate (from 0.5 to 0.7), but closer to the lowest boundary, suggesting a week correlation.  
# Therefore, it is possible to see the dominance of other aspects of life quality that are more relevant in the cities' GDP than the number of ICU beds. In fact, when analyzing the composing of the cities' GDP, this hypothesis is corroborated. Check the useful links at the end.

# In[ ]:


populacao['idhm'] = [float(x.replace(',','.')) for x in populacao['idhm']]
populacao['idhm'].corr(populacao['leitos'])


# ### 🇧🇷 A taxa de letalidade pelo COVID-19 em cada Estado é correlacionada com o PIB per capita?
# 
# - Se o PIB total é alto, mas o PIB per capita é baixo, há grande desigualdade na população
# - Nesses casos, ao examinar o PIB per capita, temos uma visão melhor das condições de vida da população e como isso impacta na saúde e qualidade de vida
# - Será que este indicador socio-econômico se correlaciona com a letalidade que estamos vendo agora em cada Estado?
# - Essa questão foi brilhantemente sugerida por [Claudio de Pizzo](http://https://www.kaggle.com/claudiodipizzo)
# - **Conclusão**: Existe uma fraca correlação negativa, isto é, quanto maior o PIB per capita, menor é a letalidade, mas não há tanta força nessa relação
# 
# ### 🇺🇸 Does the letality rate due to COVID-19 is correlated with GDP per capita among states?
# 
# - If the total GDP is high, but the GDP per capita is low, there is great inequality in the population
# - In these cases, when examining the GDP per capita, we have a better view of the living conditions of the population and how it impacts on health and quality of life.
# - Does this socio-economic indicator correlate with the lethality that we are seeing now in each state?
# - This question was  brilliantly suggested by [Claudio de Pizzo](http://https://www.kaggle.com/claudiodipizzo)
# - **Conclusion**: There is a weak negative correlation, i.e, as higher the GDP per capita, the lower the letality, but there is very few strenght in this relation

# In[ ]:


newData = pd.read_csv('/kaggle/input/testdata/testData.csv')
newData['name'] = newData['name'].apply(remove_accents)
newData = newData.merge(populacao, left_on = 'name', right_on = 'name')
newData['pibpercapita'].corr(newData['letalidade'])


# ### 🇧🇷 A taxa de letalidade pelo COVID-19 em cada Estado é correlacionada com o número de leitos de UTI disponíveis?
# 
# - Se há um bom número de leitos, é provável que a população seja bem amparada, diminuindo a perda de vidas.
# - **Conclusão**: Existe uma fraca correlação positiva, o que é um pouco contraditório. Provavelmente, na prática, estas duas variáveis não estão correlacionadas.
# 
# ### 🇺🇸 Does the letality rate due to COVID-19 is correlated with the number of ICU beds among states?
# 
# - If there is plenty ICU beds, it is likely that the population is having good health support, which might decrease deaths
# - **Conclusion**: There is a weak positive correlation, which is somewhat contradictory. In practice, these two variables may not be correlated at all.

# In[ ]:


newData['letalidade'].corr(newData['leitos'])


# ### 🇧🇷 O número de casos de COVID-19 em cada Estado é correlacionado com o PIB per capita?
# 
# - Será que o PIB per capita alto favorece a diminuição da contaminação? Pode haver uma indicação de que acesso a recursos materiais favoreça acesso à saneamento, educação e outros elementos que podem ser estratégicos para uma menor exposição ao vírus?
# - Essa questão foi brilhantemente sugerida por [Claudio de Pizzo](http://https://www.kaggle.com/claudiodipizzo)
# - **Conclusão**: Existe uma fraca correlação positiva, não é possível supor tal hipótese.
# 
# ### 🇺🇸 Is the number of COVID-19 cases per State is correlated with GDP per capita?
# 
# - Does the high GDP per capita favor the reduction of contamination? Could there be an indication that access to material resources favors access to sanitation, education and other elements that may be strategic for less exposure to the virus?
# - This question was  brilliantly suggested by [Claudio de Pizzo](http://https://www.kaggle.com/claudiodipizzo)
# - **Conclusion**: There is a weak positive correlation, therefore it is not possible to suppose such hypothesis.

# In[ ]:


newData = pd.read_csv('/kaggle/input/testdata/testData.csv')
newData['name'] = newData['name'].apply(remove_accents)
newData = newData.merge(df4, left_on = 'name', right_on = 'name')
newData['cases'].corr(newData['pibpercapita'])


# # 🇧🇷 Estratégia de distribuição de testes / 🇺🇸 Test distribution strategy
# 
# - 🇧🇷 [De acordo com o Ministério da Saúde](http://https://saude.gov.br/noticias/agencia-saude/46632-comeca-hoje-a-distribuicao-de-500-mil-testes-rapidos-para-todo-o-pais), em 01/04/2020 houve a distribuição de 500 mil testes para a população, em todos os estados
# - Os testes foram de dois tipos:  
#     1. *Testes rápidos*: Com resultados obtidos em cerca de 20min, são indicados apenas para os profissionais dos serviços de saúde e da segurança. Devem ser feitos após o sétimo dia do início dos sintomas e detectam a presença de anticorpos contra o vírus SARS-CoV-2;
#     2. *TESTES RT-PCR*: Baseados em Biologia Molecular, eles identificam o COVID-19 em seus estágios iniciais. Tais testes são usados para casos graves internados.
# - De acordo com o governo, há mais testes a caminho.  
#   
# 
# - 🇺🇸 [According to the Ministry of Health](http://https://saude.gov.br/noticias/agencia-saude/46632-comeca-hoje-a-distribuicao-de-500-mil-testes-rapidos-para-todo-o-pais), on April 1st,2020, 500 thousand tests were distributed to the population, in all states
# - The tests were of two types:
#      1. *Rapid tests*: With results obtained in about 20 minutes, they are indicated only for health and safety professionals. They must be done after the seventh day of the onset of symptoms and detect the presence of antibodies against the SARS-CoV-2 virus;
#      2. *RT-PCR TESTS*: Based on Molecular Biology, they identify COVID-19 in its early stages. Such tests are used for severe hospitalized cases.

# ### 🇧🇷 Os testes foram distribuídos para os Estados com maior número de casos? 
# 
# - Vamos considerar o número de casos por estado em 31/03/2020
# - **Conclusão**: No caso dos testes rápidos, há fortes evidências que o número de casos foi determinante para a estratégia de distribuição dos mesmos. No caso dos testes RT-PRC, não há igual força de evidência na afirmação.
# 
# ### 🇺🇸 Have the tests been distributed to the states with the highest number of cases?
# 
# - Let's consider the number of cases per state in 03/31/2020
# - **Conclusion**: In the case of rapid tests, there is strong evidence that the number of cases was decisive in  distribution strategy. In the case of RT-PRC tests, there is no equal strength of evidence in the statement.

# In[ ]:


## Cases on March 31/2020 per state
data = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19.csv')
df3 = data.loc[data['date'] == '2020-03-31'].groupby(['state'])['cases','deaths'].agg('sum')
df3 = df3.reset_index()
df3['name'] = df3['state'].apply(remove_accents)
df3.drop(['state'],axis=1,inplace=True)
newData = pd.read_csv('/kaggle/input/testdata/testData.csv')
newData['name'] = newData['name'].apply(remove_accents)
newData = newData.merge(df3, left_on = 'name', right_on = 'name')
newData['cases'].corr(newData['testesRapidos'])


# In[ ]:


newData['cases'].corr(newData['testesRTPCR'])


# ### 🇧🇷 Os testes foram distribuídos para os Estados com maior número de óbitos? 
# 
# - Considerando dados por estado coletados em 31/03/2020
# - **Conclusão**: As mesmas conclusões anteriores se aplicam.
# 
# ### 🇺🇸 Have the tests been distributed to the states with the highest number of deaths?
# 
# - Considering data from state available in 03/31/2020
# - **Conclusion**: The same conclusions as above apply.

# In[ ]:


newData['deaths'].corr(newData['testesRapidos'])


# In[ ]:


newData['deaths'].corr(newData['testesRTPCR'])


# ### 🇧🇷 Os testes foram distribuídos para os Estados com maior PIB? 
# 
# - Considerando dados por estado coletados em 31/03/2020
# 
# ### 🇺🇸 Have the tests been distributed to the states with the highest GDP?
# 
# - Considering data from state available in 03/31/2020
# 

# In[ ]:


subset = populacao[['name','pib']]
newData = newData.merge(subset, left_on = 'name', right_on = 'name')
newData['testesRapidos'].corr(newData['pib'])


# In[ ]:


newData['testesRTPCR'].corr(newData['pib'])


# - 🇧🇷 Antes de concluir, resta a pergunta: os estados com maior PIB tem maior número de casos e óbitos?
# - 🇺🇸 Before concluding, the question remains: do the states with the highest GDP have a higher number of cases and deaths?

# In[ ]:


newData['pib'].corr(newData['cases'])


# In[ ]:


newData['pib'].corr(newData['deaths'])


# - 🇧🇷 **Conclusão**: Estados com maior PIB receberam maior quantidade de testes porque também registraram maior número de casos e óbitos. A distribuição de testes na ocasião parece ter considerado o panorama disponível de maneira consistente e estratégica.
# - 🇺🇸 **Conclusion**: Higher GDP states received more tests because they also recorded a higher number of cases and deaths. The distribution of tests at the time seems to have considered the panorama available in a consistent and strategic way.

# ### 🇧🇷 Os testes foram distribuídos para os Estados com número de casos e óbitos per capita?
# 
# - Considerando dados por estado coletados em 31/03/2020
# 
# ### 🇺🇸 Have the tests been distributed to the states with higher cases and deaths per capita?
# 
# - Considering data from state available in 03/31/2020
# 

# In[ ]:


populacao = pd.read_csv('/kaggle/input/dadosbrasil/populacao.csv',sep=";")
populacao['name'] = populacao['name'].apply(remove_accents)
populacao = populacao.sort_values('name')
newData = newData.merge(populacao,left_on='name', right_on='name')


# In[ ]:


newData['casescapita'] = (newData['cases']/newData['populacao'])
newData['deathscapita'] = (newData['deaths']/newData['populacao'])


# In[ ]:


newData['testesRapidos'].corr(newData['casescapita'])


# In[ ]:


newData['testesRapidos'].corr(newData['deathscapita'])


# In[ ]:


newData['testesRTPCR'].corr(newData['casescapita'])


# In[ ]:


newData['testesRTPCR'].corr(newData['deathscapita'])


# - 🇧🇷 **Conclusão**: Na minha opinião, a distribuição de testes poderia ter sido ainda melhor, considerando a proporção de doentes e óbitos nos estados, não o quantitativo geral.
# - 🇺🇸 **Conclusion**: In my opinion, the distribution of tests could have been even better, considering the proportion of patients and deaths in the states, not the general number.

# ### 🇧🇷 Como foi a distribuição geográfica dos testes disponibilizados em 01/04/2020?
# ### 🇺🇸 How was the geographic distribution of the tests made available on April 1st, 2020?

# In[ ]:


# Abertura do mapa
brazil = gpd.read_file('/kaggle/input/brazil-states-geojson/brazil.geojson')
brazil['name'] = brazil['name'].apply(remove_accents)
brazil = brazil.sort_values('name')
## mesclagem das bases
pop_states = brazil.merge(newData, left_on = 'name', right_on = 'name')
# Input GeoJSON source that contains features for plotting
geosource = GeoJSONDataSource(geojson = pop_states.to_json())


# In[ ]:


print("Dados de 31/03/2020, ignorar cabeçalho -- Data from 03/31/2020, ignore header")
show(row(myplot3(geosource = geosource,tema = 'testesRapidos', jump = 100, high = max(newData['testesRapidos']), complemento = ''),myplot3(geosource = geosource,tema = 'testesRTPCR', jump = 100, high = max(newData['testesRTPCR']), complemento = '')))


# # 🇧🇷 Examinando a Série Temporal de Casos e Prevendo o Número de Casos / 🇺🇸 Examinating the Time Series of Cases and Predicting the Number of Cases
# 
# 🇧🇷 Vamos ignorar agora a distribuição geográfica e considerar apenas o quantitativo de casos.
# 
# 🇺🇸 Let's ignore for now the geographic distribution and consider only the quantitative aspect of cases.
# 

# In[ ]:


import plotly.graph_objects as go
df2 = data.groupby(['date'])['cases','deaths'].agg('sum')
df2 = df2.reset_index()

layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title= "Série temporal de Casos",
    xaxis_title="Data",
    yaxis_title="Quantidade",
)

fig = go.Figure(data=[
    go.Scatter(x=df2['date'], y=df2['cases'])
    
])
fig['layout'].update(layout)

fig.show()


# ### 🇧🇷 Pergunta: Em que ordem o número de casos está crescendo a cada dois dias?
# 
# - Vamos começar a análise pelo primeiro dia em que houve casos confirmados
# - Iremos representar as linhas que denotam diferentes ordens crescimento de casos
# - Começaremos a análise a partir do 27o. dia, que é onde foi registrado o primeiro caso
# 
# ### 🇺🇸 Question: How much is the number of cases rising every two days?
# 
# - Let's start our analysis from the day where the first case happened
# - We will denote lines that illustrate distinct growth orders
# - Our analysis will start from day 27, where the first case was reported

# In[ ]:


dfy = df2.copy()
dfy.drop(['date','deaths'],axis=1,inplace = True)
dfy = dfy.reset_index()
dfy['dias'] = dfy['index']

# Cases double by rate every 2 days
def casesDouble(rate, doubleDays):
    supposedCases = [1]
    for i in range(len(doubleDays)-1):
        supposedCases.append(rate*supposedCases[-1])
    return supposedCases

doubleDays = list(range(0,max(dfy['dias']),2))
dfSupposed = pd.DataFrame({'dias':doubleDays,'2x':casesDouble(2,doubleDays),'3x':casesDouble(3,doubleDays),'1.5x':casesDouble(1.5,doubleDays)})


# In[ ]:


layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title= "Suposição do crescimento de casos a cada 2 dias (Escala logarítmica)",
    xaxis_title="Dias desde o primeiro caso",
    yaxis_title="Quantidade (escala log)",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 3
    ),
    yaxis_type="log"
)

fig = go.Figure(data=[
    go.Scatter(x=dfy['dias'], y=dfy['cases'], name='Casos Reais',mode="lines+markers"),
    go.Scatter(x=dfSupposed['dias'], y=dfSupposed['2x'], name = '2x',mode="lines+markers"),
    go.Scatter(x=dfSupposed['dias'], y=dfSupposed['3x'], name = '3x',mode="lines+markers"),
    go.Scatter(x=dfSupposed['dias'], y=dfSupposed['1.5x'], name = '1.5x',mode="lines+markers")
])

fig['layout'].update(layout)

fig.show()


# 🇧🇷 **Conclusão**: O número de casos parecem crescer 1,5 vezes a cada dois dias por muito tempo dentre o período observado.
# 🇺🇸 **Conclusion**: The number of cases seems to grow 1.5 times every two days during the most part of days observed.

# ### 🇧🇷 E a cada 3 dias?
# ### 🇺🇸 And every 3 days?

# In[ ]:


dfy = df2.copy()
dfy.drop(['date','deaths'],axis=1,inplace = True)
dfy = dfy.reset_index()
dfy['dias'] = dfy['index']

# Cases double by rate every 3 days
def casesDouble(rate, doubleDays):
    supposedCases = [1]
    for i in range(len(doubleDays)-1):
        supposedCases.append(rate*supposedCases[-1])
    return supposedCases

doubleDays = list(range(0,max(dfy['dias']),3))
dfSupposed = pd.DataFrame({'dias':doubleDays,'2x':casesDouble(2,doubleDays),'3x':casesDouble(3,doubleDays),'1.5x':casesDouble(1.5,doubleDays)})


# In[ ]:


layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title= "Suposição do crescimento de casos a cada 3 dias (Escala logarítmica)",
    xaxis_title="Dias desde o primeiro caso",
    yaxis_title="Quantidade (escala log)",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 3
    ),
    yaxis_type="log"
)

fig = go.Figure(data=[
    go.Scatter(x=dfy['dias'], y=dfy['cases'], name='Casos Reais',mode="lines+markers"),
    go.Scatter(x=dfSupposed['dias'], y=dfSupposed['2x'], name = '2x',mode="lines+markers"),
    go.Scatter(x=dfSupposed['dias'], y=dfSupposed['3x'], name = '3x',mode="lines+markers"),
    go.Scatter(x=dfSupposed['dias'], y=dfSupposed['1.5x'], name = '1.5x',mode="lines+markers")
])

fig['layout'].update(layout)

fig.show()


# 🇧🇷 **Conclusão**: Por muito tempo, o número de casos pareceu mais que dobrar a cada três dias!  
# 🇺🇸 **Conclusion**: For a long time the number of cases seemed to more than double every three days!

# ### 🇧🇷 Pergunta: Como se comporta um estimador bastante simples, baseado em regressão linear, para prever o número de casos?
# 
# 
# * **Holdout**: Treinar com 90% dos dados (90% dos primeiros dias) e testar nos 10% restantes
# * **Avaliação de performance**: Raiz do erro médio quadrático e $R^2$ Score
# * **Precauções**: Não há 'lookahead'
# * **Conclusão**: Como esperado, não há como uma regressão linear capturar bem o formato da tendência de crescimento. Portanto, o regressor subestima o número de casos em relação ao cenário real.
# 
# ### 🇺🇸 Question: How a simple estimator, based on linear regression, performs when predicting the number of cases?
# 
# * **Holdout**: Training with 90% of the data (90% first days) and test with the last 10%.
# * **Performance Evaluation**: RMSE (Root Mean Squared Error) and $R^2$ Score.
# * **Precautions**: There is no lookahead.
# * **Conclusion**: As expected, it is not possible for a linear regression to capture the format and tendency of growth. Therefore, the regressor underestimates the number of cases when compared to the real scenario.
# 

# In[ ]:


import plotly.graph_objects as go
import datetime
import numpy as np

df2['date'] = pd.to_datetime(df2['date'])
df2 = df2.loc[df2['date'] >= '02-26-2020']
df2['dias'] = range(1,len(df2) + 1,1)

## Treino
dias_train = df2['dias'][:int(0.9*len(df2))]
cases_train = df2['cases'][:int(0.9*len(df2))]

## Teste
dias_test = df2['dias'][int(0.9*len(df2)):]
cases_test =  df2['cases'][int(0.9*len(df2)):]

previsao = len(df2) - len(dias_test)

print("Holdout: Dados Totais: %d, Treino: %d dias, Teste: %d dias" % (len(df2),len(dias_train),len(dias_test)))


# In[ ]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(dias_train.values.reshape(-1,1), cases_train)
y_previsto = reg.predict(dias_test.values.reshape(-1,1))


# In[ ]:


layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title= "Estimador linear para o número de casos",
    xaxis_title="Dias desde a primeira notificação",
    yaxis_title="Quantidade de casos",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 3
    )
)

fig = go.Figure(data=[
    go.Scatter(x=dias_train, y=df2['cases'][:int(0.9*len(df2))], name='Dados de Treinamento',mode="lines+markers"),
    go.Scatter(x=dias_test, y=y_previsto, name = 'Casos Estimados',mode="lines+markers"),
    go.Scatter(x=dias_test, y=df2['cases'][int(0.9*len(df2)):], name = 'Casos Reais',mode="lines+markers")
])
fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0= previsao + 0.5,
            y0=120,
            x1=previsao + 0.5,
            y1=max(df2['cases']),
            line=dict(
                width=1.5,
                dash= "dash"
            )
))

fig.add_trace(go.Scatter(
    x=[previsao + 0.5],
    y=[2],
    text=["Início da previsão"],
    mode="text",
))
fig['layout'].update(layout)

fig.show()


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
print("Erro médio quadrático: ",mean_squared_error(cases_test,y_previsto))
print("R^2 Score: ", r2_score(cases_test,y_previsto))


# ### 🇧🇷 Pergunta: A tendência de crescimento dos casos é de ordem exponencial?
# 
# * Modificação: Quantidade de dias desde o primeiro caso versus casos no dia
# * Vamos fazer um scatterplot do logaritmo dos casos versus o número de dias.
# * Caso tenda a uma reta, há fortes evidências positivas para a pergunta em questão
# * **Conclusão**: Considerando os dados atuais, não estamos fortemente em ordem exponencial, embora a tendência ainda seja crescente. Há que se salientar que esta conclusão baseia-se apenas nos dados disponíveis no dataset e pode estar havendo subnotificação de casos em razão de dificuldades na ampla testagem.
# 
# ### 🇺🇸 Question: The cases growth tendency is exponencial?
# 
# * Modification: Number of days since the first case vs. Number of cases in the day.
# * Let's make a scatter plot of the logarithm of cases versus the number of days.
# * If it tends to a straight line, there are strong positive evidences to the standing question.
# * **Conclusion**: Considering the current version of the data, there is no strength in the exponential tendency, even that the tendency is growing. We need to point out that this conclusion is based only on the data available in the dataset, and sub notification might be occurring due to difficulties in extensive wide testing.

# In[ ]:


import plotly.graph_objects as go
import datetime
import numpy as np

df2['date'] = pd.to_datetime(df2['date'])
df2 = df2.loc[df2['date'] >= '02-26-2020']
df2['dias'] = range(1,len(df2) + 1,1)
log_y_data = np.log(df2['cases'])

layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title="Log Casos versus Dias"
)

fig = go.Figure(data=[go.Scatter(name='log Casos',x=df2['dias'], y=log_y_data, mode='markers'),
                     go.Scatter(name='Referência',x=df2['dias'], y=df2['dias'], line=dict(color='firebrick', width=0.5,
                              dash='dash'))])
fig['layout'].update(layout)

fig.show()


# ### 🇧🇷 Pergunta: Como se comporta uma estimador baseado em regressão exponencial para o número de casos?
# 
# * **Holdout**: Treinar com 90% dos dados (90% dos primeiros dias) e testar nos 10% restantes
# * **Avaliação de performance**: Raiz do erro médio quadrático e R^2 Score
# * **Precauções**: Não há 'lookahead'
# * **Conclusão**: Não é um bom estimador para o problema. O valor de R^2 negativo e alto revela que este estimador é pior que uma reta horizontal para o cenário
# 
# ### 🇺🇸 Question: How a exponencial estimator performs when predicting the number of cases?
# 
# * **Holdout**: Training with 90% of the data (90% first days) and test with the last 10%.
# * **Performance Evaluation**: RMSE (Root Mean Squared Error) and $R^2$ Score.
# * **Precautions**: There is no lookahead.
# * **Conclusion**: It is not a good estimator for the problem. The highly negative R^2 Score reveals that the estimator is worst than a horizontal line for the scenario. 
# 

# In[ ]:


import plotly.graph_objects as go
import datetime
import numpy as np

log_y_data = np.log(df2['cases'])

cases_train_log = log_y_data[:int(0.9*len(df2))]
cases_test_log = log_y_data[int(0.9*len(df2)):]

print("Holdout: Dados Totais: %d, Treino: %d dias, Teste: %d dias" % (len(df2),len(dias_train),len(dias_test)))


# In[ ]:


# Treino do modelo (interpolação da curva)
curve_fit = np.polyfit(dias_train, cases_train_log, 1)
y_train = (np.exp(curve_fit[1]) * np.exp(curve_fit[0]*dias_train)).astype(int)
y_estimado = (np.exp(curve_fit[1]) * np.exp(curve_fit[0]*dias_test)).astype(int)


# In[ ]:


layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title= "Estimador exponencial para o número de casos",
    xaxis_title="Dias desde a primeira notificação",
    yaxis_title="Quantidade de casos",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 3
    )
)

fig = go.Figure(data=[
    go.Scatter(x=dias_train, y=df2['cases'][:int(0.9*len(df2))], name='Dados de Treinamento',mode="lines+markers"),
    go.Scatter(x=dias_test, y=y_estimado, name = 'Casos Estimados',mode="lines+markers"),
    go.Scatter(x=dias_test, y=df2['cases'][int(0.9*len(df2)):], name = 'Casos Reais',mode="lines+markers")
])
fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0= previsao + 0.5,
            y0=120,
            x1=previsao + 0.5,
            y1=max(y_estimado),
            line=dict(
                width=1.5,
                dash= "dash"
            )
))

fig.add_trace(go.Scatter(
    x=[previsao + 0.5],
    y=[2],
    text=["Início da previsão"],
    mode="text",
))
fig['layout'].update(layout)

fig.show()


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
print("Erro médio quadrático: ",mean_squared_error(cases_test_log,y_estimado))
print("R^2 Score: ", r2_score(cases_test,y_estimado))


# ### 🇧🇷 Pergunta: Como se comporta uma estimador baseado em Rede Neural Artificial Multi-Layer Perceptron para um dia à frente?
# 
# * **Hipótese**: RNAs MLPs são aproximadoras universais de qualquer função  
# * **Metodologia**: Treinar com n-1 dias para prever o dia seguinte
# * **Avaliação de performance**: 
#     1. Erro médio absoluto que, para uma única amostra, se reduz a: $\left| x_i - \hat{x}_i \right|$
# * **Busca de parâmetros**: Foi feita de maneira ad-hoc em relação ao número de neurônios nas camadas ocultas e à função de ativação. Vários testes foram realizados. O otimizador escolhido leva em conta que há poucos dados disponíveis sobre o problema. O número de iterações até a convergência foi continuamente aumentado até atingir valores satisfatórios.
# * **Precauções**: Não há 'lookahead'
# * **Conclusão**: É um estimador excelente para o problema!!
# 
# ### 🇺🇸 Question: How an estimator based in an Artificial Neural Network Multi-Layer Perceptron performs predicting a day ahead?
# 
# * **Hypothesis**: Artificial Neural Networks (ANNs) Multi-layer Perceptrons (MLPs) can approximate any function.  
# * **Methodology**: Train with n-1 days and predict the last day.
# * **Performance Evaluation**: 
#     1. Mean Absolute Error that, for a single sample, reduces to $\left| x_i - \hat{x}_i \right|$
# * **Parameter Search**: It was done in an ad-hoc manner in relation to the number of neurons in each hidden layer and the activation function. Many tests have been carried out. The optimizer was chosen considering that there is little data available. The number of epochs until the convergence increases until reaching satisfactory values.
# * **Precautions**: There is no lookahead.
# * **Conclusion**: It is an excelent estimator for the problem!!
# 

# In[ ]:


## Treino
dias_train = df2['dias'][:-1]
cases_train = df2['cases'][:-1]

## Teste
dias_test = df2['dias'][-1:]
cases_test =  df2['cases'][-1:]

previsao = len(df2) - len(dias_test)

print("Novo Holdout: Dados Totais: %d, Treino: %d dias, Teste: %d dia" % (len(df2),len(dias_train),len(dias_test)))


# In[ ]:


from sklearn.neural_network import MLPRegressor
# Treino da rede neural
mlp = MLPRegressor(hidden_layer_sizes=(200,200),activation='relu',solver='lbfgs',max_iter=1000, shuffle=True)
mlp.fit(X=dias_train.values.reshape(-1,1),y=cases_train.values.ravel())


# In[ ]:


y_previsto = mlp.predict(dias_test.values.reshape(-1,1))


# In[ ]:


layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title= "Estimador baseado em RNA MLP para o número de casos",
    xaxis_title="Dias desde a primeira notificação",
    yaxis_title="Quantidade de casos",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 3
    )
)

fig = go.Figure(data=[
    go.Scatter(x=dias_train, y=df2['cases'][:-1], name='Dados de Treinamento',mode="lines+markers"),
    go.Scatter(x=dias_test, y=y_previsto, name = 'Casos Estimados',mode="lines+markers"),
    go.Scatter(x=dias_test, y=df2['cases'][-1:], name = 'Casos Reais',mode="lines+markers")
])
fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0= previsao + 0.5,
            y0=120,
            x1=previsao + 0.5,
            y1=max(df2['cases']) + 100,
            line=dict(
                width=1.5,
                dash= "dash"
            )
))

fig.add_trace(go.Scatter(
    x=[previsao - 0.5],
    y=[2],
    text=["Início da previsão"],
    mode="text",
))
fig['layout'].update(layout)

fig.show()


# In[ ]:


print("Erro Médio Absoluto: {0:6.3f} casos".format(mean_absolute_error(cases_test,y_previsto)))


# ### 🇧🇷 Pergunta: Como se comporta uma estimador baseado em Rede Neural Artificial Multi-Layer Perceptron para um dia à frente + ACF da série temporal?
# 
# * **Hipóteses**:
#     1. RNAs MLPs são aproximadoras universais de qualquer função  
#     2. Algumas séries temporais são auto-correlacionadas
# * **Metodologia**: 
#     1. Calcular o ACF da Série Temporal
#     2. Defasar a série temporal conforme o resultado anterior
#     3. Treinar a mesma arquitetura de RNA MLP anterior com n-1 dias para prever o dia seguinte
# * **Avaliação de performance**: 
#     1. Erro médio absoluto que, para uma única amostra, se reduz a: $\left| x_i - \hat{x}_i \right|$
# * **Precauções**: Não há 'lookahead'
# * **Conclusão**:
# 
# ### 🇺🇸 Question: How an ANN MLP estimator performs in the task of predicting a day ahead + Time Series ACF?
# 
# * **Hypothesis**:
#     1. ANN MLPs can approximate any function.
#     2. Some Time Series are autocorrelated.
# * **Methodology**: 
#     1. Calculate the autocorrelation function of the Time Series.
#     2. Lag the Time Series based on the previous result.
#     3. Train an MLP with the same architecture that we used in the last example using $n-1$ days and predict the $n-th$ day.
# * **Performance Evaluation**: 
#     1. Mean Absolute Error that, for a single sample, reduces to $\left| x_i - \hat{x}_i \right|$
# * **Precauções**: There is no lookahead.
# * **Conclusion**:

# #### 🇧🇷 Parte 1: Organizando os dados
# #### 🇺🇸 Step 1: Tidying the data

# In[ ]:


import matplotlib.pyplot as plt
import statsmodels.api as sm

df3 = None
df3 = df2.copy()
df3.head()
df3.set_index('dias',inplace=True)
df3.drop(['date','deaths'],axis=1,inplace=True)


# #### 🇧🇷 Parte 2: Obtendo o ACF
# 
# Observe: a série é muito bem autocorrelacionada com janela (lag) = 1
# 
# #### 🇺🇸 Step 2: Obtaining the autocorrelation function
# 
# Observe: The time series is very well autocorrelated with lag = 1.

# In[ ]:


sm.graphics.tsa.plot_acf(df3.values.squeeze(), lags=10)
plt.show()


# #### 🇧🇷 Parte 3: Obtendo também o PACF
# 
# Também nota-se que a série é muito bem parcialmente auto-correlacionada com lag = 1
# 
# #### 🇺🇸 Step 3: Obtaining the partial autocorrelation function
# 
# It is also noticeable that the series is very well partially autocorrelated with lag = 1.

# In[ ]:


sm.graphics.tsa.plot_pacf(df3.values.squeeze(), lags=10)
plt.show()


# #### 🇧🇷 Parte 4: Preparando os dados
# 
# Com vistas a obter a seguinte preparação dos dados
# * Atributos preditores: 
#     1. Dia $t$
#     2. Número de casos no dia $t$
#     3. Número de casos no dia $t - 1$
# * Atributo alvo: número de casos no dia $t + 1$
# 
# #### 🇺🇸 Step 4: Preparing the data
# 
# Looking forward to obtaining the following preparation of data:
# * Features:
#     1. Day $t$
#     2. Number of cases in $t$
#     3. Number of cases in $t - 1$
# * Target variable: number of cases in $t + 1$

# In[ ]:


df3['yesterday'] = df3['cases'].shift(1,fill_value=0)
df3.reset_index(level=0, inplace=True)
df3.head()


# #### 🇧🇷 Parte 5: Preparação do Holdout
# 
# #### 🇺🇸 Step 5: Holdout preparation

# In[ ]:


## Treino
dias_train = df3[['dias','yesterday']][:-1]
cases_train = df3['cases'][:-1]

## Teste
dias_test = df3[['dias','yesterday']][-1:]
cases_test =  df3['cases'][-1:]

previsao = len(df3) - len(dias_test)

print("Novo Holdout: Dados Totais: %d, Treino: %d dias, Teste: %d dia" % (len(df2),len(dias_train),len(dias_test)))


# #### 🇧🇷 Parte 6: Treino da mesma MLP com nova estratégia
# #### 🇺🇸 Step 6: Training the same MLP architecture using this new strategy

# In[ ]:


from sklearn.neural_network import MLPRegressor
# Treino da rede neural
mlp = MLPRegressor(hidden_layer_sizes=(200,200),activation='relu',solver='lbfgs',max_iter=1000, shuffle=True)
mlp.fit(X=dias_train.values,y=cases_train.values.ravel())


# In[ ]:


y_previsto = mlp.predict(dias_test.values)


# In[ ]:


layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title= "RNA MLP para previsão um dia à frente com dia anterior nos atributos",
    xaxis_title="Dias desde a primeira notificação",
    yaxis_title="Quantidade de casos",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 4
    )
)

fig = go.Figure(data=[
    go.Scatter(x=dias_train['dias'], y=df3['cases'][:-1], name='Dados de Treinamento',mode="lines+markers"),
    go.Scatter(x=dias_test['dias'], y=y_previsto, name = 'Casos Estimados',mode="lines+markers"),
    go.Scatter(x=dias_test['dias'], y=df3['cases'][-1:], name = 'Casos Reais',mode="lines+markers")
])
fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0= previsao + 0.5,
            y0=120,
            x1=previsao + 0.5,
            y1=max(df2['cases']) + 100,
            line=dict(
                width=1.5,
                dash= "dash"
            )
))

fig.add_trace(go.Scatter(
    x=[previsao - 0.5],
    y=[2],
    text=["Início da previsão"],
    mode="text",
))
fig['layout'].update(layout)

fig.show()


# In[ ]:


print("Erro Médio Absoluto: {0:6.3f} casos".format(mean_absolute_error(cases_test,y_previsto)))


# # 🇧🇷 Considerações Finais / 🇺🇸 Final Remarks
# 
# - 🇧🇷 A utilização de conhecimentos sobre a auto-correlação total e parcial da série foi a melhor estratégia para previsão um dia à frente
# - Utilizou-se dados dos casos do dia anterior para auxiliar no aprendizado dos padrões implícitos no crescimento da série
# - O melhor modelo obtido foi uma Rede Neural Multi Layer Perceptron com duas camadas ocultas e 200 neurônios em cada camada, função de ativação 'relu'e otimizador LBFGS em virtude da pequena quantidade de dados
# - A RNA se mostrou bastante tolerante aos ruídos e capturou adequadamente flutuações que não estão nos dados em si, tais como: 
#    - Chegada de mais kits de testes
#    - Subnotificação
#    - Notificação tardia
#    
# 
# - 🇺🇸 The use of knowledge about autocorrelation and partial autocorrelation was the best strategy to predict a day ahead.
# - The number of cases in the previous day was used to help the models to learn implicit patterns in the series growth.
# - The best model obtained was an ANN MLP with two hidden layers and 200 neurons each, using `relu` as the activation function and `LBFGS` as the optimizer, in virtue of the small amount of data.
# - The ANN has shown tolerance to noise and appropriately captured fluctuations that are not in the data itself, such as:
#     * The arrival of more testing kits
#     * Subnotification
#     * Late notification

# # 🇧🇷 Palpite para o futuro: Número de casos amanhã/ 🇺🇸 Guessing the future: Number of cases tomorrow
# 
#  - 🇧🇷 Obtido automaticamente a partir do mesmo modelo retreinado com todos os dados
#  - Serão realizadas 20 execuções, para minimizar o viés estocástico da inicialização aleatória dos pesos
#  - Será considerada a previsão mais otimista, com o menor número de casos
#  
#  
#  - 🇺🇸 Automatically obtained using the same model retrained with all the data
#  - Will be run 20 times, to minimize the stochastic bias of the random weight initialization
#  - The most optimistic prediction will be considered, with the lowest amount of cases

# In[ ]:


df3.tail()


# In[ ]:


from sklearn.neural_network import MLPRegressor

# Para uso nos dias que fiz previsão à posteriori
#df3.drop(df3.tail(1).index,inplace=True) 

tomorrow = max(df3['dias']) + 1
today_cases = df3.loc[df3['dias'] == max(df3['dias'])]['cases']

results = []
for i in range(20):

    # Treino da rede neural
    mlp = MLPRegressor(hidden_layer_sizes=(200,200),activation='relu',solver='lbfgs',max_iter=3000, shuffle=True)
    mlp.fit(X=df3[['dias','yesterday']].values,y=df3['cases'].values.ravel())
    

    x = pd.Series([tomorrow,int(today_cases)]).values.reshape(1,-1)
    tomorrow_cases = mlp.predict(x)
    results.append(tomorrow_cases)


# In[ ]:


tomorrow_data = (datetime.datetime.now()).strftime('%d/%m/%Y')
print("Previsão de casos para {0} no Brasil, a conferir na coletiva diária das 17h30min: {1}".format(tomorrow_data,int(min(results))))


# | **Data da Previsão** | **Número de Casos Previstos** | **Número de Casos Real** | **Observação**|
# | --- | --- | ---| --- |  
# | 03/04/2020 | 9152 | 9056 |  |
# | 04/04/2020 | 10432 | 10278 |   | 
# | 05/04/2020 | 11766 | 11130|    |
# | 06/04/2020 |  12521     |12056 |  |
# | 07/04/2020 |  13379     |  14347   |  |
# | 08/04/2020 | 15316     | 15927    |   |
# | 09/04/2020 | 17943 | 17857 | A posteriori em 10/04  |
# | 10/04/2020 | 20071 | 19638 |  |
# | 11/04/2020 | 21923      |20727      |  |
# | 12/04/2020 |  21919     | 22169       |  |
# | 13/04/2020 | 24097   |   23430   |  |
# | 14/04/2020 | 25353     | 25262   |  |
# | 15/04/2020 | 27010      | 28320 |      |
# | 16/04/2020 | 30788   | 30425    | A posteriori em 17/04  |
# | 17/04/2020 | 32730   | 33682    |  |
# | 18/04/2020 |36598       | 36599      | A posteriori em 20/04 - Erro por um caso! |
# | 19/04/2020| 39721         |   38654      | A posteriori em 20/04  | 
# | 20/04/2020| 41664     |40581     | A posteriori em 22/04 |
# | 21/04/2020|  43517       |    43079     | A posteriori em 22/04    | 
# | 22/04/2020|  45985       |  45757  |    | 
# | 23/04/2020|   48708         |   49492    | (Buffer de testes divulgado de uma vez só?)   |
# | 24/04/2020| 52806 |   52995 |   |
# | 25/04/2020| 56585 | 58509  |   |
# | 26/04/2020| 62838      | 63584    |    |
# | 27/04/2020| 66246   |  66501  |    | 
# | 28/04/2020| 71186 | 71886 | |
# | 29/04/2020| 77052      | 78162   |.  |
# | 30/04/2020| 83951       | 85380   |.  |
# |01/05/2020 | 91967  | 91589   |   | 
# | 02/05/2020 | 98552 | 96396 | | 
# | 03/05/2020 | 103467 | 101147 | A posteriori em 04/05|
# | 04/05/2020 | 107909 | 107780  | | 
# | 05/05/2020|  111746   | 114715  | (Erro no preenchimento do dataset? -- Corrigido!)|
# | 06/05/2020|  122221  | 125218    |.    | 
# | 07/05/2020 | 133920     | 135106   |.  |
# | 08/05/2020 | 144682 |145328  |. |
# | 09/05/2020 | 155719 |155939  |. |
# | 10/05/2020 |167089   | 162699  |  | 
# | 11/05/2020 | 173487   |168331  |  .| 
# | 12/05/2020| 178559    |177589   |. |
# | 13/05/2020|  188156 | 188974 | . |
# | 14/05/2020| 200292 | 202918 | .|
# | 15/05/2020| 215428 | 218223 | .|
# | 16/05/2020|  232085   | 233142   |.   |
# | 17/05/2020|248059      | 241080    | (A posteriori em 18/05/2020) |
# | 18/05/2020|255226    |254220  |. |
# | 19/05/2020| 268995  |271628   |.  |
# | 20/05/2020|287681    |291579    |.    |  
# | 21/05/2020| 309425 | 310087 | | 
# | 22/05/2020| 329081 | 330890 | |
# | 23/05/2020| 351431 | 347398 | (A posteriori em 24/05/2020) | 
# | 24/05/2020| 368255 | 363211 | | 
# | 25/05/2020| 384151| 374898| |
# | 26/05/2020| 394788| 391222 |(A posteriori em 27/05/2020) |
# | 27/05/2020| 411194| 411821 | |
# | 28/05/2020|431782      |438238     |.  |
# | 29/05/2020| 461557     |  465166    |.  |
# |30/05/2020|  490163    |  498440    |.  |
# | 31/05/2020| 526460     |  514849    |.  |
# | 01/06/2020| 542052    | 526447   |. | 
# | 02/06/2020| 551830 | 555383 | .|
# | 03/06/2020|582685  | 584016  | (A posteriori em 04/06/2020) |
# | 04/06/2020|612518  | 614941  | (Mudança no horário de divulgação dos resultados)  |
# | 05/06/2020 |645351  |.   |.  |
# 

# In[ ]:


y_true = [9056, 10278, 11130, 12056,14347, 15927, 17857, 19638, 20727, 22169, 23430, 25262, 28320, 30425, 33682, 36599, 38654, 40581, 43079, 45757, 52995, 58509, 63584, 66501, 71886, 78162, 85380, 91589, 96396,101147,107780, 114715,125218, 135106, 145328, 155939, 162699, 168331, 177589, 188974, 202918, 218223, 233142, 241080, 254220, 271628, 291579, 310087, 330890, 347398, 363211, 374898, 391222, 411821, 438238, 465166, 498440, 514849, 526447, 555393,584016, 614941]
y_previsto = [9152, 10432, 11766, 12521, 13379, 15316, 17943, 20071, 21923, 21919, 24097, 25353, 27010, 30788, 32730, 36598, 39721, 41664, 43517, 45985, 52806, 56585, 62838, 66246, 71186, 77052,83951, 91967,98552, 103467, 107909,111746,122221, 133920, 144682, 155719, 167089, 173487, 178559, 188156, 200292, 215428, 232085, 248059, 255226, 268995, 287681, 309425, 329081, 351431, 368255, 384151, 394788, 411194, 431782, 461557, 490163, 526460, 542052,551830,582685, 612518] 

print("Raiz do Erro Médio Quadrático (RMSE): {0:6.4f}".format(mean_squared_error(y_true,y_previsto)**0.5))
print("R2-Score: {0:6.4f}".format(r2_score(y_true,y_previsto)))


# ### 🇧🇷 Visualizando real versus previsão a partir de 03/04/2020
# 
# ### 🇺🇸 Visualizing ground truth versus prediction starting in April 3rd, 2020

# In[ ]:


# Criação dos rótulos para o eixo x com datas
labels = []
start_date = datetime.date(2020, 4, 3)
end_date = datetime.date.today()
delta = datetime.timedelta(days=1)
while start_date < end_date:
    labels.append(start_date.strftime('%d/%m/%Y'))
    start_date += delta

layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title= "Visualizando previsões one-day-ahead realizadas com o modelo proposto",
    xaxis_title="Datas",
    yaxis_title="Quantidade de casos",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 3,
        tickangle = 295
    )
)

fig = go.Figure(data=[
    go.Scatter(x=labels, y=y_true, name='Casos Reais',mode="lines+markers"),
    go.Scatter(x=labels, y=y_previsto, name = 'Casos Estimados',mode="lines+markers"),
])

fig['layout'].update(layout)

fig.show()


# ### 🇧🇷 Visualizando os Resíduos
# 
# - Os resíduos na previsão são a diferença entre os valores previstos e observados ao quadrado
# - Resultam em pontos que são comparados com a reta-0
# - A reta-0 representa o modelo perfeito, em que a distância entre o valor previsto e observado é zero
# - Visualizar os resíduos ajuda a interpretar visualmente a qualidade do modelo
# 
# ### 🇺🇸 Visualizing the residuals
# 
# - The residual in the prediction are the square of the difference between the predicted ($x'$) and the real ($x$) values: $(x' - x)^2$
# - The result is a set of points that are compared with the line-0
# - The line-0 represents the perfect model, in which the distance between every predicted and real value is zero.
# - Visualizing the residual values helps to visually understand the quality of the model
# 
# 

# In[ ]:


residuos = []
datas = []
inicio = datetime.datetime.now() - datetime.timedelta(days=len(y_true))
for (x,y) in zip(y_true,y_previsto):
    r = (x-y)
    residuos.append(r)
    datas.append(inicio.strftime('%d/%m/%Y'))
    inicio += datetime.timedelta(days=1)


# In[ ]:


layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title= "Visualização dos Resíduos",
    xaxis_title="Dia da Previsão",
    yaxis_title="Resíduos",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 3,
        tickangle = 295
    )
)

fig = go.Figure(data=[
    go.Scatter(x=datas, y=residuos, name='Residuos',mode="markers")
])
fig.add_shape(
        # Horizontal Line
        dict(
            type="line",
            x0= 0,
            y0= 0,
            x1= len(y_true),
            y1=0,
            name = "Reta Zero",
            line=dict(
                width=3,
                dash= "dash"
            ),
            
))

fig.add_trace(go.Scatter(
    x=[len(y_true)],
    y=[300],
    text=["Reta Zero"],
    mode="text",
))
fig['layout'].update(layout)

fig.show()


# ### 🇧🇷 Links Úteis / 🇺🇸 Useful links
# 
# * https://docs.bokeh.org/en/latest/docs/user_guide/geo.html  
# * https://towardsdatascience.com/walkthrough-mapping-basics-with-bokeh-and-geopandas-in-python-43f40aa5b7e9  
# * https://pt.wikipedia.org/wiki/Lista_de_unidades_federativas_do_Brasil_por_popula%C3%A7%C3%A3o
# * http://www.atlasbrasil.org.br/2013/data/rawData/publicacao_atlas_municipal_pt.pdf
# * http://portal.cfm.org.br/images/PDF/leitosdeutiestados2018.pdf
# * https://www.ssp.sp.gov.br/fale/estatisticas/answers.aspx?t=6
# * https://github.com/codeforamerica/click_that_hood/blob/master/public/data/brazil-states.geojson
# * https://docs.python.org/3/library/datetime.html

# # Agradecimento
# 
# A autora Elloá B. Guedes agradece o apoio financeiro provido pela FAPEAM no âmbito do Projeto PPP 04/2017.
# 
# ![](http://www.fapeam.am.gov.br/downloads/57415/)
