#!/usr/bin/env python
# coding: utf-8

# #Valorant é o E-sports que vem mais crescendo nos ultimos tempos, desde seu lançamento em 2020, o jogo é o novo FPS competitivo da Riot Games, a mesma desenvolvedora de League of Legends, que mistura bastante elementos de Overwatch e CS:GO.
# 
# #O VALORANT Masters Reykjavik 2022 é o primeiro campeonato internacional de VALORANT do ano. Ao todo são 12 times lutando pelo título, premiação em dinheiro e pontos no circuito Challengers. O Brasil terá dois representantes na competição: LOUD e NiP (Ninjas in Pyjamas).
# 
# #"O Valorant conseguiu ultrapassar a marca dos 15 milhões de jogadores ativos por mês, de acordo com o portal Talk Esports. O número foi registrado no mês de janeiro deste ano, mostrando um aumento considerável de acordo com o último dado apresentado pela Riot Games." de : https://ge.globo.com/esports/valorant/noticia/valorant-jogo-atingiu-marca-de-15-milhoes-de-jogadores-por-mes.ghtml

# In[ ]:


#primeiro vamos importar nossas bibliotecas padrões.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px


# In[ ]:


dados_all = pd.read_csv('/kaggle/input/vct-masters-stage-1-map-statistics/Combined Data.csv')
dados_players = pd.read_csv('/kaggle/input/firstvalorantmasters-2k22/Players_stats.csv', sep=';' ,encoding= "ISO-8859-1")
dados_agents = pd.read_csv('/kaggle/input/firstvalorantmasters-2k22/agents_master_2022.csv',sep=';')


# # **Analise dos agentes que foram escolhidos no mundial**

# In[ ]:


#Lembrando que todos os agentes aparentes na tabela são apenas os primarios dos jogadores, os mais jogados.
dados_players


# **Vamos retirar as sujeiras e limpar nossa tabela**

# In[ ]:


#Primeiro vamos arrumar o nome das colunas e depois as linhas.

dados_players.columns = dados_players.columns.str.replace('[#,@,\t]','') 


# In[ ]:


dados_players.head(2)


# In[ ]:


#Agora vamos retirar esses \t das linhas do nosso trabalho.


dados_players['Player'] = dados_players['Player'].str.replace('\t','')
dados_players['Team'] = dados_players['Team'].str.replace('\t','')
dados_players['Agents'] = dados_players['Agents'].str.replace('\t','')


# **Fazendo analise da nossa tabela**

# In[ ]:


dados_players.head(3)


# In[ ]:


#Analisando os times de acordo com os KDA.

kdas = ['Team','Maps','KDA']
kda = dados_players.filter(items=kdas)
kda.groupby(['Team']).mean()


# **Conseguimos ver a quantidade de jogo por time x KD/A dos mesmo e podemos ver que a influencia de mais jogos impactou nesse resultado.**

# In[ ]:


#Fazendo nosso grafico para compararmos os mapas jogados com os kd/as

kda.groupby(['Team']).mean().plot(kind='bar',figsize=(12,8))
plt.title("mapas jogados x KDA",fontsize=15, fontweight='bold')
plt.xlabel("Times", fontsize=15, fontweight='bold')
plt.ylabel('MAPAS/KDA', fontsize=12, fontweight='bold');


# In[ ]:


kda_players = ['Player', 'KDA']
kda_player = dados_players.filter(items=kda_players)


# In[ ]:


dados = kda_player.sort_values(by=['KDA','Player'],ascending=False).head(5)


# In[ ]:


dados.head(5)


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(y='KDA', x='Player', data=dados, color = '#0000FF')
ax.set_title("TOP 5 KDA PLAYERS", fontsize=15, fontweight='bold')
ax.set_xlabel("Players", fontsize=10, fontweight='bold')
ax.set_ylabel("KD/A", fontsize=10, fontweight='bold');


# In[ ]:


#Os personagens mais utilizados pelos profissionais.

agente = ['Player','Agents']
agentes = dados_players.filter(items=agente)


# In[ ]:


#Quantidade de vezes que o personagem foi o escolhido pelos profissionais.

plt.figure(figsize=(12,7))

agentes['Agents'].value_counts().plot(kind="bar", color = "royalblue")
plt.title("Principais Agentes dos profissionais",fontsize=15, fontweight='bold')
plt.xlabel("Agentes", fontsize=15, fontweight='bold')
plt.ylabel('Qtd de Vezes Escolhidos', fontsize=12, fontweight='bold');


# # Agora vamos estudar a porcentagem de vitorias e picks dos agentes 

# In[ ]:


#Dado dos agentes em porcentagem durante os seus picks no campeonato.
dados_agents


# In[ ]:


#Renomeando o pickrate para ser melhor estudado
dados_agents.rename(columns={'Overall Pick Rate':'PickRate'},inplace=True)


# In[ ]:


#Separando os melhores agentes em cada mapa para analise.
pick1 = ['Agents', 'PickRate']
PickRate = dados_agents.filter(items=pick1)

pick2 = ['Agents', 'Split']
Split = dados_agents.filter(items=pick2)

pick3 = ['Agents', 'Bind']
Bind = dados_agents.filter(items=pick3)

pick4 = ['Agents', 'Haven']
Haven = dados_agents.filter(items=pick4)

pick5 = ['Agents', 'Ascent']
Ascent = dados_agents.filter(items=pick5)

pick6 = ['Agents', 'Icebox']
Icebox = dados_agents.filter(items=pick6)

pick7 = ['Agents', 'Breeze']
Breeze = dados_agents.filter(items=pick7)

pick8 = ['Agents', 'Fracture']
Fracture = dados_agents.filter(items=pick8)


# In[ ]:


#agora vamos organizar para temos os melhores de cada mapa e os mais utilizados.

Split= Split.sort_values(by=['Split','Agents'],ascending=False)
PickRate= PickRate.sort_values(by=['PickRate','Agents'],ascending=False)
Bind= Bind.sort_values(by=['Bind','Agents'],ascending=False)
Haven= Haven.sort_values(by=['Haven','Agents'],ascending=False)
Ascent= Ascent.sort_values(by=['Ascent','Agents'],ascending=False)
Icebox= Icebox.sort_values(by=['Icebox','Agents'],ascending=False)
Breeze= Breeze.sort_values(by=['Breeze','Agents'],ascending=False)
Fracture= Fracture.sort_values(by=['Fracture','Agents'],ascending=False)


# In[ ]:


Split.head()


# In[ ]:


PickRate.head()


# **Os agentes preferidos dos profissionais durante o campeonato**

# In[ ]:


plt.figure(figsize=(15,6))
plt.title('Vezes em % que os agentes foram pegos no campeonato', fontsize= 15, fontweight='bold')
plt.xlabel('Agents', fontsize= 15, fontweight='bold')
plt.ylabel('% em Partidas', fontsize= 15, fontweight='bold')
plt.plot(PickRate['Agents'], PickRate['PickRate'], linestyle='--', color='b', marker='s', 
         linewidth=3.0)
plt.xticks(rotation=45, ha= 'right')
plt.show()


# **Agora vamos analisar por mapas, quais foram os melhores agentes de cada um deles**

# In[ ]:


#Agentes com a % de vitoria na split
plt.figure(figsize=(15, 8)) 
plots = sns.barplot(x="Agents", y="Split", data=Split) 
for bar in plots.patches: 
  
    
    
    plots.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center', 
                   size=10, xytext=(0, 5), 
                   textcoords='offset points') 
  
  
plt.title("MELHORES AGENTES X SPLIT",fontsize= 12, fontweight='bold')
plt.xlabel('Agents',fontsize= 12, fontweight='bold')
plt.ylabel('% DE VITORIAS',fontsize= 12, fontweight='bold');


#Agentes com a % de vitoria na Bind
plt.figure(figsize=(15, 8)) 
plots = sns.barplot(x="Agents", y="Bind", data=Bind) 
for bar in plots.patches: 
  
    
    
    plots.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center', 
                   size=10, xytext=(0, 5), 
                   textcoords='offset points') 
  
  
plt.title("MELHORES AGENTES X BIND",fontsize= 12, fontweight='bold')
plt.xlabel('Agents',fontsize= 12, fontweight='bold')
plt.ylabel('% DE VITORIAS',fontsize= 12, fontweight='bold');

#Agentes com a % de vitoria na Ascent
plt.figure(figsize=(15, 8)) 
plots = sns.barplot(x="Agents", y="Ascent", data=Ascent) 
for bar in plots.patches: 
  
    
    
    plots.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center', 
                   size=10, xytext=(0, 5), 
                   textcoords='offset points') 
  
  
plt.title("MELHORES AGENTES X ASCENT",fontsize= 12, fontweight='bold')
plt.xlabel('Agents',fontsize= 12, fontweight='bold')
plt.ylabel('% DE VITORIAS',fontsize= 12, fontweight='bold');


# In[ ]:


#Agentes com a % de vitoria na Icebox
plt.figure(figsize=(15, 8)) 
plots = sns.barplot(x="Agents", y="Icebox", data=Icebox) 
for bar in plots.patches: 
  
    
    
    plots.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center', 
                   size=10, xytext=(0, 5), 
                   textcoords='offset points') 
  
  
plt.title("MELHORES AGENTES X ICEBOX",fontsize= 12, fontweight='bold')
plt.xlabel('Agents',fontsize= 12, fontweight='bold')
plt.ylabel('% DE VITORIAS',fontsize= 12, fontweight='bold');


#Agentes com a % de vitoria na Haven
plt.figure(figsize=(15, 8)) 
plots = sns.barplot(x="Agents", y="Haven", data=Haven) 
for bar in plots.patches: 
  
    
    
    plots.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center', 
                   size=10, xytext=(0, 5), 
                   textcoords='offset points') 
  
  
plt.title("MELHORES AGENTES X HAVEN",fontsize= 12, fontweight='bold')
plt.xlabel('Agents',fontsize= 12, fontweight='bold')
plt.ylabel('% DE VITORIAS',fontsize= 12, fontweight='bold');


#Agentes com a % de vitoria na Breeze
plt.figure(figsize=(15, 8)) 
plots = sns.barplot(x="Agents", y="Breeze", data=Breeze) 
for bar in plots.patches: 
  
    
    
    plots.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center', 
                   size=10, xytext=(0, 5), 
                   textcoords='offset points') 
  
  
plt.title("MELHORES AGENTES X BREEZE",fontsize= 12, fontweight='bold')
plt.xlabel('Agents',fontsize= 12, fontweight='bold')
plt.ylabel('% DE VITORIAS',fontsize= 12, fontweight='bold');

#Agentes com a % de vitoria na Breeze
plt.figure(figsize=(15, 8)) 
plots = sns.barplot(x="Agents", y="Fracture", data=Fracture) 
for bar in plots.patches: 
  
    
    
    plots.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center', 
                   size=10, xytext=(0, 5), 
                   textcoords='offset points') 
  
  
plt.title("MELHORES AGENTES X FRACTURE",fontsize= 12, fontweight='bold')
plt.xlabel('Agents',fontsize= 12, fontweight='bold')
plt.ylabel('% DE VITORIAS',fontsize= 12, fontweight='bold');


# # **Vamos fazer as analises dos dados e gráficas dos nossos mapas**

# In[ ]:


dados_all


# In[ ]:


dados_all.isnull().sum()


# In[ ]:


dados_all.describe()


# In[ ]:


lista1 = ['Map','TotalPicks','GroupAPicks','GroupBPicks','PlayoffsPicks']
escolha_mapas = dados_all.filter(items=lista1)


# In[ ]:


escolha_mapas


# In[ ]:


lista2 = ['Map','TotalPicks', 'AtkWins','AtkWin%','DefWins','DefWin%']
lados = dados_all.filter(items=lista2)


# In[ ]:


lados


# In[ ]:


lista3 = ['Map', 'TotalBans','GroupABans','GroupBBans','PlayoffsBans']
ban_mapas = dados_all.filter(items=lista3)


# In[ ]:


ban_mapas


# # **Agora vamos fazer a nossa visualização gráfica dos itens acima em ordem.**

# In[ ]:


#Primeiro vamos fazer das escolhas dos mapas.

fig, ax = plt.subplots(figsize=(10, 8))

#Plot de barra
sns.barplot(x='Map', y='TotalPicks', data=escolha_mapas, color = '#0000FF', alpha = 0.85)

#Plot de torta 

plt.figure(figsize=(15,6))
plt.title('Porcentagem dos picks - MAPAS',fontsize=12, fontweight='bold')
Analise = round(escolha_mapas['Map'].value_counts(normalize=True) * 100, 1)
plt.pie(escolha_mapas['TotalPicks'].head(10),
        labels = Analise.index[0:10],
        shadow=True, 
        startangle = 90, 
        autopct='%1.1f%%')

#Arrumando o grafico de barra.

ax.set_title("QTD DE PICKS - MAPAS", fontsize=12, fontweight='bold')
ax.set_xlabel("Mapas", fontsize=12, fontweight='bold')
ax.set_ylabel("Quantidade de picks", fontsize=12, fontweight='bold');


# In[ ]:


#Vamos fazer agora dos banimentos dos MAPAS.

ban_mapas


# In[ ]:


count = ban_mapas['TotalBans']
mapa = ban_mapas['Map']

plt.figure(figsize = (20, 8))
plt.subplot(1, 2, 1)
#Plot de barra
plt.bar(mapa, count, ec = "k", alpha = .6, color = "royalblue")
plt.xlabel("Mapas", fontsize=12, fontweight='bold')
plt.ylabel('Quantidade', fontsize=12, fontweight='bold')
plt.title("Número de Qtd de bans x MAPAS")


#Plot de torta 
plt.subplot(1, 2, 2)
plt.pie(count, labels= mapa, autopct='%1.1f%%')
plt.axis("equal")
plt.title("Porcentagem de bans - Mapas",fontsize=15, fontweight='bold')
plt.legend();


# In[ ]:


#Agora vamos pegar os lados que mais obteram derrotas e vitorias nesses mapas. (ataque/defesa)

lados.info()


# In[ ]:


lados


# In[ ]:


#Grafico sobre o total de picks e a % de vitorias atk e defesa 
total = lados['TotalPicks']
atk = lados['AtkWin%']
defe = lados['DefWin%']
maps = lados['Map']

#Configurando o tamanho do grafico
barWidth = 0.2
fig, ax = plt.subplots(figsize=(20,10))

#Definindo a posição das barras

b1 = np.arange(len(total))
b2 = [x + barWidth for x in b1]
b3 = [x + barWidth for x in b2]


#Criando as barras
ax1 = ax.bar(b2, atk, width=barWidth, label='AtkWins%', color= 'Blue')
ax2 = ax.bar(b3, defe ,width=barWidth, label="DefWins%", color='Green')

#Adicionando os dados do grafico.

ax.set_xlabel('MAPAS',fontsize=15, fontweight='bold')
ax.set_xticks([r + barWidth for r in range(len(total))], ['Ascent', 'Bind', 'Breeze', 'Fracture', 'Haven', 'Icebox', 'Split'])
ax.set_ylabel("% de vitorias",fontsize=15, fontweight='bold')
ax.set_title('Atk wins/Def Wins x MAPAS', fontsize=15, fontweight='bold')
ax.legend()

#Botando rotulos nos graficos

ax.bar_label(ax1, fmt="%.01f%%", size=15, label_type="edge");
ax.bar_label(ax2, fmt="%.01f%%", size=15, label_type="edge");


# # O Valorant é um game lançado em 2020 pela RIOT GAMES e desde então tem crescido absurdamente entre a comunidade de FPS durante esses dois anos online, e nesse tempo alguns agentes e mapas sempre foram mais queridinhos da galera que outros. Os agentes tem kits de habilidades que fazem cada um deles unicos. A Jett por exemplo possui uma agressividade absurda com um poder de posicionamento aonde não oferece risco caso você seja um bom jogador com a mesma, por isso ela vem dominando as estatisticas de não só a mais pegada do campeonato mas também como a preferida de todos os profissionais.
# # Assim como os agentes, os mapas também tem suas individualidades, contando com:  teleportes, cordas para atravessar os mesmos, alturas e também toda uma mecânica. Icebox e a Ascent foram as preferidas para esse campeonato de nivel mundial por serem mapas mais faceis de trabalhar e pela sua rotatividade.

# In[ ]:




