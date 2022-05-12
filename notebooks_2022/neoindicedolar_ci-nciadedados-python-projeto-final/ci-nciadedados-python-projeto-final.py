#!/usr/bin/env python
# coding: utf-8

# <img src="https://" alt="arc-tecnologia">
# 
# ---
# 
# # **Ciência de Dados** | Python: Projeto Final
# Caderno de **Aula**<br> 
# Aprendiz [Alexandre R.](https://www.linkedin.com/in/ale2301/)
# 
# ---

# # **Tópicos**
# 
# <ol type="1">
#   <li>Introdução;</li>
#   <li>Exploração de dados;</li>
#   <li>Transformação e limpeza de dados;</li>
#   <li>Visualização de dados;</li>
#   <li>Storytelling.</li>
# </ol>

# ---

# ## 1\. Introdução

# Vamos explorar dados de crédito presentes neste neste [link](https://raw.githubusercontent.com/andre-marcos-perez/ebac-course-utils/develop/dataset/credito.csv). Os dados estão no formato CSV e contém informações sobre clientes de uma instituição financeira. Em especial, estamos interessados em explicar a segunda coluna, chamada de **default**, que indica se um cliente é adimplente(`default = 0`), ou inadimplente (`default = 1`), ou seja, queremos entender o porque um cliente deixa de honrar com suas dívidas baseado no comportamento de outros atributos, como salário, escolaridade e movimentação financeira. 

# ## 2\. Exploração de Dados

# >Uma descrição sobre os atributos que serão usados:
# 
# >O atributo de interesse (`default`) é conhecido como **variável resposta** ou **variável dependente**, já os demais atributos que buscam explicá-la (`idade`, `salário`, etc.) são conhecidas como **variáveis explicatívas**, **variáveis independentes** ou até **variáveis preditoras**.

# 
# 
# | Coluna  | Descrição |
# | ------- | --------- |
# | id      | Número da conta |
# | default | Indica se o cliente é adimplente (0) ou inadimplente (1) |
# | idade   | --- |
# | sexo    | --- |
# | depedentes | --- |
# | escolaridade | --- |
# | estado_civil | --- |
# | salario_anual | Faixa do salario mensal multiplicado por 12 |
# | tipo_cartao | Categoria do cartao: blue, silver, gold e platinium |
# | meses_de_relacionamento | Quantidade de meses desde a abertura da conta |
# | qtd_produtos | Quantidade de produtos contratados |
# | iteracoes_12m | Quantidade de iteracoes com o cliente no último ano |
# | meses_inatico_12m | Quantidade de meses que o cliente ficou inativo no último ano |
# | limite_credito | Valor do limite do cartão de crédito |
# | valor_transacoes_12m | Soma total do valor das transações no cartão de crédito no último ano |
# | qtd_transacoes_12m | Quantidade total de transações no cartão de crédito no último ano |
# 
# 

# Vamos começar lendos os dados num dataframe `pandas`.

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('https://raw.githubusercontent.com/andre-marcos-perez/ebac-course-utils/develop/dataset/credito.csv', na_values='na')


# In[ ]:


df.head(n=20)


# Com o dados em mãos, vamos conhecer um pouco melhor a estrutura do nosso conjunto de dados.

# ### **2.1. Estrutura** 

# In[ ]:


df.shape # retorna uma tupla (qtd linhas, qtd colunas)


# In[ ]:


df[df['default'] == 0].shape


# In[ ]:


df[df['default'] == 1].shape


# In[ ]:


qtd_total, _ = df.shape
qtd_adimplentes, _ = df[df['default'] == 0].shape
qtd_inadimplentes, _ = df[df['default'] == 1].shape


# In[ ]:


print(f"A proporcão clientes adimplentes é de {round(100 * qtd_adimplentes / qtd_total, 2)}%")
print(f"A proporcão clientes inadimplentes é de {round(100 * qtd_inadimplentes / qtd_total, 2)}%")


# ### **2.2. Schema** 

# In[ ]:


df.head(n=5)


#  - Colunas e seus respectivos tipos de dados.

# In[ ]:


df.dtypes


#  - Atributos **categóricos**.

# In[ ]:


df.select_dtypes('object').describe().transpose()


#  - Atributos **numéricos**.

# In[ ]:


df.drop('id', axis=1).select_dtypes('number').describe().transpose()


# ### **2.3. Dados faltantes** 

# Dados faltantes podem ser:
# 
#  - Vazios (`""`);
#  - Nulos (`None`);
#  - Não disponíveis ou aplicaveis (`na`, `NA`, etc.);
#  - Não numérico (`nan`, `NaN`, `NAN`, etc).

# In[ ]:


df.head()


# Podemos verificar quais colunas possuem dados faltantes.

# In[ ]:


df.isna().any()


#  - A função abaixo levanta algumas estatisticas sobre as colunas dos dados faltantes.

# In[ ]:


def stats_dados_faltantes(df: pd.DataFrame) -> None:

  stats_dados_faltantes = []
  for col in df.columns:
    if df[col].isna().any():
      qtd, _ = df[df[col].isna()].shape
      total, _ = df.shape
      dict_dados_faltantes = {col: {'quantidade': qtd, "porcentagem": round(100 * qtd/total, 2)}}
      stats_dados_faltantes.append(dict_dados_faltantes)

  for stat in stats_dados_faltantes:
    print(stat)


# In[ ]:


stats_dados_faltantes(df=df)


# In[ ]:


stats_dados_faltantes(df=df[df['default'] == 0])


# In[ ]:


stats_dados_faltantes(df=df[df['default'] == 1])


# ## 3\. Transformação e limpeza de dados

# Agora que conhecemos melhor a natureza do nosso conjunto de dados, vamos conduzir uma atividade conhecida como *data wrangling* que consiste na transformação e limpeza dos dados do conjunto para que possam ser melhor analisados. Em especial, vamos remover:
# 
#  - Corrigir o *schema* das nossas colunas;
#  - Remover os dados faltantes.

# ### **3.1. Correção de schema** 

# Na etapa de exploração, notamos que as colunas **limite_credito** e **valor_transacoes_12m** estavam sendo interpretadas como colunas categóricas (`dtype = object`).

# In[ ]:


df[['limite_credito', 'valor_transacoes_12m']].dtypes


# In[ ]:


df[['limite_credito', 'valor_transacoes_12m']].head(n=5)


# Vamos criar uma função `lambda` para limpar os dados. Mas antes, vamos testar sua aplicação através do método funcional `map`:

# In[ ]:


fn = lambda valor: float(valor.replace(".", "").replace(",", "."))

valores_originais = ['12.691,51', '8.256,96', '3.418,56', '3.313,03', '4.716,22']
valores_limpos = list(map(fn, valores_originais))

print(valores_originais)
print(valores_limpos)


# Com a função `lambda` de limpeza pronta, basta aplica-la nas colunas de interesse.

# In[ ]:


df['valor_transacoes_12m'] = df['valor_transacoes_12m'].apply(fn)
df['limite_credito'] = df['limite_credito'].apply(fn)


# Vamos descrever novamente o *schema*:

# In[ ]:


df.dtypes


#  - Atributos **categóricos**.

# In[ ]:


df.select_dtypes('object').describe().transpose()


#  - Atributos **numéricos**.

# In[ ]:


df.drop('id', axis=1).select_dtypes('number').describe().transpose()


# ### **3.2. Remoção de dados faltantes** 

# Como o pandas está ciente do que é um dados faltante, a remoção das linhas problemáticas é trivial.

# In[ ]:


df.dropna(inplace=True)


# Vamos analisar a estrutura dos dados novamente.

# In[ ]:


df.shape


# In[ ]:


df[df['default'] == 0].shape


# In[ ]:


df[df['default'] == 1].shape


# In[ ]:


qtd_total_novo, _ = df.shape
qtd_adimplentes_novo, _ = df[df['default'] == 0].shape
qtd_inadimplentes_novo, _ = df[df['default'] == 1].shape


# In[ ]:


print(f"A proporcão adimplentes ativos é de {round(100 * qtd_adimplentes / qtd_total, 2)}%")
print(f"A nova proporcão de clientes adimplentes é de {round(100 * qtd_adimplentes_novo / qtd_total_novo, 2)}%")
print("")
print(f"A proporcão clientes inadimplentes é de {round(100 * qtd_inadimplentes / qtd_total, 2)}%")
print(f"A nova proporcão de clientes inadimplentes é de {round(100 * qtd_inadimplentes_novo / qtd_total_novo, 2)}%")


# ## 4\. Visualização de dados

# Os dados estão prontos, vamos criar diversas visualizações para correlacionar variáveis explicativas com a variável resposta para buscar entender qual fator leva um cliente a inadimplencia. E para isso, vamos sempre comparar a base com todos os clientes com a base de adimplentes e inadimplentes.

# Começamos então importando os pacotes de visualização e separando os clientes adimplentes e inadimplentes 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


# In[ ]:


df_adimplente = df[df['default'] == 0]


# In[ ]:


df_inadimplente = df[df['default'] == 1]


# ### **4.1. Visualizações categóricas** 

# Nesta seção, vamos visualizar a relação entre a variável resposta **default** com os atributos categóricos.

# In[ ]:


df.select_dtypes('object').head(n=5)


#  - Tido de Cartão

# In[ ]:


coluna = 'tipo_cartao'
titulos = ['Tipo de Cartão dos Clientes', 'Tipo de Cartão dos Clientes Adimplentes', 'Tipo de Cartão dos Clientes Inadimplentes']

eixo = 0
max_y = 0
max = df.select_dtypes('object').describe()[coluna]['freq'] * 1.1

figura, eixos = plt.subplots(1,3, figsize=(20, 5), sharex=True)

for dataframe in [df, df_adimplente, df_inadimplente]:

  df_to_plot = dataframe[coluna].value_counts().to_frame()
  df_to_plot.rename(columns={coluna: 'frequencia_absoluta'}, inplace=True)
  df_to_plot[coluna] = df_to_plot.index
  df_to_plot.sort_values(by=[coluna], inplace=True)
  df_to_plot.sort_values(by=[coluna])

  f = sns.barplot(x=df_to_plot[coluna], y=df_to_plot['frequencia_absoluta'], ax=eixos[eixo])
  f.set(title=titulos[eixo], xlabel=coluna.capitalize(), ylabel='Frequência Absoluta')
  f.set_xticklabels(labels=f.get_xticklabels(), rotation=90)

  _, max_y_f = f.get_ylim()
  max_y = max_y_f if max_y_f > max_y else max_y
  f.set(ylim=(0, max_y))

  eixo += 1

figura.show()


#  - Salário Anual

# In[ ]:


coluna = 'salario_anual'
titulos = ['Salário Anual dos Clientes', 'Salário Anual dos Clientes Adimplentes', 'Salário Anual dos Clientes Inadimplentes']

eixo = 0
max_y = 0
figura, eixos = plt.subplots(1,3, figsize=(20, 5), sharex=True)

for dataframe in [df, df_adimplente, df_inadimplente]:

  df_to_plot = dataframe[coluna].value_counts().to_frame()
  df_to_plot.rename(columns={coluna: 'frequencia_absoluta'}, inplace=True)
  df_to_plot[coluna] = df_to_plot.index
  df_to_plot.reset_index(inplace=True, drop=True)
  df_to_plot.sort_values(by=[coluna], inplace=True)

  f = sns.barplot(x=df_to_plot[coluna], y=df_to_plot['frequencia_absoluta'], ax=eixos[eixo])
  f.set(title=titulos[eixo], xlabel=coluna.capitalize(), ylabel='Frequência Absoluta')
  f.set_xticklabels(labels=f.get_xticklabels(), rotation=90)
  _, max_y_f = f.get_ylim()
  max_y = max_y_f if max_y_f > max_y else max_y
  f.set(ylim=(0, max_y))
  eixo += 1

figura.show()


# ### **4.2. Visualizações numéricas** 

# Nesta seção, vamos visualizar a relação entre a variável resposta **default** com os atributos numéricos.

# In[ ]:


df.drop(['id', 'default'], axis=1).select_dtypes('number').head(n=5)


#  - Quantidade de Transações nos Últimos 12 Meses

# In[ ]:


coluna = 'qtd_transacoes_12m'
titulos = ['Qtd. de Transações no Último Ano', 'Qtd. de Transações no Último Ano de Adimplentes', 'Qtd. de Transações no Último Ano de Inadimplentes']

eixo = 0
max_y = 0
figura, eixos = plt.subplots(1,3, figsize=(20, 5), sharex=True)

for dataframe in [df, df_adimplente, df_inadimplente]:

  f = sns.histplot(x=coluna, data=dataframe, stat='count', ax=eixos[eixo])
  f.set(title=titulos[eixo], xlabel=coluna.capitalize(), ylabel='Frequência Absoluta')

  _, max_y_f = f.get_ylim()
  max_y = max_y_f if max_y_f > max_y else max_y
  f.set(ylim=(0, max_y))

  eixo += 1

figura.show()


#  - Valor das Transações nos Últimos 12 Meses

# In[ ]:


coluna = 'valor_transacoes_12m'
titulos = ['Valor das Transações no Último Ano', 'Valor das Transações no Último Ano de Adimplentes', 'Valor das Transações no Último Ano de Inadimplentes']

eixo = 0
max_y = 0
figura, eixos = plt.subplots(1,3, figsize=(20, 5), sharex=True)

for dataframe in [df, df_adimplente, df_inadimplente]:

  f = sns.histplot(x=coluna, data=dataframe, stat='count', ax=eixos[eixo])
  f.set(title=titulos[eixo], xlabel=coluna.capitalize(), ylabel='Frequência Absoluta')

  _, max_y_f = f.get_ylim()
  max_y = max_y_f if max_y_f > max_y else max_y
  f.set(ylim=(0, max_y))

  eixo += 1

figura.show()


#  - Valor de Transações nos Últimos 12 Meses x Quantidade de Transações nos Últimos 12 Meses

# In[ ]:


coluna = 'tipo_cartao'
titulos = ['Tipo de Cartão dos Clientes', 'Tipo de Cartão dos Clientes Adimplentes', 'Tipo de Cartão dos Clientes Inadimplentes']

eixo = 0
max_y = 0
figura, eixos = plt.subplots(1,3, figsize=(20, 5), sharex=True)

for dataframe in [df, df_adimplente, df_inadimplente]:

  f = sns.histplot(x=coluna, data=dataframe, stat='count', ax=eixos[eixo])
  f.set(title=titulos[eixo], xlabel=coluna.capitalize(), ylabel='Frequência Absoluta')

  _, max_y_f = f.get_ylim()
  max_y = max_y_f if max_y_f > max_y else max_y
  f.set(ylim=(0, max_y))

  eixo += 1

figura.show()


# In[ ]:


f = sns.relplot(x='valor_transacoes_12m', y='qtd_transacoes_12m', data=df, hue='default')
_ = f.set(
    title='Relação entre Valor e Quantidade de Transações no Último Ano', 
    xlabel='Valor das Transações no Último Ano', 
    ylabel='Quantidade das Transações no Último Ano'
  )


# In[ ]:


f = sns.relplot(x='tipo_cartao', y='valor_transacoes_12m', data=df, hue='default')
_ = f.set(
    title='Relação entre Valor e Tipo do cartão do cliente', 
    xlabel='Tipo do cartão do cliente', 
    ylabel='Valor das Transações no Último Ano'
  )


# ## 5\. Storytelling

# Uma primeira impressão que podemos tirar de nossas 10.127 linhas do banco de dados de forma bruta é que 8.500 são referentes a clientes adimplentes o que corresponde a 83,93% dos clientes. Por outro lado temos 1.627 linhas de dados representando os clientes inadimplentes, o que corresponde a 16,07% dos clientes, mostrando uma desproporcionalidade entre os números, o que não deixa de ser uma coisa boa no caso em questão, pois nos mostram quais seriam nossos clientes preferenciais na hora de concedermos crédito. Após a limpeza do banco e readequação da estrutura nós obtivemos novos valores e estes ficaram bem próximos dos números anteriores, reforçando nossa assertiva. Os novos números são de 7.081 linhas totais analisadas  e destes, 5.968 são referentes aos clientes adimplentes, o que corresponde a 84,28% do total de clientes e temos 1.113 linhas de dados que representam os clientes inadimplentes, o que corresponde a um total de 15,72%. Um dados importante vem sobre o tipo do cartão usados pelos clientes. Estes dados nos mostram que as pessoas  com cartão do tipo platium tem menos probabilidade de se tornarem inadimplentes, já os clientes com os cartões do tipo blue e silver com valores de transação entre R$0,00(Zero) e R$10.000,00(Dez mil reais) precisam ser acompanhados mais de perto e os clientes com cartão gold  com valores de transação entre R$7.000,00(Sete mil reais) e R$9.500,00(Nove mil e quinhentos reais) precisam de mais atenção Podemos ver os números que indicam a quantidade de transações no ultimo ano nos indicam que as pessoas que utilizaram o cartão de 30 a 50 vezes por ano estão nas pessoas com maior risco de se tornarem inadimplentes, portando precisam ser acompanhadas mais de perto. Uma outra variável que nos trás uma relação interessante é a sobre os valores das transações feitas durante o ultimo ano  nos mostra que os clientes que gastaram entre R$1.500,00(Um mil e quinhentos reais) e R$3.000(Três mil reais) estão mais sujeitos a se tornarem inadimplentes, nos remetendo novamente a necessidade de um acompanhamento mais constante, já pessoas que utilizaram o cartão entre 55 e 90 vezes com um gasto entre R$7.000,00(Sete mil) e R$11.000,00(Onze mil) também tem maior probabilidade de se tornarem inadimplentes precisando de um acompanhamento mais direcionado. Essas são algumas informações que podemos tirar depois de observarmos os dados apresentados.

# ____
