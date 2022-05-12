#!/usr/bin/env python
# coding: utf-8

# I struggled with the following APIs so you don't have to:
# 
# - [**IMF API (Inflation, Trade Balance)**](#IMF) 
# 
# A bit difficult to use. It seems that most IMF data is on db-nomics (see below). I might switch to it.
# 
# - [**Japanese Ministry of Finance (Interest Rates, Government Debt)**](#MOF)
# 
# Not so difficult once I found the direct download links.
# 
# - [**db-nomics - JSTAT (Unemployment)**](#JSTAT)
# 
# Relatively easy to use.
# 
# - [**E-stat adaptator - (National Statistics)**](#ESTAT)
# 
# Lots of available data but not everything seems to works.
# 
# - [**Japan COVID-19 Bulletin Board - Covid 19 Data HUB**](#COVID)
# 
# Very easy to use python API. Basically just need a country name.
# 
# I also looked at the National Central Bank (Bank of Japan), but couldn't really find anything. Might look into weather data too.

# # Imports
# requests to get data, the rest is to manipulate and plot data.

# In[ ]:


import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# competition dates

# In[ ]:


train_start = '2017-01-04'
train_end = '2021-12-03'
eval_start = '2022-07-05'
eval_end = '2022-10-07'


# <a id='IMF'></a>
# # International Monetary Fund - API
# 
# Some of the code adapted to this competition from: https://www.bd-econ.com/imfapi1.html
# 
# Databases names are required to get data. Here is an overwiew of what is available:
# 
# https://data.imf.org/?sk=388DFA60-1D26-4ADE-B505-A05A558D9A42&sId=1479329132316

# There is a search engine, but it is not always easy to use. 
# For exemple if we start with inflation we have to know this is the differentiation of Consummer Price Index and look for it instead...

# In[ ]:


# parameters
url = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/'
key = 'Dataflow'  # Method with series information
search_term = 'Consumer'  # Term to find in series names

# request
series_list = requests.get(f'{url}{key}').json()['Structure']['Dataflows']['Dataflow']

# Use dict keys to navigate through results:
for series in series_list:
    if search_term in series['Name']['#text']:
        print(f"{series['Name']['#text']}: {series['KeyFamilyRef']['KeyFamilyID']}")


# Learn how to request data set: with a dataset key we can look the way to request it:

# In[ ]:


Data_set = 'CPI'

key = 'DataStructure/' + Data_set  # Method / series
dimension_list = requests.get(f'{url}{key}').json()            ['Structure']['KeyFamilies']['KeyFamily']            ['Components']['Dimension']

for n, dimension in enumerate(dimension_list):
    print(f'Dimension {n+1}: {dimension["@codelist"]}')


# There is three dimensions to request the data set:
# 
# Freq is usually 'M, 'Q' or 'Y', as we want granular data we will use M.
# 
# Area is the area of interest ('JP' here)
# 
# We need to look further the 3rd dimension to know which serie to request.

# In[ ]:


# Example: codes for third dimension, which is 2 in python
key = f"CodeList/{dimension_list[2]['@codelist']}"
code_list = requests.get(f'{url}{key}').json()['Structure']['CodeLists']['CodeList']['Code']

for code in code_list[:10]:
    print(f"{code['Description']['#text']}: {code['@value']}")
    
print(f'Plus {len(code_list)-10} other series')


# As we are interested in inflation we start with the general price indice: PCPI_IX.

# In[ ]:


Data_set = 'CPI'
freq = 'M'
area = 'JP'
name = 'Consumer Price Index, All items'
indice = 'PCPI_IX'

key = 'CompactData/'+Data_set+'/'+freq+'.'+area+'.'+indice

data = (requests.get(f'{url}{key}').json()['CompactData']['DataSet']['Series'])

baseyr = data['@COMMON_REFERENCE_PERIOD'] 

data_list = [[obs.get('@TIME_PERIOD'), obs.get('@OBS_VALUE')] for obs in data['Obs']]
df = pd.DataFrame(data_list, columns=['date', indice])
df = df.set_index(pd.to_datetime(df['date']))[indice].astype('float')


# Basic plot - CPI

# In[ ]:


title = f'{area} {name} (index, {baseyr})'
source = f'Source: IMF {Data_set}'
plot = df.plot(title=title, colormap='Set1')


# I gathered a list of interesting data; but only two first sources seems to work from the IMF API. 
# It seems that dbnomics also offer data from IMF. Might be a better solution (see below).

# In[ ]:


# BOP = {
#     'Assets - Euros': ('BOP','M','JP','IAFR_BP6_EUR'),
#     'Assets - National Currency': ('BOP','M','JP','IAFR_BP6_XDC'),
#     'Assets - US Dollars': ('BOP','M','JP','IAFR_BP6_USD')
# }

# DOT = {
#     'Exports, US Dollars': ('DOT','M','JP','TXG_FOB_USD','W00'),
#     'Imports, Cost, Insurance, Freight': ('DOT','M','JP','TMG_CIF_USD','W00'),
#     'Imports, US Dollars': ('DOT','M','JP','TMG_FOB_USD','W00'),
#     'Trade Balance, US Dollars': ('DOT','M','JP','TBG_USD','W00')
# }

# FM ={
#     'Cyclically adjusted balance (% of potential GDP)': ('FM','M','JP','GGCB_G01_PGDP_PT'),
#     'Cyclically adjusted primary balance (% of potential GDP)': ('FM','M','JP','GGCBP_G01_PGDP_PT'),
#     'Expenditure (% of GDP)': ('FM','M','JP','G_X_G01_GDP_PT'),
#     'Gross debt (% of GDP)':('FM','M','JP','G_XWDG_G01_GDP_PT'),
#     'Net debt (% of GDP)': ('FM','M','JP', 'GGXWDN_G01_GDP_PT'),
#     'Net lending/borrowing (also referred as overall balance) (% of GDP)': ('FM','M','JP','GGXCNL_G01_GDP_PT '),
#     'Primary net lending/borrowing (also referred as primary balance) (% of GDP)': ('FM','M','JP','GGXONLB_G01_GDP_PT'),
#     'Revenue (% of GDP)':  ('FM','M','JP','GGR_G01_GDP_PT')
# }

# IFS ={
#     'National Accounts, National Currency - Nominal': ('FM','M','JP','NFIAXD_XDC'),
#     'National Accounts, National Currency - Real': ('FM','M','JP','NFIAXD_R_XDC'),
#     'Net acquisition of financial assets, US Dollars': ('FM','M','JP','IAFR_BP6_USD'),
#     'Debt instruments, US Dollars': ('FM','M','JP','IADD_BP6_USD'),
#     'Equity and investment fund shares , US Dollars': ('FM','M','JP','IADE_BP6_USD'),
#     'Assets, Direct investment, US Dollars': ('FM','M','JP','IAD_BP6_USD')
# }

# APDREO = {'Balance of Payments, Current Account, Total, Net(BPM6), percent of GDP in U.S. dollars': 'BCA_GDP_BP6',
#   'Consumer Prices, end of period, percent change': 'PCPIE_PCH',
#   'Consumer Prices, period average, percent change': 'PCPI_PCH',
#   'General government net lending/borrowing, percent of fiscal year GDP': 'GGXCNL_GDP',
#   'Gross domestic product, constant prices, National Currency, percent change': 'NGDP_RPCH',
#   'Gross domestic product, constant prices, purchasing power parity, per capita, percent change': 'NGDP_R_PPP_PC_PCH',
#   'Unemployment rate': 'LUR'}

# HPDD = {'Public Balance': 'GGXWDG_GDP'}

# FAS = {'Total Population, Female': 'LPAF_NUM',
#   'Total Population, Male': 'LPAM_NUM',
#   'Claims on nonfinancial corporations and households, Domestic Currency': 'FODMANCH_XDC',
#   'Claims on nonfinancial corporations and households, Euros': 'FODMANCH_EUR',
#   'Claims on nonfinancial corporations and households, US Dollars': 'FODMANCH_USD',
#   'Geographical Outreach, Mobile Money, Number of active mobile money agent outlets': 'FCMOA_NUM',
#   'Geographical Outreach, Mobile Money, Number of registered mobile money agent outlets': 'FCMOR_NUM',
#   'Geographical Outreach, Number of Automated Teller Machines (ATMs), Country wide, Number of': 'FCAC_NUM',
#   'Geographical Outreach, Number of Branches, Excluding Headquarters, Other Depository Corporations, Commercial banks, Number of': 'FCBODC_NUM'}

# COFER = {'Foreign Exchange, US Dollars': 'RAXGFX_USD',
#   'Allocated Reserves, US Dollars': 'RAXGFXAR_USD'}

# FSI = {'Domestic government securities owned (market value), National Currency': 'FS_ODX_GSD_MV_XDC',
#   'Gross loans to the public sector, National Currency': 'FS_ODX_AFLG_PS_XDC',
#   'Gross new deposits during the period, National Currency': 'FS_ODX_AFCDGN_XDC'}


# Inflation is the difference in price index. Here we use a 12-months rolling difference.

# In[ ]:


Data_set = 'CPI'
freq = 'M'
area = 'JP'
name = 'Consumer Price Index, All items'
indice = 'PCPI_IX'

key = 'CompactData/'+Data_set+'/'+freq+'.'+area+'.'+indice

data = (requests.get(f'{url}{key}').json()['CompactData']['DataSet']['Series'])

baseyr = data['@COMMON_REFERENCE_PERIOD'] 

data_list = [[obs.get('@TIME_PERIOD'), obs.get('@OBS_VALUE')] for obs in data['Obs']]
df = pd.DataFrame(data_list, columns=['date', indice])
df = df.set_index(pd.to_datetime(df['date']))[indice].astype('float')

df.to_csv(f'{area}_{Data_set}_{indice}.csv', header=True)

plt.plot(df.diff(12));
plt.title(f'YoY Inflation - {source}');
plt.show()

inflation_recent = df.diff(12)[df.index>'2010-01-01']

plt.plot(inflation_recent);
plt.title(f'YoY Inflation - {source}');

plt.axvspan(train_start, train_end, color="grey", alpha=0.25)
plt.axvspan(eval_start, eval_end, color="grey", alpha=0.5)
plt.hlines(0,xmin=inflation_recent.index.min(),xmax=pd.to_datetime(eval_end),color='k',linestyles='dotted');
plt.show()


# Trade Balance

# In[ ]:


Data_set = 'DOT'
freq = 'M'
area = 'JP'
name = 'test'
indice = 'TBG_USD'

key = 'CompactData/'+Data_set+'/'+freq+'.'+area+'.'+indice+'.W00'

data = (requests.get(f'{url}{key}').json()['CompactData']['DataSet']['Series'])

#baseyr = data['@COMMON_REFERENCE_PERIOD'] 

data_list = [[obs.get('@TIME_PERIOD'), obs.get('@OBS_VALUE')] for obs in data['Obs']]
df = pd.DataFrame(data_list, columns=['date', indice])
df = df.set_index(pd.to_datetime(df['date']))[indice].astype('float')

df.to_csv(f'{area}_{Data_set}_{indice}.csv', header=True)

plt.plot(df.rolling(12).mean());
plt.title(f'Trade Balance - IMF {Data_set}');
plt.show()

df_recent = df.rolling(12).mean()[df.index>'2010-01-01']

plt.plot(df_recent);
plt.title(f'Trade Balance - IMF {Data_set}');

plt.axvspan(train_start, train_end, color="grey", alpha=0.25)
plt.axvspan(eval_start, eval_end, color="grey", alpha=0.5)
plt.hlines(0,xmin=df_recent.index.min(),xmax=pd.to_datetime(eval_end),color='k',linestyles='dotted');
plt.show()


# <a id='MOF'></a>
# 
# # Japanese ministry of finance - Interest Rates

# In[ ]:


japan_hist = pd.read_csv("https://www.mof.go.jp/english/jgbs/reference/interest_rate/historical/jgbcme_all.csv", header=1, parse_dates=['Date'])
japan_hist = japan_hist.set_index('Date').replace('-',np.nan).astype('float')


# In[ ]:


plt.plot(japan_hist[['1Y','5Y','10Y','25Y']]);


# In[ ]:


japan_hist_recent = japan_hist[japan_hist.index>'2010-01-01']

horizons = ['1Y','5Y','10Y','25Y']

plt.plot(japan_hist_recent[horizons]);
plt.title(f'Interest Rates - Japan MoF');
plt.legend(horizons);

plt.axvspan(train_start, train_end, color="grey", alpha=0.25);
plt.axvspan(eval_start, eval_end, color="grey", alpha=0.5);
plt.hlines(0,xmin=df_recent.index.min(),xmax=pd.to_datetime(eval_end),color='k',linestyles='dotted');
plt.show()


# In[ ]:


cols = japan_hist.columns[japan_hist.columns.str.endswith('Y')]

horizons = cols.str.replace('Y','').astype('int').values
values = japan_hist[cols].iloc[-1].astype('float')

plt.scatter(horizons,values);


# # Government Debt

# In[ ]:


get_ipython().system('pip install xlrd')


# In[ ]:


df = pd.read_excel("https://www.mof.go.jp/english/jgbs/reference/gbb/suii.xls", header=2)

Total = (df[df.Category=='Total'].values)[0][3:]
Data = pd.DataFrame(Total,index=pd.date_range('2017-03','2022-01',freq='Q'),columns=['Total Government Debt'])

plt.plot(Data);
plt.title(f'Government Debt - Japan MoF');

plt.axvspan(train_start, train_end, color="grey", alpha=0.25);
plt.axvspan(eval_start, eval_end, color="grey", alpha=0.5);
plt.show()


# <a id='JSTAT'></a>
# 
# # db nomics - employment rate

# In[ ]:


get_ipython().system('pip install dbnomics')

import dbnomics


# In[ ]:


df = dbnomics.fetch_series('STATJP', 'MIm')
df.series_name.unique()


# In[ ]:


df = df[df.series_name == 'Monthly – Unemployment rate – Both sexes – Percent – seasonally adjusted'][['period','value']].set_index('period')

plt.plot(df);
plt.title(f'Unemployment rate - STAT JP');


# In[ ]:


plt.plot(df[df.index>'2010-01-01']);

plt.axvspan(train_start, train_end, color="grey", alpha=0.25);
plt.axvspan(eval_start, eval_end, color="grey", alpha=0.5);
plt.show()


# <a id='ESTAT'></a>
# 
# # E-stat adaptator - national statistics

# code from https://github.com/e-stat-api/adaptor

# In[ ]:


get_ipython().system('pip install xlrd')


# In[ ]:


import urllib
import requests
import csv
import json
import xlrd
import zipfile
import requests
import functools
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import io
import os
from pprint import pprint


# Complete parser (hidden code).

# In[ ]:




class EstatRestAPI_URLParser:
    """
    This is a simple python module class for e-Stat API (ver.3.0).
    See more details at https://www.e-stat.go.jp/api/api-info/e-stat-manual3-0
    """

    def __init__(self, api_version=None, app_id=None):
        # base url
        self.base_url = "https://api.e-stat.go.jp/rest"

        # e-Stat REST API Version
        if api_version is None:
            self.api_version = "3.0"
        else:
            self.api_version = api_version

        # Application ID
        if app_id is None:
            self.app_id = "****************" #Enter the application ID here
        else:
            self.app_id = app_id

    def getStatsListURL(self, params_dict, format="csv"):
        """
        2.1 Get statistical table information(HTTP GET)
        """
        params_str = urllib.parse.urlencode(params_dict)
        if format == "xml":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/getStatsList?{params_str}"
            )
        elif format == "json":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/json/getStatsList?{params_str}"
            )
        elif format == "jsonp":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/jsonp/getStatsList?{params_str}"
            )
        elif format == "csv":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/getSimpleStatsList?{params_str}"
            )
        return url

    def getMetaInfoURL(self, params_dict, format="csv"):
        """
        2.2 Meta information acquisition(HTTP GET)
        """
        params_str = urllib.parse.urlencode(params_dict)
        if format == "xml":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/getMetaInfo?{params_str}"
            )
        elif format == "json":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/json/getMetaInfo?{params_str}"
            )
        elif format == "jsonp":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/jsonp/getMetaInfo?{params_str}"
            )
        elif format == "csv":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/getSimpleMetaInfo?{params_str}"
            )
        return url

    def getStatsDataURL(self, params_dict, format="csv"):
        """
        2.3 Statistical data acquisition(HTTP GET)
        """
        params_str = urllib.parse.urlencode(params_dict)
        if format == "xml":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/getStatsData?{params_str}"
            )
        elif format == "json":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/json/getStatsData?{params_str}"
            )
        elif format == "jsonp":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/jsonp/getStatsData?{params_str}"
            )
        elif format == "csv":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/getSimpleStatsData?{params_str}"
            )
        return url

    def postDatasetURL(self):
        """
        2.4 Data set registration(HTTP POST)
        """
        url = (
            f"{self.base_url}/{self.api_version}"
            "/app/postDataset"
        )
        return url

    def refDataset(self, params_dict, format="xml"):
        """
        2.5 Dataset reference(HTTP GET)
        """
        params_str = urllib.parse.urlencode(params_dict)
        if format == "xml":
            url = (
                f"{self.base_url}/{self.api_version}"
                + f"/app/refDataset?{params_str}"
            )
        elif format == "json":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/json/refDataset?{params_str}"
            )
        elif format == "jsonp":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/jsonp/refDataset?{params_str}"
            )
        return url

    def getDataCatalogURL(self, params_dict, format="xml"):
        """
        2.6 Data catalog information acquisition(HTTP GET)
        """
        params_str = urllib.parse.urlencode(params_dict)
        if format == "xml":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/getDataCatalog?{params_str}"
            )
        elif format == "json":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/json/getDataCatalog?{params_str}"
            )
        elif format == "jsonp":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/jsonp/getDataCatalog?{params_str}"
            )
        return url

    def getStatsDatasURL(self, params_dict, format="xml"):
        """
        2.7 Collective statistical data(HTTP GET)
        """
        params_str = urllib.parse.urlencode(params_dict)
        if format == "xml":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/getStatsDatas?{params_str}"
            )
        elif format == "json":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/json/getStatsDatas?{params_str}"
            )
        elif format == "csv":
            url = (
                f"{self.base_url}/{self.api_version}"
                f"/app/getSimpleStatsDatas?{params_str}"
            )
        return url


def get_json(url):
    """
    Request a HTTP GET method to the given url (for REST API)
    and return its response as the dict object.

    Args:
    ====
    url: string
        valid url for REST API
    """
    try:
        print("HTTP GET", url)
        r = requests.get(url)
        json_dict = r.json()
        return json_dict
    except requests.exceptions.RequestException as error:    
        print(error)


def download_json(url, filepath):
    """
    Request a HTTP GET method to the given url (for REST API)
    and save its response as the json file.

    Args:
    url: string
        valid url for REST API
    filepath: string
        valid path to the destination file
    """
    try:
        print("HTTP GET", url)
        r = requests.get(url)
        json_dict = r.json()
        json_str = json.dumps(json_dict, indent=2, ensure_ascii=False)
        with open(filepath, "w") as f:
            f.write(json_str)
    except requests.exceptions.RequestException as error:
        print(error)


def download_csv(url, filepath, enc="utf-8", dec="utf-8", logging=False):
    """
    Request a HTTP GET method to the given url (for REST API)
    and save its response as the csv file.

    Args:
    =====
    url: string
        valid url for REST API
    filepathe: string
        valid path to the destination file
    enc: string
        encoding type for a content in a given url
    dec: string
        decoding type for a content in a downloaded file
            dec = 'utf-8' for general env
            dec = 'sjis'  for Excel on Win
            dec = 'cp932' for Excel with extended JP str on Win
    logging: True/False
        flag whether putting process log
    """
    try:
        if logging:
            print("HTTP GET", url)
        r = requests.get(url, stream=True)
        with open(filepath, 'w', encoding=enc) as f:
            f.write(r.content.decode(dec))
    except requests.exceptions.RequestException as error:
        print(error)


def download_all_csv(
        urls,
        filepathes,
        max_workers=10,
        enc="utf-8",
        dec="utf-8"):
    """
    Request some HTTP GET methods to the given urls (for REST API)
    and save each response as the csv file.
    (!! This method uses multi threading when calling HTTP GET requests
    and downloading files in order to improve the processing speed.)

    Args:
    =====
    urls: list of strings
        valid urls for REST API
    filepathes: list of strings
        valid pathes to the destination file
    max_workers: int
        max number of working threads of CPUs within executing this method.
    enc: string
        encoding type for a content in a given url
    dec: string
        decoding type for a content in a downloaded file
            dec = 'utf-8' for general env
            dec = 'sjis'  for Excel on Win
            dec = 'cp932' for Excel with extended JP str on Win
    logging: True/False
    """
    func = functools.partial(download_csv, enc=enc, dec=dec)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(executor.map(func, urls, filepathes), total=len(urls))
        )
        del results

def search_tables(appId, params_dict):
    """
    Prams (dictionary) to search eStat tables.
    For more details, see also
    https://www.e-stat.go.jp/api/api-info/e-stat-manual3-0#api_3_2

        - appId: Application ID (*required)
        - lang:language(J:Japanese, E:English)
        - surveyYears:Survey date(YYYYY or YYYYMM or YYYYMM-YYYYMM)
        - openYears:Same as the survey date
        - statsField:Statistics field(2 digits:Statistical classification,4 digits:Statistical subclass)
        - statsCode:Government statistics code(8-digit)
        - searchWord:Search keyword
        - searchKind:Data type(1:Statistics, 2:Subregion / regional mesh)     
        - collectArea:Aggregate area classification(1:Nationwide, 2:Prefectures, 3:Municipality)        
        - explanationGetFlg:Existence of commentary information(Y or N)
        - ...
    """

    url = estatapi_url_parser.getStatsListURL(params_dict, format="json")   
    json_dict = get_json(url)
    # pprint(json_dict)

    if json_dict['GET_STATS_LIST']['DATALIST_INF']['NUMBER'] != 0:
        tables = json_dict["GET_STATS_LIST"]["DATALIST_INF"]["TABLE_INF"]
    else:
        tables = []
    return tables


def parse_table_id(table):
    return table["@id"]


def parse_table_raw_size(table):
    return table["OVERALL_TOTAL_NUMBER"]


def parse_table_urls(table_id, table_raw_size, csv_raw_size=100000):
    urls = []
    for j in range(0, int(table_raw_size / csv_raw_size) + 1):
        start_pos = j * csv_raw_size + 1
        params_dict = {
            "appId": appId,  # Application ID
            "lang": "E",  #language(J:Japanese, E:English)
            "statsDataId": str(table_id),  #Statistical table ID
            "startPosition": start_pos,  #Start line
            "limit": csv_raw_size,  #Number of data acquisitions
            "explanationGetFlg": "N",  #Existence of commentary information(Y or N)
            "annotationGetFlg": "N",  #Presence or absence of annotation information(Y or N)
            "metaGetFlg": "N",  #Presence or absence of meta information(Y or N)
            "sectionHeaderFlg": "2",  #CSV header flag(1:Get, 2:Get無)
        }
        url = estatapi_url_parser.getStatsDataURL(params_dict, format="csv")
        urls.append(url)
    return urls


# From the e-stat website we have these code for the main datatsets:

# In[ ]:


dict_map = {
'00100401':'Machinery Orders Detail ',
'00100406':'Indexes of Business Conditions Detail ',
'00100409':'National Accounts Detail ',
'00130002':'Statistics about Road Traffic Detail ',
'00200502':'System of Social and Demographic Statistics Detail ',
'00200521':'Population Census Detail ',
'00200522':'Housing and Land Survey Detail ',
'00200523':'Report on Internal Migration in Japan Detail ',
'00200524':'Population Estimates Detail ',
'00200531':'Labour Force Survey Detail ',
'00200532':'Employment Status Survey Detail ',
'00200533':'Survey on Time Use and Leisure Activities Detail ',
'00200541':'Unincorporated Enterprise Survey Detail ',
'00200544':'Monthly Survey on Service Industries Detail ',
'00200545':'Survey on Service Industries Detail ',
'00200551':'Establishment and Enterprise Census Detail ',
'00200552':'Economic Census for Business Frame Detail ',
'00200553':'Economic Census for Business Activity Detail ',
'00200555':'Annual Business Survey Detail ',
'00200561':'Family Income and Expenditure Survey Detail ',
'00200564':'National Survey of Family Income, Consumption and Wealth Detail ',
'00200566':'National Income and Expenditure Survey for one-person households Detail ',
'00200567':'Consumption Trend Index Detail ',
'00200571':'Retail price survey Detail ',
'00200572':'National Survey of Prices Detail ',
'00200573':'Consumer Price Index Detail ',
'00200603':'Input-Output Tables for Japan Detail ',
'00200604':'The Special Data Dissemination Standard Plus Detail ',
'00350300':'Trade Statistics Detail ',
'00350310':'Other Trade Related Statistics Detail ',
'00350320':'Statistics on Arrival of Aircraft & Entrance of Vessels Detail ',
'00400601':'Science Information Infrastructure Statistics of Colleges and Universities Detail ',
'00450011':'Vital Statistics Detail ',
'00450013':'Specified Report of Vital Statistics Detail ',
'00450432':'Annual Population and Social Security Surveys (The National Survey on Migration) Detail ',
'00500209':'Census of Agriculture and Forestry Detail ',
'00500210':'Census of Fisheries Detail ',
'00550035':'Current Survey of Mass Merchandise Specialty Retailers Detail ',
'00550560':'Production by Kind of Iron and Steel Detail ',
'00550710':'Spot LNG Price Statistics Detail ',
'00551020':'Current Survey of Supply and Demand for Petroleum Products Detail ',
'00600120':'Building Starts Detail ',
'00600130':'Statistics on Construction Works Detail ',
'00600260':'Quick Estimate of Construction Investment Detail ',
'00600470':'Corporations Survey on Land and Buildings Detail ',
'00600475':'Household Survey on Land Detail ',
'00600480':'Corporations Survey on Buildings Detail ',
'00600870':'Estimate of Construction Investment'}


# In[ ]:


CSV_RAW_SIZE = 100000

appId = "481173f9e1eef4a259ebdd575636d389451fa5bf" #Enter the application ID here
estatapi_url_parser = EstatRestAPI_URLParser()  # URL Parser

params_dict = {
    "appId": appId,
    "lang": "E",
    "statsCode": '00100406',
}

# list of tables
tables = search_tables(appId,params_dict)

if len(tables) == 0:
    print("No tables were found.")
elif len(tables) == 1:
    table_ids = [parse_table_id(tables[0])]
else:
    table_ids = list(map(parse_table_id, tables))
    
table_ids


# In[ ]:


# list of urls
table_urls = []
table_raw_size = list(map(parse_table_raw_size, tables))
for i, table_id in enumerate(table_ids):
    table_urls = table_urls + parse_table_urls(table_id, table_raw_size[i])


# list of filepathes
filepathes = []
for i, table_id in enumerate(table_ids):
    table_name = tables[i]["TITLE_SPEC"]["TABLE_NAME"]
    table_dir = f"./downloads/tmp/{table_name}_{table_id}"
    os.makedirs(table_dir, exist_ok=True)
    for j in range(0, int(table_raw_size[i] / CSV_RAW_SIZE) + 1):
        filepath = f"{table_dir}/{table_name}_{table_id}_{j}.csv"
        filepathes.append(filepath)


# In[ ]:


urlData = requests.get(table_urls[0]).content
rawData = pd.read_csv(io.StringIO(urlData.decode('utf-8')))

rawData.index = pd.to_datetime(rawData['time_code'].astype('str').str[:4]+'/'+rawData['time_code'].astype('str').str[-2:])
df = rawData[(rawData['tab_code']==100)&(rawData['cat01_code']==100)&(rawData['unit']=='2015=100')]

plt.plot(df.value);
plt.title(f'Indexes of Business Conditions Detail - e-stat');


# In[ ]:


plt.plot(df[df.index>'2010-01-01'].value);
plt.title(f'Indexes of Business Conditions Detail - e-stat');

plt.axvspan(train_start, train_end, color="grey", alpha=0.25);
plt.axvspan(eval_start, eval_end, color="grey", alpha=0.5);
plt.show()


# <a id='COVID'></a>
# # Japan COVID-19 Bulletin Board - COVID-19 Data Hub
# 
# usage of a very intuitive API from Guidotti, E., Ardia, D., (2020), "COVID-19 Data Hub", Journal of Open Source Software 5(51):2376, doi: 10.21105/joss.02376.

# In[ ]:


get_ipython().system('pip install covid19dh')

from covid19dh import covid19
import matplotlib.pyplot as plt


# In[ ]:


x, src = covid19("Japan")
x = x.set_index('date')


# In[ ]:


plt.plot(x['deaths'].diff().rolling(7).mean());
plt.title(f'Covid Deaths - COVID-19 Data Hub');
plt.axvspan(train_start, train_end, color="grey", alpha=0.25);
plt.axvspan(eval_start, eval_end, color="grey", alpha=0.5);
plt.show()


# In[ ]:


plt.plot(x['workplace_closing']);

plt.title(f'workplace_closing - COVID-19 Data Hub');
plt.axvspan(train_start, train_end, color="grey", alpha=0.25);
plt.axvspan(eval_start, eval_end, color="grey", alpha=0.5);
plt.show()

