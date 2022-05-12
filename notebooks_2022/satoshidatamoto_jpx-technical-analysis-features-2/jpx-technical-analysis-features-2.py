#!/usr/bin/env python
# coding: utf-8

# >### Technical Analysis Indicators #2
# >- Here are some simple indexes to analyze the charts. some can even be used as features to a model.
# >- Ta-lib is very good and very helpful library for calculating various indexes, but kernel doesn't support.
# >- Enjoy the short scripts to obtain them! 
# >
# 
# [1]: TBD

# #### Code starts here ⬇

# In[ ]:


import os
import gc
import traceback
import numpy as np
import pandas as pd
import datatable as dt
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.core.common.SettingWithCopyWarning)
    
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = [14, 8]  # width, height


# # <span class="title-section w3-xxlarge" id="loading">Data Loading 🗃️</span>
# <hr>
# 
# In the real competition data, the number of datapoints per day (that is per "group") is not constant as it was in the spoofed data. We need to confirm that the time series split respects that there are different counts of samples in the the days. We load the data and reduce memory footprint.

# In[ ]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# # <span class="title-section w3-xxlarge" id="loading">Data Loading 🗃️</span>
# <hr>
# 
# The data organisation has already been done and saved to Kaggle datasets. Here we choose which years to load. We can use either 2017, 2018, 2019, 2020, 2021, Original, Supplement by changing the `INC2021`, `INC2020`, `INC2019`, `INC2018`, `INC2017`, `INCCOMP`, `INCSUPP` variables in the preceeding code section. These datasets are discussed [here][1].
# 
# [1]: https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/285726
# 

# In[ ]:


stock_list = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
stock_list = stock_list.loc[stock_list['SecuritiesCode'].isin(prices['SecuritiesCode'].unique())]
stock_name_dict = {stock_list['SecuritiesCode'].tolist()[idx]: stock_list['Name'].tolist()[idx] for idx in range(len(stock_list))}

def load_training_data(asset_id = None):
    prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
    supplemental_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
    df_train = pd.concat([prices, supplemental_prices]) if INCSUPP else prices
    df_train = pd.merge(df_train, stock_list[['SecuritiesCode', 'Name']], left_on = 'SecuritiesCode', right_on = 'SecuritiesCode', how = 'left')
    df_train['date'] = pd.to_datetime(df_train['Date'])
    df_train['year'] = df_train['date'].dt.year
    if not INC2022: df_train = df_train.loc[df_train['year'] != 2022]
    if not INC2021: df_train = df_train.loc[df_train['year'] != 2021]
    if not INC2020: df_train = df_train.loc[df_train['year'] != 2020]
    if not INC2019: df_train = df_train.loc[df_train['year'] != 2019]
    if not INC2018: df_train = df_train.loc[df_train['year'] != 2018]
    if not INC2017: df_train = df_train.loc[df_train['year'] != 2017]
    # asset_id = 1301 # Remove before flight
    if asset_id is not None: df_train = df_train.loc[df_train['SecuritiesCode'] == asset_id]
    # df_train = df_train[:1000] # Remove before flight
    return df_train


# In[ ]:


# WHICH YEARS TO INCLUDE? YES=1 NO=0
INC2022 = 1
INC2021 = 1
INC2020 = 1
INC2019 = 1
INC2018 = 1
INC2017 = 1
INCSUPP = 1
print("Loaded all data!")

train = load_training_data().sort_values('date').set_index("date")
train_data = train.copy()
train_data['date'] = pd.to_datetime(train_data['Date'])
df = train_data.loc[train_data['SecuritiesCode'] == 1301]


# # <span class="title-section w3-xxlarge" id="features">Feature Engineering 🔬</span>
# <hr>

# In[ ]:




import numpy
import pandas as pd


def rolling_mean(series, window): return series.rolling(window).mean()
def rolling_std(series, window): return series.rolling(window).std()
def rolling_sum(series, window): return series.rolling(window).sum()
def ewma(series, span, min_periods): return series.ewm(span = span, min_periods = min_periods).mean()
def get_value(df, idx, col): return df.iloc[idx][col]

#Moving Average
def MA(df, n):
    MA = pd.Series(rolling_mean(df['Close'], n), name = 'MA_' + str(n))
    df['MA'] = MA
    return df

#Exponential Moving Average
def EMA(df, n):
    EMA = pd.Series(ewma(df['Close'], span = n, min_periods = n - 1), name = 'EMA_' + str(n))
    df['EMA'] = EMA
    return df

#Momentum
def MOM(df, n):
    M = pd.Series(df['Close'].diff(n), name = 'Momentum_' + str(n))
    df['MOM'] = M
    return df

#Rate of Change
def ROC(df, n):
    M = df['Close'].diff(n - 1)
    N = df['Close'].shift(n - 1)
    ROC = pd.Series(M / N, name = 'ROC_' + str(n))
    df['ROC'] = ROC
    return df

#Average True Range
def ATR(df, n):
    i = 0
    TR_l = [0]
    while i < len(df) - 1:
        TR = max(get_value(df, i + 1, 'High'), get_value(df, i, 'Close')) - min(get_value(df, i + 1, 'Low'), get_value(df, i, 'Close'))
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(ewma(TR_s, span = n, min_periods = n), name = 'ATR_' + str(n))
    df['ATR'] = ATR
    return df

#Bollinger Bands
def BBANDS(df, n):
    MA = pd.Series(rolling_mean(df['Close'], n))
    MSD = pd.Series(rolling_std(df['Close'], n))
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name = 'BollingerB_' + str(n))
    df['B1'] = B1
    b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name = 'Bollinger%b_' + str(n))
    df['B2'] = B2
    return df

#Pivot Points, Supports and Resistances
def PPSR(df):
    PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)
    R1 = pd.Series(2 * PP - df['Low'])
    S1 = pd.Series(2 * PP - df['High'])
    R2 = pd.Series(PP + df['High'] - df['Low'])
    S2 = pd.Series(PP - df['High'] + df['Low'])
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}
    PSR = pd.DataFrame(psr)
    for col in PSR.columns:
        df['PSR_' + col] = PSR[col]
    return df

#Stochastic oscillator %K
def STOK(df):
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')
    df['SOk'] = SOk
    return df

# Stochastic Oscillator, EMA smoothing, nS = slowing (1 if no slowing)
def STO(df,  nK, nD, nS=1):
    SOk = pd.Series((df['Close'] - df['Low'].rolling(nK).min()) / (df['High'].rolling(nK).max() - df['Low'].rolling(nK).min()), name = 'SO%k'+str(nK))
    SOd = pd.Series(SOk.ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'SO%d'+str(nD))
    SOk = SOk.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()
    SOd = SOd.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()
    df['SOk'] = SOk
    df['SOd'] = SOd
    return df

#Trix
def TRIX(df, n):
    EX1 = ewma(df['Close'], span = n, min_periods = n - 1)
    EX2 = ewma(EX1, span = n, min_periods = n - 1)
    EX3 = ewma(EX2, span = n, min_periods = n - 1)
    i = 0
    ROC_l = [0]
    while i + 1 <= len(df) - 1:
        ROC = (EX3.iloc[i + 1] - EX3.iloc[i]) / EX3.iloc[i]
        ROC_l.append(ROC)
        i = i + 1
    Trix = pd.Series(ROC_l, name = 'Trix_' + str(n))
    df['Trix'] = Trix
    return df

#Average Directional Movement Index
def ADX(df, n, n_ADX):
    i = 0
    UpI = []
    DoI = []
    while i + 1 <= len(df) - 1:
        UpMove = get_value(df, i + 1, 'High') - get_value(df, i, 'High')
        DoMove = get_value(df, i, 'Low') - get_value(df, i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    i = 0
    TR_l = [0]
    while i < len(df) - 1:
        TR = max(get_value(df, i + 1, 'High'), get_value(df, i, 'Close')) - min(get_value(df, i + 1, 'Low'), get_value(df, i, 'Close'))
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(ewma(TR_s, span = n, min_periods = n))
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(ewma(UpI, span = n, min_periods = n - 1) / ATR)
    NegDI = pd.Series(ewma(DoI, span = n, min_periods = n - 1) / ATR)
    ADX = pd.Series(ewma(abs(PosDI - NegDI) / (PosDI + NegDI), span = n_ADX, min_periods = n_ADX - 1), name = 'ADX_' + str(n) + '_' + str(n_ADX))
    df['ADX'] = ADX
    return df

#MACD, MACD Signal and MACD difference
def MACD(df, n_fast, n_slow):
    EMAfast = pd.Series(ewma(df['Close'], span = n_fast, min_periods = n_slow - 1))
    EMAslow = pd.Series(ewma(df['Close'], span = n_slow, min_periods = n_slow - 1))
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(ewma(MACD, span = 9, min_periods = 8), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df['MACD'] = MACD
    df['MACDsign'] = MACDsign
    df['MACDdiff'] = MACDdiff
    return df

#Mass Index
def MassI(df):
    Range = df['High'] - df['Low']
    EX1 = ewma(Range, span = 9, min_periods = 8)
    EX2 = ewma(EX1, span = 9, min_periods = 8)
    Mass = EX1 / EX2
    MassI = pd.Series(rolling_sum(Mass, 25), name = 'Mass Index')
    df['MassI'] = MassI
    return df

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF
def Vortex(df, n):
    i = 0
    TR = [0]
    while i < len(df) - 1:
        Range = max(get_value(df, i + 1, 'High'), get_value(df, i, 'Close')) - min(get_value(df, i + 1, 'Low'), get_value(df, i, 'Close'))
        TR.append(Range)
        i = i + 1
    i = 0
    VM = [0]
    while i < len(df) - 1:
        Range = abs(get_value(df, i + 1, 'High') - get_value(df, i, 'Low')) - abs(get_value(df, i + 1, 'Low') - get_value(df, i, 'High'))
        VM.append(Range)
        i = i + 1
    VI = pd.Series(rolling_sum(pd.Series(VM), n) / rolling_sum(pd.Series(TR), n), name = 'Vortex_' + str(n))
    df['VI'] = VI
    return df

#KST Oscillator
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):
    M = df['Close'].diff(r1 - 1)
    N = df['Close'].shift(r1 - 1)
    ROC1 = M / N
    M = df['Close'].diff(r2 - 1)
    N = df['Close'].shift(r2 - 1)
    ROC2 = M / N
    M = df['Close'].diff(r3 - 1)
    N = df['Close'].shift(r3 - 1)
    ROC3 = M / N
    M = df['Close'].diff(r4 - 1)
    N = df['Close'].shift(r4 - 1)
    ROC4 = M / N
    KST = pd.Series(rolling_sum(ROC1, n1) + rolling_sum(ROC2, n2) * 2 + rolling_sum(ROC3, n3) * 3 + rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))
    df['KST'] = KST
    return df

#Relative Strength Index
def RSI(df, n):
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= len(df) - 1:
        UpMove = get_value(df, i + 1, 'High') - get_value(df, i, 'High')
        DoMove = get_value(df, i, 'Low') - get_value(df, i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(ewma(UpI, span = n, min_periods = n - 1))
    NegDI = pd.Series(ewma(DoI, span = n, min_periods = n - 1))
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))
    df['RSI'] = RSI
    return df

#True Strength Index
def TSI(df, r, s):
    M = pd.Series(df['Close'].diff(1))
    aM = abs(M)
    EMA1 = pd.Series(ewma(M, span = r, min_periods = r - 1))
    aEMA1 = pd.Series(ewma(aM, span = r, min_periods = r - 1))
    EMA2 = pd.Series(ewma(EMA1, span = s, min_periods = s - 1))
    aEMA2 = pd.Series(ewma(aEMA1, span = s, min_periods = s - 1))
    TSI = pd.Series(EMA2 / aEMA2, name = 'TSI_' + str(r) + '_' + str(s))
    df['TSI'] = TSI
    return df

#Accumulation/Distribution
def ACCDIST(df, n):
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC = M / N
    AD = pd.Series(ROC, name = 'Acc/Dist_ROC_' + str(n))
    df['AD'] = AD
    return df

#Chaikin Oscillator
def Chaikin(df):
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    Chaikin = pd.Series(ewma(ad, span = 3, min_periods = 2) - ewma(ad, span = 10, min_periods = 9), name = 'Chaikin')
    df['Chaikin'] = Chaikin
    return df

#Money Flow Index and Ratio
def MFI(df, n):
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    i = 0
    PosMF = [0]
    while i < len(df) - 1:
        if PP.iloc[i + 1] > PP.iloc[i]:
            PosMF.append(PP.iloc[i + 1] * get_value(df, i + 1, 'Volume'))
        else:
            PosMF.append(0)
        i = i + 1
    PosMF = pd.Series(PosMF)
    TotMF = PP * df['Volume']
    MFR = pd.Series(PosMF / TotMF)
    MFI = pd.Series(rolling_mean(MFR, n), name = 'MFI_' + str(n))
    df['MFI'] = MFI
    return df

#On-balance Volume
def OBV(df, n):
    i = 0
    OBV = [0]
    while i < len(df) - 1:
        if get_value(df, i + 1, 'Close') - get_value(df, i, 'Close') > 0:
            OBV.append(get_value(df, i + 1, 'Volume'))
        if get_value(df, i + 1, 'Close') - get_value(df, i, 'Close') == 0:
            OBV.append(0)
        if get_value(df, i + 1, 'Close') - get_value(df, i, 'Close') < 0:
            OBV.append(-get_value(df, i + 1, 'Volume'))
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(rolling_mean(OBV, n), name = 'OBV_' + str(n))
    df['OBV_ma'] = OBV_ma
    return df

#Force Index
def FORCE(df, n):
    F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name = 'Force_' + str(n))
    df['F'] = F
    return df

#Ease of Movement
def EOM(df, n):
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])
    Eom_ma = pd.Series(rolling_mean(EoM, n), name = 'EoM_' + str(n))
    df['Eom_ma'] = Eom_ma
    return df

#Commodity Channel Index
def CCI(df, n):
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = pd.Series((PP - rolling_mean(PP, n)) / rolling_std(PP, n), name = 'CCI_' + str(n))
    df['CCI'] = CCI
    return df

#Coppock Curve
def COPP(df, n):
    M = df['Close'].diff(int(n * 11 / 10) - 1)
    N = df['Close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df['Close'].diff(int(n * 14 / 10) - 1)
    N = df['Close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    Copp = pd.Series(ewma(ROC1 + ROC2, span = n, min_periods = n), name = 'Copp_' + str(n))
    df['Copp'] = Copp
    return df

#Keltner Channel
def KELCH(df, n):
    KelChM = pd.Series(rolling_mean((df['High'] + df['Low'] + df['Close']) / 3, n), name = 'KelChM_' + str(n))
    KelChU = pd.Series(rolling_mean((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3, n), name = 'KelChU_' + str(n))
    KelChD = pd.Series(rolling_mean((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3, n), name = 'KelChD_' + str(n))
    df['KelChM'] = KelChM
    df['KelChU'] = KelChU
    df['KelChD'] = KelChD
    return df

#Ultimate Oscillator
def ULTOSC(df):
    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < len(df) - 1:
        TR = max(get_value(df, i + 1, 'High'), get_value(df, i, 'Close')) - min(get_value(df, i + 1, 'Low'), get_value(df, i, 'Close'))
        TR_l.append(TR)
        BP = get_value(df, i + 1, 'Close') - min(get_value(df, i + 1, 'Low'), get_value(df, i, 'Close'))
        BP_l.append(BP)
        i = i + 1
    UltO = pd.Series((4 * rolling_sum(pd.Series(BP_l), 7) / rolling_sum(pd.Series(TR_l), 7)) + (2 * rolling_sum(pd.Series(BP_l), 14) / rolling_sum(pd.Series(TR_l), 14)) + (rolling_sum(pd.Series(BP_l), 28) / rolling_sum(pd.Series(TR_l), 28)), name = 'Ultimate_Osc')
    df['UltO'] = UltO
    return df

#Donchian Channel
def DONCH(df, n):
    i = 0
    DC_l = []
    while i < n - 1:
        DC_l.append(0)
        i = i + 1
    i = 0
    while i + n - 1 < len(df) - 1:
        DC = max(df['High'].iloc[i:i + n - 1]) - min(df['Low'].iloc[i:i + n - 1])
        DC_l.append(DC)
        i = i + 1
    DonCh = pd.Series(DC_l, name = 'Donchian_' + str(n))
    DonCh = DonCh.shift(n - 1)
    df['DonCh'] = DonCh
    return df

#Standard Deviation
def STDDEV(df, n):
    std_dev = pd.Series(rolling_std(df['Close'], n), name = 'STD_' + str(n))
    df['std_dev'] = std_dev
    return df





n = 15
nK = 5
nD = 7
n_fast = 5
n_slow = 10
n_ADX = 10
r = 3
s = 7

df = MA(df, n)
df = EMA(df, n)
df = MOM(df, n)
df = ROC(df, n)
df = ATR(df, n)
df = BBANDS(df, n)
df = PPSR(df)
df = STOK(df)
df = STO(df,  nK, nD, nS=1)
df = TRIX(df, n)
df = ADX(df, n, n_ADX)
df = MACD(df, n_fast, n_slow)
df = MassI(df)
df = Vortex(df, n)
df = RSI(df, n)
df = TSI(df, r, s)
df = ACCDIST(df, n)
df = Chaikin(df)
df = MFI(df, n)
df = OBV(df, n)
df = FORCE(df, n)
df = EOM(df, n)
df = CCI(df, n)
df = COPP(df, n)
df = KELCH(df, n)
df = ULTOSC(df)
df = DONCH(df, n)
df = STDDEV(df, n)


# # More to come..
