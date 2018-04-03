
import urllib.request
import time
import csv
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
import codecs
import os


def stock_data_download(stocknum, years, rawdatadir):
    for year in range(years, time.localtime().tm_year+1):
        for m in range(1, 13):
            if year == time.localtime().tm_year and m >= time.localtime().tm_mon+1:
                continue  # 超過當前時間，終止

            for stockid in stocknum:
                url = ('http://www.twse.com.tw/exchangeReport/STOCK_DAY?response=csv&date='
                       + str(year)+str(m).zfill(2)+'01'+'&stockNo='+stockid)
                try:
                    pd.read_csv(rawdatadir + stockid+'_' + str(year) +
                                '_' + str(m).zfill(2) + '.csv', 'r', encoding='cp950')

                    if m == time.localtime().tm_mon and year == time.localtime().tm_year:
                        urllib.request.urlretrieve(url, rawdatadir
                                                   + stockid+'_' + str(year) + '_' + str(m).zfill(2) + '.csv')
                        time.sleep(3)  # 重新下載當月資料

                except FileNotFoundError:  # 沒有資料，就下載
                    try:
                        urllib.request.urlretrieve(url, rawdatadir
                                                   + stockid+'_' + str(year) + '_' + str(m).zfill(2) + '.csv')
                        time.sleep(3)
                    except urllib.error.URLError:
                        print('urllib.error.URLError')
                        continue
                    # except FileExistsError:
                    #     continue


def stock_data_process(stocknum, years, rawdatadir, h5datadir):
    count = np.zeros((111))
    countstock = 0

    for stockid in stocknum:
        for year in range(years, time.localtime().tm_year+1):
            for m in range(1, 13):
                try:
                    os.rename(rawdatadir + stockid+'_' + str(year) + '_' + str(m)
                              + '.csv',
                              rawdatadir + stockid+'_' +
                              str(year) + '_' + str(m).zfill(2)
                              + '.csv')
                except:
                    pass

    for stockid in stocknum:
        mixed_data = pd.DataFrame()
        for year in range(years, time.localtime().tm_year+1):
            for m in range(1, 13):

                if year == time.localtime().tm_year and m >= time.localtime().tm_mon+1:
                    continue  # 超過當前時間，終止
                try:
                    f = open(rawdatadir + stockid+'_' + str(year) + '_' + str(m).zfill(2)
                             + '.csv', 'r', encoding='cp950')
                except FileNotFoundError:
                    #print('file not found! = ' + stockid+str(year)+str(m))
                    continue
                try:
                    df = pd.read_csv(f, header=1)
                except:
                    print('error reading csv = '+stockid+str(year)+str(m))
                    continue

                df = df.iloc[:, 0:7]
                nan_flag = 0
                # try:
                #     for col in df.columns[3:7]:
                #         df[col] = df[col].astype('float64')#一列一列改變
                # except:
                #nan_flag = 1
                for i in range(len(df['開盤價'])):
                    try:
                        df.iloc[i, 3:7] = df.iloc[i, 3:7].astype('float64')
                        #print(df.iloc[i,3:7])
                    except:
                        #print(df.iloc[i,3:7])
                        nan_flag = 1
                        df.iloc[i, 2] = np.nan
                        #print(df)

                df.dropna(how='any', inplace=True)

                if nan_flag == 1:
                    #print(df)
                    for col in df.columns[3:7]:
                        df[col] = df[col].astype('float64')
                    #print (df.dtypes)

                mixed_data = pd.concat([mixed_data, df], ignore_index=True)
                #print(mixed_data)

        #print(mixed_data.dtypes)
        count[countstock] = len(mixed_data)

        print(mixed_data.shape)

        for i, value in enumerate(mixed_data.iloc[:, 2]):
            try:
                mixed_data.iloc[i, 2] = value.replace(',', '')
            except AttributeError:
                continue

        mixed_data.iloc[i, 2] = float(mixed_data.iloc[i, 2])
        mixed_data['成交金額'] = mixed_data['成交金額'].astype('float64')

        mixed_data.to_hdf(h5datadir + stockid+'new.h5', 'stock_data',
                          mode='w', dropna=True, format='table')
        print(str(stockid)+'  個股總天數 = '+str(count[countstock]))
        countstock += 1
    print(count.sum())
    #########################################################   合併  轉float64 輸出hdf5
