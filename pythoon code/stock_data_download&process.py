import urllib.request
import time
import csv
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
import codecs

def stock_data_download(stocknum,years,months):
    for year in range(years,time.localtime().tm_year):
        for m in range(months,13):
            for stockid in stocknum:
                url =('http://www.twse.com.tw/exchangeReport/STOCK_DAY?response=csv&date='+
                str(year)+str(m).zfill(2)+'01'+'&stockNo='+stockid)

                try:
                    open('D:/專題使用/新股票資料/' +stockid+'_' +str(year) + '_' + str(m).zfill(2) + '.csv','r')
                except FileNotFoundError:    
                    try:
                        f = urllib.request.urlretrieve(url,'D:/專題使用/新股票資料/' +stockid+'_' +str(year) + '_' + str(m).zfill(2) + '.csv')
                        time.sleep(3)
                    except urllib.error.URLError:
                        continue
                    except FileExistsError:
                        continue






def stock_data_process(stocknum,years,months):
    count = np.zeros((111))
    countstock = 0

    for stockid in stocknum:
        mixed_data = pd.DataFrame()
        for year in range(years,2018):
            for m in range(months,13):
                try:
                    f = open('D:/program/新股票資料/' +stockid+'_'+ str(year) + '_' + str(m).zfill(2) 
                    + '.csv','r',encoding='cp950')  
                except FileNotFoundError:
                    print('file not found! = ' +stockid+str(year)+str(m))
                    continue
                try:
                    df = pd.read_csv(f,header=1)
                except:
                    print('error reading csv = '+stockid+str(year)+str(m))
                    continue

                df = df.iloc[:,0:7]
                nan_flag = 0
                # try:
                #     for col in df.columns[3:7]:
                #         df[col] = df[col].astype('float64')#一列一列改變
                # except:
                    #nan_flag = 1
                for i in range(len(df['開盤價'])):
                    try:
                        df.iloc[i,3:7] = df.iloc[i,3:7].astype('float64')
                        #print(df.iloc[i,3:7])
                    except :
                        #print(df.iloc[i,3:7])
                        nan_flag = 1
                        df.iloc[i,2]= np.nan
                        #print(df)
                            
                df.dropna(how='any',inplace=True) 
                
                if nan_flag==1:
                    #print(df)
                    for col in df.columns[3:7]:
                        df[col] = df[col].astype('float64')
                    #print (df.dtypes)
                
                
                
                mixed_data = pd.concat([mixed_data,df],ignore_index=True)
                #print(mixed_data)
        
        
        #print(mixed_data.dtypes)
        count[countstock]=len(mixed_data)
        
        
        mixed_data.to_hdf(stockid+'.h5','stock_data',mode='w',dropna=True,format='table')
        print(stockid+'  個股總天數 = '+count[countstock])
        countstock += 1
    print(count.sum())
    #########################################################   合併  轉float64 輸出hdf5
