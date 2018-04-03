#import twstock
import pandas as pd
import numpy as np
#import urllib.request
import time
import csv
#import tensorflow as tf
import matplotlib.pyplot as plt
#from mpl_finance import candlestick2_ochl,volume_overlay
#import talib
from math import log, exp
#from matplotlib import dates as mdates
#from matplotlib import ticker as mticker
#from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY,YEARLY
#from matplotlib.dates import MonthLocator,MONTHLY
import datetime as dt
#import pylab
import h5py
import math
import os


def stock_recordchange(stocknum, X_window, Y_slicing, K_changedays, h5datapath):  # 將變化量新建h5

    for stockid in stocknum:
        df = pd.read_hdf(h5datapath + stockid+'new.h5', 'stock_data')
        # 建立儲存變化量矩陣
        recordchangedata_np_ar = np.zeros(shape=(int((len(df) - X_window) / Y_slicing + 1 - K_changedays), 2))

        for Y in range(int((len(df) - X_window) / Y_slicing + 1 - K_changedays)):
            sumchange_plus = 0
            sumchange_minus = 0
            #取收盤價，做後K天的漲跌量
            for i, K in enumerate(df.iloc[Y+X_window: Y+X_window + K_changedays, 6]):

                if K - df.iloc[Y + X_window - 1, 6] > 0:
                    sumchange_plus += K - df.iloc[Y + X_window - 1, 6]

                if K - df.iloc[Y + X_window - 1, 6] < 0:
                    sumchange_minus += abs(K - df.iloc[Y + X_window - 1, 6])

            recordchangedata_np_ar[Y][0] = sumchange_plus
            recordchangedata_np_ar[Y][1] = sumchange_minus

        se_rcd = pd.DataFrame(recordchangedata_np_ar,
                              columns=['plus', 'minus'])
        se_rcd.to_hdf(h5datapath + stockid+'_table.h5',
                      'stock_data_table', format='table', mode='w')


def stock_tablemake(stocknum, h5datapath):
    ##########################################   將分類套用並存h5   依變化量
    for stockid in stocknum:
        df_table = pd.read_hdf(h5datapath + stockid +
                               '_table.h5', 'stock_data_table', mode='r')
        table_sumchange = np.zeros((len(df_table), 3))

        for i, value in enumerate(df_table.values):
            if value[0] > 50 and value[1] <= 50:
                table_sumchange[i][0] = 1
            if value[0] <= 50 and value[1] > 50:
                table_sumchange[i][1] = 1
            if (value[0] < 50 and value[1] < 50) or (value[0] > 50 and value[1] > 50):
                table_sumchange[i][2] = 1
        print(stockid+' shape = ' + str(table_sumchange.shape))
        table_sumchange_df = pd.DataFrame(table_sumchange, columns=[
                                          'plus', 'minus', 'unchange'])
        table_sumchange_df.to_hdf(
            h5datapath + stockid+'_table_sumchange.h5', 'stock_data_table', format='table', mode='w')
    ########################################       求證總數無誤  將所有變化量統合
    sum_df = pd.DataFrame()

    for stockid in stocknum:
        df_table = pd.read_hdf(h5datapath + stockid +
                               '_table_sumchange.h5', 'stock_data_table')
        sum_df = pd.concat([sum_df, df_table], ignore_index=True)

    print(sum_df.describe())


def observe_relation(stockid, h5datapath):
    ######################################         觀察變化量與價格關係
    df_table = pd.read_hdf(h5datapath + str(stockid) +
                           '_table.h5', 'stock_data_table')
    df = pd.read_hdf(h5datapath + str(stockid)+'.h5', 'stock_data')

    fig = plt.figure()

    plt.plot(df.loc[50:150, '收盤價'], 'r')
    plt.show()
    plt.plot(df_table[0:50])
    plt.show()
