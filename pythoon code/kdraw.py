import pandas as pd
import numpy as np
import time
import csv
#import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_finance import candlestick2_ochl,volume_overlay
import talib
from math import log, exp
from matplotlib import dates as mdates
from matplotlib import ticker as mticker
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY,YEARLY
from matplotlib.dates import MonthLocator,MONTHLY
import datetime as dt
import pylab
import h5py
from PIL import Image
import os
import threading
import multiprocessing as mp
from multiprocessing import Pool
import random
import imageio

def computeMACD(x, slow=26, fast=12):
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslow
def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a
def rsiFunc(prices, n=12):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)
    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter
        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    return rsi

def drawpic(stockid,X_window=50,Y_slicing=1,K_changedays=50,pic_check=0):

    fig = plt.figure(figsize=(24, 24))
    ax = plt.subplot2grid((20,4), (0,0), rowspan=7, colspan=4, facecolor='#07000d')

    df = pd.read_hdf(stockid + '.h5','stock_data')

    sma_10 = talib._ta_lib.SMA(np.array(df['收盤價']), 10)
    sma_30 = talib._ta_lib.SMA(np.array(df['收盤價']), 30)    
    
    countpic = 0

    for X_pics in range( 0,int((len(df)- X_window)/ Y_slicing + 1  - K_changedays) ):

        if pic_check == 1:                    
            try:
                #open(stockid + 'pic/'+ str(X_pics) + '_'+stockid+'.jpg','r')
                imageio.imread(stockid + 'pic/'+ str(X_pics) + '_'+stockid+'.jpg',format='jpg')
                countpic +=1
                print(stockid +' = '+ countpic)
                continue
            except :
                print(stockid + 'pic/'+ str(X_pics) + '_'+stockid+'.jpg'+'  =  error!!')
                pass
        
        ax.clear()
        #ax.set_xticks(range(0,50), 10)
        #ax.set(ylim=[-2, 2])
        df_slice = df.iloc[X_pics:X_pics + X_window,:]
        #df_slice.index=range(0,50)

        candlestick2_ochl(ax, df_slice['開盤價'], df_slice['收盤價'],
                          df_slice['最高價'], df_slice['最低價'],
                           width=1, colorup='r', colordown='green',alpha=0.6)
        ax.plot(range(0,50),sma_10[X_pics:X_pics+50],color='#ffff00',lw=5)
        ax.plot(range(0,50),sma_30[X_pics:X_pics+50],color='#0066ff',lw=5)
        #try:
         #   candlestick2_ochl(ax, df_diff_sliced['開盤價'], df_diff_sliced['收盤價'],
          #                df_diff_sliced['最高價'], df_diff_sliced['最低價'],
           #               width=0.75, colorup='r', colordown='green',alpha=0.6)
        #except:
         #   print(df_diff_sliced.dtypes)

        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())

        ##########################################################       MACD

        ax2 = plt.subplot2grid((20,4), (7,0), rowspan=2, colspan=4, facecolor='#07000d')
        fillcolor = '#c58fff'

        #nslow = 26
        #nfast = 12
        nema = 9
        emaslow, emafast, macd = computeMACD(df['收盤價'])
        ema9 = ExpMovingAverage(macd, nema)
        ax2.plot(df_slice['日期'], macd[X_pics:X_pics+50], color='#4ee6fd', lw=5)
        ax2.plot(df_slice['日期'], ema9[X_pics:X_pics+50], color='#ffc78f', lw=5)
        ax2.fill_between(df_slice['日期'], (macd-ema9)[X_pics:X_pics+50]
        , 0, alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)


        posCol = '#ffff00'
        negCol = '#ff3300'
        ax2.axhline(0.5, color=negCol)
        ax2.axhline(-0.5, color=posCol)

        ####################################################        RSI

        ax3 = plt.subplot2grid((20,4), (9,0),  rowspan=2, colspan=4, facecolor='#07000d')
        rsi = rsiFunc(df['收盤價'])
        rsiCol = '#ff99ff'
        posCol = '#ffff00'
        negCol = '#ff3300'

        ax3.plot(df_slice['日期'], rsi[X_pics:X_pics+50], rsiCol, linewidth=5)
        ax3.axhline(70, color=negCol)
        ax3.axhline(30, color=posCol)

        ax3.fill_between(df_slice['日期'], rsi[X_pics:X_pics+50], 70,
                         where=(rsi[X_pics:X_pics+50]>=70), facecolor=negCol, edgecolor=negCol, alpha=0.5)
        ax3.fill_between(df_slice['日期'], rsi[X_pics:X_pics+50], 30,
                         where=(rsi[X_pics:X_pics+50]<=30), facecolor=posCol, edgecolor=posCol, alpha=0.5)

        ax3.set_yticks([30,70])

        ######################################################   Momentum
        ax4 = plt.subplot2grid((20,4), (11,0),    rowspan=2, colspan=4, facecolor='#07000d')
        mom = talib._ta_lib.MOM(np.array(df['收盤價']), 10)
        #ax4.axhline(0, color='#ffffff')
        ax4.plot(df_slice['日期'],mom[X_pics:X_pics+50],'#668cff',linewidth=5)

        ###################################################    價格關係   一年參考關係
        ax5 = plt.subplot2grid((20,4),(13,0), rowspan=4, colspan=4, facecolor='#07000d')

        if X_pics < 301:
            ax5.set(ylim=[  df.iloc[1:301,6].quantile(0.1),df.iloc[1:301,6].quantile(0.9)  ] )
        else :
            ax5.set(ylim=[  df.iloc[X_pics-300:X_pics,6].quantile(0.1),df.iloc[X_pics-300:X_pics,6].quantile(0.9)  ])

        ax5.fill_between(range(0,50),df_slice['收盤價'], 0, alpha=0.5, facecolor='#ccffff', edgecolor='#ccffff')
        ax5.plot(range(0,50),df_slice['收盤價'],'#ffffff',linewidth=5)

        #################################################################### 成交量   一年參考關係
        ax6 = plt.subplot2grid((20,4),(17,0), rowspan=4, colspan=4, facecolor='#07000d') 

        if X_pics < 301:
            ax6.set(ylim=[  df.iloc[1:301,2].quantile(0.1),df.iloc[1:301,2].quantile(0.9)  ] )
        else :
            ax6.set(ylim=[  df.iloc[X_pics-300:X_pics,2].quantile(0.1),df.iloc[X_pics-300:X_pics,2].quantile(0.9)  ])

        ax6.bar(range(0,50), df_slice['成交金額'], facecolor='#ff9933') 

        ax2.xaxis.set_major_formatter(plt.NullFormatter())
        ax2.yaxis.set_major_formatter(plt.NullFormatter())
        ax3.xaxis.set_major_formatter(plt.NullFormatter())
        ax3.yaxis.set_major_formatter(plt.NullFormatter())
        ax4.xaxis.set_major_formatter(plt.NullFormatter())
        ax4.yaxis.set_major_formatter(plt.NullFormatter())
        ax5.xaxis.set_major_formatter(plt.NullFormatter())
        ax5.yaxis.set_major_formatter(plt.NullFormatter())
        ax6.xaxis.set_major_formatter(plt.NullFormatter())
        ax6.yaxis.set_major_formatter(plt.NullFormatter())

        if not os.path.isdir(stockid + 'pic/'):
            os.mkdir(stockid + 'pic/')

        plt.savefig(stockid + 'pic/'+ str(X_pics) + '_'+stockid+'.jpg',dpi=20,bbox_inches='tight',mode='w')
        countpic+=1
        print(stockid +' = '+ countpic)
