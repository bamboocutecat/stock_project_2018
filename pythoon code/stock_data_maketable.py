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

def stock_recordchange(stocknum,X_window,Y_slicing,K_changedays):      ##############################       將變化量新建h5  

    for stockid in stocknum:
        df = pd.read_hdf(stockid+'.h5','stock_data')
        # 建立儲存變化量矩陣
        recordchangedata_np_ar = np.zeros(int((len(df)- X_window)/ Y_slicing + 1 - K_changedays),2)
        
        for Y in range(int((len(df)- X_window)/ Y_slicing + 1 - K_changedays)):
            sumchange_plus = 0
            sumchange_minus = 0
            #取收盤價，做後K天的漲跌量
            for i,K in enumerate(df.iloc[Y+X_window : Y+X_window + K_changedays,6]): 

                if  K- df.iloc[Y + X_window -1,6] > 0:
                    sumchange_plus += K- df.iloc[Y + X_window -1,6]

                if  K- df.iloc[Y + X_window -1,6] < 0:
                    sumchange_minus += abs(K- df.iloc[Y + X_window -1,6])

            recordchangedata_np_ar[Y][0] = sumchange_plus
            recordchangedata_np_ar[Y][1] = sumchange_minus

        se_rcd = pd.DataFrame(recordchangedata_np_ar,columns=['plus','minus'])
        se_rcd.to_hdf(stockid+'_table.h5','stock_data_table',format='table',mode='w')
            