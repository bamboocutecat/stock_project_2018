from PyQt5 import QtCore  #import QUrl, QObject, pyqtSlot
from PyQt5 import QtGui  #import QGuiApplication
from PyQt5 import QtQuick  #import QQuickView
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
import sys


def stock_data_download(stocknum, years, rawdatadir):

    for year in range(years, time.localtime().tm_year + 1):
        for m in range(1, 13):
            if year == time.localtime(
            ).tm_year and m >= time.localtime().tm_mon + 1:
                continue  # 超過當前時間，終止
            for stockid in stocknum:
                url = (
                    'http://www.twse.com.tw/exchangeReport/STOCK_DAY?response=csv&date='
                    + str(year) + str(m).zfill(2) + '01' + '&stockNo=' +
                    stockid)
                try:
                    open(rawdatadir + stockid + '_' + str(year) + '_' +
                         str(m).zfill(2) + '.csv', 'r')

                    if  year == time.localtime().tm_year:
                        urllib.request.urlretrieve(
                            url, rawdatadir + stockid + '_' + str(year) + '_' +
                            str(m).zfill(2) + '.csv')
                        print(
                            str(rawdatadir + stockid + '_' + str(year) + '_' +
                                str(m).zfill(2) + '.csv'))
                        time.sleep(2.5)  # 重新下載當月資料

                except FileNotFoundError:  # 沒有資料，就下載
                    try:
                        urllib.request.urlretrieve(
                            url, rawdatadir + stockid + '_' + str(year) + '_' +
                            str(m).zfill(2) + '.csv')
                        time.sleep(2.5)
                    except urllib.error.URLError:
                        print('urllib.error.URLError')
                        continue


def stock_data_process(stocknum, years=1991, rawdatadir='raw_data', h5datadir='h5_data'):

    count = np.zeros((111))
    countstock = 0

    for stockid in stocknum:
        for year in range(years, time.localtime().tm_year + 1):
            for m in range(1, 13):
                try:
                    os.rename(rawdatadir + stockid + '_' + str(year) + '_' +
                              str(m) + '.csv', rawdatadir + stockid + '_' +
                              str(year) + '_' + str(m).zfill(2) + '.csv')
                except:
                    pass

    for stockid in stocknum:
        mixed_data = pd.DataFrame()
        for year in range(years, time.localtime().tm_year + 1):
            for m in range(1, 13):

                if year == time.localtime(
                ).tm_year and m >= time.localtime().tm_mon + 1:
                    continue  # 超過當前時間，終止
                try:
                    f = open(
                        rawdatadir + stockid + '_' + str(year) + '_' +
                        str(m).zfill(2) + '.csv',
                        'r',
                        encoding='cp950')
                except FileNotFoundError:
                    #print('file not found! = ' + stockid+str(year)+str(m))
                    continue
                try:
                    df = pd.read_csv(f, header=1)
                except:
                    print(
                        'error reading csv = ' + stockid + str(year) + str(m))
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

        mixed_data.to_hdf(
            h5datadir + stockid + 'new.h5',
            'stock_data',
            mode='w',
            dropna=True,
            format='table')
        print(str(stockid) + '  個股總天數 = ' + str(count[countstock]))
        countstock += 1
    print(count.sum())
    #########################################################   合併  轉float64 輸出hdf5


# class MyClass(QtCore.QObject):
#     stocknum = {
#         '0051', '1102', '1216', '1227', '1314', '1319', '1434', '1451', '1476',
#         '1477', '1504', '1536', '1560', '1590', '1605', '1704', '1717', '1718',
#         '1722', '1723', '1789', '1802', '1909', '2015', '2049', '2059', '2106',
#         '2201', '2204', '2207', '2227', '2231', '2312', '2313', '2324', '2327',
#         '2337', '2344', '2347', '2352', '2353', '2356', '2360', '2371', '2376',
#         '2377', '2379', '2385', '2439', '2448', '2449', '2451', '2478', '2492',
#         '2498', '2542', '2603', '2606', '2610', '2615', '2618', '2723', '2809',
#         '2812', '2834', '2845', '2867', '2888', '2912', '2915', '3019', '3034',
#         '3044', '3051', '3189', '3231', '3406', '3443', '3532', '3673', '3682',
#         '3702', '3706', '4137', '4915', '4943', '4958', '5264', '5522', '5871',
#         '6005', '6116', '6176', '6239', '6269', '6285', '6409', '6414', '6415',
#         '6452', '6456', '8454', '8464', '9910', '9914', '9917', '9921', '9933',
#         '9938', '9941', '9945'
#     }
#     busysig = QtCore.pyqtSignal(bool, arguments=['indicator'])
#     def action(self,from_year,raw_path):
#         stock_data_download(self.stocknum,from_year,raw_path)
#         self.busysig.emit(0)

# if __name__ == '__main__':

#     filepath = os.path.abspath('.')
#     filepath = filepath + '/'

#     path = filepath + 'stockgui/main.qml'
#     app = QtGui.QGuiApplication([])
#     view = QtQuick.QQuickView()
#     process = MyClass()

#     process.action(sys.argv[1],sys.argv[2])
#     context = view.rootContext()
#     context.setContextProperty("download", process)
#     view.engine().quit.connect(app.quit)
#     view.setSource(QtCore.QUrl(path))
#     view.show()
#     app.exec_()