# coding:utf-8
import multiprocessing as mp
import os
import math
import time
from multiprocessing import Pool
import glob
import h5py
import imageio
import talib
import multiprocessing
import keras

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import pandas as pd
import tensorflow as tf

import kdraw
# import stock_data_download_process
# import stock_data_maketable

# loadmodel
model = keras.models.load_model('my_model_0.81.h5', custom_objects={"tf": tf})
print('read model OK')

stocknum = {
    '0051', '1102', '1216', '1227', '1314', '1319', '1434', '1451', '1476',
    '1477', '1504', '1536', '1560', '1590', '1605', '1704', '1717', '1718',
    '1722', '1723', '1789', '1802', '1909', '2015', '2049', '2059', '2106',
    '2201', '2204', '2207', '2227', '2231', '2312', '2313', '2324', '2327',
    '2337', '2344', '2347', '2352', '2353', '2356', '2360', '2371', '2376',
    '2377', '2379', '2385', '2439', '2448', '2449', '2451', '2478', '2492',
    '2498', '2542', '2603', '2606', '2610', '2615', '2618', '2723', '2809',
    '2812', '2834', '2845', '2867', '2888', '2912', '2915', '3019', '3034',
    '3044', '3051', '3189', '3231', '3406', '3443', '3532', '3673', '3682',
    '3702', '3706', '4137', '4915', '4943', '4958', '5264', '5522', '5871',
    '6005', '6116', '6176', '6239', '6269', '6285', '6409', '6414', '6415',
    '6452', '6456', '8454', '8464', '9910', '9914', '9917', '9921', '9933',
    '9938', '9941', '9945'
}
Y_slicing = 1
X_window = 50
K_changedays = 50
from_years = 2018
#from_months = 1
rawdata_path = 'raw data/'
h5data_path = 'h5 data/'
table_path = 'table/'
pic_path = 'stock_pic/'
if_retrain = 0

if not os.path.isdir(rawdata_path):
    os.mkdir(rawdata_path)
if not os.path.isdir(h5data_path):
    os.mkdir(h5data_path)
if not os.path.isdir(pic_path):
    os.mkdir(pic_path)
    for stockid in stocknum:
        if not os.path.isdir(pic_path + stockid + 'pic/'):
            os.mkdir(pic_path + stockid + 'pic/')

if if_retrain == 1:
    trained_data_count = []
    for stockid in stocknum:
        df = pd.read_hdf(h5data_path + stockid + '.h5', 'stock_data')
        trained_data_count.append((stockid, int(len(df))))

    trained_data_countnp = np.array(trained_data_count)
    pd.DataFrame(trained_data_countnp).to_hdf(
        'train_data_count.h5', 'train_data_count', mode='w')


def check_pic_h5data(stocknum, h5data_path, pic_path):
    for stockid in stocknum:
        df = pd.read_hdf(h5data_path + stockid + 'new.h5', 'stock_data')
        #     print(len(df)-50+1-50)

        piclist = glob.glob(pic_path + stockid + 'pic/*.jpg')
        #     print(len(piclist))
        if (len(df) - 50 + 1) != len(piclist):
            print('error')
        print('%d  =  %d' % ((len(df) - 50 + 1), len(piclist)))


check_pic_h5data(stocknum, h5data_path, pic_path)

# ######stock strategy

# stock_data_download_process.stock_data_download(
#    stocknum, from_years, rawdata_path)
# #download csv

# stock_data_download_process.stock_data_process(
#     stocknum, 1991, rawdata_path, h5data_path)
# #into hdf5

# stock_data_maketable.stock_recordchange(
#     stocknum, X_window, Y_slicing, K_changedays, h5data_path)
# #into _table

# stock_data_maketable.stock_tablemake(stocknum, h5data_path)
# #into_sumchange

# #########################   multprocess draw

# # kdraw.drawpic('0051')

# if __name__ == '__main__':
#     # mp.set_start_method('spawn')
#     stocknum_list = list(stocknum)
#     pool = Pool(mp.cpu_count())
#     res = pool.map(kdraw.drawpic, stocknum_list)
#     print(res)
# # into pic

# def show_predict(stockid,)

# def control():
# df = pd.read_hdf(h5data_path + '0051new.h5', 'stock_data')
# train_datacountdf = pd.read_hdf('train_data_count.h5', 'train_data_count')
# df_table = pd.read_hdf(h5data_path + '0051_table_sumchange.h5',
#                     'stock_data_table')

# def buy

# def sell

# def creat_user_data?

#選一天
#   排除訓練天數

money = 100
mystocklist = []

df = pd.read_hdf(h5data_path + '0051new.h5', 'stock_data')
train_datacountdf = pd.read_hdf('train_data_count.h5', 'train_data_count')
df_table = pd.read_hdf(h5data_path + '0051_table_sumchange.h5',
                       'stock_data_table')

for i, stock in enumerate(train_datacountdf[0]):
    if stock == '0051':
        traindaycount = int(train_datacountdf[1][i])

# randday = randint(traindaycount, len(df))

print('untrain pic is from  =  ' + str(df.iloc[traindaycount - 1, 0]) + ' ~ ' +
      str(df.iloc[len(df) - 1, 0]))
print('pic num from  =  ' + str(traindaycount - 50 + 1) + ' ~ ' +
      str(len(df) - 50 + 1))

today = input('choose a pic num as your day : ')

print('today is : ' + str(df.iloc[today + 50 - 1 - 1, 0]))

while True:

    ##### 把當天圖片show出來
    today_pic = imageio.imread(
        pic_path + '0051pic/' + str(today - 1).zfill(4) + '_0051.jpg',
        format='jpg')
    plt.imsave('today.jpg',today_pic)

    ##### 進入predict
    X_list = []

    for i in range(50):
        pic = imageio.imread(
            pic_path + '0051pic/' + str(today - 1 - 49 + i).zfill(4) +
            '_0051.jpg',
            format='jpg')
        X_list.append(pic)

    X_pridict = np.array(X_list).reshape(len(X_list), 224, 224, 3)
    prob_array = model.predict(x=X_pridict, batch_size=1, verbose=0)

    # loss, acc = model.evaluate(
    #     x=X_pridict, y=df_table[today-1], verbose=1)

    # print('loss = ' + str(loss) + '  acc = ' + str(acc))

    print(prob_array.shape)

    plt.figure()
    plt.plot(prob_array[:, 0], label='plus')
    plt.plot(prob_array[:, 1], label='minus')
    plt.plot(prob_array[:, 2], label='unchange')
    plt.legend(loc='best')
    plt.savefig('prob_pic.pdf', mode='w')

    input_order = input('buy or sell : \n')
    if input_order == 'buy':
        money -= df.iloc[today + 50 - 1 - 1, 6]
        mystocklist.append(df.iloc[today + 50 - 1 - 1, 6])

    if input_order == 'sell':
        print(mystocklist)
        sellnum = input('sell num : \n')
        money += df.iloc[today + 50 - 1 - 1, 6]
        mystocklist.pop(int(sellnum))
        print(mystocklist)

    today += 1
    print('my money have : ' + str(money))
    print('today is : ' + str(df.iloc[today + 50 - 1 - 1, 0]))

# decision = input('make a decision!!~ buy or sell')
# if decision == 'buy':
#     money -= df['收盤價'][randday+X_window-1]
# if decision == 'sell':
#     pass
# else:
#     pass

# for day in range(50):

# print(date +:    -:)
# print(mystocklist)
# 影印前50天的資料  3張圖
# print(buy? sell?)

# sell(mystocklist)
# buy(mystocklist)

# df_0051 = pd.read_hdf(h5data_path+'0051.h5','stock_data',mode='r')
# print(df_0051.describe())
# print(len(df_0051.iloc[:,6]))

# 1.製造圖片
# (1)新增資料
# (2)
# 2.輸入圖片到模型pridict  return 機率分布array
# 3.畫出2個直方圖  一個代表正起伏+負起伏  一個代表小變化
#  一格直方代表後50天的起伏預測 --> 轉負起伏就開始買
