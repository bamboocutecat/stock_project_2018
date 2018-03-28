import multiprocessing as mp
import os
import math
import time
from multiprocessing import Pool

import h5py
import imageio
#import talib
#import multiprocessing
import keras

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import kdraw
import stock_data_download_process
import stock_data_maketable

# loadmodel
model = keras.models.load_model('my_model_0.81.h5', custom_objects={"tf": tf})
print('read model OK')

stocknum = {'0051', '1102', '1216', '1227', '1314', '1319', '1434', '1451', '1476', '1477', '1504', '1536', '1560', '1590',
            '1605', '1704', '1717', '1718', '1722', '1723', '1789', '1802', '1909', '2015', '2049', '2059', '2106', '2201',
            '2204', '2207', '2227', '2231', '2312', '2313', '2324', '2327', '2337', '2344', '2347', '2352', '2353', '2356',
            '2360', '2371', '2376', '2377', '2379', '2385', '2439', '2448', '2449', '2451', '2478', '2492', '2498', '2542',
            '2603', '2606', '2610', '2615', '2618', '2723', '2809', '2812', '2834', '2845', '2867', '2888', '2912', '2915',
            '3019', '3034', '3044', '3051', '3189', '3231', '3406', '3443', '3532', '3673', '3682', '3702', '3706', '4137',
            '4915', '4943', '4958', '5264', '5522', '5871', '6005', '6116', '6176', '6239', '6269', '6285', '6409', '6414',
            '6415', '6452', '6456', '8454', '8464', '9910', '9914', '9917', '9921', '9933', '9938', '9941', '9945'}
Y_slicing = 1
X_window = 50
K_changedays = 50
from_years = 2018
from_months = 1
rawdata_path = 'raw data/'
h5data_path = 'h5 data/'
table_path = 'table/'
pic_path = 'stock_pic/'
if_retrain = 1

if not os.path.isdir(rawdata_path):
    os.mkdir(rawdata_path)
if not os.path.isdir(h5data_path):
    os.mkdir(h5data_path)

if if_retrain == 1:
    trained_data_count = []
    for stockid in stocknum:
        df = pd.read_hdf(h5data_path+stockid+'.h5', 'stock_data')
        trained_data_count.append(len(df))
    trained_data_countnp = np.array(trained_data_count)
    pd.Series(trained_data_countnp).to_hdf('train_data_count', mode='w')


# ######stock strategy

# stock_data_download_process.stock_data_download(
#     stocknum, from_years, from_months, rawdata_path)
# #download csv
# stock_data_download_process.stock_data_process(
#     stocknum, from_years, from_months, rawdata_path, h5data_path)
# #into hdf5
# stock_data_maketable.stock_recordchange(
#     stocknum, X_window, Y_slicing, K_changedays, h5data_path)
# #into _table
# stock_data_maketable.stock_tablemake(stocknum, h5data_path)
# #into_sumchange

# #########################   multprocess draw
# if __name__ == '__main__':
#     mp.set_start_method('spawn')
#     stocknum_list = list(stocknum)
#     pool = Pool(mp.cpu_count())
#     res = pool.map(kdraw.drawpic, stocknum_list)
#     print(res)
#into pic


# use old data as inputdata first!!~
# see how much money can i earn in old data

#read(h5)  ->  get len of days
money = 1000

df = pd.read_hdf(h5data_path+'0051.h5', 'stock_data')

randday = np.random.randint((len(df)-X_window+1)/1-K_changedays)
print('today is randday = %d' % randday+50-1)
# np.randint() (days-X)/Y
X_list = []

for i in range(51):
    pic = imageio.imread(pic_path+'0051pic/'+str(randday-50+i).zfill(4)+'_0051.jpg',
                         format='jpg')
    #print(str(randday-50+i)+'  day')
    X_list.append(pic)

X_pridict = np.array(X_list).reshape(len(X_list), 224, 224, 3)
prob_array = model.predict(x=X_pridict, batch_size=1, verbose=0)

for row in prob_array:
    print(row)

print(prob_array.shape)
# print(prob_array[:,0])

plt.figure
plt.plot(prob_array[:,0],label='plus')
plt.plot(prob_array[:,1],label='minus')
plt.plot(prob_array[:,2],label='unchange')
plt.legend(loc='best')
plt.savefig('prob_pic.pdf',mode='w')







# decision = input('make a decision!!~ buy or sell')
# if decision == 'buy':
#     money -= df['收盤價'][randday+X_window-1]
# if decision == 'sell':
#     pass
# else:
#     pass


#for day in range(50):

# print(date +:    -:)
# print(mystocklist)
# 影印前50天的資料  3張圖
# print(buy? sell?)

# sell(mystocklist)
# buy(mystocklist)

# df_0051 = pd.read_hdf(h5data_path+'0051.h5','stock_data',mode='r')
# print(df_0051.describe())
# print(len(df_0051.iloc[:,6]))

#1.製造圖片
#(1)新增資料
#(2)
#2.輸入圖片到模型pridict  return 機率分布array
#3.畫出2個直方圖  一個代表正起伏+負起伏  一個代表小變化
#  一格直方代表後50天的起伏預測 --> 轉負起伏就開始買
