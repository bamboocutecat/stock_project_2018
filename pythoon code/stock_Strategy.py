import numpy
import pandas as pd
#import talib
#import multiprocessing
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import h5py
import time
import kdraw
import stock_data_download_process
import stock_data_maketable

# loadmodel
model = keras.models.load_model('my_model.h5')
stock_data = h5py.File('shuffleed_data.h5')
print('read OK')


# for addr in stock_data['test_img'][:10]:
#     pic = imageio.imread(addr)
#     pic = pic.reshape(1,380,383,3)
#     print(model.predict(pic,batch_size=1,verbose=1))

#plt.figure()
#for i in range(1):
# data = imageio.imread(stock_data['test_addrs'][0])
# data = data.reshape(1,380,383,3)
# print(list(model.predict(data,batch_size=1,verbose=1)))
#plt.show()
stocknum = {'0051','1102','1216','1227','1314','1319','1434','1451','1476','1477','1504','1536','1560','1590',
            '1605','1704','1717','1718','1722','1723','1789','1802','1909','2015','2049','2059','2106','2201',
            '2204','2207','2227','2231','2312','2313','2324','2327','2337','2344','2347','2352','2353','2356',
            '2360','2371','2376','2377','2379','2385','2439','2448','2449','2451','2478','2492','2498','2542',
            '2603','2606','2610','2615','2618','2723','2809','2812','2834','2845','2867','2888','2912','2915',
            '3019','3034','3044','3051','3189','3231','3406','3443','3532','3673','3682','3702','3706','4137',
            '4915','4943','4958','5264','5522','5871','6005','6116','6176','6239','6269','6285','6409','6414',
            '6415','6452','6456','8454','8464','9910','9914','9917','9921','9933','9938','9941','9945'}
Y_slicing = 1
X_window = 50
K_changedays = 50
#交易策略
stock_data_download_process.stock_data_download(stocknum,2018,1)



    #1.製造圖片
        #(1)新增資料
        #(2)
    #2.輸入圖片到模型pridict  return 機率分布array
    #3.畫出2個直方圖  一個代表正起伏+負起伏  一個代表小變化
    #  一格直方代表後50天的起伏預測 --> 轉負起伏就開始買



