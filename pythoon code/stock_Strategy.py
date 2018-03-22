import numpy
import pandas as pd
#import talib
#import multiprocessing
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import h5py

# loadmodel
model = keras.models.load_model('my_model.h5')
stock_data = h5py.File('shuffleed_data.h5')
print('read OK')


for addr in stock_data['test_img'][:10]:
    pic = imageio.imread(addr)
    pic = pic.reshape(1,380,383,3)
    print(model.predict(pic,batch_size=1,verbose=1))







#plt.figure()
#for i in range(1):
# data = imageio.imread(stock_data['test_addrs'][0])
# data = data.reshape(1,380,383,3)
# print(list(model.predict(data,batch_size=1,verbose=1)))
#plt.show()

#交易策略
    #1.製造圖片
        #(1)新增資料
        #(2)
    #2.輸入圖片到模型pridict  return 機率分布array
    #3.畫出2個直方圖  一個代表正起伏+負起伏  一個代表小變化
    #  一格直方代表後50天的起伏預測 --> 轉負起伏就開始買



