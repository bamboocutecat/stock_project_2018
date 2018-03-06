import numpy
import pandas as pd
import talib
import multiprocessing
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import h5py

# 載入模型
model = tf.contrib.keras.models.load_model('my_model.h5')
stock_data = h5py.File('pic_sum_new.h5','r')

#plt.figure()
#for i in range(1):
data = imageio.imread(stock_data['test_addrs'][0])
data = data.reshape(1,380,383,3)
print(list(model.predict(data,batch_size=1,verbose=1)))
#plt.show()

#

