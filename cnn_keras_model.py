import pandas as pd
import numpy as np
import time
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from math import log, exp
from matplotlib import dates as mdates
from matplotlib import ticker as mticker
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY,YEARLY
from matplotlib.dates import MonthLocator,MONTHLY
import datetime as dt

import h5py
from PIL import Image
import os
import threading
import multiprocessing as mp
import multiprocessing
from multiprocessing import Pool
import random

from numpy.random import randint
import imageio
import glob
from random import shuffle

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation,BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.utils import Sequence
from keras.utils import multi_gpu_model
from keras.preprocessing.image import img_to_array
from numpy import float
from multi_gpu_model_fixed import multi_gpu_model



stocknum = {'0051','1102','1216','1227','1314','1319','1434','1451','1476','1477','1504','1536','1560','1590',
            '1605','1704','1717','1718','1722','1723','1789','1802','1909','2015','2049','2059','2106','2201',
            '2204','2207','2227','2231','2312','2313','2324','2327','2337','2344','2347','2352','2353','2356',
            '2360','2371','2376','2377','2379','2385','2439','2448','2449','2451','2478','2492','2498','2542',
            '2603','2606','2610','2615','2618','2723','2809','2812','2834','2845','2867','2888','2912','2915',
            '3019','3034','3044','3051','3189','3231','3406','3443','3532','3673','3682','3702','3706','4137',
            '4915','4943','4958','5264','5522','5871','6005','6116','6176','6239','6269','6285','6409','6414',
            '6415','6452','6456','8454','8464','9910','9914','9917','9921','9933','9938','9941','9945'}


hdf5_path = "pic_sum_new.h5"

def generate_train_from_file(x,y,batch_size):
    
    ylen = len(y)
    loopcount = ylen // batch_size
    
    while(True):

        #X_output = np.zeros((batch_size,) + (380, 383, 3), dtype=np.float)
        Data = []

        i = randint(0,loopcount)
        labels = np.array(y[i * batch_size : (i + 1) * batch_size])

        for i,pic_addr in enumerate(x[i * batch_size : (i + 1) * batch_size]) :
            pic = Image.open(pic_addr)
            #X_output[i] = pic
            Data.append(img_to_array(pic))
            
            
        Data = np.array(Data).reshape((batch_size,) + (380, 383, 3))
        
        yield Data, labels

        
def generate_val_from_file(x,y,batch_size):

    ylen = len(y)
    loopcount = ylen // batch_size
    
    while(True):
        #X_output = np.zeros((batch_size,) + (380, 383, 3), dtype=np.float)
        Data = []

        i = randint(0,loopcount)
        labels = np.array(y[i * batch_size : (i + 1) * batch_size])

        for i,pic_addr in enumerate(x[i * batch_size : (i + 1) * batch_size]) :
            pic = Image.open(pic_addr)
            #X_output[i] = pic
            Data.append(img_to_array(pic))
            
            
        Data = np.array(Data).reshape((batch_size,) + (380, 383, 3))

        yield Data, labels


def main():


    pic_sum = []
    table_sum=[]
    
    for stockid in stocknum:
        df = pd.read_hdf('table/'+stockid+'_table_sumchange.h5','stock_data_table')
        pic_addr = glob.glob('stock_pic/' +stockid + 'pic/'+'*.jpg')
        
        for item in pic_addr:
            item = item.replace('\\','/')
            #print (item)
            pic_sum.append(item)
        for table in df.values:
            table_sum.append(table)
        print(len(pic_sum))
        print(len(table_sum))
        
    c = list(zip(pic_sum,table_sum))
    shuffle(c)
    addrs, labels = zip(*c)
    addrs = list(addrs)
    labels = list(labels)
    
    train_addrs = addrs[0:int(0.6*len(addrs))]
    train_labels = labels[0:int(0.6*len(labels))]
    val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
    val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
    test_addrs = addrs[int(0.8*len(addrs)):]
    test_labels = labels[int(0.8*len(labels)):]
    
    
    
    with tf.device('/cpu:0'):
        model = ResNet50(input_shape=(380,383,3),classes=3,pooling='max',include_top=True,weights=None)
      
    parallel_model = multi_gpu_model(model, gpus=8)
    parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  
    parallel_model.summary()
    print ('\n\n\n\n')

    #filepath="best_acc.h5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # early_stop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=5, verbose=1, mode='max')
 
    # callbacks_list = [checkpoint, early_stop]
    # myAdam = Adam(decay=0.1) 

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # train_history = model.fit(x=x_train, y=y_train, epochs=3, batch_size=100, shuffle=True, callbacks = callbacks_list)
    
    batch_size = 50
    #train_generator = train_datagen.flow_from_directory(  '')
    #print multiprocessing.cpu_count()
    
    train_history = parallel_model.fit_generator(generate_train_from_file(train_addrs,train_labels,batch_size), 
                                        steps_per_epoch=10,epochs=20
                                        ,verbose=1,workers=multiprocessing.cpu_count(), use_multiprocessing=True)
    
    loss, accuracy = parallel_model.evaluate_generator(generate_val_from_file(val_addrs,val_labels,batch_size),
                                        steps=100,workers=multiprocessing.cpu_count(), use_multiprocessing=True)
                                        
    #steps=len(val_labels)//batch_size
    print(train_history)
    print('\ntest loss: ', loss) 
    print('\ntest accuracy: ', accuracy)
    
    #plt.figure()
    for i in range(10):
        data = imageio.imread(test_addrs[i])
        data = data.reshape(1,380,383,3)
        print(parallel_model.predict(data))
        
    #plt.show()

    #print(parallel_model.predict(data,batch_size=1,verbose=1))
    
    #accPrint = int(accuracy * 10000)
    #parallel_model.save('my_model.h5') 

    # ------------ save the template model rather than the gpu_mode ----------------
    # serialize model to JSON
    # model_json = parallel_model.to_json()
    # with open('model_json.json','w') as jsonfile:
    #     jsonfile.write(model_json)
    # parallel_model.save_weights('model.h5')
    # print('save model done!\n')



main()