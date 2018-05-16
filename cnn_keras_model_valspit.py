# coding:utf-8
# from keras.backend.tensorflow_backend import set_session
import pandas as pd
import numpy as np
import time
import csv
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
from math import log, exp
from matplotlib import dates as mdates
from matplotlib import ticker as mticker
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, YEARLY
from matplotlib.dates import MonthLocator, MONTHLY
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
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.utils import Sequence
from keras.utils import multi_gpu_model
from keras.preprocessing.image import img_to_array
from numpy import float
# from multi_gpu_model_fixed import multi_gpu_model

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


def generate_from_file(x, y, batch_size):

    ylen = len(y)

    while (True):
        #X_output = np.zeros((batch_size,) + (224, 224, 3), dtype=np.float)
        Data = []

        randpick = randint(0, ylen, size=batch_size)
        labels = np.zeros((batch_size, 3), dtype=np.uint8)

        for i, value in enumerate(randpick):
            labels[i] = y[value]

            pic_addr = x[value]
            pic = imageio.imread(pic_addr)
            #X_output[i] = pic
            Data.append(pic)

        Data = np.array(
            Data, dtype=np.uint8).reshape((batch_size, ) + (224, 224, 3))
        # print(Data.shape)

        yield Data, labels


def main():
    #將單一一支股票當作validation
    pic_train = []
    table_train = []
    pic_validation = []
    table_validation = []

    for stockid in stocknum:
        df = pd.read_hdf('h5_data/' + stockid + '_table_sumchange.h5',
                         'stock_data_table')
        pic_addrs = []
        for i in range(len(df)):
            if (os.path.exists('stock_pic/' + stockid + 'pic/' + str(i)
                               .zfill(4) + '_' + stockid + '.jpg')) == False:
                print('FileNotFoundError error 圖片不足\n')
                raise FileNotFoundError
            pic_addrs.append('stock_pic/' + stockid + 'pic/' +
                             str(i).zfill(4) + '_' + stockid + '.jpg')

        randpickstock = randint(0, 111, size=15)
        valflag = 0
        for num in randpickstock:
            if stockid == '0051':
                break
            if stockid == list(stocknum)[num]:
                for item in pic_addrs:
                    item = item.replace('\\', '/')
                    pic_validation.append(item)
                for table in df.values:
                    table_validation.append(table)
                valflag = 1

        if valflag == 0:
            for item in pic_addrs:
                item = item.replace('\\', '/')
                pic_train.append(item)
            for table in df.values:
                table_train.append(table)

    # for i,labels in enumerate(table_train) :
    #     if labels.shape !=(3,):
    #         raise ValueError
    #     print(i)
    # for i,labels in enumerate(table_validation) :
    #     if labels.shape !=(3,):
    #         raise ValueError
    #     print(i)

    # for i,picaddr in enumerate(pic_train) :
    #     pic = imageio.imread(picaddr,format='jpg')
    #     if pic.shape !=(224,224,3):
    #         raise ValueError
    #     print(i)
    # for i,picaddr in enumerate(pic_validation) :
    #     pic = imageio.imread(picaddr,format='jpg')
    #     if pic.shape !=(224,224,3):
    #         raise ValueError
    #     print(i)
    print('pic_train shape = ' + str(len(pic_train)))
    print('table_train shape = ' + str(len(table_train)))
    print('pic_validation shape = ' + str(len(pic_validation)))
    print('table_validation shape = ' + str(len(table_validation)))

    c = list(zip(pic_train, table_train))
    shuffle(c)
    addrs, labels = zip(*c)
    train_addrs = list(addrs)
    train_labels = list(labels)

    vc = list(zip(pic_validation, table_validation))
    shuffle(vc)
    addrs, labels = zip(*vc)
    val_addrs = list(addrs)
    val_labels = list(labels)

    
    print('reading model......\n')
    parallel_model = keras.models.load_model(
        'best_acc.h5', custom_objects={"tf": tf})
    print('read model OK')
    # with tf.device('/cpu:0'):
        # model = ResNet50(
        #     input_shape=(224, 224, 3),
        #     classes=3,
        #     pooling='max',
        #     include_top=True,
        #     weights=None)
        # model = InceptionResNetV2(
        #     input_shape=(224, 224, 3), classes=3, include_top=True,weights=None)
    # parallel_model = Xception(
    #     input_shape=(224, 224, 3),
    #     classes=3,
    #     include_top=True,
    #     weights=None)

    # # parallel_model = multi_gpu_model(model, gpus=4)
    # parallel_model.compile(
    #     loss='categorical_crossentropy',
    #     optimizer='adam',
    #     metrics=['accuracy'])

    parallel_model.summary()
    print('\n\n\n\n')

    # if not os.path.exists('sum_pic.h5'):
    #     sum_pic = h5py.File('sum_pic.h5', mode='w')

    #     sum_pic.create_dataset(
    #         'pic', shape=(len(addrs), 224, 224, 3), dtype=np.uint8)

    #     for i, addr in enumerate(addrs):
    #         pic = imageio.imread(addr)
    #         sum_pic['pic'][i] = pic
    #         if i % 100 == 0:
    #             print(str(i) + ' / ' + str(len(addrs)) + '  pic done')

    #     sum_pic.create_dataset('table', data=labels, dtype=np.uint8)

    # else:
    #     sum_pic = h5py.File('sum_pic.h5', mode='r')

    batch_size = 10

    model_checkpoint_save = ModelCheckpoint(
        "best_acc.h5",
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    model_checkpoint = EarlyStopping(
        monitor='val_acc', verbose=1, mode='max', min_delta=0.0001, patience=8)

    # train_history = parallel_model.fit(x=sum_pic['pic'][0:int(len(sum_pic['table'])*0.9)],
    #                                    y=sum_pic['table'][0:int(
    #                                        len(sum_pic['table'])*0.9)],
    #                                    #steps_per_epoch=None,
    #                                    steps_per_epoch=50000,
    #                                    epochs=1,
    #                                    #batch_size=batch_size,
    #                                    verbose=1, callbacks=[model_checkpoint],
    #                                    validation_split=0.2, shuffle=True)

    train_history = parallel_model.fit_generator(
        generate_from_file(train_addrs, train_labels, batch_size),
        steps_per_epoch=len(train_addrs) // batch_size // 3,
        # steps_per_epoch=1000,
        epochs=50,
        verbose=1,
        validation_data=generate_from_file(
            val_addrs[0:int(len(val_addrs) * 0.7)],
            val_labels[0:int(len(val_addrs) * 0.7)], batch_size),
        validation_steps=len(val_addrs[0:int(len(val_addrs) * 0.7)]) //
        batch_size//3,
        #validation_steps=100,
        # workers=mp.cpu_count(),
        workers=7,
        max_queue_size=1000,
        shuffle=True,
        callbacks=[model_checkpoint,model_checkpoint_save],
        use_multiprocessing=True)

    loss, accuracy = parallel_model.evaluate_generator(
        generate_from_file(val_addrs[int(len(val_addrs) * 0.7):],
                           val_labels[int(len(val_addrs) * 0.7):], batch_size),
        steps=len(val_addrs[int(len(val_addrs) * 0.7):]) // batch_size,
        # workers=mp.cpu_count(),
        workers=7,
        use_multiprocessing=True,
        max_queue_size=1000)

    print(train_history)
    plt.figure()
    plt.plot(train_history.history['acc'])
    plt.savefig(
        'model_accuracy' + str(round(accuracy, 2)) + '.jpg',
        dpi=200,
        bbox_inches='tight',
        mode='w')
    plt.figure()
    plt.plot(train_history.history['loss'])
    plt.savefig(
        'model_loss' + str(round(loss, 2)) + '.jpg',
        dpi=200,
        bbox_inches='tight',
        mode='w')

    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)

    parallel_model.save('my_model_' + str(round(accuracy, 3)) + '.h5')


main()
