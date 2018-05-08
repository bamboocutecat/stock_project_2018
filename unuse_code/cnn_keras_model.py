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
from keras.utils import Sequence
from keras.utils import multi_gpu_model
from keras.preprocessing.image import img_to_array
from numpy import float
from multi_gpu_model_fixed import multi_gpu_model

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

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # set_session(tf.Session(config=config))

    pic_sum = []
    table_sum = []

    for stockid in stocknum:
        df = pd.read_hdf('table/' + stockid + '_table_sumchange.h5',
                         'stock_data_table')
        #pic_addr = glob.glob('stock_pic/' +stockid + 'pic/'+'*.jpg')
        pic_addrs = []
        for addr in range(len(df)):
            if (os.path.exists('stock_pic/' + stockid + 'pic/' + str(addr)
                               .zfill(4) + '_' + stockid + '.jpg')) == False:
                print('FileNotFoundError error\n')
                raise FileNotFoundError
            pic_addrs.append('stock_pic/' + stockid + 'pic/' +
                             str(addr).zfill(4) + '_' + stockid + '.jpg')
        #pic_addr = pic_addr.sort()
        # print(pic_addrs)
        for item in pic_addrs:
            item = item.replace('\\', '/')
            #print(item)
            pic_sum.append(item)
        for table in df.values:
            table_sum.append(table)

        print(len(pic_sum))
        print(len(table_sum))

    print(pic_sum[:100])
    c = list(zip(pic_sum, table_sum))
    shuffle(c)
    addrs, labels = zip(*c)
    addrs = list(addrs)
    labels = list(labels)

    train_addrs = addrs[0:int(0.7 * len(addrs))]
    train_labels = labels[0:int(0.7 * len(labels))]
    val_addrs = addrs[int(0.7 * len(addrs)):int(0.9 * len(addrs))]
    val_labels = labels[int(0.7 * len(addrs)):int(0.9 * len(addrs))]
    test_addrs = addrs[int(0.9 * len(addrs)):]
    test_labels = labels[int(0.9 * len(labels)):]

    with tf.device('/cpu:0'):
        model = ResNet50(
            input_shape=(224, 224, 3),
            classes=3,
            pooling='max',
            include_top=True,
            weights=None)

    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

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

    model_checkpoint = ModelCheckpoint(
        "best_acc.h5",
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')

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
        steps_per_epoch=len(train_addrs) // batch_size,
        #steps_per_epoch=100,
        epochs=10,
        verbose=1,
        validation_data=generate_from_file(val_addrs, val_labels, batch_size),
        validation_steps=len(val_addrs) // batch_size,
        #validation_steps=100,
        workers=mp.cpu_count(),
        max_queue_size=1000,
        shuffle=True,
        callbacks=[model_checkpoint],
        use_multiprocessing=True)

    loss, accuracy = parallel_model.evaluate_generator(
        generate_from_file(test_addrs, test_labels, batch_size),
        # val_samples=len(test_addrs),
        steps=len(test_addrs) // batch_size,
        workers=mp.cpu_count(),
        use_multiprocessing=True,
        verbose=1,
        max_queue_size=1000)

    print(train_history)
    plt.figure()
    plt.plot(train_history.history['acc'])
    plt.savefig(
        'model_accuracy' + str(round(accuracy, 2)) + '.pdf',
        dpi=200,
        bbox_inches='tight',
        mode='w')
    plt.figure()
    plt.plot(train_history.history['loss'])
    plt.savefig(
        'model_loss' + str(round(loss, 2)) + '.pdf',
        dpi=200,
        bbox_inches='tight',
        mode='w')

    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)
    # plt.figure()
    # for i in range(10):
    #     data = imageio.imread(test_addrs[i])
    #     data = data.reshape(1, 380, 383, 3)
    #     print(parallel_model.predict(data))
    #     print(parallel_model.predict(data).shape)

    # plt.show()

    # print(list(parallel_model.predict(data,batch_size=10,verbose=1)))

    #accPrint = int(accuracy * 10000)
    parallel_model.save('my_model_' + str(round(accuracy, 2)) + '.h5')

    # ------------ save the template model rather than the gpu_mode ----------------
    # serialize model to JSON
    # model_json = parallel_model.to_json()
    # with open('model_json.json','w') as jsonfile:
    #     jsonfile.write(model_json)
    # parallel_model.save_weights('model.h5')
    # print('save model done!\n')


main()