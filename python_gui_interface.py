from PyQt5 import QtCore  #import QUrl, QObject, pyqtSlot
from PyQt5 import QtGui  #import QGuiApplication
from PyQt5 import QtQuick  #import QQuickView
import pandas as pd
import numpy as np
import os
import threading
from stock_data_download_process import stock_data_process, stock_data_download
from stock_data_maketable import stock_tablelize
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Pool
import kdraw
import sys
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import glob2
import imageio
if sys.platform.startswith('linux'):
    from OpenGL import GL
import tensorflow as tf
import keras
import subprocess
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))

filepath = os.path.abspath('.')
filepath = filepath + '/'


class MyClass(QtCore.QObject):
    # Var declare
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
    select = None
    today = 0
    modelpath = filepath + 'best_acc_1.h5'
    model = None
    graph = None
    max_stockidpic_num = 0
    stockid = None
    df = None
    mystocklist = []
    money = 1000
    Y_slicing = 1
    X_window = 50
    K_changedays = 50
    from_years = 2018
    #from_months = 1
    rawdata_path = filepath + 'raw_data/'
    h5data_path = filepath + 'h5_data/'
    table_path = filepath + 'table/'
    pic_path = filepath + 'stock_pic/'
    predict_pic = filepath + 'predict_pic/'
    if_retrain = 0
    busysig = QtCore.pyqtSignal(bool, arguments=['indicator'])
    picsig = QtCore.pyqtSignal(str, arguments=['predict_pic'])

    def set_value(self, stockid):
        self.set_stockinfo(stockid)

        processthread = threading.Thread(
            target=self.busy_thread, name='processthread')
        processthread.start()

        self.creatpath()
        if self.if_retrain == 1:
            self.retrain_adjust()

    def set_stockinfo(self, stockid):
        self.stockid = stockid
        self.df = pd.read_hdf(self.h5data_path + stockid + 'new.h5',
                              'stock_data')
        self.max_stockidpic_num = len(self.df) - 50 + 1

    def creatpath(self):
        if not os.path.isdir(self.rawdata_path):
            os.mkdir(self.rawdata_path)
        if not os.path.isdir(self.h5data_path):
            os.mkdir(self.h5data_path)
        if not os.path.isdir(self.predict_pic):
            os.mkdir(self.predict_pic)
            for stockid in self.stocknum:
                if not os.path.isdir(
                        self.predict_pic + stockid + 'predictpic/'):
                    os.mkdir(self.predict_pic + stockid + 'predictpic/')
        if not os.path.isdir(self.pic_path):
            os.mkdir(self.pic_path)
            for stockid in self.stocknum:
                if not os.path.isdir(self.pic_path + stockid + 'pic/'):
                    os.mkdir(self.pic_path + stockid + 'pic/')

    def check_pic_h5data(self, stocknum, h5data_path, pic_path):
        for stockid in stocknum:
            df = pd.read_hdf(self.h5data_path + stockid + 'new.h5',
                             'stock_data')
            #     print(len(df)-50+1-50)

            piclist = glob2.glob(self.pic_path + stockid + 'pic/*.jpg')
            #     print(len(piclist))
            if (len(df) - 50 + 1) != len(piclist):
                print(stockid + '   pic !=  h5data  error')
                print('%d  =  %d' % ((len(df) - 50 + 1), len(piclist)))

    def retrain_adjust(self):
        trained_data_count = []
        for stockid in self.stocknum:
            df = pd.read_hdf(self.h5data_path + stockid + '.h5', 'stock_data')
            trained_data_count.append((stockid, int(len(df))))

        trained_data_countnp = np.array(trained_data_count)
        pd.DataFrame(trained_data_countnp).to_hdf(
            'train_data_count.h5', 'train_data_count', mode='w')

    @QtCore.pyqtSlot(result=str)
    def return_today(self):
        print(str(self.df.iloc[self.today + 50 - 1 , 0]))
        return str(self.df.iloc[self.today + 50 - 1 , 0])

    @QtCore.pyqtSlot(result=str)
    def return_picaddr(self):
        return str('../stock_pic/' + self.stockid + 'pic/' +
                   str(self.today).zfill(4) + '_' + self.stockid + '.jpg')

    @QtCore.pyqtSlot(result=int)
    def return_maxpicnum(self):
        return self.max_stockidpic_num -1

    @QtCore.pyqtSlot(result=str)
    def return_todayprice(self):
        return str(self.df.iloc[self.today + 50 - 1, 6])

    @QtCore.pyqtSlot(result=str)
    def return_money(self):
        return str(round(self.money, 3)) + ' $$'

    @QtCore.pyqtSlot(result=str)
    def return_income(self):
        income = self.money
        oriincome = self.money
        for i in range(len(self.mystocklist)):
            income += self.df.iloc[self.today + 50 - 1 - 1, 6]
        for val in self.mystocklist:
            oriincome += float(val)

        return str(round((income / oriincome), 3) * 100) + '%'

    @QtCore.pyqtSlot()
    def add_today(self):
        self.today += 1

    @QtCore.pyqtSlot()
    def minus_today(self):
        self.today -= 1

    @QtCore.pyqtSlot(int)
    def set_today(self, sliderval):
        self.today = sliderval

    @QtCore.pyqtSlot()
    def processdata(self):
        self.select = 'process'

    @QtCore.pyqtSlot()
    def downloaddata(self):
        self.select = 'download'

    @QtCore.pyqtSlot()
    def tablelize(self):
        self.select = 'tablelize'

    @QtCore.pyqtSlot()
    def drawpic(self):
        self.select = 'drawpic'

    @QtCore.pyqtSlot()
    def predict(self):
        self.select = 'predict'

    def busy_thread(self):
        check_picprocess = Process(
            target=self.check_pic_h5data,
            args=(self.stocknum, self.h5data_path, self.pic_path))
        check_picprocess.start()

        while True:

            if self.select == 'download':
                self.select = None
                downloadprocess = Process(
                    target=stock_data_download,
                    args=([
                        self.stockid,
                    ], self.from_years, self.rawdata_path))
                downloadprocess.start()
                while True:
                    if not downloadprocess.is_alive():
                        break
                self.busysig.emit(0)

            if self.select == 'process':
                self.select = None
                self.select = None
                processdataprocess = Process(
                    target=stock_data_process,
                    args=([
                        self.stockid,
                    ], 1991, self.rawdata_path, self.h5data_path))
                processdataprocess.start()
                while True:
                    if not processdataprocess.is_alive():
                        break

                self.set_stockinfo(self.stockid)
                self.busysig.emit(0)

            if self.select == 'drawpic':
                self.select = None
                drawpicprocess = Process(
                    target=kdraw.drawpic, args=([
                        self.stockid,
                    ]))
                drawpicprocess.start()
                while True:
                    if not drawpicprocess.is_alive():
                        break
                # stocknum_list = list(self.stocknum)
                # pool = Pool(mp.cpu_count())
                # res = pool.map(kdraw.drawpic, stocknum_list)
                # print(res)
                self.busysig.emit(0)

            if self.select == 'tablelize':
                self.select = None
                tablelizeprocess = Process(
                    target=stock_tablelize,
                    args=([
                        self.stockid,
                    ], self.h5data_path))
                tablelizeprocess.start()
                self.busysig.emit(0)

            if self.select == 'predict':
                self.select = None

                if self.model == None:
                    print('reading model......\n')
                    self.model = keras.models.load_model(
                        self.modelpath, custom_objects={"tf": tf})
                    print('read model OK')
                self.graph = tf.get_default_graph()

                X_list = []
                for i in range(50):
                    pic = imageio.imread(
                        self.pic_path + self.stockid + 'pic/' +
                        str(self.today - 1 - 49 + i).zfill(4) + '_' +
                        self.stockid + '.jpg',
                        format='jpg')
                    X_list.append(pic)
                X_pridict = np.array(X_list).reshape(len(X_list), 224, 224, 3)

                with self.graph.as_default():
                    prob_array = self.model.predict(
                        X_pridict, batch_size=10, verbose=0)

                print(prob_array.shape)
                plt.figure()
                plt.plot(prob_array[:, 0], label='plus')
                plt.plot(prob_array[:, 1], label='minus')
                plt.plot(prob_array[:, 2], label='unchange')
                plt.legend(loc='best')
                plt.savefig(
                    self.predict_pic + self.stockid + 'predictpic/' +
                    str(self.today).zfill(4) + '.jpg',
                    mode='w',
                    bbox_inches='tight')
                self.picsig.emit(
                    str(self.predict_pic + self.stockid + 'predictpic/' +
                        str(self.today).zfill(4) + '.jpg'))
                print(
                    str(self.predict_pic + self.stockid + 'predictpic/' +
                        str(self.today).zfill(4) + '.jpg'))
                self.busysig.emit(0)

    @QtCore.pyqtSlot(int)
    def buystock(self, buynum):
        for i in range(buynum):
            self.mystocklist.append(
                str(self.df.iloc[self.today + 50 - 1 - 1, 6]))
            self.money -= self.df.iloc[self.today + 50 - 1 - 1, 6]
            print(str(self.df.iloc[self.today + 50 - 1 - 1, 6]))

    @QtCore.pyqtSlot(str, int)
    def sellstock(self, sellcount, sellnum):
        for i in range(sellnum):
            self.mystocklist.pop(int(sellcount))
            self.money += self.df.iloc[self.today + 50 - 1 - 1, 6]

    @QtCore.pyqtSlot(result=str)
    def showstocklist(self):
        if len(self.mystocklist) == 0:
            return str('no stock')
        self.mystocklist.sort()
        strlist = '已購買股票列表\n'
        for i, stock in enumerate(self.mystocklist):
            strlist = strlist + str(i) + '  ' + str(stock) + '\n'
        return str(strlist)


if __name__ == '__main__':

    path = filepath + 'stockgui/main.qml'
    app = QtGui.QGuiApplication([])
    view = QtQuick.QQuickView()
    stock = MyClass()

    context = view.rootContext()
    stock.set_value('0051')
    context.setContextProperty("stock", stock)
    view.engine().quit.connect(app.quit)
    view.setSource(QtCore.QUrl(path))
    view.show()
    app.exec_()
