from PyQt5 import QtCore  #import QUrl, QObject, pyqtSlot
from PyQt5 import QtGui  #import QGuiApplication
from PyQt5 import QtQuick  #import QQuickView
import pandas as pd
import numpy as np
import os
import threading
from stock_data_download_process import stock_data_process, stock_data_download
from stock_data_maketable import stock_recordchange, stock_tablemake
import multiprocessing as mp
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
    modelpath = filepath + 'my_model_0.81.h5'
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
        self.stockid = stockid
        self.df = pd.read_hdf(self.h5data_path + stockid + 'new.h5',
                              'stock_data')
        self.max_stockidpic_num = len(self.df) - 50 + 1

        processthread = threading.Thread(
            target=self.busy_thread, name='processthread')
        processthread.start()

        self.creatpath()
        if self.if_retrain == 1:
            self.retrain_adjust()
        # self.check_pic_h5data(self.stocknum, self.h5data_path, self.pic_path)

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
                print('error')
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
        print(str(self.df.iloc[self.today + 50 - 1 - 1, 0]))
        return str(self.df.iloc[self.today + 50 - 1 - 1, 0])

    @QtCore.pyqtSlot(result=str)
    def return_picaddr(self):
        return str('../stock_pic/' + self.stockid + 'pic/' +
                   str(self.today).zfill(4) + '_' + self.stockid + '.jpg')

    @QtCore.pyqtSlot(result=int)
    def return_maxpicnum(self):
        return self.max_stockidpic_num

    @QtCore.pyqtSlot(result=str)
    def return_todayprice(self):
        return str(self.df.iloc[self.today + 50 - 1 - 1, 6])

    @QtCore.pyqtSlot(result=str)
    def return_money(self):
        return str(round(self.money, 3)) + ' $$'

    @QtCore.pyqtSlot(result=str)
    def return_income(self):
        income = self.money
        for i in range(len(self.mystocklist)):
            income += self.df.iloc[self.today + 50 - 1 - 1, 6]
        return str(round(income,3))

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
        time.sleep(3)
        self.busysig.emit(1)
        if self.model == None:
            print('reading model......\n')
            self.model = keras.models.load_model(
                self.modelpath, custom_objects={"tf": tf})
            print('read model OK')
        self.graph = tf.get_default_graph()
        self.busysig.emit(0)

        while True:
            if self.select == 'download':
                self.select = None
                stock_data_download(self.stocknum, self.from_years,
                                    self.rawdata_path)
                self.busysig.emit(0)

            if self.select == 'process':
                self.select = None
                stock_data_process(self.stocknum, 1991, self.rawdata_path,
                                   self.h5data_path)
                self.busysig.emit(0)

            if self.select == 'drawpic':
                self.select = None
                stocknum_list = list(self.stocknum)
                pool = Pool(mp.cpu_count())
                res = pool.map(kdraw.drawpic, stocknum_list)
                print(res)
                self.busysig.emit(0)

            if self.select == 'tablelize':
                self.select = None
                stock_recordchange(self.stocknum, self.X_window,
                                   self.Y_slicing, self.K_changedays,
                                   self.h5data_path)
                stock_tablemake(self.stocknum, self.h5datapath)
                self.busysig.emit(0)

            if self.select == 'predict':
                self.select = None
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
                    mode='w')
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

# from PyQt5.QtCore import QUrl, QObject, pyqtSlot
# from PyQt5.QtGui import QGuiApplication
# from PyQt5.QtQuick import QQuickView

# class MyClass(QObject):
#     @pyqtSlot(int, result=str)    # 声明为槽，输入参数为int类型，返回值为str类型
#     def returnValue(self, value):
#         return str(value+10)

# if __name__ == '__main__':
#     path = 'test2.qml'   # 加载的QML文件
#     app = QGuiApplication([])
#     view = QQuickView()
#     con = MyClass()
#     context = view.rootContext()
#     context.setContextProperty("con", con)
#     view.engine().quit.connect(app.quit)
#     view.setSource(QUrl(path))
#     view.show()
#     app.exec_()