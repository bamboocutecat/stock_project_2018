# coding:utf-8
from stock_data_download_process import stock_data_process, stock_data_download
from stock_data_maketable import stock_tablelize
import kdraw
import os
from multiprocessing import Pool
import multiprocessing as mp

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


filepath = os.path.abspath('.')
filepath = filepath + '/'
Y_slicing = 1
X_window = 50
K_changedays = 50
from_years = 2018
rawdata_path = filepath + 'raw_data/'
h5data_path = filepath + 'h5_data/'
pic_path = filepath + 'stock_pic/'

# stock_data_download(stocknum,from_years,rawdata_path)
# stock_data_process(stocknum,1991,rawdata_path,h5data_path)
# stock_tablelize(stocknum,h5data_path)




stocknum_list = list(stocknum)
pool = Pool(mp.cpu_count())
res = pool.map(kdraw.drawpic, stocknum_list)
print(res)