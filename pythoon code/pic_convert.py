import PIL
import glob
from PIL import Image


def pic_convert(stockid):

    piclist = glob.glob('stock_pic/'+stockid+'pic/*.jpg')

    for pic in piclist:
        picdata = PIL.Image.open(pic)
        picdata = picdata.resize((224, 224), Image.ANTIALIAS)
        #picdata = picdata.convert('RGB')
        picdata.save(pic[:-4]+'.jpg')
        print(pic[:-4]+'.jpg')
