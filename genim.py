from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageFilter
import random
from random import randint
import time
import markovify
import os
import sys
sys.path.insert(1, "/home/jack/anaconda2/envs/py27/lib/python2.7/site-packages")
import twython
from twython import Twython

def verify_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception:
            pass
# creates a tmmpp/ directory for tracking and debug of image generation            
verify_path('tmmpp/')

def GenIm0():
    imsize0 = (640,640)
    img0 = Image.new('RGBA', imsize0, (255, 0, 0, 0))
    img0.save('tmmpp/img0.png')
    return img0
# generate and save blank image tmmpp/img0.png
GenIm0()

def GenIm1():
    imsize1 = (640,640)
    img1 = Image.new('RGBA', imsize1, (0,255, 0, 0))
    img1.save('tmmpp/img1.png')
    return img1
# generate and save blank image tmmpp/img0.png
GenIm1()    


def BlendIms(img0a,img1a,alpha):
    resulta = ImageChops.blend(img0a,img1a,alpha)
    resulta.save('tmmpp/resulta.png')
    return resulta

    
def RanFile(path):
    base_image = random.choice([
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
        ])
    ranfile=(path+base_image)
    return ranfile

def SaveTempU(TempU):
    tempU = time.strftime("tmmpp/SaveTempU_%Y%m%d%H%M%S.png")
    TempU.save(tempu)
    return filepath

def SaveTemp(Temp):
    Tempname = time.strftime("tmmpp/SaveTemp.png")
    Temp.save(Tempname)
    return Tempname

def SaveName(savepath,imName):
    filename = time.strftime("SaveName_%Y%m%d%H%M%S.png")
    filepath = os.path.join(savepath,filename)
    imName.save(filepath)
    return filepath
