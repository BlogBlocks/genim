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



def GenIm0():
    imsize0 = (640,640)
    img0 = Image.new('RGBA', imsize, (0, 0, 0, 0))

def GenIm1():
    imsize1 = (640,640)
    img1 = Image.new('RGBA', imsize, (0, 0, 0, 0))


def blendIms(img0,img1,alpha):
    halo = Image.new('RGBA', img.size, (0, 0, 0, 0))

    
def RanFile(path):
    base_image = random.choice([
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
        ])
    ranfile=(path+base_image)
    return ranfile
    # gets a random file from a directory. The directory needs to contain only the filetypes required
    # Use:
    # path = r"/home/user/images/"
    # ranfile = RanFile(path)
    # print ranfile