#ONE TIME MANUAL POSTS
#!/home/jack/anaconda2/python
import random
from random import randint
import time
import markovify
import os
import sys
sys.path.insert(1, "/home/jack/anaconda2/envs/py27/lib/python2.7/site-packages")
import twython
from twython import Twython
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageFilter

def generate_the_word(infile):
        with open(infile) as f:
            contents_of_file = f.read()
        lines = contents_of_file.splitlines()
        line_number = random.randrange(0, len(lines))
        return lines[line_number]

def draw_text_with_halo(img, position, text, font, col, halo_col):
    halo = Image.new('RGBA', img.size, (0, 0, 0, 0))
    ImageDraw.Draw(halo).text(position, text, font = font, fill = halo_col)
    blurred_halo = halo.filter(ImageFilter.BLUR)
    ImageDraw.Draw(blurred_halo).text(position, text, font = font, fill = col)
    return Image.composite(img, blurred_halo, ImageChops.invert(blurred_halo))

def rndcolor():
    r = randint(50,255)
    g = randint(50,255)
    b = randint(50,255)
    rndcolor = (r,g,b) 
    return rndcolor

def get_random_line(file_name):
    total_bytes = os.stat(file_name).st_size 
    random_point = random.randint(0, total_bytes)
    file = open(file_name)
    file.seek(random_point)
    file.readline() # skip this line to clear the partial line
    return file.readline()



if __name__ == '__main__':
    nap = randint(500,1200)
    time.sleep(nap)
    isize = (640,640)     
    inp = Image.new('RGB', isize)
    font = ImageFont.truetype("/home/jack/.fonts/Exo-Black.ttf", 40)
    text_col = (255, 255,230) # bright green
    halo_col = (0, 0, 0)   # black
    textin = (generate_the_word("wordcloud.txt"))
    i2 = draw_text_with_halo(inp, (15, 8), "SaltMan", font, text_col, halo_col)
    
    #txt = Image.new('RGBA', i.size, (255,255,255,0))

    # get a font
    fnt = ImageFont.truetype("/home/jack/.fonts/Exo-Black.ttf", 20)
    # get a drawing context
    width, height = inp.size
    marginx = 225
    marginy = 35
    x = width - marginx
    y = height - marginy
    signature_ = "The TwitterBot Project" 
    #text_col2 = (150, 255, 150) # bright green
    #halo_col2 = (0, 0, 0)   # black
    text_col2 = (255, 255,230) # bright green
    halo_col2 = (0, 0, 0)   # black
    #text_col2 = (0, 0, 0)  # bright green
    #halo_col2 = (255, 255,230)  # black    
    txt1=draw_text_with_halo(i2,(x,y), signature_, fnt, text_col2, halo_col2)
    
    
    
    # get a font
    fs=randint(15,24)
    fnt = ImageFont.truetype("/home/jack/.fonts/Exo-Black.ttf", fs)
    # get a drawing context
    width, height = inp.size
    marginx = 225
    marginy = 35
    x = width - marginx
    y = height - marginy
    signature_ = "The TwitterBot Project" 
    #text_col2 = (150, 255, 150) # bright green
    #halo_col2 = (0, 0, 0)   # black
    #text_col2 = (255, 255,230) # bright green
    text_col2 = rndcolor()
    halo_col2 = (0, 0, 0)   # black
    #text_col2 = (0, 0, 0)  # bright green
    #halo_col2 = (255, 255,230)  # black 
    yy=randint(70,290)
    xx=randint(5,60)
    #iword = (text_model.make_short_sentence(50))
    file_name = 'saltman.txt'
    iword = get_random_line(file_name)
    
    txt3=draw_text_with_halo(txt1,(xx,yy), iword, fnt, text_col2, halo_col2)
   
    vv=randint(320,530)
    vvv=randint(5,10)
    #iword = (text_model.make_short_sentence(50))
    file_name = 'saltman.txt'
    lword = get_random_line(file_name)        
    text_col3 = rndcolor()
    fs2=randint(15,24)
    fnt = ImageFont.truetype("/home/jack/.fonts/Exo-Black.ttf", fs2)
    txt=draw_text_with_halo(txt3,(vvv,vv), lword, fnt, text_col3, halo_col2)
     
    out = Image.alpha_composite(i2, txt)
    out.save("tmp/TM_POST.jpg")

#removed keys for privacy reasons
CONSUMER_KEY = 'YazCRIfWX4VICiRCOiph08jDL'
CONSUMER_SECRET = 'QOkLHou6NMwkghSHjMFXMdffQKJlDzttKtP6uBCcZ4VlQtvJyc'
ACCESS_KEY = '296906916-AWggjhqpEWIS7EzXXhc2pOPBeCVJczpOm11cQGIf'
ACCESS_SECRET = 'zFrCiyaPt8gCBVVs1bLCmdCSyQQ3DKxT5wHJq2tOu2AMj'

twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_KEY, ACCESS_SECRET)
#path = 'images/NewFolder'
#f = open("Mine.txt")
#text = f.read()
# Build the model.
#text_model = markovify.Text(text)
# Print randomly-generated sentences of no more than 140 characters
#http://paulbourke.net/fractals/
#STR = (text_model.make_short_sentence(140))
#random.choice(open('Mine.txt').readlines())



file_name = 'Mine.txt'
STR = get_random_line(file_name)

#STR = ("Sometimes we have visitors at night. If lucky we don't wake up.")
#PATH = "/home/jack/Desktop/deep-dream-generator/notebooks/STUFF/experiment/experiment8.jpg"
PATH = "tmp/TM_POST.jpg"
# 1 , 2, 3, 12, 5, 15, 8, 6
#photo = open('/home/jack/Desktop/deep-dream-generator/notebooks/images/'+file_list[rnd]+'.jpg','rb')

photo = open(PATH,'rb')
response = twitter.upload_media(media=photo)
twitter.update_status(status=STR, media_ids=[response['media_id']])
#print STR