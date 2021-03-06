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

def Verify_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception:
            pass
# creates a tmmpp/ directory for tracking and debug of image generation            
Verify_path('tmmpp/')

def Resize640(image, output):
    Bp=Image.open(image)
    width, height = Bp.size
    if width>height:
        w1=(width-height)/2
        w2 = width-w1
        h1=height-height
        h2=height
        Cc=Bp.crop((w1,h1,w2,h2))
        resizeIm = Cc.resize((640,640), Image.NEAREST)
        resizeIm.save(output)
    else:
        w1= width - width
        w2 = width
        h1 = (height-width)/2
        h2= height - h1
        Cc=Bp.crop((w1,h1,w2,h2))
        #Cc=Bp.crop((260,680,680,940))
        #Cc=Bp.crop((0,260,680,940))
        resizeIm = Cc.resize((640,640), Image.NEAREST)
        resizeIm.save(output)
        return resizeIm
        # USE example
        #image = "junk/wide.png"
        #output = "junk/wideTest4.png"
        #nim = genim.Resize640(image, output)    

def CustIm(color):
    imsize0 = (640,640)
    img0 = Image.new('RGBA', imsize0, color)
    img0.save('tmmpp/img0.png')
    return img0

    # generate and save Custom color/transparency image as: tmmpp/img0.png
    #CustIm(color)
        
    
def GenIm0():
    imsize0 = (640,640)
    img0 = Image.new('RGBA', imsize0, (255, 0, 0, 0))
    img0.save('tmmpp/img0.png')
    return img0
# generate and save blank image as: tmmpp/img0.png
GenIm0()

def GenIm1():
    imsize1 = (640,640)
    img1 = Image.new('RGBA', imsize1, (0,255, 0, 0))
    img1.save('tmmpp/img1.png')
    return img1
# generate and save blank image as: tmmpp/img1.png
GenIm1()    


def BlendIms(img0a,img1a,alpha):
    op1 = Image.open(img0a).convert("RGBA")
    op2 = Image.open(img1a).convert("RGBA")
    resulta = ImageChops.blend(op1,op2,alpha)
    resulta.save('tmmpp/resulta.png')
    return resulta

def Brightness(fileN, output, alpha):
    alpha = float(alpha)
    im3 = Image.open(fileN)
    # multiply each pixel by < 1.0 (darker image)
    # multiply each pixel by > 1.0 (lighter image)
    # works best with .jpg and .png files
    # note that lambda is akin to a one-line function
    #im2 = im1.point(lambda p: p * 0.9)
    im4 = im3.point(lambda p: p * alpha)
    im4.save(output)
    return im4
    #Usage Example:
    #fileN = "junk/wideTest4.png"
    #output= "tmmpp/brightnessTest3.png"
    #alpha = 1.2
    #genim.brightness(fileN, output, alpha)
    #view = Image.open("tmmpp/brightnessTest3.png")
    #view
def wordcloud(inputF,outputF):
    title = "Python Stuff"
    signature_ = "Jack Northrup" 
    count = 1
    start = Image.open(inputF).convert('RGBA')
    start.save('tmmpp/textbacktmp.jpg')
    while count < 256 :
        base = Image.open('tmmpp/textbacktmp.jpg').convert('RGBA')

        #8 5 4 6 3 2
        # make a blank image for the text, initialized to transparent text color
        txt = Image.new('RGBA', base.size, (255,255,255,0))
        # get a font
        #generate a random size for the font
        int_n = int(count*.2)
        Fsize = randint(15,100-int_n)
        fnt = ImageFont.truetype("/home/jack/.fonts/Exo-Black.ttf", Fsize)
        # get a drawing context
        d = ImageDraw.Draw(txt)

        width, height = base.size


        def generate_the_word(infile):
            with open(infile) as f:
                contents_of_file = f.read()
            lines = contents_of_file.splitlines()
            line_number = random.randrange(0, len(lines))
            return lines[line_number]
        textin = (generate_the_word("wordcloud.txt"))

        # calculate the x,y coordinates of the text
        w, h = base.size
        Lw = randint(-150,w-50)
        Lh = randint(-50,h-30)
        #textin = "The TwitterBot Project" 
        #generate random color and opacity
        r = randint(0,256)
        g = randint(0,256)
        b = randint(0,256)
        a = randint(0,count)
        d.text((Lw,Lh), textin, font=fnt, fill=(r,g,b,a))

        out = Image.alpha_composite(base, txt)
        out.save("tmmpp/textbacktmp.jpg", "JPEG")
        count=count+1

    #base = Image.open('images/NewFolder/lightning01.jpg').convert('RGBA')
    #8 5 4 6 3 2
    # make a blank image for the text, initialized to transparent text color
    txt = Image.new('RGBA', out.size, (255,255,255,0))

    # get a font
    fnt = ImageFont.truetype("/home/jack/.fonts/Exo-Black.ttf", 20)
    # get a drawing context
    d = ImageDraw.Draw(txt)

    width, height = out.size
    # calculate the x,y coordinates of the text
    #marginx = 325
    #marginy = 75
    marginx = 225
    marginy = 50
    x = width - marginx
    y = height - marginy

    d.text((x,y), signature_, font=fnt, fill=(0,0,0,256))

    out = Image.alpha_composite(out, txt)
    out.save("tmmpp/tmp.jpg", "JPEG")
    # save the image then reopen to put a title
    base = Image.open('tmmpp/tmp.jpg').convert('RGBA')
    #8 5 4 6 3 2
    # make a blank image for the text, initialized to transparent text color
    txt = Image.new('RGBA', base.size, (255,255,255,0))

    # get a font
    fnt = ImageFont.truetype("/home/jack/.fonts/Exo-Black.ttf", 50)
    # get a drawing context
    d = ImageDraw.Draw(txt)

    width, height = base.size
    # calculate the x,y coordinates of the text
    #marginx = 325
    #marginy = 75
    x = 90
    y = 10

    d.text((x,y), title , font=fnt, fill=(0,0,0,250))

    out2 = Image.alpha_composite(base, txt)
    out2.save(outputF, "JPEG")
    out2.show()
    
    #outputF = 'tmmpp/wordcloud_01.jpg'
    #inputF = 'tmmpp/brightnessTest3.png'
    #wordcloud(inputF,outputF)

def RanFile(path):
    base_image = random.choice([
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
        ])
    ranfile=(path+base_image)
    return ranfile

def Draw_text_with_halo(img, position, text, font, col, halo_col):
    halo = Image.new('RGBA', img.size, (0, 0, 0, 0))
    font = ImageFont.truetype("/home/jack/.fonts/Exo-Black.ttf", 40)
    ImageDraw.Draw(halo).text(position, text, font = font, fill = halo_col)
    blurred_halo = halo.filter(ImageFilter.BLUR)
    ImageDraw.Draw(blurred_halo).text(position, text, font = font, fill = col)
    return Image.composite(img, blurred_halo, ImageChops.invert(blurred_halo))

def Draw_text_with_effects(img, (x,y), (xf,yf), text, font, col, halo_col):
    halo = Image.new('RGBA', img.size, (0, 0, 0, 0))
    font = ImageFont.truetype("/home/jack/.fonts/Exo-Black.ttf", 40)
    ImageDraw.Draw(halo).text((x-xf,y+xf), text, font = font, fill = halo_col)
    blurred_halo = halo.filter(ImageFilter.BLUR)
    ImageDraw.Draw(blurred_halo).text((x,y), text, font = font, fill = col)
    return Image.composite(img, blurred_halo, ImageChops.invert(blurred_halo))
    '''
    Usage:
    img = Image.open("tmmpp/imblend.png")
    x=200;y=200
    font = ImageFont.truetype("/home/jack/.fonts/Exo-Black.ttf", 40)
    text = "Place Text in Image"
    col= (255,0,0)
    xf=10;yf=10
    halo_col = (255,255,255)
    Draw_text_with_effects(img, (x,y), (xf,yf), text, font, col, halo_col)
    '''
def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array

def applybin(inputA, outputA):
    im = Image.open(inputA)
    im_grey = im.convert('LA') # convert to grayscale
    width,height = im.size

    total=0
    for i in range(0,width):
        for j in range(0,height):
            total += im_grey.getpixel((i,j))[0]

    mean = total / (width * height)
    image_file = Image.open(inputA)
    imagex = image_file.convert('L')  # convert image to monochrome
    imagey = np.array(imagex)
    binim = binarize_array(imagey, mean)
    cv2.imwrite(outputA, binim)
    return binim
    #inputA ="tmmpp/640_iguana.jpg"
    #outputA = "tmmpp/binim02.png"
    #binim = applybin(inputA, outputA)
    
def bw(imgFile,outFile):
    #convert image to blackand white
    col = Image.open(imgFile)
    gry = col.convert('L')
    grarray = np.asarray(gry)
    bw = (grarray > grarray.mean())*255
    cv2.imwrite(outFile, bw)
    return bw
    #imgFile= 'tmmpp/Effects.png'
    #outFile= 'tmmpp/bw3.png'
    #genim.bw(imgFile,outFile)

def Generate_the_word(infile):
    with open(infile) as f:
        contents_of_file = f.read()
        lines = contents_of_file.splitlines()
        line_number = random.randrange(0, len(lines))
        return lines[line_number]
    
def Rndcolor():
    r = randint(50,255)
    g = randint(50,255)
    b = randint(50,255)
    rndcolor = (r,g,b) 
    return rndcolor

def Get_random_line(file_name):
    total_bytes = os.stat(file_name).st_size 
    random_point = random.randint(0, total_bytes)
    file = open(file_name)
    file.seek(random_point)
    file.readline() # skip this line to clear the partial line
    return file.readline()

def Adtext(inputIm,text):
    font = ImageFont.truetype("/home/jack/.fonts/Exo-Black.ttf", 40)
    col = (255, 255,230) # bright green
    halo_col = (0, 0, 0)   # black
    Draw_text_with_halo(img, position, text, font, col, halo_col)
    # USE: img = draw_text_with_halo(inputIm, (15, 8), text, font, text_color, halo_color)
    return text
    
def Signat(imfile,imout):
    fnt = ImageFont.truetype("/home/jack/.fonts/Exo-Black.ttf", 25)
    width, height = imfile.size
    marginx = 325
    marginy = 35
    x = width - marginx
    y = height - marginy
    signature_ = "The TwitterBot Project" 
    text_col2 = (255, 255,230) # bright green
    halo_col2 = (0, 0, 0)   # black
    imfile = Draw_text_with_halo(imfile,(x,y), signature_ , fnt, text_col2, halo_col2) 
    imfile.save(imout)
    return imfile

def Centercut(path, img1, new_width, new_height):
    img = Image.open(path+"/"+img1)
    w_var=new_width/2
    h_var=new_height/2
    half_the_width = img.size[0] / 2
    half_the_height = img.size[1] / 2
    img4 = img.crop(
        (
            half_the_width - w_var,
            half_the_height - h_var,
            half_the_width + w_var,
            half_the_height + h_var
        )
    )
    return img4

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
