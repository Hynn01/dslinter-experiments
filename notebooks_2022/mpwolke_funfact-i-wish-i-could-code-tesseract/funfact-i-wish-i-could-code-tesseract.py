#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import io #The io module provides Python’s main facilities for dealing with various types of I/O.
import json #JSON (JavaScript Object Notation) is a lightweight data-interchange format
import cv2 # cv2.imread(), cv2.imshow() , cv2.imwrite()
import requests #Make a request to a web page, and print the response text
import matplotlib.pyplot as plt  #Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <center style="font-family:verdana;"><h1 style="font-size:200%; padding: 10px; background: #DC143C;"><b style="color:white;">Heartstopper is about love, friendship, loyalty.</b></h1></center>

# "Now an acclaimed live-action Netflix series! Boy meets boy. Boys become friends. Boys fall in love. A bestselling LGBTQ+ graphic novel about life, love, and everything that happens in between."
# 
# "Charlie and Nick are at the same school, but they’ve never met … until one day when they’re made to sit together. They quickly become friends, and soon Charlie is falling hard for Nick, even though he doesn’t think he has a chance."
# 
# "By Alice Oseman, winner of the YA Book Prize, Heartstopper is about love, friendship, loyalty and mental illness. It encompasses all the small stories of Nick and Charlie’s lives that together make up something larger, which speaks to all of us."
# 
# https://aliceoseman.com/graphic-novel/heartstopper-volume-one-tv/

# ![](https://aliceoseman.com/wp-content/uploads/2022/04/Character-profiles-featured-image-wpv_600x450_center_center.png)https://aliceoseman.com/heartstopper/

# In[ ]:


#import the necessary libraries and load the image using matplotlib. 
#
img = cv2.imread("../input/cusersmarildownloadsfactpng/fact.png")
height, width, _ = img.shape
height
width,height


# In[ ]:


plt.figure(figsize=(20,12))
plt.imshow(img);


# In[ ]:


url_api = "https://api.ocr.space/parse/image"


# In[ ]:


# Ocr
url_api = "https://api.ocr.space/parse/image"
_, compressedimage = cv2.imencode(".jpg", img, [1, 90])
file_bytes = io.BytesIO(compressedimage)


# In[ ]:


#you execute this code 
"""
result = requests.post(url_api,
              files = {"screenshot.jpg": file_bytes},
              data = {"apikey": "YOURAPIKEYHERE",
                      "language": "eng"})

"""


# In[ ]:


result = requests.post(url_api,
              files = {"../input/cusersmarildownloadsfactpng/fact.png": file_bytes},
              data = {"apikey": "eb516eb1f288957",
                      "language": "eng"})


# In[ ]:


result = result.content.decode()
result = json.loads(result)


# In[ ]:


result


# In[ ]:


parsed_results = result.get("ParsedResults")[0]
text_detected = parsed_results.get("ParsedText")
text_detected


# #Extract text using tesseract

# In[ ]:


# Generic Libraries
from PIL import Image

import re,string,unicodedata

#Tesseract Library
import pytesseract

#Warnings
import warnings
warnings.filterwarnings("ignore")

#Garbage Collection
import gc

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pytesseract


# In[ ]:


# Let's start with a simple image
img = cv2.imread("../input/cusersmarildownloadsfactpng/fact.png") # image in BGR format
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize = [10,10])
height,width,channel = img.shape
plt.imshow(img)
print(type(img))
print(height,width,channel)


# In[ ]:


# as the image is simple enough, image_to_string method reads all characters almost perfectly!
text = pytesseract.image_to_string(img)
print(text)


# In[ ]:


# the output of OCR can be saved in a file in necessary
file = open('output.txt','a') # file opened in append mode
file.write(text)
file.close()


# #Third Method

# In[ ]:


get_ipython().system('pip install https://github.com/myhub/tr/archive/1.5.1.zip')


# In[ ]:


from tr import *
from PIL import Image, ImageDraw, ImageFont


# In[ ]:


img_pil = Image.open("../input/cusersmarildownloadsfactpng/fact.png")
MAX_SIZE = 2000
if img_pil.height > MAX_SIZE or img_pil.width > MAX_SIZE:
    scale = max(img_pil.height / MAX_SIZE, img_pil.width / MAX_SIZE)

    new_width = int(img_pil.width / scale + 0.5)
    new_height = int(img_pil.height / scale + 0.5)
    img_pil = img_pil.resize((new_width, new_height), Image.BICUBIC)

print(img_pil.width, img_pil.height)
#img_pil


# #Now it's colorful!

# In[ ]:


gray_pil = img_pil.convert("L")

rect_arr = detect(img_pil, FLAG_RECT)

img_draw = ImageDraw.Draw(img_pil)
colors = ['red', 'green', 'blue', "yellow", "pink"]

for i, rect in enumerate(rect_arr):
    x, y, w, h = rect
    img_draw.rectangle(
        (x, y, x + w, y + h),
        outline=colors[i % len(colors)],
        width=4)

img_pil


# In[ ]:


#Code by Olga Belitskaya https://www.kaggle.com/olgabelitskaya/sequential-data/comments
from IPython.display import display,HTML
c1,c2,f1,f2,fs1,fs2='#DC143C','#DC143C','Akronim','Smokum',30,15
def dhtml(string,fontcolor=c1,font=f1,fontsize=fs1):
    display(HTML("""<style>
    @import 'https://fonts.googleapis.com/css?family="""\
    +font+"""&effect=3d-float';</style>
    <h1 class='font-effect-3d-float' style='font-family:"""+\
    font+"""; color:"""+fontcolor+"""; font-size:"""+\
    str(fontsize)+"""px;'>%s</h1>"""%string))
    
    
dhtml('Thanks for your patience – please keep coming back to see my improvements, @mpwolke Was Here.' )


# #Acknowledgements:
# 
# Olga Belitskaya https://www.kaggle.com/olgabelitskaya/sequential-data/comments
# 
# Sabbir Ahmed https://www.kaggle.com/ggck43/ocr-using-pytesseract-bengali-english
# 
# Naim Mhedhbi. https://www.kaggle.com/naim99/ocr-text-recognition-ocr-space-api-tesseract/data
