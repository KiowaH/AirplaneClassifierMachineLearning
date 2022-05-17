# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:45:33 2022

To Resize Images:
    Open CV
    imageMagick

@author: Kiowa

FGVC-Aircraft
https://visualdata.io/discovery
"""


"""
1522867 C-130
0774035 C-130
1364712 C-130
1085194 C-130
1414395 C-130
1794564 C-130
1033590 C-130
0939510 C-130
2260054 C-130
0967838 C-130
0773222 C-130
0280348 C-130
1940180 C-130
1543648 C-130
1931897 C-130
2049579 C-130
0434345 C-130
0771762 C-130
1404391 C-130
1196802 C-130
0610659 C-130
1534869 C-130
0573367 C-130
0875344 C-130
0962825 C-130
0772253 C-130
1522562 C-130
0958085 C-130
1373821 C-130
0883339 C-130
1055698 C-130
0978113 C-130
0977928 C-130
1048528 C-130

0735402 F-16
1307230 F-16
1223924 F-16
1564426 F-16
0934121 F-16
1499817 F-16
0736498 F-16
2136394 F-16
2118954 F-16
2147025 F-16
2243568 F-16
1754078 F-16
1596393 F-16
1601715 F-16
2114079 F-16
2145444 F-16
1588718 F-16
1196995 F-16
1123669 F-16
1364706 F-16
1259625 F-16
1223797 F-16
1476963 F-16
1706712 F-16
1463710 F-16
1613947 F-16
2123710 F-16
1592597 F-16
2114113 F-16
1922017 F-16
2123565 F-16
1552593 F-16
0737605 F-16
1588720 F-16
"""

from matplotlib import image
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from glob import glob
import re
import numpy as np
import matplotlib.image as img
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential 
from keras.layers import Dense
from keras.utils import np_utils
import keras
import theano
import tensorflow as tf

boxes = pd.read_csv("images_box.txt", delimiter=" ", names=["id", "x1", "y1", "x2", "y2"])

#val = pd.concat([boxes[1], boxes[2], boxes[3]], axis=1)

#dict = {key: boxes[0], value: val}

boundingDictionary = {}
x = 0

with open("images_box.txt") as f:
    Lines = f.readlines()
    for line in Lines:
        linetmp = line.split(" ")
        #print(linetmp[0])
        values = linetmp[1], linetmp[2], linetmp[3], linetmp[4]
        tmp = {linetmp[0]: values}
        boundingDictionary.update(tmp)
           

ac1301 = Image.open(r'images/1522867.jpg')
ac1302 = Image.open(r'images/0774035.jpg')

f161 = Image.open(r'images/0735402.jpg')
f162 = Image.open(r'images/1307230.jpg')

"""
To crop images create dictionary with key being Image_box plane# and value being x1, y1, x2, y2
then read in images and search dictionary to get cropping values then resize
"""

files = glob("images/*.jpg")
y=0
imgNames = []
while y < (len(files)):
    test = [files[y][7:14]]
    imgNames.append(test)
    #print(test)
    y+=1
    
#for x in imgNames:
    #print("imgName = " + str(x))
    
    

#for key, value in boundingDictionary.items():
    #print("KeyVal = " + str(key))
imgList = []

for i in imgNames:
    for x in i:
        imgList.append(x.split("'")[0])
str1 = ' '
z = 0
#Crop and resize img function
"""
for x in imgList:
    for key,value in boundingDictionary.items():
        if (x == key):
            keys = list(boundingDictionary.keys())
            index = keys.index(key)
            vals = list(boundingDictionary.values())
            left = vals[index][0]
            top = vals[index][1]
            right = vals[index][2]
            bottom = vals[index][3]
            tmp2 = "images/" + key + ".jpg"
            tmpImage = Image.open(tmp2)
            resizeIMG = tmpImage.crop((int(left), int(top), int(right), int(bottom)))
            newsize = (128, 128)
            resizeD = resizeIMG.resize(newsize)
            #picture = resizeD.save("resizedImages/" + key + ".jpg")
"""



        
#[i[0].split("'", 2) for i in imgNames]

tempSizeDF = pd.DataFrame()
pixelDF = pd.DataFrame()

file = open("resizedImage.csv", "w")
z=0

#Writing to a csv


for i in range(0, 16383):
    tmp = "pixel" + str(i)
    file.write(tmp + ',')
    
file.write('pixel16384')
file.write('\n')

for i in imgNames:
    name = str(i[0])
    #i[0].split("'", 2)
    tmp3 = "resizedImages/" + i[0] + ".jpg"
    tmpR = img.imread(tmp3)
    #tmpR = Image.open(tmp3)
    #tmpR = np.asarray(tmpR)
    tempSizeDF = tmpR[:,:,0].flatten()
    #tempSizeDF = tempSizeDF/255
    string = [str(p) for p in tempSizeDF]
    line = ','.join(string)
    file.write(name + ',' + line + '\n')

  
#pixelDF.append(tempSizeDF)

    

        #values = boundingDictionary[i]
        #tmpCrp = 'images/'+imgNames[i]+".jpg".crop(())
  
    
    
left = 18
top = 122
right = 1015
bottom = 458

ac1301 = ac1301.crop((left, top, right, bottom))

newsize = (128, 128)

ac1301 = ac1301.resize(newsize)
ac1302 = ac1302.resize(newsize)
f161 = f161.resize(newsize)
f162 = f162.resize(newsize)


pyplot.imshow(ac1301)
pyplot.show()
pyplot.imshow(ac1302)
pyplot.show()
pyplot.imshow(f161)
pyplot.show()
pyplot.imshow(f162)
pyplot.show()

df = pd.read_csv('resizedImage.csv', delimiter=',')

num_pixels = df.shape[1]

def CNN_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid',
                    input_shape=(32,32,1), data_format='channels_last',
                    activation='selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=num_pixels, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model

familys = pd.DataFrame()
familyVals = open('images_family_val.txt')
familyFile = open("familyFile.csv", "w")
familyFile.write('name' + ',' + 'label' + '\n')
    
for vals in familyVals:
    tmp = vals.split(" ", 1)
    famid = str(tmp[0])
    label = tmp[1]
    print(famid)
    print(label)
    #name = df['name']
    data = df.iloc[[str(famid)]]    
    familyFile.write(famid + ',' + data + ','+ label + '\n')
    
#famdf = pd.read_csv('familyFile.csv', delimiter =',')
#familyLabel = famdf['label']

#img = img.reshape((img.shape[0], 1200, 3))
#img.squeeze()
#img = img.reshape(888,1200)

"""

cropBox = pd.read_csv("images_box.txt", 'r', delimiter = " ", names = ['x1', 'y1', 'x2', 'y2'])
print(type(img))

imgdf = pd.DataFrame.from_records(img)
                        
img_crop = imgdf.crop((3, 144, 998, 431))
plt.imshow(img_crop)

"""