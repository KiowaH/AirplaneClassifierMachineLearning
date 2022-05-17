# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:45:33 2022

To Resize Images:
    Open CV
    imageMagick

@author: Kiowa
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
#import keras
#import theano

#Specify to not use and GPU's during computation
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder 

#physical_devices = tf.config.list_physical_devices('GPU')
#for device in physical_devices:
#    tf.config.experimental.set_memory_growth(device, True)

#val = pd.concat([boxes[1], boxes[2], boxes[3]], axis=1)

#dict = {key: boxes[0], value: val}



#Creating bounding dictionary to loop through and resize imgs
"""
boxes = pd.read_csv("images_box.txt", delimiter=" ", names=["id", "x1", "y1", "x2", "y2"])


boundingDictionary = {}


with open("images_box.txt") as f:
    Lines = f.readlines()
    for line in Lines:
        linetmp = line.split(" ")
        #print(linetmp[0])
        values = linetmp[1], linetmp[2], linetmp[3], linetmp[4]
        tmp = {linetmp[0]: values}
        boundingDictionary.update(tmp)
 """ 



"""
To crop images create dictionary with key being Image_box plane# and value being x1, y1, x2, y2
then read in images and search dictionary to get cropping values then resize
"""

"""
files = glob("images/*.jpg")
y=0
imgNames = []
while y < (len(files)):
    test = [files[y][7:14]]
    imgNames.append(test)
    #print(test)
    y+=1
"""
    
#for x in imgNames:
    #print("imgName = " + str(x))
    
    

#for key, value in boundingDictionary.items():
    #print("KeyVal = " + str(key))
"""
imgList = []

for i in imgNames:
    for x in i:
        imgList.append(x.split("'")[0])



#Crop and resize initial img's

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
            picture = resizeD.save("resizedImages/" + key + ".jpg")
            tmp = Image.open(r'resizedImages/'+ key +'.jpg')
            pyplot.imshow(tmp)
            pyplot.show()
"""



        
#[i[0].split("'", 2) for i in imgNames]





#Writing pixel vals of resized imgs to a csv
"""
tempSizeDF = pd.DataFrame()
pixelDF = pd.DataFrame()


file = open("resizedImage.csv", "w")
file.write('name, ')
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
"""


#pixelDF.append(tempSizeDF)

    

        #values = boundingDictionary[i]
        #tmpCrp = 'images/'+imgNames[i]+".jpg".crop(())


#Basic visualization experimentation
"""
ac1301 = Image.open(r'images/1522867.jpg')
ac1302 = Image.open(r'images/0774035.jpg')

f161 = Image.open(r'images/0735402.jpg')
f162 = Image.open(r'images/1307230.jpg')
    
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
"""



#num_pixels = df.shape[1]




#Writing train grp pixel vals and label to csv
"""
df = pd.read_csv('resizedImage.csv', delimiter=',', dtype={"name":'str'})

#familyVals = open('images_family_val.txt')
familyFile = open("familyFile.csv", "w")
familyFile1 = open("familyFile1.csv", "w")
familyFile2 = open("familyFile2.csv", "w")

#tu-154 vs yak-42
tyfamilyVals = open('cut_images_family_val.txt')
#AC-130 vs f16
affamilyVals = open('cut2_images_family_val.txt')
#f18 vs c47
fcfamilyVals = open('cut3_images_family_val.txt')


testCumulativeFile = open("testCumulativeFile.csv", "w")
testCumulativeFamilyVals = open('testCumulativeFamVal.txt')

cumulativeFile = open("cumulativeFile.csv", "w")
cumulativeFamilyVals = open('cumulativeFamVal.txt')


familyFile.write('name' + ',')
for i in range(0, 16384):
    tmp = "pixel" + str(i)
    familyFile.write(tmp + ',')

familyFile.write('label' + '\n')


familyFile1.write('name' + ',')
for i in range(0, 16384):
    tmp = "pixel" + str(i)
    familyFile1.write(tmp + ',')

familyFile1.write('label' + '\n')
                 

familyFile2.write('name' + ',')
for i in range(0, 16384):
    tmp = "pixel" + str(i)
    familyFile2.write(tmp + ',')

familyFile2.write('label' + '\n')
     

cumulativeFile.write('name' + ',')
for i in range(0, 16384):
    tmp = "pixel" + str(i)
    cumulativeFile.write(tmp + ',')

cumulativeFile.write('label' + '\n')


testCumulativeFile.write('name' + ',')
for i in range(0, 16384):
    tmp = "pixel" + str(i)
    testCumulativeFile.write(tmp + ',')

testCumulativeFile.write('label' + '\n')


for vals in tyfamilyVals:
    tmp = vals.split(" ", 1)
    famid = str(tmp[0])
    label = tmp[1]
    print(famid)
    print(label)
    data = df.loc[df['name'] == famid]
    data['family'] = label
    data.to_csv(familyFile, index=False, header=False) 
    
for vals in affamilyVals:
    tmp1 = vals.split(" ", 1)
    famid = str(tmp1[0])
    label = tmp1[1]
    print(famid)
    print(label)
    data = df.loc[df['name'] == famid]
    data['family'] = label
    data.to_csv(familyFile1, index=False, header=False) 
    
for vals in fcfamilyVals:
    tmp2 = vals.split(" ", 1)
    famid = str(tmp2[0])
    label = tmp2[1]
    print(famid)
    print(label)
    data = df.loc[df['name'] == famid]
    data['family'] = label
    data.to_csv(familyFile2, index=False, header=False) 
  

for vals in cumulativeFamilyVals:
    tmp3 = vals.split(" ", 1)
    famid = str(tmp3[0])
    label = tmp3[1]
    print(famid)
    print(label)
    data = df.loc[df['name'] == famid]
    data['family'] = label
    data.to_csv(cumulativeFile, index=False, header=False) 
    
for vals in testCumulativeFamilyVals:
    tmp4 = vals.split(" ", 1)
    famid = str(tmp4[0])
    label = tmp4[1]
    print(famid)
    print(label)
    data = df.loc[df['name'] == famid]
    data['family'] = label
    data.to_csv(testCumulativeFile, index=False, header=False) 
"""

#famdf = pd.read_csv('familyFile.csv', delimiter =',')
#familyLabel = famdf['label']

#img = img.reshape((img.shape[0], 1200, 3))
#img.squeeze()
#img = img.reshape(888,1200)

#Cnn Models
def CNN_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid',
                    input_shape=(128,128,1), data_format='channels_last',
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

def A_CNN_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid',
                    input_shape=(128,128,1), data_format='channels_last',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=num_pixels, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model

def B_CNN_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(5,5), padding='valid',
                    input_shape=(128,128,1), data_format='channels_last',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=64, kernel_size=(4,4),
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=64, kernel_size=(3,3),
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=num_pixels, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model


def C_CNN_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(5,5), padding='valid',
                    input_shape=(128,128,1), data_format='channels_last',
                    activation='selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=64, kernel_size=(4,4),
                    activation='selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=64, kernel_size=(3,3),
                    activation='selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=num_pixels, activation='selu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model


def test_CNN_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid',
                    input_shape=(128,128,1), data_format='channels_last',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=32, kernel_size=(3,3),
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=(3,3),
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=num_pixels, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model
"""
def D_CNN_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid',
                    input_shape=(128,128,1), data_format='channels_last',
                    activation='selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=32, kernel_size=(5,5),
                    activation='selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=(5,5),
                    activation='selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=32, kernel_size=(5,5),
                    activation='selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=(5,5),
                    activation='selu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=num_pixels, activation='selu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model
"""
"""
#prepping first data set
trainDF = pd.read_csv('familyFile.csv', delimiter=',', index_col=False, dtype={"name":'str', "label":'str'})
trainLabel = trainDF['label']
trainName = trainDF['name']
print(trainLabel)
trainDF = trainDF.drop(columns=['name', 'label'])
trainDF = trainDF/255
print(trainDF.tail())
testTrainDF = trainDF.values.reshape((trainDF.shape[0],128,128,1))
encTrainLabel = LabelEncoder().fit_transform(trainLabel)
catLabel = np_utils.to_categorical(encTrainLabel)
num_classes = 2
num_pixels = trainDF.shape[1]

#Prepping second dataset
trainDF1 = pd.read_csv('familyFile1.csv', delimiter=',', index_col=False, dtype={"name":'str', "label":'str'})
trainLabel1 = trainDF1['label']
trainName1 = trainDF1['name']
print(trainLabel1)
trainDF1 = trainDF1.drop(columns=['name', 'label'])
trainDF1 = trainDF1/255
print(trainDF1.tail())
testTrainDF1 = trainDF1.values.reshape((trainDF1.shape[0],128,128,1))
encTrainLabel1 = LabelEncoder().fit_transform(trainLabel1)
catLabel1 = np_utils.to_categorical(encTrainLabel1)
num_classes1 = 2
num_pixels1 = trainDF1.shape[1]

#prepping 3rd data set
trainDF2 = pd.read_csv('familyFile2.csv', delimiter=',', index_col=False, dtype={"name":'str', "label":'str'})
trainLabel2 = trainDF2['label']
trainName2 = trainDF2['name']
print(trainLabel2)
trainDF2 = trainDF2.drop(columns=['name', 'label'])
trainDF2 = trainDF2/255
print(trainDF2.tail())
testTrainDF2 = trainDF2.values.reshape((trainDF2.shape[0],128,128,1))
encTrainLabel2 = LabelEncoder().fit_transform(trainLabel2)
catLabel2 = np_utils.to_categorical(encTrainLabel2)
num_classes2 = 2
num_pixels2 = trainDF2.shape[1]


cnn = CNN_model(num_classes)
Acnn = A_CNN_model(num_classes)
Bcnn = B_CNN_model(num_classes)
Ccnn = C_CNN_model(num_classes)
#Dcnn = D_CNN_model(num_classes)
"""
"""
print("Start first data set Test Model Selu")
cnn.fit(testTrainDF, catLabel, epochs=5)
print("Test Model Relu")
Acnn.fit(testTrainDF, catLabel, epochs=5)
print("Relu 3 conv layers")
Bcnn.fit(testTrainDF, catLabel, epochs=5)
print("Selu 3 conv layers")
Ccnn.fit(testTrainDF, catLabel, epochs=5)
#Dcnn.fit(testTrainDF, catLabel, epochs=5)

print("Start second data set Test Model Selu")
cnn.fit(testTrainDF1, catLabel1, epochs=5)
print("Test Model Relu")
Acnn.fit(testTrainDF1, catLabel1, epochs=5)
print("Relu 3 conv layers")
Bcnn.fit(testTrainDF1, catLabel1, epochs=5)
print("Selu 3 conv layers")
Ccnn.fit(testTrainDF1, catLabel1, epochs=5)

print("Start third data set Test Model Selu")
cnn.fit(testTrainDF2, catLabel2, epochs=5)
print("Test Model Relu")
Acnn.fit(testTrainDF2, catLabel2, epochs=5)
print("Relu 3 conv layers")
Bcnn.fit(testTrainDF2, catLabel2, epochs=5)
print("Selu 3 conv layers")
Ccnn.fit(testTrainDF2, catLabel2, epochs=5)
"""

#prepping cumulative data set
test = pd.read_csv('cumulativeFile.csv', delimiter=',', index_col=False, dtype={"name":'str', "label":'str'})
testLabel = test['label']
testName = test['name']
print(testLabel)
test = test.drop(columns=['name', 'label'])
test = test/255
print(test.tail())
finalTest = test.values.reshape((test.shape[0],128,128,1))
encTestLabel = LabelEncoder().fit_transform(testLabel)
catTestLabel = np_utils.to_categorical(encTestLabel)
num_classes_test = 6
num_pixels = test.shape[1]


testCNN = test_CNN_model(num_classes_test)
print("Cumulative data test")
testCNN.fit(finalTest, catTestLabel, epochs=10)

#Prep unseen test data
unseenTest = pd.read_csv('testCumulativeFile.csv', delimiter=',', index_col=False, dtype={"name":'str', "label":'str'})
utestLabel = unseenTest['label']
utestName = unseenTest['name']
print(utestLabel)
unseenTest = unseenTest.drop(columns=['name', 'label'])
unseenTest = unseenTest/255
print(unseenTest.tail())
ruTest = unseenTest.values.reshape((unseenTest.shape[0],128,128,1))


yp = testCNN.predict(ruTest)
pred = np.argmax(yp, axis=1)

names = pd.DataFrame({"name": utestName})
results = pd.DataFrame({"class": pred})
finalRes = pd.concat([names,results], axis=1)
finalFile = open("Horendeck.csv", "w")
finalRes.to_csv('Horendeck.csv', sep=',', index=False) 


finalDF = pd.read_csv("Horendeck.csv", delimiter=",", dtype={"name":'str'})


for index, row in finalDF.iterrows():
    n=row['name']
    c=row['class']
    if c == 0:
        c = "AC-130"
    elif c == 1:
        c = "C-47"
    elif c == 2:
        c = "F-16"
    elif c == 3:
        c = "F/A-18"
    elif c == 4:
        c = "Tu-154"
    elif c == 5:
        c = "Yak-42"
    tmp = Image.open(r'resizedImages/'+ n +'.jpg')
    pyplot.imshow(tmp)
    pyplot.title(c)
    pyplot.show()