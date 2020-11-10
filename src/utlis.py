import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2

def getName(filePath):
    return filePath.split('\\')[-1]

def importDatainfo(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names = columns)

    data['Center'] = data['Center'].apply(getName)
    print('Total Images Imported: ', data.shape[0])
    
    return data

def balanceData(data, display = True):
    nBins = 31 # This has to be an odd number because it has positive site and negative site. 
    samplesPerBin = 1000 
    hist, bins = np.histogram(data['Steering'], nBins)
    
    #print(bins)
    
    if display:
        center = (bins[:-1]+ bins[1:]) * 0.5
        #print(center)
        plt.bar(center, hist, width = 0.06)
        plt.plot((-1,1), (samplesPerBin, samplesPerBin))
        plt.show()
    
    #remove redudent data
    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    
    print('Removed Images: ', len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace = True)
    print('Remaining Images: ', len(data))
    
    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        center = (bins[:-1]+ bins[1:]) * 0.5
        #print(center)
        plt.bar(center, hist, width = 0.06)
        plt.plot((-1,1), (samplesPerBin, samplesPerBin))
        plt.show()
    
    return data


def loadData(path, data):
    imagesPath = [] #importing the path not the actual image
    steering = []
    
    for i in range( len(data)):
        indexedData = data.iloc[i]
        #print(indexedData)
        imagesPath.append(os.path.join(path, 'IMG', indexedData[0]))
        #print(os.path.join(path, 'IMG', indexedData[0]))
        steering.append(float(indexedData[3]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering

def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)
    
    ## PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y':(-0.1,0.1)})
        img = pan.augment_image(img)
        
    ## ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)
    
    ## BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4, 1.2))
        img = brightness.augment_image(img)
    
    ## FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img,1)
        steering = steering * (-1)
    
    return img, steering
    
### Testing augmentImage method    
# imgRe, st = augmentImage('test.jpg', 0)
# plt.imshow(imgRe)
# plt.show()
    