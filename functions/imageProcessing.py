#CONTAINS:
#   -BATCH normalization
#   -CONVOLUTIONS
#   -RELU
#
#
#
#

import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import os

#given a folder path with a set of images, function that returns the mean and standard deviation
#of the entire set for batch normalization
def returnBatchNormMeanAndStandarDeviationTensors(folderPath):
  meanTensor = np.array([])
  standardDeviationTensor = np.array([])

  subimagesCount = 0
  for subimagename in os.listdir(folderPath):
    subimage = cv2.imread(folderPath + "/" + subimagename)
    if meanTensor.size == 0:
      meanTensor = subimage
    else:
      meanTensor += subimage
    subimagesCount += 1
  meanTensor = meanTensor / subimagesCount

  for subimagename in os.listdir(folderPath):
    subimage = cv2.imread(folderPath + "/" + subimagename)
    if standardDeviationTensor.size == 0:
      standardDeviationTensor = (subimage - meanTensor)**2
    else:
      standardDeviationTensor += (subimage - meanTensor)**2

  standardDeviationTensor = (standardDeviationTensor / (subimagesCount - 1))**0.5

  return [meanTensor, standardDeviationTensor]

#Normalize the images with Batch Norm inside a specific folder
def normalizeSubimageWithBN(meanTensor, standardDeviationTensor, subimage):
  if 0 in standardDeviationTensor:
    print("Error: standard deviation tensor contains null value")
    return -1
  return (subimage - meanTensor) / standardDeviationTensor



#Convolution
def returnConvolution(image, kernel):
  kwidth = kernel.shape[1]
  kheight = kernel.shape[0]
  imagewidth = image.shape[1]
  imageheight = image.shape[0]

  imagepaddedwidth = imagewidth + 2*(kwidth // 2)
  imagepaddedheight = imageheight + 2*(kheight // 2)

  imagepadded = np.zeros((imagepaddedheight, imagepaddedwidth, image.shape[2]))

  imagepadded[kheight // 2: -1*(kheight // 2), kwidth // 2:-1*(kwidth // 2), :] = image[:,:,:].copy()

  convo = np.zeros((image.shape[0], image.shape[1]))

  for y in range(image.shape[0]):
    for x in range(image.shape[1]):
      convo[y,x] = (imagepadded[ y : y + kheight, x: x + kwidth, :] * kernel).sum()

  return convo

#create convoluted image with b bands
def returnBbandsConvoImage(image, nkernelsArray):
  bBandsConvoImage = np.zeros((image.shape[0], image.shape[1], len(nkernelsArray)))
  for index, kernel in enumerate(nkernelsArray):
    bBandsConvoImage[:,:,index] = returnConvolution(image, kernel)
  return bBandsConvoImage


def reLU(image):
  image[image < 0] = 0
  return image
