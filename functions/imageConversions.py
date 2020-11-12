import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import os


#function that saves an image
def saveImage(image, imagefullpath):
  cv2.imwrite(imagefullpath, image)

#Mapping from grayscale pixel to RGB using color table
def fromGrayscaleToRGBMapping(pixel, H):
  V = pixel / H
  V = (6 - 2) * V + 1
  r_p = H * max(0, (3- abs(V-4) - abs(V-5))/2 )
  g_p = H * max(0, (4- abs(V-2) - abs(V-4))/2 )
  b_p = H * max(0, (3- abs(V-1) - abs(V-2))/2 )
  return [r_p, g_p, b_p]

#Transform the grayscale image with one channel into an RGB image with 3 channels
def fromGrayscaleToRGB(grayscaleimage, bits):
  if len(grayscaleimage.shape) !=2:
    print ("Error: image is not a grayscale image")
    return 0

  H = 2**bits -1

  #we initiate an rgb image with 3 channels with same dimensions as grayscale image
  rgbimage = np.zeros((grayscaleimage.shape[0], grayscaleimage.shape[1], 3))
  i = 0
  while i < grayscaleimage.shape[0]:
    j = 0
    while j < grayscaleimage.shape[1]:
      r_p, g_p, b_p = fromGrayscaleToRGBMapping(grayscaleimage[i][j], H)
      rgbimage[i][j][0] = r_p
      rgbimage[i][j][1] = g_p
      rgbimage[i][j][2] = b_p

      j +=1
    i +=1

  return rgbimage

#Transform an RGB image with 3 channels into a YCbCr image with 3 channels
def fromRGBToYCbCr(rgbimage, bits):
  conversionMatrix = np.array([[0.299, 0.587, 0.144], [-0.169, -0.331, 0.500], [0.500, -0.419, -0.081]])
  bias = np.array([0, 2**(bits - 1), 2**(bits -1)])
  YCbCrimage = np.zeros((rgbimage.shape[0], rgbimage.shape[1], rgbimage.shape[2]))

  i = 0
  while i < rgbimage.shape[0]:
    j = 0
    while j < rgbimage.shape[1]:
      YCbCrimage[i][j] = conversionMatrix @ rgbimage[i][j] + bias
      j += 1
    i += 1

  return YCbCrimage

#Main function that converts a grayscale image with 1 channel to a YCbCr image with 3 channels
def fromGrayscaleToYCbCr(image, bits):
  rgbimage = fromGrayscaleToRGB(image, bits)
  if len(rgbimage.shape) < 3:
    return 0
  return fromRGBToYCbCr(rgbimage, bits)
