

import numpy as np
import pandas as pd
import cv2
import os


#returns the number of pixels representing the plate in the mask
def returnPlateNumberPixelsFromMask(maskImg):
  #maskImg = cv2.imread(maskPath)
  return maskImg.sum()/(3*255)

#given width x height, and center coordinates of subimage, returns number of plate pixels contained in subimage
def returnPlateNumberPixelsCovBySubimage(width, height, xcenter, ycenter, maskImg):
  xstart = xcenter - width//2
  xend = xcenter + width//2
  ystart = ycenter - height//2
  yend = ycenter + height//2
  subImg = maskImg[ystart:yend+1,xstart: xend+1,:]

  return subImg.sum()/(3*255)


def scanOriginalImageUsingSubImagesWith(width, height, xstride, ystride, imagepath, thresholdForPlateClass):

  imageToScan = cv2.imread(imagepath)
  plateNumberPixelsInImageToScan = returnPlateNumberPixelsFromMask(imageToScan)

  #statistics
  numberSubImages = 0
  numberSubImagesClassPlate = 0 #number of images wich contain the threshold of plate percentage
  numberSubImagesContainingAllPlate = 0

  #initializing scanning parameters
  xcenter_initial = width // 2
  ycenter_initial = height // 2
  yfinal = imageToScan.shape[0] - height // 2 - 1
  xfinal = imageToScan.shape[1] - width // 2 - 1
  ycenter = ycenter_initial
  xcenter = xcenter_initial

  #we scan the original image from left to right, top to bottom
  while ycenter <= yfinal:

    #We make sure we cover the entire original image
    #on the last strides on vertical and/or horizontal axes
    if xfinal - xcenter < xstride:
      xcenter = xfinal
    if yfinal - ycenter < ystride:
      ycenter = yfinal

    plateNumberPixelsInSubimage = returnPlateNumberPixelsCovBySubimage(width, height, xcenter, ycenter, imageToScan)
    #percentage of plate
    platePercentageCovered = plateNumberPixelsInSubimage / plateNumberPixelsInImageToScan

    #percentage of total pixels in subimage
    #platePercentageCovered = plateNumberPixelsInSubimage / (width*height)

    #Increment
    xcenter += xstride
    if xcenter > xfinal:
      xcenter = xcenter_initial
      ycenter += ystride

    #statistics
    numberSubImages += 1
    if platePercentageCovered >= thresholdForPlateClass:
      numberSubImagesClassPlate += 1
    if platePercentageCovered == 1.0:
      numberSubImagesContainingAllPlate += 1

  #print statistics:
  print("Number of subimages created ", numberSubImages - 1)
  print("Number of subimages with threshold pixels of plate ", numberSubImagesClassPlate)
  print("Number of subimages containing all the plate ", numberSubImagesContainingAllPlate)

  return [numberSubImagesContainingAllPlate, numberSubImagesClassPlate, numberSubImages]


  #saving subimage into folder
def saveSubimage(width, height, xcenter, ycenter, subimageClass, outputFolderPath, imagename, subImg):
  originalImagename = returnOriginalimagename(imagename)
  subImagename = originalImagename + "_" + str(width) + "_" + str(height) + "_" + str(int(xcenter)) + "_" + str(int(ycenter)) + "_" + subimageClass
  subimageFullpath = outputFolderPath + subImagename + ".png"
  # Saving the image
  cv2.imwrite(subimageFullpath, subImg)

#class as percentage of pixels from plate
def returnSubimgClass(percentagePlateCovered, thresholdForPlateClass):
  if percentagePlateCovered >= thresholdForPlateClass:
    return "2"
  return "1"

#class as percentage of total pixels in subimage
def returnSubimageClass(percentagePlateCovered, thresholdForPlateClass):
  if percentagePlateCovered >= thresholdForPlateClass:
    return "2"
  return "1"

 #return the subimage of interest
def returnSubimageWith(width, height, xcenter, ycenter, maskImg):
  xstart = xcenter - width//2
  xend = xcenter + width//2
  ystart = ycenter - height//2
  yend = ycenter + height//2
  return maskImg[ystart:yend+1,xstart: xend+1,:]

#same code as above, with the addition of creating the subimages and storing them
#information included in filename: originalimagename_width_heigth-xcenter_ycenter_class
def createSubimagesFromTrainingImage(width, height, xstride, ystride, thresholdForPlateClass, imagename, imageFolderPath, maskFolderPath, outputFolderpath):

  imagePath = imageFolderPath + imagename
  imageToScan = cv2.imread(imagePath)
  maskname = returnMaskPathFromImagePath(imagename)
  imageMask = cv2.imread(maskFolderPath + maskname, 1)

  plateNumberPixelsInImageToScan = returnPlateNumberPixelsFromMask(imageMask)

  #statistics
  numberSubImages = 0
  numberSubImagesClassPlate = 0 #number of images wich contain the threshold of plate percentage

  #initializing scanning parameters
  xcenter_initial = width // 2
  ycenter_initial = height // 2
  yfinal = imageMask.shape[0] - height // 2 - 1
  xfinal = imageMask.shape[1] - width // 2 - 1
  ycenter = ycenter_initial
  xcenter = xcenter_initial

  #class parameter depending on the threshold value
  subImageClass = ""

  #we scan the original image from left to right, top to bottom
  while ycenter <= yfinal:

    #We make sure we cover the entire original image
    #on the last strides on vertical and/or horizontal axes
    if xfinal - xcenter < xstride:
      xcenter = xfinal
    if yfinal - ycenter < ystride:
      ycenter = yfinal

    #getting the number of plate pixels contained in the subimage
    plateNumberPixelsInSubimage = returnPlateNumberPixelsCovBySubimage(width, height, xcenter, ycenter, imageMask)

    #getting the class of the subimage
    subImageClass = returnSubimgClass(plateNumberPixelsInSubimage / plateNumberPixelsInImageToScan, thresholdForPlateClass)

    #saving the subimage
    saveSubimage(width, height, xcenter, ycenter, subImageClass, outputFolderpath, imagename, returnSubimageWith(width, height, xcenter, ycenter, imageToScan))

    #Increment
    xcenter += xstride
    if xcenter > xfinal:
      xcenter = xcenter_initial
      ycenter += ystride

    #statistics
    numberSubImages += 1
    if plateNumberPixelsInSubimage / plateNumberPixelsInImageToScan >= thresholdForPlateClass:
      numberSubImagesClassPlate += 1

  print("Number of sub images classified as plate/foreground: ", numberSubImagesClassPlate)
  print("Number of sub images created:", numberSubImages )
  return [numberSubImagesClassPlate, numberSubImages]
