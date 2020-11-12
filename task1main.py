import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import os

#EVALUATES MODEL DURING TRAINING
def launchModelEvaluation(trainingsetName, trainFolder, maskFolderPath, modelOptions):

  meanTensor = np.array([])
  standardDeviationTensor = np.array([])

  kernelresultsDF = pd.DataFrame(columns = ["TrainingSet", 'Imagename', "kernelID", 'percentagePlatePixelsActivated', 'percentageDeactivatedPixelsOutsidePlate'])

  if modelOptions["batchnormalization"] == True:
    meanTensor, standardDeviationTensor = returnBatchNormMeanAndStandarDeviationTensors(trainFolder)

  bKernels = np.array(returnRandombKernels(modelOptions["kernelBank"]["nkernels"], modelOptions["kernelBank"]["width"], modelOptions["kernelBank"]["height"], modelOptions["kernelBank"]["nbands"], modelOptions["kernelBank"]["normalized"]))

  imageprocessed = 0

  for imagename in os.listdir(trainFolder):
    image = cv2.imread(trainFolder + imagename)

    #Batch Normalization if set to True
    if modelOptions["batchnormalization"] == True:
      image = normalizeSubimageWithBN(meanTensor, standardDeviationTensor, image)

    #Convolution with kernel bank
    bBandsConvoImage = returnBbandsConvoImage(image, bKernels)

    #ReLU
    bBandsImages = reLU(bBandsConvoImage)

    imageID, width, height, xcenter, ycenter, imageclass = returnSubimageDimensionsCenterCoordinates(imagename)
    imagemask = cv2.imread(maskFolderPath + "mask_" + imageID + ".png", 1)
    submask = returnSubimageWith(width, height, xcenter, ycenter, imagemask)

    #getting model scores on image
    index = 0
    while index < bBandsImages.shape[2]:
      #percentagePlatePixelsActivated, percentageDeactivatedPixelsOutsidePlate = returnPercentageActivatedPixelsInPlateRegion(bBandsImages[:,:,index], submask)
      percentagePlatePixelsActivated, percentageDeactivatedPixelsOutsidePlate = returnScores(bBandsImages[:,:,index], submask)
      kernelresultsDF = kernelresultsDF.append({"TrainingSet": trainingsetName,'Imagename' : imagename, 'kernelID' : index, 'percentagePlatePixelsActivated' : percentagePlatePixelsActivated, "percentageDeactivatedPixelsOutsidePlate": percentageDeactivatedPixelsOutsidePlate},
                ignore_index = True)

      index += 1

    imageprocessed +=1
    if imageprocessed % 2000 == 0:
      print("Number of images processed ", imageprocessed)

  return [bKernels, kernelresultsDF]


#EVALUATE MODEL ON TEST SET
 def launchModelTestEvaluation(trainingsetName, trainFolder, maskFolderPath, modelOptions, meanTensor, standardDeviationTensor, bKernels, kernelIDsArray):

  kernelresultsDF = pd.DataFrame(columns = ["TestSet", 'Imagename', "kernelID", 'percentagePlatePixelsActivated', 'percentageDeactivatedPixelsOutsidePlate'])

  imageprocessed = 0

  for imagename in os.listdir(trainFolder):
    image = cv2.imread(trainFolder + imagename)

    #Batch Normalization if set to True
    if modelOptions["batchnormalization"] == True:
      image = normalizeSubimageWithBN(meanTensor, standardDeviationTensor, image)

    #Convolution with kernel bank
    bBandsConvoImage = returnBbandsConvoImage(image, bKernels)

    #ReLU
    bBandsImages = reLU(bBandsConvoImage)

    imageID, width, height, xcenter, ycenter, imageclass = returnSubimageDimensionsCenterCoordinates(imagename)
    imagemask = cv2.imread(maskFolderPath + "mask_" + imageID + ".png", 1)
    submask = returnSubimageWith(width, height, xcenter, ycenter, imagemask)

    #getting model scores on image
    index = 0
    while index < bBandsImages.shape[2]:
      #percentagePlatePixelsActivated, percentageDeactivatedPixelsOutsidePlate = returnPercentageActivatedPixelsInPlateRegion(bBandsImages[:,:,index], submask)
      percentagePlatePixelsActivated, percentageDeactivatedPixelsOutsidePlate = returnScores(bBandsImages[:,:,index], submask)
      kernelresultsDF = kernelresultsDF.append({"TestSet": trainingsetName,'Imagename' : imagename, 'kernelID' : kernelIDsArray[index], 'percentagePlatePixelsActivated' : percentagePlatePixelsActivated, "percentageDeactivatedPixelsOutsidePlate": percentageDeactivatedPixelsOutsidePlate},
                ignore_index = True)

      index += 1

    imageprocessed +=1
    if imageprocessed % 2000 == 0:
      print("Number of images processed ", imageprocessed)

  return kernelresultsDF
