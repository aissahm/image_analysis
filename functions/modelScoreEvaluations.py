#The first score is the percentage of pixels activated in the plate region of the subimage,
#over the total number of pixels included inside the plate region in the subimage.

#The second score, similar to the first one, is the percentage of deactivated pixels,
#laying outside the plate region, over the total number of pixels outside the plate region in the subimage.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def computeScoresPerKernel(resultsDf):
  nkernelsarray = resultsDf.kernelID.unique()
  pixelsActivatedScoresArray = []
  outsidePlateDeactivatedPixelsScoresArray = []
  kernelIDArray = []

  for kernelID in nkernelsarray:
    kernelPlatePixelsActivatedDf = resultsDf[(resultsDf.kernelID == kernelID ) & (resultsDf.percentagePlatePixelsActivated > -1)].copy()
    kernelOutsidePlateDeactivatedDf = resultsDf[resultsDf.kernelID == kernelID ].copy()
    averageOutsidePixelsDeactivated = kernelOutsidePlateDeactivatedDf.percentageDeactivatedPixelsOutsidePlate.mean()
    averagePlatePixelsActivated = 0
    if kernelPlatePixelsActivatedDf.percentagePlatePixelsActivated.count() > 0:
      averagePlatePixelsActivated = kernelPlatePixelsActivatedDf.percentagePlatePixelsActivated.mean()
    pixelsActivatedScoresArray.append( 100*averagePlatePixelsActivated)
    outsidePlateDeactivatedPixelsScoresArray.append(100*(1- averageOutsidePixelsDeactivated))
    kernelIDArray.append("kernel_" + str(kernelID))

  return [kernelIDArray, pixelsActivatedScoresArray, outsidePlateDeactivatedPixelsScoresArray]

#plots the two scores to view results per kernel
def plotScores(modelScoresArray):

  fig, ax = plt.subplots()
  ax.scatter(modelScoresArray[1], modelScoresArray[2])

  plt.title("Average scores per kernel")
  plt.xlabel("% activated pixels inside plate region")
  plt.ylabel("% deactivated pixels outside plate region")

  for i, txt in enumerate(modelScoresArray[0]):
      ax.annotate(txt, (modelScoresArray[1][i], modelScoresArray[2][i]))

def returnScores(subimage, submask):
  submaskOutsidePlate = submask[:, :, 0].copy()
  submaskInsidePlate = submask[:, :, 0].copy()

  submaskOutsidePlate[submaskOutsidePlate > 0] = 2 #inside plate is equal to 2
  submaskOutsidePlate[submaskOutsidePlate == 0] = 1 #outside plate is equal to 1
  submaskOutsidePlate[submaskOutsidePlate == 2] = 0 #inside plate is equal to 0

  submaskInsidePlate[submaskInsidePlate > 0] = 1

  subimagecopy = subimage.copy()
  subimagecopy[subimagecopy > 0] = 1

  nPlatePixels = np.sum(submaskInsidePlate)
  nActivatedPlatePixels = np.sum(subimagecopy*submaskInsidePlate)

  nDeactivatedPixelsOutsidePlate = np.sum(subimagecopy*submaskOutsidePlate)

  percentagePlatePixelsActivated = -1
  if nPlatePixels > 0:
    percentagePlatePixelsActivated = nActivatedPlatePixels/nPlatePixels

  width = submask.shape[1]
  height = submask.shape[0]

  percentageDeactivatedPixelsOutsidePlate = nDeactivatedPixelsOutsidePlate/(width * height - nPlatePixels)

  return [percentagePlatePixelsActivated, percentageDeactivatedPixelsOutsidePlate]

#Returns the percentage of activated pixels over the total number of pixels that figure
#inside the plate region in the subimage
def returnPercentageActivatedPixelsInPlateRegion(subimage, submask):
  width = submask.shape[1]
  height = submask.shape[0]

  nPlatePixels = 0
  nActivatedPlatePixels = 0
  nDeactivatedPixelsOutsidePlate = 0

  i = 0
  while i < height:
    j = 0
    while j < width:
      #pixel value != 0 => inside plate region
      if submask[i][j][0] != 0:
        nPlatePixels += 1

        #pixel was activated
        if subimage[i][j] > 0:
          nActivatedPlatePixels += 1

      #pixel value = 0 => outside plate region
      else:
        #pixel was deactivated
        if subimage[i][j] == 0:
          nDeactivatedPixelsOutsidePlate += 1

      j += 1
    i += 1

  percentagePlatePixelsActivated = -1
  if nPlatePixels > 0:
    percentagePlatePixelsActivated = nActivatedPlatePixels/nPlatePixels

  percentageDeactivatedPixelsOutsidePlate = nDeactivatedPixelsOutsidePlate/(width * height - nPlatePixels)

  return [percentagePlatePixelsActivated, percentageDeactivatedPixelsOutsidePlate]
