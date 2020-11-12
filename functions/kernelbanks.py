#creates b random kernels
#coefficients inside the kernels could be normalized or not

import numpy as np
import pandas as pd

def returnRandombKernels(nkernels, width, height, nbands, normalized = True):
  # we create n kernels randomly without normalization
  if normalized == False:

    #return [torch.randn(height, width, nbands) for i in range(0, nkernels)]
    return [np.random.rand(height, width, nbands) for i in range(0, nkernels)]

  # we create n kernels randomly then normalize each one of them
  nkernelsArray = [np.random.rand(height, width, nbands) for i in range(0, nkernels)]
  #nkernelsArray = [torch.randn(height, width, nbands) for i in range(0, nkernels)]

  i = 0
  while i < nkernels:
      #substracting the mean
      kernelMean = np.sum(nkernelsArray[i]) / (height*width*nbands)
      nkernelsArray[i] = nkernelsArray[i] - kernelMean

      #normalizing the kernel
      kernelnorm = np.sum(nkernelsArray[i]**2)/(height*width*nbands)
      kernelnorm = kernelnorm**0.5
      nkernelsArray[i] = nkernelsArray[i] / kernelnorm

      i += 1

  return nkernelsArray
