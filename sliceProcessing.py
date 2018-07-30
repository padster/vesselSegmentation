import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import leastsq

import files
import util

def doCVProcessing():
  scanID = '022'
  zSlice = 60
  cnnPath = "/%s/Normal%s-MRA-CNN.mat" % (scanID, scanID)
  data = files.loadCNN(files.BASE_PATH + cnnPath)

  img = (data[:, :, zSlice] * 255).astype(np.uint8)
  plt.imshow(img)
  plt.show()
  plt.hist(img.flatten(), bins=64)
  plt.show()


def main():
  doCVProcessing()

if __name__ == '__main__':
  main()