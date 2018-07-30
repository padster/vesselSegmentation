import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import leastsq

import files
import util


# https://stackoverflow.com/questions/33178790/how-to-fit-a-double-gaussian-distribution-in-python
def double_gaussian( x, params ):
    (c1, mu1, sigma1, c2, mu2, sigma2) = params
    res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res

BINS = 400
def calculateFirstPeaks(data, plot=False):
  ys, xs = np.histogram(data.flatten(), bins=BINS, normed=True)
  xs = (xs[:-1] + xs[1:]) / 2.0

  def double_gaussian_fit( params ):
      fit = double_gaussian( xs, params )
      return (fit - ys)

  fit = leastsq( double_gaussian_fit, [0.8, 0.015, 0.1,  0.5, 0.12, 0.1] )
  fitYs = double_gaussian(xs, fit[0])
  if plot:
    plt.hist(data.flatten(), bins=BINS, normed=True)
    plt.plot(xs, fitYs)
    plt.show()
  a, b = fit[0][1], fit[0][4]
  return min(a, b), max(a, b)

  """
  localPeak = np.logical_and(ys[:-2] < ys[1:-1], ys[1:-1] > ys[2:])
  binMids = (xs[1:-2][localPeak] + xs[2:-1][localPeak]) / 2
  return binMids[0], binMids[1]
  """



"""
002 : 0.109 -> 0.015
019 : 0.114 -> 0.017
022 : 0.122 -> 0.015
023 : 0.119 -> 0.016
034 : 0.113 -> 0.017
056 : 0.129 -> 0.019
058 : 0.126 -> 0.019
066 : 0.130 -> 0.017
082 : 0.102 -> 0.013
084 : 0.010 -> 0.056
"""
def printHistogramPeaks():
  scanIDs = ['002', '019', '022', '023', '034', '056', '058', '066', '082', '084']
  for scanID in scanIDs:
    mraPath = "/%s/Normal%s-MRA-FS.mat" % (scanID, scanID)
    data, _, _, _ = files.loadFeat(files.BASE_PATH + mraPath)
    p1, p2 = calculateFirstPeaks(data)
    print ('%s : %.3f -> %.3f' % (scanID, p1, p2))

def normalizeHistogram(scanID, newP1=0.017, newP2=0.125, plot=False):
  mraPath = "/%s/Normal%s-MRA-FS.mat" % (scanID, scanID)
  data, em, jv, pc = files.loadFeat(files.BASE_PATH + mraPath)
  if plot:
    plt.hist(data.flatten(), bins=400, normed=True, label='before')
  p1, p2 = calculateFirstPeaks(data)
  print ('Normalizing %s: From %.3f -> %.3f, To %.3f -> %.3f' % (scanID, p1, p2, newP1, newP2))
  # Case 1: 
  idA, idB, idC = (data <= p1), np.logical_and(p1 < data, data <= p2), (p2 < data)
  data[idA] =     0 + (data[idA] -  0) * (newP1 -     0) / (p1 -  0) #  0 ->     0, p1 -> newP1
  data[idB] = newP1 + (data[idB] - p1) * (newP2 - newP1) / (p2 - p1) # p1 -> newP1, p2 -> newP2 
  data[idC] = newP2 + (data[idC] - p2) * (    1 - newP2) / ( 1 - p2) # p2 -> newP2,  1 ->     1
  q1, q2 = calculateFirstPeaks(data)
  print ('Normalized  %s: Peaks now %.3f -> %.3f' % (scanID, q1, q2))
  if plot:
    plt.hist(data.flatten(), bins=400, normed=True, label='after')
    plt.legend()
    plt.show()

  newMRAPath = "/%s/Normal%s-MRA-FS-preproc.mat" % (scanID, scanID)
  data = {'volMRA': data, 'EM': em, 'JV': jv, 'PC': pc}
  scipy.io.savemat(files.BASE_PATH + newMRAPath, data)
  print ("Saved to " + files.BASE_PATH + newMRAPath)



def main():
  # printHistogramPeaks()
  normalizeHistogram('084', plot=True)

if __name__ == '__main__':
  main()