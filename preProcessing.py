import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import leastsq, least_squares

import files
import util


def gauss( x, c1, mu1, sigma1 ):
  return c1 * np.exp( -(x - mu1)**2.0 / (2.0 * sigma1**2.0) )

def rayleigh( x, c1, sigma1 ):
  return c1 * x * np.exp( - x**2.0 / (2.0 * sigma1**2.0) )

# https://stackoverflow.com/questions/33178790/how-to-fit-a-double-gaussian-distribution-in-python
def multiDistribution( x, params ):
  (c1, sigma1, c2, mu2, sigma2, c3, mu3, sigma3, c4, mu4, sigma4) = params
  return rayleigh(x, c1, sigma1) + \
    gauss(x, c2, mu2, sigma2) + \
    gauss(x, c3, mu3, sigma3) + \
    gauss(x, c4, mu4, sigma4)




BINS = 400
def calculateVesselPeak(data, plot=False, scanID=None):
  ys, xs = np.histogram(data.flatten(), bins=BINS, normed=True)
  xs = (xs[:-1] + xs[1:]) / 2.0

  def distributionFit( params ):
      return multiDistribution( xs, params ) - ys

  # fit = leastsq( double_gaussian_fit, [0.8, 0.015, 0.1,  0.5, 0.12, 0.1] )
  I = np.inf
  POS, NIL = [0, I], [-I, I]
  boundsWrong = (
    POS, NIL, \
    POS, POS, POS, \
    POS, POS, POS, \
    POS, POS, POS
  )
  bounds = (tuple(b[0] for b in boundsWrong), tuple(b[1] for b in boundsWrong))
  # print (bounds)
  # bounds = (-I, I)
  fit = leastsq( distributionFit, [30, -0.03, 8, 0.01, 0.05, 1, 0.08, 0.05, 5, 0.12, 0.1])[0] #, ftol=1e-11, method='dogbox', bounds=bounds).x
  fitYs = multiDistribution(xs, fit)
  vPeak = max(fit[3], fit[6], fit[9])
  if plot:
    plt.title("Distribution for Scan %s (vessel peak %.3f)" % (scanID, vPeak))
    plt.hist(data.flatten(), bins=BINS, normed=True, label="Intensity histogram", alpha=0.5)
    plt.plot(xs, fitYs, label="Best Fit", linewidth=3)
    plt.plot(xs, rayleigh(xs, fit[0], fit[1]), linestyle='--', label="%.3f * R(%.3f)" % (fit[0], fit[1]))
    plt.plot(xs, gauss(xs, fit[2], fit[3], fit[4]), linestyle='--', label="%.3f * N(%.3f, %.3f)" % (fit[2], fit[3], fit[3]))
    plt.plot(xs, gauss(xs, fit[5], fit[6], fit[7]), linestyle='--', label="%.3f * N(%.3f, %.3f)" % (fit[5], fit[6], fit[7]))
    plt.plot(xs, gauss(xs, fit[8], fit[9], fit[10]), linestyle='--', label="%.3f * N(%.3f, %.3f)" % (fit[8], fit[9], fit[10]))
    plt.legend()
    plt.show()
  return vPeak




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

def normalizeHistogram(scanID, newPeak=0.125, plot=False):
  mraPath = "/%s/Normal%s-MRA-FS.mat" % (scanID, scanID)
  data, em, jv, pc = files.loadFeat(files.BASE_PATH + mraPath)
  if plot:
    plt.hist(data.flatten(), bins=400, normed=True, label='before')
  vPeak = calculateVesselPeak(data, plot=True, scanID=scanID)
  print ('Normalizing %s: From %.3f to %.3f' % (scanID, vPeak, newPeak))

  print ("TODO")
  """
  # Case 1:
  idA, idB, idC = (data <= p1), np.logical_and(p1 < data, data <= p2), (p2 < data)
  data[idA] =     0 + (data[idA] -  0) * (newP1 -     0) / (p1 -  0) #  0 ->     0, p1 -> newP1
  data[idB] = newP1 + (data[idB] - p1) * (newP2 - newP1) / (p2 - p1) # p1 -> newP1, p2 -> newP2
  data[idC] = newP2 + (data[idC] - p2) * (    1 - newP2) / ( 1 - p2) # p2 -> newP2,  1 ->     1
  q1, q2 = calculateFirstPeaks(data)
  print ('Normalized  %s: Peaks now %.3f -> %.3f' % (scanID, q1, q2))
  if plot:
    # plt.hist(data.flatten(), bins=400, normed=True, label='after')
    # plt.legend()
    plt.show()
  newMRAPath = "/%s/Normal%s-MRA-FS-preproc.mat" % (scanID, scanID)
  data = {'volMRA': data, 'EM': em, 'JV': jv, 'PC': pc}
  scipy.io.savemat(files.BASE_PATH + newMRAPath, data)
  print ("Saved to " + files.BASE_PATH + newMRAPath)
  """



def main():
  # printHistogramPeaks()
  normalizeHistogram('088', plot=False)

if __name__ == '__main__':
  main()
