# Collection of processing steps to run before CNNs are used.

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import leastsq, least_squares

import files
import util

BINS = 400

# Calculate a Gaussian distribution for a given series of X values
def gauss( x, c1, mu1, sigma1 ):
  return c1 * np.exp( -(x - mu1)**2.0 / (2.0 * sigma1**2.0) )

# Calculate a Rayleigh distribution for a given series of X values
def rayleigh( x, c1, sigma1 ):
  return c1 * x * np.exp( - x**2.0 / (2.0 * sigma1**2.0) )

# Given parameters, caculate the sum of a Rayleigh and three Gaussians
# https://stackoverflow.com/questions/33178790/how-to-fit-a-double-gaussian-distribution-in-python
def multiDistribution( x, params ):
  (c1, sigma1, c2, mu2, sigma2, c3, mu3, sigma3, c4, mu4, sigma4) = params
  return rayleigh(x, c1, sigma1) + \
    gauss(x, c2, mu2, sigma2) + \
    gauss(x, c3, mu3, sigma3) + \
    gauss(x, c4, mu4, sigma4)

# Fit the four distributions, and return peaks of the gaussians.
def calculateVesselPeak(data, plot=False, scanID=None):
  ys, xs = np.histogram(data.flatten(), bins=BINS, normed=True)
  xs = (xs[:-1] + xs[1:]) / 2.0

  def distributionFit( params ):
      return multiDistribution( xs, params ) - ys

  I = np.inf
  POS, NIL = [0, I], [-I, I]
  boundsWrong = (
    POS, NIL, \
    POS, POS, POS, \
    POS, POS, POS, \
    POS, POS, POS
  )
  bounds = (tuple(b[0] for b in boundsWrong), tuple(b[1] for b in boundsWrong))
  fit = leastsq( distributionFit, [30, -0.03, 8, 0.01, 0.05, 1, 0.08, 0.05, 5, 0.12, 0.1])[0] #, ftol=1e-11, method='dogbox', bounds=bounds).x
  fitYs = multiDistribution(xs, fit)
  vPeak = max(fit[3], fit[6], fit[9]) # mu2, mu3, mu4
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

# Show histograms for all input volumes
def printHistogramPeaks(useFsFile=False, plot=False):
  scanIDs = util.SCAN_IDS
  for scanID in scanIDs:
    if useFsFile:
      mraPath = "/%s/Normal%s-MRA-FS.mat" % (scanID, scanID)
      data, _, _, _ = files.loadFeat(files.BASE_PATH + mraPath)
    else:
      mraPath = "/%s/Normal%s-MRA.mat" % (scanID, scanID)
      data = files.loadMRA(files.BASE_PATH + mraPath)
      data = data / np.max(data) # Need to normalize
    vPeak = calculateVesselPeak(data, plot=plot, scanID=scanID)
    print ('%s : peak at %.3f' % (scanID, vPeak))

def main():
  printHistogramPeaks(plot=False)

if __name__ == '__main__':
  main()
