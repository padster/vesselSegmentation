# Run this as 'python paperCode/<code>.py'
import os
import sys
sys.path.append(os.getcwd())

# Train the classifier using all volumes of annotated data, and save the result to file.
import numpy as np
import pandas as pd
import random
np.set_printoptions(precision=5)
pd.set_option('precision', 5)
random.seed(0)

import classifier
import cnn
import files
import util

SCAN_IDS = ['002', '019', '022', '023', '034', '056', '058', '066', '082']

def scanAnnotation(scanID):
  _, lTrain, lTest = files.loadAllInputsUpdated(scanID, classifier.PAD, allFeatures=True, moreFeatures=False)
  trainV  = lTrain.sum(axis=0)[-1]
  trainNV = lTrain.shape[0] - trainV
  testV   =  lTest.sum(axis=0)[-1]
  testNV  =  lTest.shape[0] - testV
  return trainV, trainNV, testV, testNV

def annotationCounts():
  rows = np.array([scanAnnotation(scanID) for scanID in SCAN_IDS])
  asDF = pd.DataFrame(data=rows, index=SCAN_IDS, columns=['trainV', 'trainNV', 'testV', 'testNV'])
  print (asDF)


def paramsAndOps(allFeat=False, scanID='002'):
  sub = ("allNet" if allFeat else "rawNet")
  netPath = "paperCode/results/%s/network.ckpt" % (sub)

  classifier.ONE_FEAT_NAME = None
  opt = ['--trans']
  #opt = []
  if allFeat:
    opt.append('--features')
  classifier.initOptions(opt)

  stats = cnn.calcStats(netPath, scanID)
  print ("Stats:\n\t%d parameters\n\t%d flops per voxel" % (stats['params'], stats['flops']))



if __name__ == '__main__':
  print ("Calculating stats for the paper...\n")
  
  #annotationCounts()

  paramsAndOps()

