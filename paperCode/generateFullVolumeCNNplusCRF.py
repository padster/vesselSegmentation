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

ALL_FEAT = False
SCAN_ID = '002'


def runExperiment(allFeat, scanID):
  classifier.ONE_FEAT_NAME = None
  opt = ['--flipx', '--flipy', '--flipz', '--trans']
  if allFeat:
    opt.append('--features')
  classifier.initOptions(opt)
  
  netPath = "paperCode/results/%s/network.ckpt" % ("allNet" if allFeat else "rawNet")
  outPath = "paperCode/results/volumes/%s-CNN.mat" % scanID

  cnn.volumeFromSavedNet(netPath, scanID, outPath)

if __name__ == '__main__':
  runExperiment(ALL_FEAT, SCAN_ID)
