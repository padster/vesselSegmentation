# Train the classifier using all volumes of annotated data, and save the resulting net to file.
# Run this as 'python paperCode/<code>.py'
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import random

import classifier
import cnn
import files
import util

np.set_printoptions(precision=5)
pd.set_option('precision', 5)
random.seed(0)

def trainOffAllVolumes(allFeat):
  classifier.ONE_FEAT_NAME = None
  opt = ['--flipx', '--flipy', '--flipz', '--flipxy', '--trans']
  if allFeat:
    opt.append('--features')
  classifier.initOptions(opt)

  # Final network will be written here:
  savePath = "paperCode/results/%s/network.ckpt" % ("allNet" if allFeat else "rawNet")
  scanIgnore = util.SCAN_IDS[0] # Need to test against something - ignore results though

  classifier.brainsToBrain(util.SCAN_IDS, scanIgnore, cnn.runOne, calcScore=True, savePath=savePath)

if __name__ == '__main__':
  trainOffAllVolumes(False)
  #trainOffAllVolumes(True)
