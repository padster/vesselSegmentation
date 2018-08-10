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

CNN_FUNC = cnn.runOne
SCAN_IDS = ['002', '019', '022', '023', '034', '056', '058', '066', '082']
METRICS = ['Accuracy', 'Sensitivity', 'Specificity', 'Dice score', 'ROC AUC']

def trainOffAllVolumes(allFeat):
  classifier.ONE_FEAT_NAME = None
  opt = ['--flipx', '--flipy', '--flipz', '--trans']
  if allFeat:
    opt.append('--features')
  classifier.initOptions(opt)

  savePath = "paperCode/results/%s/network.ckpt" % ("allNet" if allFeat else "rawNet")
  scanIgnore = SCAN_IDS[0] # Need to test against something - ignore results though

  classifier.brainsToBrain(SCAN_IDS, scanIgnore, CNN_FUNC, calcScore=True, writeVolume=False, savePath=savePath)

if __name__ == '__main__':
  trainOffAllVolumes(False)
  #trainOffAllVolumes(True)
