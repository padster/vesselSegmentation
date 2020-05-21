# Train the classifier using all volumes of annotated data, and save the result to file.
# Run this as 'python paperCode/<code>.py'
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import random
np.set_printoptions(precision=5)
pd.set_option('precision', 5)
random.seed(0)

import classifier
import cnn
import files
import postProcessing
import util

# Generates a full CNN volume segmentation.
# Can be done in subsets of X-slices.
def generateFullVolumeCNN(allFeat, scanID):
  classifier.ONE_FEAT_NAME = None
  opt = ['--trans'] # Use all transforms for inference
  if allFeat:
    opt.append('--features')
  classifier.initOptions(opt)

  # Load network - can be trained by all features or just raw MRA
  sub = ("allNet" if allFeat else "rawNet")
  netPath = "paperCode/results/%s/network.ckpt" % (sub)

  # Set these to however much to run:
  xFr, xTo = 250, 300
  outPath = "paperCode/results/%s/volumes/%s-CNN-%d-%d.mat" % (sub, scanID, xFr, xTo)

  cnn.volumeFromSavedNet(netPath, scanID, outPath, xFr, xTo, useMask=True)


if __name__ == '__main__':
  generateFullVolumeCNN(False, '002')
