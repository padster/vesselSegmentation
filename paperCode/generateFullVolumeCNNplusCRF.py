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
import postProcessing
import util

def generateFullVolumeCRF(allFeat, scanID):
  sigmaXYZ, A, B = 1, 0.5, 0.5

  sub = ("allNet" if allFeat else "rawNet")
  cnnPath = "paperCode/results/%s/volumes/%s-CNN.mat" % (sub, scanID)
  crfPath = "paperCode/results/%s/volumes/%s-CRF.mat" % (sub, scanID)

  cnnData = files.loadCNN(cnnPath)
  print ("  - running CRF")
  crfData = postProcessing.process3D(cnnData, sxyz=sigmaXYZ, compat=postProcessing.makeCompat(a=A, b=B))
  print ("  - saving CRF")
  files.writePrediction(crfPath, "crf", crfData)


def generateFullVolumeCNN(allFeat, scanID):
  classifier.ONE_FEAT_NAME = None
  opt = ['--flipx', '--flipy', '--flipz', '--trans']
  if allFeat:
    opt.append('--features')
  classifier.initOptions(opt)

  sub = ("allNet" if allFeat else "rawNet")
  netPath = "paperCode/results/%s/network.ckpt" % (sub)
  outPath = "paperCode/results/%s/volumes/%s-CNN.mat" % (sub, scanID)

  cnn.volumeFromSavedNet(netPath, scanID, outPath)


if __name__ == '__main__':
  generateFullVolumeCNN(True, '990')

  # generateFullVolumeCRF(True, '018')
