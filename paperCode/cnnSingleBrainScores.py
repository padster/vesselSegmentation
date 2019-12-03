# Run this as 'python paperCode/<code>.py'
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import pickle
import random

import classifier
import cnn

random.seed(0)

CNN_FUNC = cnn.runOne

SCAN_IDS = ['002', '019', '022', '023', '034', '056', '058', '066', '082']
METRICS = ['Accuracy', 'Sensitivity', 'Specificity', 'Dice score', 'ROC AUC']

def generateResults(scanID):
  opt = ['--flipx', '--flipy', '--flipz', '--flipxy', '--trans', '--features']
  classifier.initOptions(opt)
  return classifier.singleBrain(scanID, CNN_FUNC, calcScore=True, writeVolume=False, savePath=None)

if __name__ == '__main__':
  random.shuffle(SCAN_IDS)

  results = {}
  for scanID in SCAN_IDS:
    results[scanID] = generateResults(scanID)

  # Write using pickle for now.
  resultsDF = pd.DataFrame(results, index=METRICS).T
  resultsDF.to_csv("paperCode/results/SINGLE_BRAIN.csv")
