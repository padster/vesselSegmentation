# Get within-brain scores for DCNN, across all annotated volumes
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
import util

random.seed(0)

CNN_FUNC = cnn.runOne

def generateResults(scanID):
  opt = ['--flipx', '--flipy', '--flipz', '--flipxy', '--trans']
  classifier.initOptions(opt)
  return classifier.singleBrain(scanID, CNN_FUNC, calcScore=True)

if __name__ == '__main__':
  random.shuffle(util.SCAN_IDS)

  results = {}
  for scanID in util.SCAN_IDS:
    results[scanID] = generateResults(scanID)

  # Write using pickle for now.
  resultsDF = pd.DataFrame(results, index=util.METRICS).T
  resultsDF.to_csv("paperCode/results/SINGLE_BRAIN.csv")
