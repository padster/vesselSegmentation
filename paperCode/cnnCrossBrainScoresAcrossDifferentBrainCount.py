# Check across-brain results, given different numbers of training brains.
# Run this as 'python paperCode/<code>.py'
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import random

import classifier
import cnn
import util

# Run every configuration 5 times.
RUN_COUNT = 5

def runClassifier(trainBrains, testBrain):
  return classifier.brainsToBrain(trainBrains, testBrain, cnn.runOne, calcScore=True)

# Generate results for a single run, given # test brains, and feature spect.
def generateResults(scanIDs, experimentName, runID):
  n = len(scanIDs)

  accs, senss, specs, dices, aucs = \
    np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

  for i, testBrain in enumerate(scanIDs):
    print ("\n\n%d / %d\n" % (i + 1, len(scanIDs)))
    trainBrains = scanIDs[:i] + scanIDs[i+1:]
    random.shuffle(trainBrains)
    results = runClassifier(trainBrains, testBrain)
    accs[i], senss[i], specs[i], dices[i], aucs[i] = results

  allValues = np.vstack((accs, senss, specs, dices, aucs))
  means = np.mean(allValues, axis=1)[np.newaxis].T
  withMean = np.hstack((allValues, means))

  withMeanDF = pd.DataFrame(withMean, util.METRICS, scanIDs + ['Mean'])
  withMeanDF.to_csv("paperCode/results/BRAINS_%d_RUN_%d_%s.csv" % (len(scanIDs), runID, experimentName))
  return withMeanDF

# Do a single run, given nBrains and whether to use all or no features.
def runExperiment(runID, nBrains, allFeat):
  # First shuffle the scans, so we can randomly pick nBrains of them
  random.seed(runID)
  random.shuffle(util.SCAN_IDS)

  expName = "NO_FEATURES"
  if allFeat:
    expName = "ALL_FEATURES"
  classifier.ONE_FEAT_NAME = None
  opt = ['--flipx', '--flipy', '--flipz', '--flipxy', '--trans']
  if allFeat:
    opt.append('--features')
  classifier.initOptions(opt)

  generateResults(util.SCAN_IDS[:nBrains], expName, runID)


if __name__ == '__main__':
  for nBrains in [3, 6]:
    for i in range(RUN_COUNT):
      runExperiment(runID=i, brainCount=nBrains, allFeat=True)
      runExperiment(runID=i, brainCount=nBrains, allFeat=False)
