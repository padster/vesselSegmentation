# Analysis for the results when trained off particular features.
# Run this as 'python paperCode/<code>.py'
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import random

import classifier
import cnn

random.seed(0)

def runClassifier(trainBrains, testBrain):
  return classifier.brainsToBrain(trainBrains, testBrain, cnn.runOne, calcScore=True)

# For all scan IDs, uses that as test, all others as train.
# Write all five metrics out to CSV
def generateResults(scanIDs, experimentName):
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
  withMeanDF.to_csv("paperCode/results/%s.csv" % experimentName)
  return withMeanDF

# Sets up a single run
def runExperiment(allFeat, singleFeat, collector):
  expName = "NO_FEATURES"
  if allFeat:
    expName = "ALL_FEATURES"
  elif singleFeat is not None:
    expName = "ONE_FEATURE_" + singleFeat

  opt = ['--flipx', '--flipy', '--flipz', '--flipxy', '--trans']
  if allFeat:
    opt.append('--features')
    classifier.ONE_FEAT_NAME = None
  elif singleFeat is not None:
    opt.append('--onefeat')
    classifier.ONE_FEAT_NAME = singleFeat
  classifier.initOptions(opt)

  collector[expName] = generateResults(util.SCAN_IDS, expName)

# Run with no features, one feature at a time, and all four.
if __name__ == '__main__':
  random.shuffle(util.SCAN_IDS)

  collector = {}
  runExperiment(False, None, collector)
  runExperiment(False, 'EM', collector)
  runExperiment(False, 'JV', collector)
  runExperiment(False, 'PC', collector)
  runExperiment(True, None, Collector)
  for k, v in collector.items():
    print (k)
    print (v)
    print ("\n\n")
