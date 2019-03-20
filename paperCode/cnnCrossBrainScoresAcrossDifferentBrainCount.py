# Run this as 'python paperCode/<code>.py'
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import random

import classifier
import cnn

RUN_COUNT = 5
BRAIN_COUNT = 6
ALL_FEATURES = False


CNN_FUNC = cnn.runOne

SCAN_IDS = ['002', '019', '022', '023', '034', '056', '058', '066', '082']
METRICS = ['Accuracy', 'Sensitivity', 'Specificity', 'Dice score', 'ROC AUC']

def runClassifier(trainBrains, testBrain):
  return classifier.brainsToBrain(trainBrains, testBrain, CNN_FUNC, calcScore=True, writeVolume=False, savePath=None)

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

  withMeanDF = pd.DataFrame(withMean, METRICS, scanIDs + ['Mean'])
  withMeanDF.to_csv("paperCode/results/BRAINS_%d_RUN_%d_%s.csv" % (len(scanIDs), runID, experimentName))
  return withMeanDF


def runExperiment(runID, brainCount, allFeat):
  random.seed(runID)
  random.shuffle(SCAN_IDS)
  if runID == 3:
    return

  expName = "NO_FEATURES"
  if allFeat:
    expName = "ALL_FEATURES"
  classifier.ONE_FEAT_NAME = None

  opt = ['--flipx', '--flipy', '--flipz', '--trans']
  if allFeat:
    opt.append('--features')
  classifier.initOptions(opt)

  generateResults(SCAN_IDS[:brainCount], expName, runID)


if __name__ == '__main__':
  runExperiment(3, BRAIN_COUNT, True)
  runExperiment(4, BRAIN_COUNT, True)
