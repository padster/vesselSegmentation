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

ROUND_1_SCAN_IDS = ['002', '019', '022', '023', '034'] #, '056', '058', '066', '082']
ROUND_2_SCAN_IDS = ['017', '027', '035', '046', '064', '077']
METRICS = ['Accuracy', 'Sensitivity', 'Specificity', 'Dice score', 'ROC AUC']

def runClassifier(trainBrains, testBrain):
  return classifier.brainsToBrain(trainBrains, testBrain, CNN_FUNC, calcScore=True, writeVolume=False, savePath=None)

def generateResults(trainIDs, testIDs):
  n = len(testIDs)

  accs, senss, specs, dices, aucs = \
    np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

  for i, testBrain in enumerate(testIDs):
    print ("\n\n%d / %d\n" % (i + 1, len(testIDs)))
    random.shuffle(trainIDs)
    results = runClassifier(trainIDs, testBrain)
    accs[i], senss[i], specs[i], dices[i], aucs[i] = results

  allValues = np.vstack((accs, senss, specs, dices, aucs))
  means = np.mean(allValues, axis=1)[np.newaxis].T
  withMean = np.hstack((allValues, means))

  withMeanDF = pd.DataFrame(withMean, METRICS, testIDs + ['Mean'])
  withMeanDF.to_csv("runNewRound.csv")
  return withMeanDF


def runNewRoundExperiment():
  random.seed(5)
  random.shuffle(ROUND_1_SCAN_IDS)
  random.shuffle(ROUND_2_SCAN_IDS)

  classifier.ONE_FEAT_NAME = None
  opt = ['--flipx', '--flipy', '--flipz', '--trans', '--features']
  classifier.initOptions(opt)

  generateResults(ROUND_1_SCAN_IDS, ROUND_2_SCAN_IDS)


if __name__ == '__main__':
  runNewRoundExperiment()
