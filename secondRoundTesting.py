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

OLD_IDS = ['002', '019', '022', '023', '034', '056', '058', '066', '082']
NEW_IDS = ['017', '027', '035', '046', '064', '077']

# Intial - train 9, test 1
# TRAIN_IDS = OLD_IDS
# TEST_IDS = NEW_IDS[0:1]

# New - train 14, test 1
TRAIN_IDS = OLD_IDS + NEW_IDS[1:]
TEST_IDS = NEW_IDS[0:1]

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
  random.shuffle(TRAIN_IDS)
  random.shuffle(TEST_IDS)

  classifier.ONE_FEAT_NAME = None
  opt = ['--flipx', '--flipy', '--flipz', '--flipxy', '--trans', '--features']
  classifier.initOptions(opt)

  generateResults(TRAIN_IDS, TEST_IDS)


if __name__ == '__main__':
  runNewRoundExperiment()
