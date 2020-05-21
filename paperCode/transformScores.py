# Run some analysis with and without particular transforms.
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
  withMeanDF.to_csv("paperCode/results/TRANSFORMS_%s.csv" % experimentName)
  return withMeanDF


def runExperiment(expName, transformsToUse, collector):
  # All features
  opt = ['--features'] + transformsToUse
  classifier.initOptions(opt)

  collector[expName] = generateResults(util.SCAN_IDS, expName)


# Run a selection of trials for particular transforms
if __name__ == '__main__':
  random.shuffle(util.SCAN_IDS)

  collector = {}
  runExperiment('NONE', [], collector)
  runExperiment('FLIPXY', ['--flipx', '--flipy'], collector)
  runExperiment('FLIPZ', ['--flipz'], collector)
  runExperiment('ALL', ['--flipx', '--flipy', '--flipz', '--flipxy', '--trans'], collector)

  for k, v in collector.items():
    print (k)
    print (v)
    print ("\n\n")
