# See how well the network trains with different numbers of inputs.

# Run this as 'python paperCode/<code>.py'


import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import random

import classifier
import cnn

CNN_FUNC = cnn.runOne

SCAN_IDS = ['002', '019', '022', '023', '034', '056', '058', '066', '082']
METRICS = ['Accuracy', 'Sensitivity', 'Specificity', 'Dice score', 'ROC AUC']

def runClassifier(trainBrains, testBrain, nTrainPerBrain):
  return classifier.brainsToBrain(trainBrains, testBrain, CNN_FUNC, calcScore=True, perBrainExamples=nTrainPerBrain)

def generateResults(scanIDs, nTrainPerBrain):
    n = len(scanIDs)

    accs, senss, specs, dices, aucs = \
        np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    for i, testBrain in enumerate(scanIDs):
        print ("\n\n%d / %d\n" % (i + 1, len(scanIDs)))
        trainBrains = scanIDs[:i] + scanIDs[i+1:]
        random.shuffle(trainBrains)
        results = runClassifier(trainBrains, testBrain, nTrainPerBrain)
        accs[i], senss[i], specs[i], dices[i], aucs[i] = results

    allValues = np.vstack((accs, senss, specs, dices, aucs))
    means = np.mean(allValues, axis=1)[np.newaxis].T
    withMean = np.hstack((allValues, means))
    withMeanDF = pd.DataFrame(withMean, METRICS, scanIDs + ['Mean'])
    return withMeanDF

def runExperiment(nTrainPerBrain, withTransforms):
    path = "paperCode/results/trainSubset_n%d_%sTransforms.csv" % (
        nTrainPerBrain, "With" if withTransforms else "No"
    )
    if os.path.exists(path):
        print ("Skipping " + path)
        return

    random.seed(nTrainPerBrain)

    opt = ['--trans']
    if withTransforms:
        opt += ['--flipx', '--flipy', '--flipz', '--flipxy']
    classifier.initOptions(opt)

    results = generateResults(SCAN_IDS, nTrainPerBrain)
    results.to_csv(path)
    print ("Result saved!\n\t%s" % path)

if __name__ == '__main__':
    for nInputs in [40, 96, 400, 1000, 4000, 10000, 40000]:
        assert nInputs % 8 == 0
        perBrain = nInputs//8
        runExperiment(nTrainPerBrain=perBrain, withTransforms=False)
    for nInputs in [40, 96, 400, 1000, 4000, 10000, 40000]:
        assert nInputs % 8 == 0
        perBrain = nInputs//8
        runExperiment(nTrainPerBrain=perBrain, withTransforms=True)
