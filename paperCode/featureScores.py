# Generate tables of results for raw & features as classifiers themselves

# Run this as 'python paperCode/<code>.py'
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from sklearn.metrics import roc_auc_score
from skimage.filters import threshold_otsu
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)
pd.set_option('precision', 3)

import classifier
import files
import util

FEATURES = ['raw', 'EM', 'JV', 'PC']

# Run a single scan agaist classifiers defined off all the input feature values.
def resultsForScan(scanID):
  data, lTrain, lTest = files.loadAllInputsUpdated(scanID, classifier.PAD, allFeatures=True, moreFeatures=False)
  assert data.shape[-1] == len(FEATURES)

  labels = np.vstack((lTrain, lTest))
  nLabels = labels.shape[0]
  xyz, labels = labels[:, :3], labels[:, 3]
  xyz = xyz - 1 # IMPORTANT!

  predictions = np.zeros(nLabels)

  accs, senss, specs, dices, aucs = \
    np.zeros(len(FEATURES)), np.zeros(len(FEATURES)), np.zeros(len(FEATURES)), np.zeros(len(FEATURES)), np.zeros(len(FEATURES))

  for i in range(len(FEATURES)):
    feat = data[:, :, :, i]
    T_ostu = threshold_otsu(feat.ravel())
    for j in range(nLabels):
      predictions[j] = feat[xyz[j, 0], xyz[j, 1], xyz[j, 2]]
    accs[i], senss[i], specs[i], dices[i], aucs[i] = util.genScores(labels, predictions, 0.5, T_ostu)
  return accs, senss, specs, dices, aucs

# For each feature, check accuracy of using that feature to classify all scans.
def generateResults():
  shape = (len(FEATURES), len(util.SCAN_IDS))

  accs, senss, specs, dices, aucs = \
    np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)

  for idx, scanID in enumerate(util.SCAN_IDS):
    fAccs, fSenss, fSpecs, fDices, fAucs = resultsForScan(scanID)
    accs[ :, idx] = fAccs
    senss[:, idx] = fSenss
    specs[:, idx] = fSpecs
    dices[:, idx] = fDices
    aucs[ :, idx] = fAucs
  print ("\n\n")

  allMeans = []
  for metric, result in zip(util.METRICS, [accs, senss, specs, dices, aucs]):
    mean = np.mean(result, axis=1)[np.newaxis].T # This is really ugly...
    allMeans.append(mean)
    withMean = np.hstack((result, mean))
    print ("\n\n%s for (scan, feature)" % (metric))
    print ("-----------")
    print (pd.DataFrame(withMean, FEATURES, util.SCAN_IDS + ['Mean']))

  print ("\n\nAverage scores across scans:")
  print ("-----------")
  print (pd.DataFrame(np.hstack(tuple(allMeans)), FEATURES, util.METRICS))


if __name__ == '__main__':
  generateResults()
