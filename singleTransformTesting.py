# SAVING incremetal results up to 6...  
 
import gc
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import random

import classifier
import cnn
import files
import model
import util

RUN_COUNT = 5
BRAIN_COUNT = 6
ALL_FEATURES = False


CNN_FUNC = cnn.runOne

# Only use the old IDs for this one, don't need a lot
OLD_IDS = ['002', '019', '022', '023', '034', '056', '058', '066', '082']
#NEW_IDS = ['017', '027', '035', '046', '064', '077']

# ['034', '002', '056', '082', '058', '019'], ['023', '066', '022']

METRICS = ['Accuracy', 'Sensitivity', 'Specificity', 'Dice score', 'ROC AUC']


def runClassifier(trainBrains, testBrains, trainTransID, collector):
  trainX, trainY = None, None
  print ("Loading points from scans: %s" % (str(trainBrains)))
  for trainBrain in trainBrains:
    print ("  ... loading %s" % (trainBrain))
    fromX, fromY = files.convertScanToXY(
        trainBrain, classifier.ALL_FEAT, classifier.MORE_FEAT, classifier.PAD, 
        False, False, False, False, merge=True,
        oneTransID=trainTransID
    )
    if trainX is None:
      trainX, trainY = fromX, fromY
    else:
      trainX.extend(fromX)
      trainY = np.append( trainY, fromY )
    gc.collect()
  print ("Train X / Y shapes = ", len(trainX), trainY.shape)

  # brain x transform x input
  testX, testY = None, None
  for transID in range(model.N_TRANSFORMS):
    for testBrain in testBrains:
      fromX, fromY = files.convertScanToXY(
          testBrain, classifier.ALL_FEAT, classifier.MORE_FEAT, classifier.PAD, 
          False, False, False, False, merge=True, 
          oneTransID=transID
      )
      if testX is None:
        testX, testY = fromX, fromY
      else:
        testX.extend(fromX)
        testY = np.append( testY, fromY )
      gc.collect()

  print ("Test X / Y shapes = ", len(testX), testY.shape)

  _, _, _, probs = CNN_FUNC(trainX, trainY, testX, testY, "-".join(testBrains), savePath=None)
  perTransformProbs = np.reshape(probs, (model.N_TRANSFORMS, -1))
  trueY = np.reshape(testY, (model.N_TRANSFORMS, -1))
  # Answers should be the same regardless of transform:
  assert np.all(trueY.min(axis=0) == trueY.max(axis=0))
  trueY = trueY[0] # Just need one copy of true results

  # Grade each test transform separately and store results
  for testTransID in range(model.N_TRANSFORMS):
    guesses = perTransformProbs[testTransID]
    scores = util.genScores(trueY, guesses)
    if trainTransID is not None:
      collector[trainTransID, testTransID] = scores
    else:
      collector[testTransID] = scores


# Find all scores for (train on transform T1, test on transform T2)
def generateResults(trainIDs, testIDs, secondHalf=False):
  nScores = 5 # Acc / Sens / Spec / Dice / ROC AUC
  scores = np.zeros((model.N_TRANSFORMS, model.N_TRANSFORMS, nScores))

  half = model.N_TRANSFORMS // 2
  transformRange = range(half, model.N_TRANSFORMS) if secondHalf else range(half)
  for transformID in transformRange:
    classifier.ONE_TRANS_ID = transformID
    print ("RUNNING FOR %s - %s, %s" % (str(classifier.ONE_TRANS_ID), str(trainIDs), str(testIDs)))
    runClassifier(trainIDs, testIDs, transformID, scores)
    print ("\n\n *** \nSAVING incremetal results up to %d...\n ***\n\n" % transformID)
    np.save("transformTest-%d.npy" % (transformID), scores)
  print ("DONE")   

# Find all scores for (train on all transforms, test on transform T)
def generateResultsAllTransforms(trainIDs, testIDs):
  nScores = 5 # Acc / Sens / Spec / Dice / ROC AUC
  scores = np.zeros((model.N_TRANSFORMS, 5))

  classifier.ONE_TRANS_ID = None
  transformID = None
  print ("RUNNING FOR ALL - %s, %s" % (str(trainIDs), str(testIDs)))
  runClassifier(trainIDs, testIDs, transformID, scores)
  print ("\n\n *** \nSAVING results...\n ***\n\n")
  np.save("transformTestAll.npy", scores)
  print ("DONE")   


def runTransformExperiment(secondHalf=False):
  random.seed(100)
  random.shuffle(OLD_IDS)
  # Train on 6, test on 3
  TRAIN_IDS = OLD_IDS[0:6]
  TEST_IDS = OLD_IDS[6:9]

  classifier.ONE_FEAT_NAME = None
  classifier.ONE_TRANS_ID = None
  classifier.PREDICT_TRANSFORM = False
  opt = ['--features']
  classifier.initOptions(opt)

  generateResults(TRAIN_IDS, TEST_IDS, secondHalf)

def runAllTransformsExperiment():
  random.seed(100)
  random.shuffle(OLD_IDS)
  # Train on 6, test on 3
  TRAIN_IDS = OLD_IDS[0:6]
  TEST_IDS = OLD_IDS[6:9]

  classifier.ONE_FEAT_NAME = None
  classifier.ONE_TRANS_ID = None
  classifier.PREDICT_TRANSFORM = False
  opt = ['--features']
  classifier.initOptions(opt)

  generateResultsAllTransforms(TRAIN_IDS, TEST_IDS)

if __name__ == '__main__':
  runTransformExperiment(secondHalf=True)
  #runAllTransformsExperiment()
