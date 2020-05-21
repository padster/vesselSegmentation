# Unused: Test out our network against a random forest classifier
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import xgboost as xgb
import matplotlib.pyplot as plt

import classifier
import files
import util

N_FOLDS = 2 # Train on 3/4, Test on 1/4
N_REPEATS = 5 # K-fold this many times
RANDOM_SEED = 194981

HACK_GUESSES = []

def predict(forest, data):
    if classifier.PREDICT_TRANSFORM:
        data = util.allRotations(data)
        testD  = xgb.DMatrix(flatCube(data))
        pred = forest.predict(testD)
        pred = pred.reshape((-1, 8))
        return util.combinePredictions(pred)
    else:
        testD  = xgb.DMatrix(flatCube(data))
        return forest.predict(testD)

# Convert 4d: (R x X x Y x Z x C) into 2d: (R x XYZC)
def flatCube(data):
    s = data.shape
    newS = s[1] * s[2] * s[3]
    if len(data.shape) == 5:
        newS = newS * s[4]
    return data.reshape((s[0], newS))

def runOne(trainX, trainY, testX, testY, scanID, savePath):
    runTest = testY is not None
    runVolume = testY is None
    assert not runVolume # TODO: implement volume writing
    testProbs = None

    # TODO: pass weights
    trainD = xgb.DMatrix(flatCube(trainX), label=trainY)
    param = {
        'max_depth': 8,  # the maximum depth of each tree
        'eta': 0.1,  # the training step for each iteration
        'silent': 0,  # logging mode - quiet
        'alpha': 1, # L1 reg
        'objective': 'binary:logistic',  # error evaluation for multiclass training
    }
    nRounds = 25  # the number of training iterations

    # Train using only the training set:
    forest = xgb.train(param, trainD, nRounds)

    # Use the trained forest to predict the remaining positions:
    testProbs = predict(forest, testX)
    return [], [], util.genScores(testY, testProbs)

if __name__ == '__main__':
    # classifier.singleBrain('002', runOne, calcScore=True, writeVolume=False)
    classifier.brainsToBrain(['002', '019', '022'], '023', runOne, calcScore=True, writeVolume=False, savePath=None)
