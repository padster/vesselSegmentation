import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import xgboost as xgb

import files

N_FOLDS = 2 # Train on 3/4, Test on 1/4
N_REPEATS = 5 # K-fold this many times
RANDOM_SEED = 194981

# Convert 4d: (R x X x Y x Z) into 2d: (R x XYZ)
def flatCube(data):
    s = data.shape
    return data.reshape((s[0], s[1] * s[2] * s[3]))

# As an example, run XGBoost on this set.
# TODO: Use different algorithms.
def runOne(trainX, trainY, testX, testY):
    trainD = xgb.DMatrix(flatCube(trainX), label=trainY)
    testD  = xgb.DMatrix(flatCube( testX), label= testY)
    param = {
        'max_depth': 3,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        'objective': 'binary:logistic',  # error evaluation for multiclass training
    }
    nRounds = 5  # the number of training iterations

    # Train using only the training set:
    trees = xgb.train(param, trainD, nRounds)
    # Use the trained forest to predict the remaining positions:
    preds = trees.predict(testD)
    # This outputs a 0 - 1 value for each cell, convert to score via ROC AUC:
    return roc_auc_score(testY, preds)

def runKFold(Xs, Ys):
    """
    Cross-validate using stratified KFold:
      * split into K folds, keeping the same true/false proportion.
      * use (K-1) to train and 1 to test
      * run a bunch of times
    """
    print ("Input: %d, %d T, %d F" % (len(Ys), sum(Ys == 1), sum(Ys == 0)))
    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    scores = []
    for trainIdx, testIdx in rskf.split(Xs, Ys):
        trainX, trainY = Xs[trainIdx], Ys[trainIdx]
        testX, testY = Xs[testIdx], Ys[testIdx]
        scores.append(runOne(trainX, trainY, testX, testY))
    print ("Average score: %.3f " % (np.mean(np.array(scores))))

def main():
    mra = files.loadMRA()
    labels = files.loadLabels()
    Xs, Ys = files.convertToInputs(mra, labels)
    print ("%d samples" % len(Xs))
    runKFold(Xs, Ys)

if __name__ == '__main__':
    main()
