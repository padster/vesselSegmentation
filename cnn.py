import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import xgboost as xgb
import matplotlib.pyplot as plt

import files

N_FOLDS = 2 # Train on 3/4, Test on 1/4
N_REPEATS = 5 # K-fold this many times
RANDOM_SEED = 194981

HACK_GUESSES = []

# Convert 4d: (R x X x Y x Z) into 2d: (R x XYZ)
def flatCube(data):
    s = data.shape
    return data.reshape((s[0], s[1] * s[2] * s[3]))

# As an example, run CNN on these given labels and test data, return the score.
def runOne(trainX, trainY, testX, testY):
    pass
    # return roc_auc_score(testY, preds)

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
        pass # HACK - REMOVE
        # trainX, trainY = Xs[trainIdx], Ys[trainIdx]
        # testX, testY = Xs[testIdx], Ys[testIdx]
        # scores.append(runOne(trainX, trainY, testX, testY))

    """
    HG = np.array(HACK_GUESSES)
    for i in range(10):
        lBound = i / 10.0
        uBound = lBound + 0.1
        print ("%0.1f - %0.1f = %d" % (lBound, uBound, ((lBound <= HG) & (HG < uBound)).sum()))
    plt.hist(HACK_GUESSES)
    plt.show()
    """
    print ("Average score: %.3f " % (np.mean(np.array(scores))))

def generatePrediction(Xs, Ys, mraAll):
    print ("TODO: Train off labels, then predict on all cells. ")
    """
    print("GENERATING PRED")
    trainD = xgb.DMatrix(flatCube(Xs), label=Ys)
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
    print ("Converting entire volume to inputs...")
    inputs = files.convertEntireVolume(mraAll)
    print ("Shape of all inputs = ")
    print (inputs.shape)
    chunking = 10
    chunk = (inputs.shape[0] + chunking - 1) // chunking
    allPreds = []
    for i in tqdm(range(chunking)):
        startIdx = chunk * i
        endIdx = min(chunk * i + chunk, inputs.shape[0])
        testD  = xgb.DMatrix(flatCube(inputs[startIdx:endIdx]))
        preds = trees.predict(testD)
        allPreds.extend(preds.tolist())
    allPreds = np.array(allPreds)
    print ("predicted " + str(allPreds.shape))

    result = np.zeros(mraAll.shape)
    result = files.fillPredictions(result, allPreds)
    return result
    """

# TODO: document
def generateAndWriteResults():
    mra = files.loadMRA()
    labels = files.loadLabels()
    Xs, Ys = files.convertToInputs(mra, labels)
    print ("%d samples" % len(Xs))
    runKFold(Xs, Ys)
    # prediction = generatePrediction(Xs, Ys, mra)
    # files.writePrediction("data/Normal001-MRA-CNN.mat", "forest", prediction)


if __name__ == '__main__':
    generateAndWriteResults()
