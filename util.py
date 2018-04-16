import numpy as np
import datetime
from sklearn.metrics import roc_auc_score

def randomShuffle(X, Y):
    assert X.shape[0] == Y.shape[0]
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]

def todayStr():
    return datetime.datetime.today().strftime('%Y-%m-%d')

# Convert 4d: (R x X x Y x Z) into 2d: (R x XYZ)
def flatCube(data):
    s = data.shape
    return data.reshape((s[0], s[1] * s[2] * s[3]))

# R x (xyz) => 8R x (xyz)
def allRotations(data):
    allData = []
    for i in range(data.shape[0]):
        for xR in [1, -1]:
            for yR in [1, -1]:
                for zR in [1, -1]:
                    allData.append(data[i, ::xR, ::yR, ::zR])
    return np.array(allData)

def combinePredictions(predictions):
    # TODO - better ways?
    return np.mean(predictions, axis=1)

def oneshotY(y):
  return np.column_stack((1 - y, y))

# Given true and predicted Y, generate the scores we care about
def genScores(trueY, predicted):
    assert len(trueY) == len(predicted)
    trueY = (np.array(trueY) > 0.5)
    predY = (np.array(predicted) > 0.5)
    TP = np.sum(  predY  &   trueY )
    FP = np.sum(  predY  & (~trueY))
    TN = np.sum((~predY) & (~trueY))
    FN = np.sum((~predY) &   trueY )
    return [
        (TP + TN) / (TP + TN + FP + FN), # Accuracy
        (TP) / (TP + FN), # Sensitivity
        (TN) / (TN + FP), # Specificity
        (TP + TP) / (TP + TP + FP + FN), # F1
        roc_auc_score(trueY, predicted)
    ]

def formatScores(scores):
  return """Accuracy:    %3.3f%%
Sensitivity: %3.3f%%
Specificity: %3.3f%%
F1 score:    %1.5f
ROC AUC:     %1.5f""" % (scores[0]*100, scores[1]*100, scores[2]*100, scores[3], scores[4])
