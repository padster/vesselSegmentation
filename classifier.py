import gc
import numpy as np
import random
np.random.seed(0)

import files
import util

SIZE = 9
PAD = (SIZE-1)//2

import sys

N_FEAT, RUN_AWS, ALL_FEAT, MORE_FEAT, SAVE_NET, LOAD_NET, FLIP_X, FLIP_Y, FLIP_Z, FLIP_XY, PREDICT_TRANSFORM, CNN_FEAT, ONE_FEAT, ONE_FEAT_NAME, ONE_TRANS_ID = \
    None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

def initOptions(argv):
    global N_FEAT, RUN_AWS, ALL_FEAT, MORE_FEAT, SAVE_NET, LOAD_NET, FLIP_X, FLIP_Y, FLIP_Z, FLIP_XY, PREDICT_TRANSFORM, CNN_FEAT, ONE_FEAT, ONE_FEAT_NAME, ONE_TRANS_ID

    RUN_AWS = "--local" not in argv
    ALL_FEAT = "--features" in argv
    MORE_FEAT = "--morefeat" in argv
    SAVE_NET = "--save" in argv
    LOAD_NET = "--load" in argv
    FLIP_X = "--flipx" in argv
    FLIP_Y = "--flipy" in argv
    FLIP_Z = "--flipz" in argv
    FLIP_XY = "--flipxy" in argv
    PREDICT_TRANSFORM = "--trans" in argv
    CNN_FEAT = "--cnnfeat" in argv
    files.CNN_FEAT = CNN_FEAT # HACK
    ONE_FEAT = "--onefeat" in argv

    N_FEAT = 1
    if ALL_FEAT:
        N_FEAT += 3
    if CNN_FEAT:
        N_FEAT += 1
    if MORE_FEAT:
        N_FEAT += 6
    if ONE_FEAT:
        N_FEAT += 1

    print ("""
====
Target: %s
# Features: %d
Volume: %dx%dx%d
Features: %s
%s%sFlip X: %s
Flip Y: %s
Flip Z: %s
Flip XY: %s
%s%s%s====
""" % (
        "GPU" if RUN_AWS else "Local",
        N_FEAT,
        SIZE, SIZE, SIZE,
        "All" if ALL_FEAT else "Intensity",
        "Loading from file\n" if LOAD_NET else "",
        "Saving to file\n" if SAVE_NET else "",
        str(FLIP_X),
        str(FLIP_Y),
        str(FLIP_Z),
        str(FLIP_XY),
        "Predict with transform\n" if PREDICT_TRANSFORM else "",
        ("With only feature %s\n" % (ONE_FEAT_NAME)) if ONE_FEAT_NAME is not None else "",
        ("With only transform %s\n" % (ONE_TRANS_ID)) if ONE_TRANS_ID is not None else "",
    ))


def singleBrain(scanID, runOneFunc, calcScore=True, writeVolume=False, savePath=None):
  data, labelsTrain, labelsTest = files.loadAllInputsUpdated(scanID, PAD, ALL_FEAT, MORE_FEAT, oneFeat=ONE_FEAT_NAME)

  toReturn = None
  if calcScore:
    trainX, trainY = files.convertToInputs(scanID, data, labelsTrain, PAD, FLIP_X, FLIP_Y, FLIP_Z, FLIP_XY)
    testX,   testY = files.convertToInputs(scanID, data,  labelsTest, PAD, False, False, False, FLIP_XY)
    print ("%d train samples, %d test" % (len(trainX), len(testX)))
    _, _, scores, _ = runOneFunc(trainX, trainY, testX, testY, scanID, savePath)
    print ("  Results\n  -------\n" + util.formatScores(scores))
    toReturn = scores

  if writeVolume:
    labels = np.vstack((labelsTrain, labelsTest))
    trainX, trainY = files.convertToInputs(scanID, data, labels, PAD, FLIP_X, FLIP_Y, FLIP_Z)
    _, _, volume, _ = runOneFunc(trainX, trainY, data, None, scanID, savePath)
    toReturn = volume
  return toReturn


def brainsToBrain(fromIDs, toID, runOneFunc, calcScore=True, writeVolume=False, savePath=None, perBrainExamples=None):
    trainX, trainY = None, None
    print ("Loading points from scans: %s" % (str(fromIDs)))
    for fromID in fromIDs:
        print ("  ... loading %s" % (fromID))
        fromX, fromY = files.convertScanToXY(fromID, ALL_FEAT, MORE_FEAT, PAD, FLIP_X, FLIP_Y, FLIP_Z, FLIP_XY, merge=True, oneFeat=ONE_FEAT_NAME, oneTransID=ONE_TRANS_ID)

        if perBrainExamples is not None:
            idxSubset = np.random.choice(len(fromX), perBrainExamples, replace=False)
            fromX = [fromX[idx] for idx in idxSubset]
            fromY = fromY[idxSubset]

        if trainX is None:
            trainX, trainY = fromX, fromY
        else:
            trainX.extend(fromX)
            trainY = np.append( trainY, fromY )
        gc.collect()

    print ("Train X / Y shapes = ", len(trainX), trainY.shape)
    toReturn = None
    if calcScore:
        toX, toY = files.convertScanToXY(toID, ALL_FEAT, MORE_FEAT, PAD, False, False, False, False, merge=True, oneFeat=ONE_FEAT_NAME, oneTransID=ONE_TRANS_ID)
        print ("Test X / Y shapes = ", len(toX), toY.shape)
        _, _, scores, _ = runOneFunc(trainX, trainY, toX, toY, toID, savePath)
        print ("  Results\n  -------\n" + util.formatScores(scores))
        print ("\n\n\n")
        toReturn = scores

    if writeVolume:
        data, _, _ = files.loadAllInputsUpdated(toID, PAD, ALL_FEAT, MORE_FEAT, oneFeat=ONE_FEAT_NAME)
        _, _, volume, _ = runOneFunc(trainX, trainY, data, None, toID, savePath)
        toReturn = volume
    return toReturn
