import gc
import numpy as np
np.random.seed(0)

import files
import util

SIZE = 9
PAD = (SIZE-1)//2

import sys

N_FEAT, RUN_AWS, ALL_FEAT, MORE_FEAT, SAVE_NET, LOAD_NET, FLIP_X, FLIP_Y, FLIP_Z, PREDICT_TRANSFORM, CNN_FEAT, ONE_FEAT, ONE_FEAT_NAME = \
    None, None, None, None, None, None, None, None, None, None, None, None, None,

def initOptions(argv):
    global N_FEAT, RUN_AWS, ALL_FEAT, MORE_FEAT, SAVE_NET, LOAD_NET, FLIP_X, FLIP_Y, FLIP_Z, PREDICT_TRANSFORM, CNN_FEAT, ONE_FEAT, ONE_FEAT_NAME

    RUN_AWS = "--local" not in argv
    ALL_FEAT = "--features" in argv
    MORE_FEAT = "--morefeat" in argv
    SAVE_NET = "--save" in argv
    LOAD_NET = "--load" in argv
    FLIP_X = "--flipx" in argv
    FLIP_Y = "--flipy" in argv
    FLIP_Z = "--flipz" in argv
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

    print ("====\nTarget: %s\n# Features: %d\nVolume: %dx%dx%d\nFeatures: %s\n%s%sFlip X: %s\nFlip Y: %s\nFlip Z: %s\n%s%s ====\n" % (
        "GPU" if RUN_AWS else "Local",
        N_FEAT,
        SIZE, SIZE, SIZE,
        "All" if ALL_FEAT else "Intensity",
        "Loading from file\n" if LOAD_NET else "",
        "Saving to file\n" if SAVE_NET else "",
        str(FLIP_X),
        str(FLIP_Y),
        str(FLIP_Z),
        "Predict with transform\n" if PREDICT_TRANSFORM else "",
        ("With only feature %s\n" % (ONE_FEAT_NAME)) if ONE_FEAT_NAME is not None else "",
    ))



def singleBrain(scanID, runOneFunc, calcScore=True, writeVolume=False, savePath=None):
  data, labelsTrain, labelsTest = files.loadAllInputsUpdated(scanID, ALL_FEAT, MORE_FEAT, oneFeat=ONE_FEAT_NAME)

  if calcScore:
    trainX, trainY = files.convertToInputs(data, labelsTrain, PAD, FLIP_X, FLIP_Y, FLIP_Z)
    testX,   testY = files.convertToInputs(data,  labelsTest, PAD, False, False, False)
    print ("%d train samples, %d test" % (len(trainX), len(testX)))
    _, _, scores = runOneFunc(trainX, trainY, testX, testY, scanID, savePath)
    print ("  Results\n  -------\n" + util.formatScores(scores))

  if writeVolume:
    labels = np.vstack((labelsTrain, labelsTest))
    trainX, trainY = files.convertToInputs(data, labels, PAD, FLIP_X, FLIP_Y, FLIP_Z)
    runOneFunc(trainX, trainY, data, None, scanID, savePath)

def brainsToBrain(fromIDs, toID, runOneFunc, calcScore=True, writeVolume=False, savePath=None):
    trainX, trainY = None, None
    print ("Loading points from scans: %s" % (str(fromIDs)))
    for fromID in fromIDs:
        print ("  ... loading %s" % (fromID))
        fromX, fromY = files.convertScanToXY(fromID, ALL_FEAT, MORE_FEAT, PAD, FLIP_X, FLIP_Y, FLIP_Z, merge=True, oneFeat=ONE_FEAT_NAME)
        if trainX is None:
            trainX, trainY = fromX, fromY
        else:
            trainX = np.vstack((trainX, fromX))
            trainY = np.append( trainY, fromY )
        gc.collect()
    print ("Train X / Y shapes = ", trainX.shape, trainY.shape)

    toReturn = None
    if calcScore:
        toX, toY = files.convertScanToXY(toID, ALL_FEAT, MORE_FEAT, PAD, False, False, False, merge=True, oneFeat=ONE_FEAT_NAME)
        print ("Test X / Y shapes = ", toX.shape, toY.shape)
        _, _, scores = runOneFunc(trainX, trainY, toX, toY, toID, savePath)
        print ("  Results\n  -------\n" + util.formatScores(scores))
        print ("\n\n\n")
        toReturn = scores

    if writeVolume:
        data, _, _ = files.loadAllInputsUpdated(toID, ALL_FEAT, MORE_FEAT, oneFeat=ONE_FEAT_NAME)
        _, _, volume = runOneFunc(trainX, trainY, data, None, toID, savePath)
        toReturn = volume
    return toReturn
