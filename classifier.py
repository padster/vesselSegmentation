import numpy as np
import files
import util


import sys
RUN_AWS = "--local" not in sys.argv
ALL_FEAT = "--features" in sys.argv
SAVE_NET = "--save" in sys.argv
LOAD_NET = "--load" in sys.argv
FLIP_X = "--flipx" in sys.argv
FLIP_Y = "--flipy" in sys.argv
FLIP_Z = "--flipz" in sys.argv
print ("====\nTarget: %s\nFeatures: %s\n%s%sFlip X: %s\nFlip Y: %s\nFlip Z: %s\n====\n" % (
    "AWS" if RUN_AWS else "Local",
    "All" if ALL_FEAT else "Intensity",
    "Loading from file\n" if LOAD_NET else "",
    "Saving to file\n" if SAVE_NET else "",
    str(FLIP_X),
    str(FLIP_Y),
    str(FLIP_Z),
))

SIZE = 7
PAD = (SIZE-1)//2

def singleBrain(scanID, runOneFunc, calcScore=True, writeVolume=False):
  data, labelsTrain, labelsTest = files.loadAllInputsUpdated(scanID, ALL_FEAT)

  if calcScore:
    trainX, trainY = files.convertToInputs(data, labelsTrain, PAD, FLIP_X, FLIP_Y, FLIP_Z)
    testX,   testY = files.convertToInputs(data,  labelsTest, PAD, False, False, False)
    print ("%d train samples, %d test" % (len(trainX), len(testX)))
    _, _, scores = runOneFunc(trainX, trainY, testX, testY, scanID)
    print ("  Results\n  -------\n" + util.formatScores(scores))

  if writeVolume:
    labels = np.vstack((labelsTrain, labelsTest))
    trainX, trainY = files.convertToInputs(data, labels, PAD, FLIP_X, FLIP_Y, FLIP_Z)
    runOneFunc(trainX, trainY, data, None, scanID)

def brainsToBrain(fromIDs, toID, runOneFunc, calcScore=True, writeVolume=False):
    trainX, trainY = None, None
    print ("Loading points from scans: %s" % (str(fromIDs)))
    for fromID in fromIDs:
        print ("  ... loading %s" % (fromID))
        fromX, fromY = files.convertScanToXY(fromID, ALL_FEAT, PAD, FLIP_X, FLIP_Y, FLIP_Z, merge=True)
        if trainX is None:
            trainX, trainY = fromX, fromY
        else:
            trainX = np.vstack((trainX, fromX))
            trainY = np.append( trainY, fromY )
    print ("Train X / Y shapes = ", trainX.shape, trainY.shape)

    if calcScore:
        toX, toY = files.convertScanToXY(toID, ALL_FEAT, PAD, False, False, False, merge=True)
        print ("Test X / Y shapes = ", toX.shape, toY.shape)
        _, _, scores = runOneFunc(trainX, trainY, toX, toY, toID)
        print ("  Results\n  -------\n" + util.formatScores(scores))

    if writeVolume:
        data, _, _ = files.loadAllInputsUpdated(toID, ALL_FEAT)
        runOneFunc(trainX, trainY, data, None, toID)

""" TODO: bring back?
def runKFold(Xs, Ys):
    Cross-validate using stratified KFold:
      * split into K folds, keeping the same true/false proportion.
      * use (K-1) to train and 1 to test
      * run a bunch of times

    print ("Input: %d, %d T, %d F" % (len(Ys), sum(Ys == 1), sum(Ys == 0)))

    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    splits = [(a, b) for a, b in rskf.split(Xs, Ys)]

    allCosts, allCorrs, allScores = [], [], []
    for i, (trainIdx, testIdx) in enumerate(splits):
        print ("Split %d / %d" % (i + 1, len(splits)))
        trainX, trainY = Xs[trainIdx], Ys[trainIdx]
        testX, testY = Xs[testIdx], Ys[testIdx]
        runCosts, runCorrs, runScores = runOne(trainX, trainY, testX, testY, i)
        allCosts.append(runCosts)
        allCorrs.append(runCorrs)
        allScores.append(runScores)
        print ("Split %d scores = %s" % (i + 1, str(runScores)))


    ax = viz.clean_subplots(1, 2, show=(not RUN_AWS))
    ax[0][0].set_title("Loss over epochs, per split")
    ax[0][0].plot(np.array(allCosts).T)
    ax[0][1].set_title("%Correct over epochs, per split ")
    ax[0][1].plot(np.array(allCorrs).T)

    image_path = "images/LossAndCorrect.png"
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.savefig(image_path)
    print ("Image saved to %s" % str(image_path))

    if not RUN_AWS:
        plt.show()

    print ("Average scores: %s" % (np.mean(allScores, axis=0)))
"""