import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime

import files
import viz

# TODO - argparse?
import sys
RUN_AWS = "--local" not in sys.argv
ALL_FEAT = "--features" in sys.argv
SAVE_NET = "--save" in sys.argv
LOAD_NET = "--load" in sys.argv
FLIP_X = "--flipx" in sys.argv
FLIP_Y = "--flipy" in sys.argv
print ("====\nTarget: %s\nFeatures: %s\n%s%sFlip X: %s\nFlip Y: %s\n====\n" % (
    "AWS" if RUN_AWS else "Local",
    "All" if ALL_FEAT else "Intensity",
    "Loading from file\n" if LOAD_NET else "",
    "Saving to file\n" if SAVE_NET else "",
    str(FLIP_X),
    str(FLIP_Y),
))


N_FOLDS = 2 # Train on 1/2, Test on 1/2
N_REPEATS = 5 if RUN_AWS else 1 # K-fold this many times
RANDOM_SEED = 194981
N_CHANNELS = 4 # Intensity, EM, JV, PC

ERROR_WEIGHT = -4 # Positive = FN down, Sensitivity up. Negative = FP down, Specificity up
ERROR_WEIGHT_FRAC = 2 ** ERROR_WEIGHT

# SET IN MAIN:
#SIZE = 0
#N_EPOCHS = 0
#BATCH_SIZE = 0
#RUN_LOCAL = False

LEARNING_RATE = 0.001 # 0.03
DROPOUT_RATE = 0.75

HACK_GUESSES = []
HACK_COSTS = []
HACK_CORRS = []

def randomShuffle(X, Y):
    assert X.shape[0] == Y.shape[0]
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]

def todayStr():
    return datetime.datetime.today().strftime('%Y-%m-%d')

def buildNetwork(dropoutRate=DROPOUT_RATE, learningRate=LEARNING_RATE, seed=RANDOM_SEED):
    nChannels = N_CHANNELS if ALL_FEAT else 1
    xInput = tf.placeholder(tf.float32, shape=[None, SIZE, SIZE, SIZE, nChannels])
    yInput = tf.placeholder(tf.float32, shape=[None, 2])

    #nFilt = [64, 128, 128]
    nFilt = [64, 32, 32]
    #nFilt = [16, 32, 32]

    with tf.name_scope("layer_a"):
        # conv => 7*7*7
        conv1 = tf.layers.conv3d(inputs=xInput, filters=nFilt[0], kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # conv => 7*7*7
        conv2 = tf.layers.conv3d(inputs=conv1, filters=nFilt[1], kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # pool => 3*3*3
        pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2,2,2], strides=2)

    """
    with tf.name_scope("layer_c"):
        # conv => 3*3*3
        conv4 = tf.layers.conv3d(inputs=pool3, filters=64, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # conv => 3*3*3
        conv5 = tf.layers.conv3d(inputs=conv4, filters=128, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # pool => 1*1*1
        pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2,2,2], strides=2)
    """

    with tf.name_scope("batch_norm"):
        cnn3d_bn = tf.layers.batch_normalization(inputs=pool3, training=True)

    with tf.name_scope("fully_con"):
        flattening = tf.reshape(cnn3d_bn, [-1, 3*3*3*nFilt[1]])
        dense = tf.layers.dense(inputs=flattening, units=nFilt[2], activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=dropoutRate, training=True)

    with tf.name_scope("y_conv"):
        prediction = tf.layers.dense(inputs=dropout, units=2)
        predictedProbs = tf.nn.softmax(prediction)

    with tf.name_scope("cross_entropy"):
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=yInput))
        cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            targets=yInput, logits=prediction, pos_weight=ERROR_WEIGHT_FRAC
        ))

    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
        # optimizer = tf.train.AdagradOptimizer(learningRate).minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(yInput, 1))
    numCorrect = tf.reduce_sum(tf.cast(correct, tf.int32))

    return xInput, yInput, optimizer, cost, numCorrect, predictedProbs

# Convert 4d: (R x X x Y x Z) into 2d: (R x XYZ)
def flatCube(data):
    s = data.shape
    return data.reshape((s[0], s[1] * s[2] * s[3]))

# Given true and predicted Y, generate the scores we care about
def genScores(trueY, predicted):
    n = len(trueY)
    assert len(predicted) == n
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

# As an example, run CNN on these given labels and test data, return the score.
def runOne(trainX, trainY, testX, testY, runID):
    epochs = N_EPOCHS
    batchSize = BATCH_SIZE

    xInput, yInput, optimizer, cost, numCorrect, scores = buildNetwork()

    costs, corrs = [], []
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        start_time = datetime.datetime.now()

        iterations = int(len(trainY)/batchSize) + 1
        # run epochs
        for epoch in range(epochs):
            trainX, trainY = randomShuffle(trainX, trainY)
            testX,   testY = randomShuffle( testX,  testY)

            start_time_epoch = datetime.datetime.now()
            print('Split %d, Epoch %d started' % (runID, epoch))

            # mini batch for trianing set:
            totalCost, totalCorr = 0.0, 0
            for itr in range(iterations):
                mini_batch_x = trainX[itr*batchSize: (itr+1)*batchSize]
                mini_batch_y = trainY[itr*batchSize: (itr+1)*batchSize]
                batchYOneshot = (np.column_stack((mini_batch_y, mini_batch_y)) == [0, 1]) * 1
                _optimizer, _cost, _corr = sess.run([optimizer, cost, numCorrect], feed_dict={xInput: mini_batch_x, yInput: batchYOneshot})
                totalCost += _cost
                totalCorr += _corr

            print (">> Epoch %d had TRAIN loss: %f\t#Correct = %5d/%5d = %f" % (
                epoch, totalCost, totalCorr, len(trainY), totalCorr / len(trainY)
            ))

            # Run against test set:
            totalCost, totalCorr = 0.0, 0
            itrs = int(len(testY)/batchSize) + 1
            for itr in range(itrs):
                mini_batch_x_test = testX[itr*batchSize: (itr+1)*batchSize]
                mini_batch_y_test = testY[itr*batchSize: (itr+1)*batchSize]

                batchYOneshotTest = (np.column_stack((mini_batch_y_test, mini_batch_y_test)) == [0, 1]) * 1

                _cost, _corr = sess.run([cost, numCorrect], feed_dict={xInput: mini_batch_x_test, yInput: batchYOneshotTest})
                totalCost += _cost
                totalCorr += _corr

            end_time_epoch = datetime.datetime.now()
            print('>> Epoch %d had  TEST loss: %f\t#Correct = %5d/%5d = %f\tTime elapsed: %s' % (
                epoch, totalCost, totalCorr, len(testY), totalCorr / len(testY), str(end_time_epoch - start_time_epoch)
            ))
            costs.append(totalCost)
            corrs.append(totalCorr/len(testY))

        end_time = datetime.datetime.now()
        print('Time elapse: ', str(end_time - start_time))

        # Run against test:
        testYOneshot = (np.column_stack((testY, testY)) == [0, 1]) * 1
        testProbs, testCorr = sess.run([scores, numCorrect], feed_dict={xInput: testX, yInput: testYOneshot})
        testProbs = np.array(testProbs)[:, 1].tolist()
        testCorr = testCorr / len(testY)

    # HACK_GUESSES.extend(testProbs)
    # HACK_COSTS.append(costs)
    # HACK_CORRS.append(corrs)
    return costs, corrs, genScores(testY, testProbs) # testCorr, roc_auc_score(testY, testProbs)

def runKFold(Xs, Ys):
    """
    Cross-validate using stratified KFold:
      * split into K folds, keeping the same true/false proportion.
      * use (K-1) to train and 1 to test
      * run a bunch of times
    """
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


def trainAndSaveNet(data, labels, path):
    epochs = N_EPOCHS
    batchSize = BATCH_SIZE
    xInput, yInput, optimizer, cost, numCorrect, scores = buildNetwork()

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        start_time = datetime.datetime.now()

        iterations = int(len(labels)/batchSize) + 1
        # run epochs
        for epoch in range(epochs):
            data, labels = randomShuffle(data, labels)
            start_time_epoch = datetime.datetime.now()
            print('Saving epoch %d started' % (epoch))

            # mini batch for trianing set:
            totalCost, totalCorr = 0.0, 0
            for itr in range(iterations):
                mini_batch_x = data[itr*batchSize: (itr+1)*batchSize]
                mini_batch_y = labels[itr*batchSize: (itr+1)*batchSize]
                batchYOneshot = (np.column_stack((mini_batch_y, mini_batch_y)) == [0, 1]) * 1
                _optimizer, _cost, _corr = sess.run([optimizer, cost, numCorrect], feed_dict={xInput: mini_batch_x, yInput: batchYOneshot})
                totalCost += _cost
                totalCorr += _corr

            print (">> Epoch %d had TRAIN loss: %f\t#Correct = %5d/%5d = %f" % (
                epoch, totalCost, totalCorr, len(labels), totalCorr / len(labels)
            ))

        end_time = datetime.datetime.now()
        print('Time elapse: ', str(end_time - start_time))
        savePath = saver.save(sess, path)
        print('Model saved to: %s ' % savePath)


# TODO: document
def generateAndWriteNet(save):
    data, labels = files.loadAllInputs(ALL_FEAT)
    Xs, Ys = files.convertToInputs(data, labels, (SIZE-1)//2, FLIP_X, FLIP_Y)
    print ("%d samples" % len(Xs))
    # runKFold(Xs, Ys)
    if save:
        path = "network/cnn_%s.ckpt" % (todayStr())
        trainAndSaveNet(Xs, Ys, path)

def loadAndWritePrediction(savePath):
    PAD = (SIZE-1)//2
    data, labels = files.loadAllInputs(ALL_FEAT)

    startX, endX = PAD, data.shape[0] - PAD
    startY, endY = PAD, data.shape[0] - PAD

    xInput, yInput, _, _, _, predictedProbs = buildNetwork()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, savePath)

        allPreds = []
        for x in tqdm(range(startX, endX)):
            for y in tqdm(range(startY, endY)):
                dataAsInput = files.convertVolumeStack(data, PAD, x, y)
                preds = sess.run(predictedProbs, feed_dict={xInput: dataAsInput})
                preds = preds[:, 1]
                allPreds.extend(preds.tolist())
        allPreds = np.array(allPreds)
        print ("Predicted shape: " + str(allPreds.shape))

    dShape = data.shape
    result = np.zeros((dShape[0], dShape[1], dShape[2]))
    result = files.fillPredictions(result, allPreds, pad=PAD)
    files.writePrediction("data/Normal001-MRA-CNN.mat", "cnn", result)


def singleBrain(scanID):
    PAD = (SIZE-1)//2
    data, labelsTrain, labelsTest = files.loadAllInputsUpdated(scanID, ALL_FEAT)
    trainX, trainY = files.convertToInputs(data, labelsTrain, PAD, FLIP_X, FLIP_Y)
    testX,   testY = files.convertToInputs(data,  labelsTest, PAD, False, False)
    print ("%d train samples, %d test" % (len(trainX), len(testX)))
    costs, corrs, scores = runOne(trainX, trainY, testX, testY, 0)
    print ("Scores: %s" % (str(scores)))
    # if save:
        # path = "network/cnn_%s.ckpt" % (todayStr())
        # trainAndSave(Xs, Ys, path)

def singleBrainWritePrediction(scanID):
    networkPath = "data/%s/CNN_%s.ckpt" % (scanID, scanID)
    PAD = (SIZE-1)//2
    data, labelsTrain, labelsTest = files.loadAllInputsUpdated(scanID, ALL_FEAT)
    labels = np.vstack((labelsTrain, labelsTest))
    data, labels = files.convertToInputs(data, labels, PAD, FLIP_X, FLIP_Y)
    print ("Part #1: Training then saving to %s" % (networkPath))
    trainAndSaveNet(data, labels, networkPath)

    print ("\n=======\nPart #2: Loading and generating all predictions")
    startX, endX = PAD, data.shape[0] - PAD
    startY, endY = PAD, data.shape[0] - PAD

    xInput, yInput, _, _, _, predictedProbs = buildNetwork()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, networkPath)

        allPreds = []
        for x in tqdm(range(startX, endX)):
            for y in tqdm(range(startY, endY)):
                dataAsInput = files.convertVolumeStack(data, PAD, x, y)
                preds = sess.run(predictedProbs, feed_dict={xInput: dataAsInput})
                preds = preds[:, 1]
                allPreds.extend(preds.tolist())
        allPreds = np.array(allPreds)
        print ("Predicted shape: " + str(allPreds.shape))

    dShape = data.shape
    result = np.zeros((dShape[0], dShape[1], dShape[2]))
    result = files.fillPredictions(result, allPreds, pad=PAD)
    resultPath = "data/%s/Normal%s-MRA-CNN.mat" % (scanID, scanID)
    files.writePrediction(resultPath, "cnn", result)


def brainToBrain(fromIDs, toID):
    PAD = (SIZE-1)//2
    trainX, trainY = None, None
    print ("Loading points from %d inputs..." % (len(fromIDs)))
    for fromID in fromIDs:
        print ("  ... loading %s" % (fromID))
        fromData, fromLabelA, fromLabelB = files.loadAllInputsUpdated(fromID, ALL_FEAT)
        fromLabels = np.concatenate((fromLabelA, fromLabelB))
        fromX, fromY = files.convertToInputs(fromData, fromLabels, PAD, FLIP_X, FLIP_Y)
        if trainX is None:
            trainX, trainY = fromX, fromY
        else:
            trainX = np.concatenate((trainX, fromX))
            trainY = np.concatenate((trainY, fromY))

    toData, toLabelA, toLabelB = files.loadAllInputsUpdated(toID, ALL_FEAT)
    toLabels = np.concatenate((toLabelA, toLabelB))
    toX, toY = files.convertToInputs(toData, toLabels, PAD, FLIP_X, FLIP_Y)
    runOne(trainX, trainY, toX, toY, 0)

if __name__ == '__main__':
    global SIZE, N_EPOCHS, BATCH_SIZE, RUN_LOCAL
    SIZE = 7
    N_EPOCHS = 25 if RUN_AWS else 2
    BATCH_SIZE = 10 * (2 if FLIP_X else 1) * (2 if FLIP_Y else 1)

    # singleBrain('002')
    singleBrainWritePrediction('002')

    # if LOAD_NET:
        # loadAndWritePrediction("network/cnn_2018-04-02.ckpt")
    # else:
        # generateAndWriteNet(SAVE_NET)
