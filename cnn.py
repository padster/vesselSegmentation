import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import tensorflow as tf
from tqdm import tqdm

import classifier
import files
import util
import viz

# TODO - argparse?
# import sys
# RUN_AWS = "--local" not in sys.argv
# ALL_FEAT = "--features" in sys.argv
# SAVE_NET = "--save" in sys.argv
# LOAD_NET = "--load" in sys.argv
# FLIP_X = "--flipx" in sys.argv
# FLIP_Y = "--flipy" in sys.argv
# FLIP_Z = "--flipz" in sys.argv
# print ("====\nTarget: %s\nFeatures: %s\n%s%sFlip X: %s\nFlip Y: %s\nFlip Z: %s\n====\n" % (
#     "AWS" if RUN_AWS else "Local",
#     "All" if ALL_FEAT else "Intensity",
#     "Loading from file\n" if LOAD_NET else "",
#     "Saving to file\n" if SAVE_NET else "",
#     str(FLIP_X),
#     str(FLIP_Y),
#     str(FLIP_Z),
# ))


N_FOLDS = 2 # Train on 1/2, Test on 1/2
N_REPEATS = 5 if classifier.RUN_AWS else 1 # K-fold this many times
RANDOM_SEED = 194981
N_CHANNELS = 4 # Intensity, EM, JV, PC

#N_FILT = [64, 128, 128]
N_FILT = [64, 32, 32]
#N_FILT = [32, 16, 16]
#N_FILT = [32, 32, 32]

ERROR_WEIGHT = -3 # Positive = FN down, Sensitivity up. Negative = FP down, Specificity up
ERROR_WEIGHT_FRAC = 2 ** ERROR_WEIGHT

# SET IN MAIN:
#SIZE = 0
#N_EPOCHS = 0
#BATCH_SIZE = 0
#RUN_LOCAL = False

LEARNING_RATE = 0.0003 # 0.03
DROPOUT_RATE = 0.65

def buildNetwork7(dropoutRate=DROPOUT_RATE, learningRate=LEARNING_RATE, seed=RANDOM_SEED):
    nChannels = N_CHANNELS if classifier.ALL_FEAT else 1
    nFilt = N_FILT
    xInput = tf.placeholder(tf.float32, shape=[None, classifier.SIZE, classifier.SIZE, classifier.SIZE, nChannels])
    yInput = tf.placeholder(tf.float32, shape=[None, 2])

    with tf.name_scope("layer_a"):
        conv1 = tf.layers.conv3d(inputs=xInput, filters=nFilt[0], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) # 7x7x7
        conv2 = tf.layers.conv3d(inputs=conv1, filters=nFilt[1], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) # 7x7x7
        pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3,3,3], strides=2) # 3x3x3

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
        cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            targets=yInput, logits=prediction, pos_weight=ERROR_WEIGHT_FRAC
        ))
    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
        # optimizer = tf.train.AdagradOptimizer(learningRate).minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(yInput, 1))
    numCorrect = tf.reduce_sum(tf.cast(correct, tf.int32))
    return xInput, yInput, optimizer, cost, numCorrect, predictedProbs


def buildNetwork9(dropoutRate=DROPOUT_RATE, learningRate=LEARNING_RATE, seed=RANDOM_SEED):
    nChannels = N_CHANNELS if classifier.ALL_FEAT else 1
    nFilt = [64, 64, 64, 32, 32]
    xInput = tf.placeholder(tf.float32, shape=[None, classifier.SIZE, classifier.SIZE, classifier.SIZE, nChannels])
    yInput = tf.placeholder(tf.float32, shape=[None, 2])

    with tf.name_scope("layer_a"):
        conv1 = tf.layers.conv3d(inputs=xInput, filters=nFilt[0], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) # 9x9x9
        conv2 = tf.layers.conv3d(inputs=conv1, filters=nFilt[1], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) # 9x9x9
        pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2,2,2], strides=2, padding='same') # 5x5x5

    with tf.name_scope("layer_b"):
        conv4 = tf.layers.conv3d(inputs=pool3, filters=nFilt[2], kernel_size=[3,3,3], padding='same', activation=tf.nn.relu) #5x5x5
        conv5 = tf.layers.conv3d(inputs=conv4, filters=nFilt[3], kernel_size=[3,3,3], padding='same', activation=tf.nn.relu) #5x5x5
        pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2,2,2], strides=2, padding='same') #3x3x3

    with tf.name_scope("batch_norm"):
        cnn3d_bn = tf.layers.batch_normalization(inputs=pool6, training=True)
    with tf.name_scope("fully_con"):
        flattening = tf.reshape(cnn3d_bn, [-1, 3*3*3*nFilt[3]])
        dense = tf.layers.dense(inputs=flattening, units=nFilt[4], activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=dropoutRate, training=True)
    with tf.name_scope("y_conv"):
        prediction = tf.layers.dense(inputs=dropout, units=2)
        predictedProbs = tf.nn.softmax(prediction)
    with tf.name_scope("cross_entropy"):
        cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            targets=yInput, logits=prediction, pos_weight=ERROR_WEIGHT_FRAC
        ))
    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
        # optimizer = tf.train.AdagradOptimizer(learningRate).minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(yInput, 1))
    numCorrect = tf.reduce_sum(tf.cast(correct, tf.int32))
    return xInput, yInput, optimizer, cost, numCorrect, predictedProbs    


# As an example, run CNN on these given labels and test data, return the score.
def runOne(trainX, trainY, testX, testY, scanID):
    epochs = N_EPOCHS
    batchSize = BATCH_SIZE

    runTest = testY is not None
    runVolume = testY is None
    testProbs = None

    buildFunc = buildNetwork9 if classifier.SIZE == 9 else buildNetwork7
    xInput, yInput, optimizer, cost, numCorrect, scores = buildFunc()

    costs, corrs = [], []
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        start_time = datetime.datetime.now()

        iterations = int(len(trainY)/batchSize) + 1
        # run epochs
        for epoch in range(epochs):
            start_time_epoch = datetime.datetime.now()
            print('Scan %s, Epoch %d started' % (scanID, epoch))
            trainX, trainY = util.randomShuffle(trainX, trainY)

            # mini batch for trianing set:
            totalCost, totalCorr = 0.0, 0
            for itr in range(iterations):
                batchX = trainX[itr*batchSize: (itr+1)*batchSize]
                batchY = trainY[itr*batchSize: (itr+1)*batchSize]
                _optimizer, _cost, _corr = sess.run([optimizer, cost, numCorrect], feed_dict={
                    xInput: batchX, 
                    yInput: util.oneshotY(batchY)
                })
                totalCost += _cost
                totalCorr += _corr

            print (">> Epoch %d had TRAIN loss: %f\t#Correct = %5d/%5d = %f" % (
                epoch, totalCost, totalCorr, len(trainY), totalCorr / len(trainY)
            ))

            # Run against test set:
            if runTest:
                # print ("\n=======\nPart #2: Running against test voxels.")
                testX, testY = util.randomShuffle(testX, testY)
                totalCost, totalCorr = 0.0, 0
                itrs = int(len(testY)/batchSize) + 1
                for itr in range(itrs):
                    batchX = testX[itr*batchSize: (itr+1)*batchSize]
                    batchY = testY[itr*batchSize: (itr+1)*batchSize]
                    _cost, _corr = sess.run([cost, numCorrect], feed_dict={
                        xInput: batchX, 
                        yInput: util.oneshotY(batchY)
                    })
                    totalCost += _cost
                    totalCorr += _corr
                end_time_epoch = datetime.datetime.now()
                print('>> Epoch %d had  TEST loss: %f\t#Correct = %5d/%5d = %f\tTime elapsed: %s' % (
                    epoch, totalCost, totalCorr, len(testY), totalCorr / len(testY), str(end_time_epoch - start_time_epoch)
                ))
                costs.append(totalCost)
                corrs.append(totalCorr/len(testY))

        # Generate entire volume:
        if runVolume:
            print ("\n=======\nPart #2: Loading and generating all predictions")
            pad = classifier.PAD
            startX, endX = pad, testX.shape[0] - pad
            startY, endY = pad, testX.shape[0] - pad

            #rewrite variable names
            predictedProbs = scores
            allPreds = []
            for x in tqdm(range(startX, endX), ascii=True):
                for y in tqdm(range(startY, endY), ascii=True):
                    dataAsInput = files.convertVolumeStack(testX, pad, x, y)
                    preds = sess.run(scores, feed_dict={xInput: dataAsInput})
                    allPreds.extend(preds[:, 1].tolist())
            allPreds = np.array(allPreds)
            print ("\n# predictions: " + str(allPreds.shape))

            volumeResult = np.zeros(testX.shape[0:3])
            volumeResult = files.fillPredictions(volumeResult, allPreds, pad)
            resultPath = "data/%s/Normal%s-MRA-CNN.mat" % (scanID, scanID)
            print ("Writing to %s" % (resultPath))
            files.writePrediction(resultPath, "cnn", volumeResult)

        end_time = datetime.datetime.now()
        print('Time elapse: ', str(end_time - start_time))

        # Run against test:
        if runTest:
            testProbs = []
            itrs = int(len(testY)/batchSize) + 1
            for itr in range(itrs):
                batchX = testX[itr*batchSize: (itr+1)*batchSize]
                batchY = testY[itr*batchSize: (itr+1)*batchSize]
                _scores = sess.run(scores, feed_dict={
                    xInput: batchX, 
                    yInput: util.oneshotY(batchY)
                })
                testProbs.extend(np.array(_scores)[:, 1].tolist())

    if testProbs is not None:
        return costs, corrs, util.genScores(testY, testProbs)
    else:
        return costs, corrs, None


# TODO: Migrate to classifier
"""
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
            data, labels = util.randomShuffle(data, labels)
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
"""

if __name__ == '__main__':
    global SIZE, N_EPOCHS, BATCH_SIZE, RUN_LOCAL
    N_EPOCHS = 30 if classifier.RUN_AWS else 2
    BATCH_SIZE = 10 * (2 if classifier.FLIP_X else 1) * (2 if classifier.FLIP_Y else 1) * (2 if classifier.FLIP_Z else 1)

    # classifier.singleBrain('002', runOne, calcScore=True, writeVolume=False)
    classifier.brainsToBrain(['002', '019', '023'], '022', runOne, calcScore=True, writeVolume=True)