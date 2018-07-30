import datetime
import gc
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import tensorflow as tf
from tqdm import tqdm


import classifier
import files
import util
import viz


N_FOLDS = 2 # Train on 1/2, Test on 1/2
N_REPEATS = 5 if classifier.RUN_AWS else 1 # K-fold this many times
RANDOM_SEED = 194981


ERROR_WEIGHT = -2 # Positive = FN down, Sensitivity up. Negative = FP down, Specificity up
ERROR_WEIGHT_FRAC = 2 ** ERROR_WEIGHT

# SET IN MAIN:
#SIZE = 0
#N_EPOCHS = 0
#BATCH_SIZE = 0
#RUN_LOCAL = False

LEARNING_RATE = 0.0003 # 0.001 # 0.03
DROPOUT_RATE = 0.65 #.5
BASE_BATCH = 5
N_EPOCHS = 9 if classifier.RUN_AWS else 2

BATCH_SIZE = BASE_BATCH * (2 if classifier.FLIP_X else 1) * (2 if classifier.FLIP_Y else 1) * (2 if classifier.FLIP_Z else 1)

def buildNetwork7(dropoutRate=DROPOUT_RATE, learningRate=LEARNING_RATE, seed=RANDOM_SEED):
    # nFilt = [64, 128, 128]
    nFilt = [64, 64, 64]
    # nFilt = [32, 16, 16]
    # nFilt = [32, 32, 32]
    xInput = tf.placeholder(tf.float32, shape=[None, classifier.SIZE, classifier.SIZE, classifier.SIZE, classifier.N_FEAT])
    yInput = tf.placeholder(tf.float32, shape=[None, 2])
    isTraining = tf.placeholder(tf.bool)

    with tf.name_scope("layer_a"):
        conv1 = tf.layers.conv3d(inputs=xInput, filters=nFilt[0], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) # 7x7x7
        conv2 = tf.layers.conv3d(inputs=conv1, filters=nFilt[1], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) # 7x7x7
        pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3,3,3], strides=2) # 3x3x3
    with tf.name_scope("batch_norm"):
        cnn3d_bn = tf.layers.batch_normalization(inputs=pool3, training=isTraining)
    with tf.name_scope("fully_con"):
        flattening = tf.reshape(cnn3d_bn, [-1, 3*3*3*nFilt[1]])
        dense = tf.layers.dense(inputs=flattening, units=nFilt[2], activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=dropoutRate, training=isTraining)
    with tf.name_scope("y_conv"):
        prediction = tf.layers.dense(inputs=dropout, units=2)
        predictedProbs = tf.nn.softmax(prediction)
    with tf.name_scope("cross_entropy"):
        cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            targets=yInput, logits=prediction, pos_weight=ERROR_WEIGHT_FRAC
        ))
    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learningRate)
        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            trainOp = optimizer.minimize(cost)
        # optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
        # optimizer = tf.train.AdagradOptimizer(learningRate).minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(yInput, 1))
    numCorrect = tf.reduce_sum(tf.cast(correct, tf.int32))
    return xInput, yInput, isTraining, trainOp, cost, numCorrect, predictedProbs


def buildNetwork9(dropoutRate=DROPOUT_RATE, learningRate=LEARNING_RATE, seed=RANDOM_SEED):
    nFilt = [64, 64, 64, 32, 32]
    xInput = tf.placeholder(tf.float32, shape=[None, classifier.SIZE, classifier.SIZE, classifier.SIZE, classifier.N_FEAT])
    yInput = tf.placeholder(tf.float32, shape=[None, 2])
    isTraining = tf.placeholder(tf.bool)

    with tf.name_scope("layer_a"):
        conv1 = tf.layers.conv3d(inputs=xInput, filters=nFilt[0], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) # 9x9x9
        conv2 = tf.layers.conv3d(inputs=conv1, filters=nFilt[1], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) # 9x9x9
        pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2,2,2], strides=2, padding='same') # 5x5x5

    with tf.name_scope("layer_b"):
        conv4 = tf.layers.conv3d(inputs=pool3, filters=nFilt[2], kernel_size=[3,3,3], padding='same', activation=tf.nn.relu) #5x5x5
        conv5 = tf.layers.conv3d(inputs=conv4, filters=nFilt[3], kernel_size=[3,3,3], padding='same', activation=tf.nn.relu) #5x5x5
        pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2,2,2], strides=2, padding='same') #3x3x3

    with tf.name_scope("batch_norm"):
        cnn3d_bn = tf.layers.batch_normalization(inputs=pool6, training=isTraining)
    with tf.name_scope("fully_con"):
        flattening = tf.reshape(cnn3d_bn, [-1, 3*3*3*nFilt[3]])
        dense = tf.layers.dense(inputs=flattening, units=nFilt[4], activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=dropoutRate, training=isTraining)
    with tf.name_scope("y_conv"):
        prediction = tf.layers.dense(inputs=dropout, units=2)
        predictedProbs = tf.nn.softmax(prediction)
    with tf.name_scope("cross_entropy"):
        cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            targets=yInput, logits=prediction, pos_weight=ERROR_WEIGHT_FRAC
        ))
    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learningRate)
        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            trainOp = optimizer.minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(yInput, 1))
    numCorrect = tf.reduce_sum(tf.cast(correct, tf.int32))
    return xInput, yInput, isTraining, trainOp, cost, numCorrect, predictedProbs

def buildNetwork11(dropoutRate=DROPOUT_RATE, learningRate=LEARNING_RATE, seed=RANDOM_SEED):
    nFilt = [64, 64, 64, 32, 32]
    xInput = tf.placeholder(tf.float32, shape=[None, classifier.SIZE, classifier.SIZE, classifier.SIZE, classifier.N_FEAT])
    yInput = tf.placeholder(tf.float32, shape=[None, 2])
    isTraining = tf.placeholder(tf.bool)

    with tf.name_scope("layer_a"):
        conv1 = tf.layers.conv3d(inputs=xInput, filters=nFilt[0], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) # 11x11x11
        conv2 = tf.layers.conv3d(inputs=conv1, filters=nFilt[1], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) # 11x11x11
        pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2,2,2], strides=2, padding='same') # 6x6x6

    with tf.name_scope("layer_b"):
        conv4 = tf.layers.conv3d(inputs=pool3, filters=nFilt[2], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) # 6x6x6
        conv5 = tf.layers.conv3d(inputs=conv4, filters=nFilt[3], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) # 6x6x6
        pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2,2,2], strides=2, padding='same') # 3x3x3

    with tf.name_scope("batch_norm"):
        cnn3d_bn = tf.layers.batch_normalization(inputs=pool6, training=isTraining)
    with tf.name_scope("fully_con"):
        flattening = tf.reshape(cnn3d_bn, [-1, 3*3*3*nFilt[3]])
        dense = tf.layers.dense(inputs=flattening, units=nFilt[4], activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=dropoutRate, training=isTraining)
    with tf.name_scope("y_conv"):
        prediction = tf.layers.dense(inputs=dropout, units=2)
        predictedProbs = tf.nn.softmax(prediction)
    with tf.name_scope("cross_entropy"):
        cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            targets=yInput, logits=prediction, pos_weight=ERROR_WEIGHT_FRAC
        ))
    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learningRate)
        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            trainOp = optimizer.minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(yInput, 1))
    numCorrect = tf.reduce_sum(tf.cast(correct, tf.int32))
    return xInput, yInput, isTraining, trainOp, cost, numCorrect, predictedProbs


def predict(sess, scores, xInput, isTraining, data):
    if classifier.PREDICT_TRANSFORM:
        data = util.allRotations(data)
        preds = sess.run(scores, feed_dict={xInput: data, isTraining: False})
        preds = preds[:, 1].reshape((-1, 8))
        return util.combinePredictions(preds)
    else:
        preds = sess.run(scores, feed_dict={xInput: data, isTraining: False})
        return preds[:, 1].tolist()


# As an example, run CNN on these given labels and test data, return the score.
def runOne(trainX, trainY, testX, testY, scanID, savePath):
    epochs = N_EPOCHS
    batchSize = BATCH_SIZE

    runTest = testY is not None
    runVolume = testY is None
    testProbs = None

    tf.reset_default_graph()
    buildFunc = None
    if classifier.SIZE == 7:
        buildFunc = buildNetwork7
    elif classifier.SIZE == 9:
        buildFunc = buildNetwork9
    elif classifier.SIZE == 11:
        buildFunc = buildNetwork11
    else:
        print ("No network for size %d" % (classifier.SIZE))
        raise 0
    xInput, yInput, isTraining, trainOp, cost, numCorrect, scores = buildFunc()

    initOp = tf.global_variables_initializer()
    saver = None if savePath is None else tf.train.Saver()

    costs, corrs = [], []
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        start_time = datetime.datetime.now()
        sess.run(tf.global_variables_initializer())

        iterations = int(len(trainY)/batchSize) + 1
        # run epochs
        for epoch in range(epochs):
            start_time_epoch = datetime.datetime.now()
            print('Scan %s, Epoch %d started' % (scanID, epoch))
            gc.collect()
            # trainX, trainY = util.randomShuffle(trainX, trainY)
            order = np.random.permutation(trainX.shape[0])

            # mini batch for trianing set:
            totalCost, totalCorr = 0.0, 0
            for itr in tqdm(range(iterations)):
                batchX = trainX[order[itr*batchSize: (itr+1)*batchSize]]
                batchY = trainY[order[itr*batchSize: (itr+1)*batchSize]]
                # batchX = trainX[itr*batchSize: (itr+1)*batchSize]
                # batchY = trainY[itr*batchSize: (itr+1)*batchSize]
                _trainOp, _cost, _corr = sess.run([trainOp, cost, numCorrect], feed_dict={
                    xInput: batchX,
                    yInput: util.oneshotY(batchY),
                    isTraining: True
                })
                totalCost += _cost
                totalCorr += _corr

            print (">> Epoch %d had TRAIN loss: %.2f\t#Correct = %5d/%5d = %f" % (
                epoch, totalCost, totalCorr, len(trainY), totalCorr / len(trainY)
            ))

            # Run against test set:
            if runTest:
                # print ("\n=======\nPart #2: Running against test voxels.")
                # testX, testY = util.randomShuffle(testX, testY)
                order = np.random.permutation(testX.shape[0])
                totalCorr = 0
                itrs = int(math.ceil(len(testY)/batchSize))
                for itr in range(itrs):
                    batchX = testX[order[itr*batchSize: (itr+1)*batchSize]]
                    batchY = testY[order[itr*batchSize: (itr+1)*batchSize]]
                    # batchX = testX[itr*batchSize: (itr+1)*batchSize]
                    # batchY = testY[itr*batchSize: (itr+1)*batchSize]
                    predictions = predict(sess, scores, xInput, isTraining, batchX)
                    totalCorr += np.sum((np.array(predictions) > 0.5) == (np.array(batchY) > 0.5))
                end_time_epoch = datetime.datetime.now()
                print('>> Epoch %d had  TEST loss:      \t#Correct = %5d/%5d = %f\tTime elapsed: %s' % (
                    epoch, totalCorr, len(testY), totalCorr / len(testY), str(end_time_epoch - start_time_epoch)
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
            allPreds = []
            for x in tqdm(range(startX, endX), ascii=True):
                for y in tqdm(range(startY, endY), ascii=True):
                    dataAsInput = files.convertVolumeStack(testX, pad, x, y)
                    preds = predict(sess, scores, xInput, isTraining, dataAsInput)
                    allPreds.extend(preds)
            allPreds = np.array(allPreds)
            print ("\n# predictions: " + str(allPreds.shape))

            volumeResult = np.zeros(testX.shape[0:3])
            volumeResult = files.fillPredictions(volumeResult, allPreds, pad)
            # resultPath = "data/%s/Normal%s-MRA-CNN-trans.mat" % (scanID, scanID)
            resultPath = "data/tmp/%s-weighted.mat" % scanID
            print ("Writing to %s" % (resultPath))
            files.writePrediction(resultPath, "cnn", volumeResult)

        end_time = datetime.datetime.now()
        print('Time elapse: ', str(end_time - start_time))

        # Run against test:
        if runTest:
            testProbs = []
            itrs = int(math.ceil(len(testY)/batchSize))
            for itr in range(itrs):
                batchX = testX[itr*batchSize: (itr+1)*batchSize]
                batchY = testY[itr*batchSize: (itr+1)*batchSize]
                _scores = sess.run(scores, feed_dict={
                    xInput: batchX,
                    yInput: util.oneshotY(batchY), 
                    isTraining: False
                })
                testProbs.extend(np.array(_scores)[:, 1].tolist())

        # Save the network:
        if savePath is not None:
            savePath = saver.save(sess, savePath)
            print ("Model saved to %s" % (savePath))

    if testProbs is not None:
        return costs, corrs, util.genScores(testY, testProbs)
    else:
        return costs, corrs, None


# TODO: Migrate to classifier
def volumeFromSavedNet(path, scanID):
    tf.reset_default_graph()

    print ("Using network %s to generate volume %s" % (path, scanID))
    volume, _, _ = files.loadAllInputsUpdated(scanID, classifier.ALL_FEAT)

    buildFunc = buildNetwork9 if classifier.SIZE == 9 else buildNetwork7
    xInput, yInput, isTraining, trainOp, cost, numCorrect, scores = buildFunc()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print ("Loading net from file...")
        start_time = datetime.datetime.now()
        saver.restore(sess, path)

        # Generate entire volume:
        print ("\nGenerating all predictions for volume %s" % (scanID))
        pad = classifier.PAD
        startX, endX = pad, volume.shape[0] - pad
        startY, endY = pad, volume.shape[0] - pad

        #rewrite variable names
        allPreds = []
        for x in tqdm(range(startX, endX), ascii=True):
            for y in tqdm(range(startY, endY), ascii=True):
                dataAsInput = files.convertVolumeStack(volume, pad, x, y)
                preds = predict(sess, scores, xInput, isTraining, dataAsInput)
                allPreds.extend(preds)
        allPreds = np.array(allPreds)
        print ("\n# predictions: " + str(allPreds.shape))

        volumeResult = np.zeros(volume.shape[0:3])
        volumeResult = files.fillPredictions(volumeResult, allPreds, pad)
        resultPath = "data/multiV/04_25/Normal%s-MRA-CNN.mat" % (scanID)
        print ("Writing to %s" % (resultPath))
        files.writePrediction(resultPath, "cnn", volumeResult)

    end_time = datetime.datetime.now()
    print('Time elapse: ', str(end_time - start_time))



if __name__ == '__main__':
    savePath = None
    # classifier.singleBrain('002', runOne, calcScore=True, writeVolume=True, savePath=savePath)
    #classifier.brainsToBrain(['002', '019', '022'], '023', runOne, calcScore=True, writeVolume=False, savePath=savePath)
    # classifier.brainsToBrain(['002', '019', '022', '023', '034', '058', '066', '082'], '056', runOne, calcScore=True, writeVolume=False, savePath=savePath)

    classifier.brainsToBrain(['002', '019', '022', '023', '034', '058', '066', '082'], '084', runOne, calcScore=True, writeVolume=False, savePath=savePath)

    # volumeFromSavedNet(savePath, '002')
    # volumeFromSavedNet(savePath, '019')
    # volumeFromSavedNet(savePath, '022')
    # volumeFromSavedNet(savePath, '023')
