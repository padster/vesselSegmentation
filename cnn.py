import datetime
import gc
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import sys
import tensorflow as tf
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


import classifier
import files
import util
import viz

tf.set_random_seed(0)


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
N_EPOCHS = 5

BATCH_SIZE = BASE_BATCH * (2 if classifier.FLIP_X else 1) * (2 if classifier.FLIP_Y else 1) * (2 if classifier.FLIP_Z else 1)

LOG_INTERVAL = 30

def kInit(seed):
    return tf.glorot_normal_initializer(seed=seed)

def buildNetwork7(dropoutRate=DROPOUT_RATE, learningRate=LEARNING_RATE, seed=RANDOM_SEED):
    nFilt = [64, 64, 64]
    # nFilt = [64, 128, 128]
    # nFilt = [32, 16, 16]
    # nFilt = [32, 32, 32]
    xInput = tf.placeholder(tf.float32, shape=[None, classifier.SIZE, classifier.SIZE, classifier.SIZE, classifier.N_FEAT])
    yInput = tf.placeholder(tf.float32, shape=[None, 2])
    isTraining = tf.placeholder(tf.bool)

    with tf.name_scope("layer_a"):
        conv1 = tf.layers.conv3d(inputs=xInput, filters=nFilt[0], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu, kernel_initializer=kInit(1)) # 7x7x7
        conv2 = tf.layers.conv3d(inputs=conv1, filters=nFilt[1], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu, kernel_initializer=kInit(2)) # 7x7x7
        pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3,3,3], strides=2) # 3x3x3
    with tf.name_scope("batch_norm"):
        cnn3d_bn = tf.layers.batch_normalization(inputs=pool3, training=isTraining)
    with tf.name_scope("fully_con"):
        flattening = tf.reshape(cnn3d_bn, [-1, 3*3*3*nFilt[1]])
        dense = tf.layers.dense(inputs=flattening, units=nFilt[2], activation=tf.nn.relu, kernel_initializer=kInit(5))
        dropout = tf.layers.dropout(inputs=dense, rate=dropoutRate, training=isTraining, seed=123)
    with tf.name_scope("y_conv"):
        prediction = tf.layers.dense(inputs=dropout, units=2, kernel_initializer=kInit(6))
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
        conv1 = tf.layers.conv3d(inputs=xInput, filters=nFilt[0], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu, kernel_initializer=kInit(1)) # 9x9x9
        conv2 = tf.layers.conv3d(inputs=conv1, filters=nFilt[1], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu, kernel_initializer=kInit(2)) # 9x9x9
        pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2,2,2], strides=2, padding='same') # 5x5x5

    with tf.name_scope("layer_b"):
        conv4 = tf.layers.conv3d(inputs=pool3, filters=nFilt[2], kernel_size=[3,3,3], padding='same', activation=tf.nn.relu, kernel_initializer=kInit(3)) #5x5x5
        conv5 = tf.layers.conv3d(inputs=conv4, filters=nFilt[3], kernel_size=[3,3,3], padding='same', activation=tf.nn.relu, kernel_initializer=kInit(4)) #5x5x5
        pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2,2,2], strides=2, padding='same') #3x3x3

    with tf.name_scope("batch_norm"):
        cnn3d_bn = tf.layers.batch_normalization(inputs=pool6, training=isTraining)
    with tf.name_scope("fully_con"):
        flattening = tf.reshape(cnn3d_bn, [-1, 3*3*3*nFilt[3]])
        dense = tf.layers.dense(inputs=flattening, units=nFilt[4], activation=tf.nn.relu, kernel_initializer=kInit(5))
        dropout = tf.layers.dropout(inputs=dense, rate=dropoutRate, training=isTraining, seed=123)
    with tf.name_scope("y_conv"):
        prediction = tf.layers.dense(inputs=dropout, units=2, kernel_initializer=kInit(6))
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
        conv1 = tf.layers.conv3d(inputs=xInput, filters=nFilt[0], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu, kernel_initializer=kInit(1)) # 11x11x11
        conv2 = tf.layers.conv3d(inputs=conv1, filters=nFilt[1], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu, kernel_initializer=kInit(2)) # 11x11x11
        pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2,2,2], strides=2, padding='same') # 6x6x6

    with tf.name_scope("layer_b"):
        conv4 = tf.layers.conv3d(inputs=pool3, filters=nFilt[2], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu, kernel_initializer=kInit(3)) # 6x6x6
        conv5 = tf.layers.conv3d(inputs=conv4, filters=nFilt[3], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu, kernel_initializer=kInit(4)) # 6x6x6
        pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2,2,2], strides=2, padding='same') # 3x3x3

    with tf.name_scope("batch_norm"):
        cnn3d_bn = tf.layers.batch_normalization(inputs=pool6, training=isTraining)
    with tf.name_scope("fully_con"):
        flattening = tf.reshape(cnn3d_bn, [-1, 3*3*3*nFilt[3]])
        dense = tf.layers.dense(inputs=flattening, units=nFilt[4], activation=tf.nn.relu, kernel_initializer=kInit(5))
        dropout = tf.layers.dropout(inputs=dense, rate=dropoutRate, training=isTraining, seed=123)
    with tf.name_scope("y_conv"):
        prediction = tf.layers.dense(inputs=dropout, units=2, kernel_initializer=kInit(6))
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

customNetworkFunc = None
def overrideNetwork(networkFunc):
    global customNetworkFunc
    customNetworkFunc = networkFunc


def _getNetworkFunc():
    if customNetworkFunc is not None:
        return customNetworkFunc()

    if classifier.SIZE == 7:
        return buildNetwork7
    elif classifier.SIZE == 9:
        return buildNetwork9
    elif classifier.SIZE == 11:
        return buildNetwork11
    else:
        print ("No network for size %d" % (classifier.SIZE))
        raise 0


def predict(sess, scores, xInput, isTraining, data):
    if classifier.PREDICT_TRANSFORM:
        nTransforms = 16  # model.N_TRANSFORMS
        data = util.allRotations(data)
        preds = sess.run(scores, feed_dict={
            xInput: util.subVolumesToTensor(data),
            isTraining: False
        })
        preds = preds[:, 1].reshape((-1, nTransforms))
        return util.combinePredictions(preds)
    else:
        preds = sess.run(scores, feed_dict={
            xInput: util.subVolumesToTensor(data),
            isTraining: False
        })
        return preds[:, 1].tolist()

PROGRESS_WRITER = SummaryWriter()

# As an example, run CNN on these given labels and test data, return the score.
def runOne(trainX, trainY, testX, testY, scanID, savePath):
    # HACK
    trainX = np.array(trainX)
    testX = np.array(testX)

    epochs = N_EPOCHS
    batchSize = BATCH_SIZE

    runTest = testY is not None
    runVolume = testY is None
    testProbs = None

    tf.reset_default_graph()

    xInput, yInput, isTraining, trainOp, cost, numCorrect, scores = (_getNetworkFunc())()

    saver = None if savePath is None else tf.train.Saver()
    iterCount = 0

    costs, corrs = [], []
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        start_time = datetime.datetime.now()
        sess.run(tf.global_variables_initializer())

        iterations = int(math.ceil(len(trainY)/batchSize))
        # run epochs
        for epoch in range(epochs):
            start_time_epoch = datetime.datetime.now()
            print('Scan %s, Epoch %d started' % (scanID, epoch))
            gc.collect()
            order = np.random.permutation(len(trainX))

            # mini batch for trianing set:
            totalCost, totalCorr = 0.0, 0
            for itr in tqdm(range(iterations)):
                batchX = trainX[order[itr*batchSize: (itr+1)*batchSize]]
                batchY = trainY[order[itr*batchSize: (itr+1)*batchSize]]
                _trainOp, _cost, _corr = sess.run([trainOp, cost, numCorrect], feed_dict={
                    xInput: util.subVolumesToTensor(batchX),
                    yInput: util.oneshotY(batchY),
                    isTraining: True
                })
                totalCost += _cost
                totalCorr += _corr
                iterCount += 1
                if ((iterCount + 1) % LOG_INTERVAL) == 0:
                    PROGRESS_WRITER.add_scalar('loss', _cost, iterCount)

            print (">> Epoch %d had TRAIN loss: %.2f\t#Correct = %5d/%5d = %f" % (
                epoch, totalCost, totalCorr, len(trainY), totalCorr / len(trainY)
            ))

            # Run against test set:
            if runTest:
                # print ("\n=======\nPart #2: Running against test voxels.")
                order = np.random.permutation(testX.shape[0])
                totalCorr = 0
                itrs = int(math.ceil(len(testY)/batchSize))
                for itr in range(itrs):
                    batchX = testX[order[itr*batchSize: (itr+1)*batchSize]]
                    batchY = testY[order[itr*batchSize: (itr+1)*batchSize]]
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
            for x in tqdm(range(testX.shape[0]), ascii=True):
                for y in tqdm(range(testX.shape[1]), ascii=True):
                    dataAsInput = files.convertVolumeStack(testX, 'testX', pad, x, y)
                    preds = predict(sess, scores, xInput, isTraining, dataAsInput)
                    allPreds.extend(preds)
            allPreds = np.array(allPreds)
            print ("\n# predictions: " + str(allPreds.shape))

            #volumeResult = np.zeros(testX.shape[0:3])
            volumeResult = allPreds
            #volumeResult = files.fillPredictions(volumeResult, allPreds, pad)
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
                    xInput: util.subVolumesToTensor(batchX),
                    yInput: util.oneshotY(batchY),
                    isTraining: False
                })
                testProbs.extend(np.array(_scores)[:, 1].tolist())

        # Save the network:
        if savePath is not None:
            savePath = saver.save(sess, savePath)
            print ("Model saved to %s" % (savePath))

    if testProbs is not None:
        return costs, corrs, util.genScores(testY, testProbs), testProbs
    else:
        return costs, corrs, volumeResult, None

# Run the classifier over a whole volume, generate a volume of results.
def volumeFromSavedNet(netPath, scanID, resultPath, xFr=None, xTo=None, useMask=True):
    pad = classifier.PAD
    # resultPath = "data/multiV/04_25/Normal%s-MRA-CNN.mat" % (scanID)
    tf.reset_default_graph()

    print ("Using network %s to generate volume %s" % (netPath, scanID))
    volume = files.loadAllInputsUpdated(scanID, pad, classifier.ALL_FEAT, classifier.MORE_FEAT, oneFeat=classifier.ONE_FEAT_NAME, noTrain=True)
    prediction = np.zeros(volume.shape[0:3])

    brainMask = np.ones(volume.shape[0:3])
    if useMask:
        brainMask = files.loadBM(scanID, maskPad=classifier.SIZE)

    xInput, yInput, isTraining, trainOp, cost, numCorrect, scores = (_getNetworkFunc())()
    saver = tf.train.Saver()

    # Change these if we need only small subsegments
    #XMIN, XMAX = 0, volume.shape[0]
    if xFr is None:
        xFr = 0
    if xTo is None:
        xTo = volume.shape[0]

    with tf.Session() as sess:
        print ("Loading net from file...")
        start_time = datetime.datetime.now()
        saver.restore(sess, netPath)

        # Generate entire volume, one column of x/y at a time:
        print ("\nGenerating all predictions for volume %s" % (scanID))

        nBrain = 0
        for x in tqdm(range(xFr, xTo)):
            for y in range(volume.shape[1]):
                zFr, zTo = util.maskBounds(brainMask[x, y, :])
                nBrain += zTo - zFr

        with tqdm(total=nBrain, ascii=True) as progress:
            for x in range(xFr, xTo):
                for y in range(volume.shape[1]):
                    zFr, zTo = util.maskBounds(brainMask[x, y, :])
                    if zFr == -1:
                        continue # No brain in this column
                    dataAsInput = files.convertVolumeStack(scanID, pad, x, y, zFr, zTo)
                    preds = predict(sess, scores, xInput, isTraining, dataAsInput)
                    prediction[x, y, zFr:zTo] = preds
                    progress.update(zTo - zFr)

        print ("Writing to %s" % (resultPath))
        files.writePrediction(resultPath, "cnn", prediction)

    end_time = datetime.datetime.now()
    print('Time elapse: ', str(end_time - start_time))


def calcStats(netPath, toID):
    params, flops = -1, -1

    tf.reset_default_graph()

    xInput, yInput, isTraining, trainOp, cost, numCorrect, scores = (_getNetworkFunc())()
    saver = tf.train.Saver()

    #toX, toY = files.convertScanToXY(toID, 
    #    classifier.ALL_FEAT, classifier.MORE_FEAT, classifier.PAD, 
    #    False, False, False, False, 
    #    merge=True, oneFeat=classifier.ONE_FEAT_NAME, oneTransID=classifier.ONE_TRANS_ID)
    #batchX, batchY = toX[0:1], toY[0:1]

    with tf.Session() as sess:
        saver.restore(sess, netPath)
        flops = tf.profiler.profile(sess.graph, 
            options=tf.profiler.ProfileOptionBuilder.float_operation(), 
            run_meta = tf.RunMetadata(), cmd='op'
        )
        params = tf.profiler.profile(sess.graph, 
            options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        )

    return {
        'params': params.total_parameters,
        'flops': flops.total_float_ops
    }

if __name__ == '__main__':
    classifier.initOptions(sys.argv)

    savePath = None
    classifier.singleBrain('002', runOne, calcScore=True, writeVolume=False, savePath=savePath)
    #classifier.brainsToBrain(['002', '019', '022', '023', '034', '058', '066', '082'], '056', runOne, calcScore=True, writeVolume=False, savePath=savePath)

    # volumeFromSavedNet(savePath, '002')
