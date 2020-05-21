# Unused: try to use a Convolutional Autoencoder to do dimensionality
# reduction to help with network shape and feature generation.
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import RepeatedStratifiedKFold
import tensorflow as tf
from tqdm import tqdm
import sys

import classifier
import files
import util
import viz

TRAIN_CAE = "--train" in sys.argv
CLASSIFY = "--classify" in sys.argv

INNER_DIMEN = 100

ERROR_WEIGHT = 0 # Positive = FN down, Sensitivity up. Negative = FP down, Specificity up
ERROR_WEIGHT_FRAC = 2 ** ERROR_WEIGHT

RANDOM_SEED = 194981
LEARNING_RATE = 0.003 # 0.03
DENSE_LEARNING_RATE = 0.003

DROPOUT_RATE = 0.65 #.5
BASE_BATCH = 30
N_EPOCHS = 10 if classifier.RUN_AWS else 2

BATCH_SIZE = BASE_BATCH * (2 if classifier.FLIP_X else 1) * (2 if classifier.FLIP_Y else 1) * (2 if classifier.FLIP_Z else 1)


def show_tsne(ax, Xs, perplexity, learnRate, colors, nDim=2):
    print ("Fitting TSNE...")
    fitted = TSNE(
        n_components=nDim, perplexity=perplexity, learning_rate=learnRate
    ).fit_transform(Xs)
    if nDim == 2:
        ax.scatter(fitted[:, 0], fitted[:, 1], color=colors, marker='x')
    elif nDim == 3:
        ax.scatter(fitted[:, 0], fitted[:, 1], fitted[:, 2], color=colors, marker='x')

def simpleTSNE(Xs, Ys):
    DOWNSAMPLE = 3
    Xs, Ys = Xs[::DOWNSAMPLE], Ys[::DOWNSAMPLE]
    PERPLEXITY_OPTIONS = [10]
    LEARN_RATE_OPTIONS = [300]

    sz = classifier.SIZE
    print ("TSNE: %d samples" % Xs.shape[0])
    xPoints = np.copy(Xs)
    if len(Xs.shape) > 2:
        xPoints = Xs.reshape(Xs.shape[0], -1)

    colours = []
    for y in Ys:
        colours.append('#ff0000' if y < 0.5 else '#00ff00')

    f, ax = viz.clean_subplots(len(PERPLEXITY_OPTIONS), len(LEARN_RATE_OPTIONS)) #, projection='3d')
    for i in tqdm(range(len(PERPLEXITY_OPTIONS))):
        perplexity = PERPLEXITY_OPTIONS[i]
        ax[i][0].get_yaxis().set_visible(True)
        ax[i][0].set_ylabel("Perplexity = %d" % perplexity)
        for j in tqdm(range(len(LEARN_RATE_OPTIONS))):
            learnRate = LEARN_RATE_OPTIONS[j]
            if i == 0:
                # ax[0][j].set_title("Learn rate = %d" % learnRate)
                ax[0][j].set_title("Use CAE to reduce to 32 dimensions, TSNE to flatten this to 2:")
            show_tsne(ax[i][j], xPoints, perplexity, learnRate, colours, nDim=2)
    plt.show()


def buildCAENetwork9(dropoutRate=DROPOUT_RATE, learningRate=LEARNING_RATE, seed=RANDOM_SEED):
    print ("Building network...")

    # nFilt = [64, 64, 32, 32, INNER_DIMEN, 32, 32, 64, 64]
    nFilt = [5, 5, 10, 10, INNER_DIMEN, 10, 10, 5, 5]
    xInput = tf.placeholder(tf.float32, shape=[None, classifier.SIZE, classifier.SIZE, classifier.SIZE, classifier.N_FEAT])
    # yInput = tf.placeholder(tf.float32, shape=[None, 2])
    isTraining = tf.placeholder(tf.bool)

    # Bx9x9x9xN -> Bx5x5x5xF[1]
    with tf.name_scope("layer_a"):
        conv1 = tf.layers.conv3d(inputs=xInput, filters=nFilt[0], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) # 9x9x9
        # conv2 = tf.layers.conv3d(inputs=conv1, filters=nFilt[1], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) # 9x9x9
        pool3 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[2,2,2], strides=2, padding='same') # 5x5x5

    # Bx5x5x5xF[1] -> Bx3x3x3xF[3]
    with tf.name_scope("layer_b"):
        conv4 = tf.layers.conv3d(inputs=pool3, filters=nFilt[2], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) #5x5x5
        # conv5 = tf.layers.conv3d(inputs=conv4, filters=nFilt[3], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu) #5x5x5
        pool6 = tf.layers.max_pooling3d(inputs=conv4, pool_size=[2,2,2], strides=2, padding='same') #3x3x3

    # Bx3x3x3xF[3] -> Bx3x3x3xF[3]
    with tf.name_scope("batch_norm"):
        cnn3d_bn = tf.layers.batch_normalization(inputs=pool6, training=isTraining)

    # Bx3x3x3xF[3] -> BxM
    with tf.name_scope("fully_con"):
        flattening = tf.reshape(cnn3d_bn, [-1, 3*3*3*nFilt[3]])
        innerFeat = tf.layers.dense(inputs=flattening, units=nFilt[4], activation=tf.nn.selu)
        # dropout = tf.layers.dropout(inputs=dense, rate=dropoutRate, training=isTraining)

    # BxM -> Bx3x3x3xF[6]
    with tf.name_scope("layer_b_inv"):
        print ("IF: ", innerFeat.shape)
        unFlat = tf.reshape(innerFeat, [-1, 1, 1, 1, nFilt[4]])
        deconv5 = tf.layers.conv3d_transpose(inputs=unFlat,  filters=nFilt[5], kernel_size=[3,3,3], strides=3, padding='same', activation=tf.nn.selu)
        print ("D5: ", deconv5.shape)
        # deconv4 = tf.layers.conv3d_transpose(inputs=deconv5, filters=nFilt[6], kernel_size=[3,3,3], strides=1, padding='valid', activation=tf.nn.selu)
        # print ("D4: ", deconv4.shape)

    # Bx3x3x3xF[6] -> Bx6x6x6xF[8]
    with tf.name_scope("layer_a_inv"):
        # deconv3 = tf.layers.conv3d_transpose(inputs=deconv4, filters=nFilt[7], kernel_size=[3,3,3], padding='same', activation=tf.nn.selu)
        deconv2 = tf.layers.conv3d_transpose(inputs=deconv5, filters=nFilt[8], kernel_size=[2,2,2], strides=2, padding='valid', activation=tf.nn.selu)
        print ("D2: ", deconv2.shape)

    # Bx6x6x6xF[8] -> Bx9x9x9xN
    with tf.name_scope("final_inv"):
        # SIGMOID to make sure result intensities \in (0, 1)
        xOutputPre = tf.layers.conv3d_transpose(inputs=deconv2, filters=1, kernel_size=[4,4,4], strides=1, padding='valid', activation=tf.nn.sigmoid)
        xOutput = tf.squeeze(xOutputPre)
        # xOutput = deconv2
        print ("XO: ", xOutput.shape)

    # Loss and optimizer
    with tf.name_scope("training"):
        intensityInput = xInput[:, :, :, :, 0]
        cost = tf.reduce_mean(tf.square(xOutput - intensityInput))
        optimizer = tf.train.AdamOptimizer(learningRate)
        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            trainOp = optimizer.minimize(cost)

    return xInput, xOutput, innerFeat, isTraining, trainOp, cost


# Build network for the reduced-dimensionality version
def buildDenseNetwork(learningRate=DENSE_LEARNING_RATE):
    MID_DIMEN = INNER_DIMEN // 4

    xInputD = tf.placeholder(tf.float32, shape=[None, INNER_DIMEN])
    yInputD = tf.placeholder(tf.float32, shape=[None, 2])

    with tf.name_scope("hiddenD"):
        hiddenD = tf.layers.dense(inputs=xInputD, units=MID_DIMEN, activation=tf.nn.selu)

    with tf.name_scope("y_convD"):
        predictionD = tf.layers.dense(inputs=hiddenD, units=2)
        predictedProbsD = tf.nn.softmax(predictionD)

    with tf.name_scope("cross_entropyD"):
        costD = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            targets=yInputD, logits=predictionD, pos_weight=ERROR_WEIGHT_FRAC
        ))

    with tf.name_scope("denseTrainingD"):
        optimizerD = tf.train.AdamOptimizer(learningRate)
        updateOpsD = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOpsD):
            trainOpD = optimizerD.minimize(costD)

    with tf.name_scope("scoresD"):
        correctD = tf.equal(tf.argmax(predictionD, 1), tf.argmax(yInputD, 1))
        numCorrectD = tf.reduce_sum(tf.cast(correctD, tf.int32))
    return {
        'x': xInputD,
        'y': yInputD,
        't': trainOpD,
        'c': costD,
        'nc': numCorrectD,
        'p': predictedProbsD
    }


##
def densePredict(sess, xInput, innerFeat, isTraining, xInputD, yInputD, probs, cost, batchX, batchY):
    #if classifier.PREDICT_TRANSFORM:
    if False: # HACK - support transforms during prediction
        data = util.allRotations(data)
        preds = sess.run(probs, feed_dict={xInput: data, isTraining: False})
        preds = preds[:, 1].reshape((-1, 8))
        return util.combinePredictions(preds)
    else:
        print (batchY.shape)
        print (util.oneshotY(batchY).shape)
        _feat = sess.run(innerFeat, feed_dict={
            xInput: batchX,
            isTraining: False
        })
        _preds, _cost = sess.run([probs, cost], feed_dict={
            xInput: batchX, # HMM
            isTraining: False, # HMM
            xInputD: _feat,
            yInputD: util.oneshotY(batchY)
        })
        return _preds[:, 1].tolist(), _cost


# Given trained CAE, train a new classifier off train data, and check against test
def runDenseNetwork(trainID, testID, loadPath, batchSize=BATCH_SIZE):
    epochs = N_EPOCHS * 5
    tf.reset_default_graph()

    volTrain, lTrainA, lTrainB = files.loadAllInputsUpdated(trainID, classifier.ALL_FEAT, classifier.MORE_FEAT)
    volTest, lTestA, lTestB = files.loadAllInputsUpdated(testID, classifier.ALL_FEAT, classifier.MORE_FEAT)
    lTrain = np.concatenate([lTrainA, lTrainB], axis=0)
    lTest = np.concatenate([lTestA, lTestB], axis=0)

    trainX, trainY = files.convertToInputs(volTrain, lTrain, classifier.PAD, classifier.FLIP_X, classifier.FLIP_Y, classifier.FLIP_Z)
    testX,   testY = files.convertToInputs(volTest, lTest, classifier.PAD, False, False, False)

    print ("Dense network: %d train samples, %d test" % (len(trainX), len(testX)))

    print ("Building networks...")
    caeX, _, innerFeat, caeIsTraining, _, _ = buildCAENetwork9()
    net = buildDenseNetwork()
    xInput, yInput, trainOp, cost, numCorrect, predictedProbs = net['x'], net['y'], net['t'], net['c'], net['nc'], net['p']

    runTest = True

    iterations = int(len(trainY)/batchSize) + 1
    saver = tf.train.Saver()

    trainCosts, corrs = [], []
    testCosts = []
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        saver.restore(sess, loadPath)

        # run epochs
        for epoch in range(epochs):
            start_time_epoch = datetime.datetime.now()
            print('Training Scan %s, Epoch %d started' % (trainID, epoch))
            trainX, trainY = util.randomShuffle(trainX, trainY)

            # mini batch for trianing set:
            totalCost, totalCorr = 0.0, 0
            for itr in range(iterations):
                batchX = trainX[itr*batchSize: (itr+1)*batchSize]
                batchY = trainY[itr*batchSize: (itr+1)*batchSize]

                _feat = sess.run(innerFeat, feed_dict={
                    caeX: batchX,
                    caeIsTraining: False
                })

                _trainOp, _cost, _corr = sess.run([trainOp, cost, numCorrect], feed_dict={
                    caeX: batchX, # HMM
                    caeIsTraining: False, # HMM
                    xInput: _feat,
                    yInput: util.oneshotY(batchY)
                })
                totalCost += _cost
                totalCorr += _corr

            print (">> Epoch %d had TRAIN loss: %.2f\t#Correct = %5d/%5d = %f" % (
                epoch, totalCost, totalCorr, len(trainY), totalCorr / len(trainY)
            ))
            trainCosts.append(totalCost)
            corrs.append(totalCorr/len(testY))

            # Run against test set:
            if runTest:
                testX, testY = util.randomShuffle(testX, testY)
                totalCost, totalCorr = 0, 0
                itrs = int(math.ceil(len(testY)/batchSize))
                for itr in range(itrs):
                    batchX = testX[itr*batchSize: (itr+1)*batchSize]
                    batchY = testY[itr*batchSize: (itr+1)*batchSize]

                    predictions, cost = densePredict(sess, caeX, innerFeat, caeIsTraining, xInput, yInput, predictedProbs, cost, batchX, batchY)
                    totalCost += cost
                    totalCorr += np.sum((np.array(predictions) > 0.5) == (np.array(batchY) > 0.5))
                end_time_epoch = datetime.datetime.now()
                print('>> Epoch %d had  TEST loss:      \t#Correct = %5d/%5d = %f\tTime elapsed: %s' % (
                    epoch, totalCorr, len(testY), totalCorr / len(testY), str(end_time_epoch - start_time_epoch)
                ))
                testCosts.append(totalCost)



        # Run against test:
        if runTest:
            print('Testing Scan %s' % (testID))
            testProbs = []
            itrs = int(math.ceil(len(testY)/batchSize))
            for itr in range(itrs):
                batchX = testX[itr*batchSize: (itr+1)*batchSize]
                batchY = testY[itr*batchSize: (itr+1)*batchSize]

                _feat = sess.run(innerFeat, feed_dict={
                    caeX: batchX,
                    caeIsTraining: False
                })

                _probs = sess.run(predictedProbs, feed_dict={
                    caeX: batchX, # HMM
                    caeIsTraining: False, # HMM
                    xInput: _feat,
                    yInput: util.oneshotY(batchY)
                })
                testProbs.extend(np.array(_probs)[:, 1].tolist())
    return trainCosts, testCosts, corrs, util.genScores(testY, testProbs)

# Build CAE, train with all brain voxels
SUB_SAMPLE = 40
def trainAndSave(scanID, savePath=None):
    epochs = N_EPOCHS
    batchSize = BATCH_SIZE

    inFeat, lTrain, lTest = files.loadAllInputsUpdated(scanID, classifier.ALL_FEAT, classifier.MORE_FEAT)
    labelled = np.concatenate([lTrain, lTest])
    lX, lY = files.convertToInputs(inFeat, labelled, classifier.PAD, False, False, False)
    # simpleTSNE(lX, lY)

    bm = files.loadBM(scanID)
    brainIndices = np.array(np.where(bm == 1)).T
    brainIndices = brainIndices[::SUB_SAMPLE, :]
    nSamples = brainIndices.shape[0]
    print ("Training on %d sub-volumes" % brainIndices.shape[0])

    xInput, xOutput, innerFeat, isTraining, trainOp, cost = buildCAENetwork9()
    denseNet = buildDenseNetwork()

    saver = None if savePath is None else tf.train.Saver()

    costs = []
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        start_time = datetime.datetime.now()
        print ("Initializing session...")
        sess.run(tf.global_variables_initializer())

        iterations = int(nSamples/batchSize) + 1
        # run epochs
        for epoch in range(epochs):
            start_time_epoch = datetime.datetime.now()
            print('Scan %s, Epoch %d started' % (scanID, epoch))
            np.random.shuffle(brainIndices)

            # mini batch for trianing set:
            totalCost = 0.0
            for itr in tqdm(range(iterations)):
                batchMids = brainIndices[itr*batchSize: (itr+1)*batchSize]
                batchX = util.xyzRowsToVolumes(inFeat, batchMids, classifier.PAD)
                _trainOp, _cost = sess.run([trainOp, cost], feed_dict={
                    xInput: batchX,
                    isTraining: True
                })
                totalCost += _cost

            avCost = totalCost / iterations
            finalCost = nSamples * avCost
            print (">> Epoch %d had TRAIN loss: %.3f\tav: %.6f" % (
                epoch, finalCost, avCost
            ))
            costs.append(finalCost)

        end_time = datetime.datetime.now()
        print('Time elapse: ', str(end_time - start_time))

        _cost, _feat = sess.run([cost, innerFeat], feed_dict={
            xInput: lX,
            isTraining: False
        })
        print ('Train/Test data has cost %.3f\ av = %.6f' % (_cost * lX.shape[0], _cost))
        print (_feat.shape)

        # Save the network:
        if savePath is not None:
            savePath = saver.save(sess, savePath)
            print ("Model saved to %s" % (savePath))

    return costs

def main():
    TRAIN_SCAN = '019'
    CLASSIFY_TRAIN_SCAN = '022'
    CLASSIFY_TEST_SCAN = '023'
    savePath = "D:/projects/vessels/cae/cae%s.ckpt" % TRAIN_SCAN

    if TRAIN_CAE:
        trainAndSave(TRAIN_SCAN, savePath=savePath)
    if CLASSIFY:
        trainCosts, testCosts, corrs, scores = runDenseNetwork(trainID=CLASSIFY_TRAIN_SCAN, testID=CLASSIFY_TEST_SCAN, loadPath=savePath)
        plt.plot(trainCosts, label='train cost')
        plt.plot(testCosts, label='test cost')
        plt.legend()
        plt.show()
        print(util.formatScores(scores))



if __name__ == '__main__':
  main()
