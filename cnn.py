import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime

import files
import viz

N_FOLDS = 2 # Train on 3/4, Test on 1/4
N_REPEATS = 5 # K-fold this many times
RANDOM_SEED = 194981

# SET IN MAIN:
#SIZE = 0
#N_EPOCHS = 0
#BATCH_SIZE = 0
#RUN_LOCAL = False

LEARNING_RATE = 0.1
DROPOUT_RATE = 0.5

HACK_GUESSES = []
HACK_COSTS = []
HACK_CORRS = []

def buildNetwork(dropoutRate=DROPOUT_RATE, learningRate=LEARNING_RATE, seed=RANDOM_SEED):
    xInput = tf.placeholder(tf.float32, shape=[None, SIZE, SIZE, SIZE])
    xInputAsOneChannel = tf.expand_dims(xInput, -1)
    yInput = tf.placeholder(tf.float32, shape=[None, 2])

    with tf.name_scope("layer_a"):
        # conv => 7*7*7
        conv1 = tf.layers.conv3d(inputs=xInputAsOneChannel, filters=32, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
        # conv => 7*7*7
        conv2 = tf.layers.conv3d(inputs=conv1, filters=64, kernel_size=[3,3,3], padding='same', activation=tf.nn.relu)
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
        flattening = tf.reshape(cnn3d_bn, [-1, 3*3*3*64])
        dense = tf.layers.dense(inputs=flattening, units=64, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=dropoutRate, training=True)

    with tf.name_scope("y_conv"):
        prediction = tf.layers.dense(inputs=dropout, units=2)
        predictedProbs = tf.nn.softmax(prediction)

    with tf.name_scope("cross_entropy"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=yInput))

    with tf.name_scope("training"):
        # optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
        optimizer = tf.train.AdagradOptimizer(learningRate).minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(yInput, 1))
    numCorrect = tf.reduce_sum(tf.cast(correct, tf.int32))

    return xInput, yInput, optimizer, cost, numCorrect, predictedProbs

# Convert 4d: (R x X x Y x Z) into 2d: (R x XYZ)
def flatCube(data):
    s = data.shape
    return data.reshape((s[0], s[1] * s[2] * s[3]))

# As an example, run CNN on these given labels and test data, return the score.
def runOne(trainX, trainY, testX, testY, firstRun):
    epochs = N_EPOCHS
    batchSize = BATCH_SIZE

    xInput, yInput, optimizer, cost, numCorrect, scores = buildNetwork()

    costs, corrs = [], []
    with tf.Session(config=tf.ConfigProto(log_device_placement=firstRun)) as sess:
        optimizer, cost,
        sess.run(tf.global_variables_initializer())
        start_time = datetime.datetime.now()

        iterations = int(len(trainY)/batchSize) + 1
        # run epochs
        for epoch in range(epochs):
            start_time_epoch = datetime.datetime.now()
            print('Epoch %d started' % (epoch))

            # mini batch
            totalCost, totalCorr = 0.0, 0
            for itr in range(iterations):
                mini_batch_x = trainX[itr*batchSize: (itr+1)*batchSize]
                mini_batch_y = trainY[itr*batchSize: (itr+1)*batchSize]

                batchYOneshot = (np.column_stack((mini_batch_y, mini_batch_y)) == [0, 1]) * 1

                _optimizer, _cost, _corr = sess.run([optimizer, cost, numCorrect], feed_dict={xInput: mini_batch_x, yInput: batchYOneshot})
                totalCost += _cost
                totalCorr += _corr

            print (">> Epoch %d had train loss %f, #Correct = %d / %d = %f" % (
                epoch, totalCost, totalCorr, len(trainY), totalCorr / len(trainY)
            ))

            #  using mini batch in case not enough memory
            totalCost, totalCorr = 0.0, 0
            itrs = int(len(testY)/batchSize) + 1
            for itr in range(itrs):
                mini_batch_x_test = trainX[itr*batchSize: (itr+1)*batchSize]
                mini_batch_y_test = trainY[itr*batchSize: (itr+1)*batchSize]

                batchYOneshotTest = (np.column_stack((mini_batch_y_test, mini_batch_y_test)) == [0, 1]) * 1

                _cost, _corr = sess.run([cost, numCorrect], feed_dict={xInput: mini_batch_x_test, yInput: batchYOneshotTest})
                totalCost += _cost
                totalCorr += _corr

            end_time_epoch = datetime.datetime.now()
            print('>> Testing loss: %f\t#Correct = %d/%d = %f\tTime elapsed: %s' % (
                totalCost, totalCorr, len(testY), totalCorr / len(testY), str(end_time_epoch - start_time_epoch)
            ))
            costs.append(totalCost)
            corrs.append(totalCorr)

        end_time = datetime.datetime.now()
        print('Time elapse: ', str(end_time - start_time))

        # Run against test:
        testYOneshot = (np.column_stack((testY, testY)) == [0, 1]) * 1
        testProbs, testCorr = sess.run([scores, numCorrect], feed_dict={xInput: testX, yInput: testYOneshot})
        testProbs = np.array(testProbs)[:, 1].tolist()
        testCorr = testCorr / len(testY)

    HACK_GUESSES.extend(testProbs)
    HACK_COSTS.append(costs)
    HACK_CORRS.append(corrs)
    return testCorr, roc_auc_score(testY, testProbs)

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

    allCorrect, allAUC = [], []
    for i, (trainIdx, testIdx) in enumerate(splits):
        print ("Split %d / %d" % (i + 1, len(splits)))
        trainX, trainY = Xs[trainIdx], Ys[trainIdx]
        testX, testY = Xs[testIdx], Ys[testIdx]
        correct, auc = runOne(trainX, trainY, testX, testY, firstRun=(i==0))
        allCorrect.append(correct)
        allAUC.append(auc)
        print ("Correct / AUC = %.3f / %.3f" % (allCorrect[-1], allAUC[-1]))


    ax = viz.clean_subplots(1, 2, show=RUN_LOCAL)
    ax[0][0].set_title("Loss over epochs, per split")
    ax[0][0].plot(np.array(HACK_COSTS).T)
    ax[0][1].set_title("%Correct over epochs, per split ")
    ax[0][1].plot(np.array(HACK_CORRS).T)

    image_path = "LossAndCorrect"
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.savefig(image_path)
    print ("Image saved to %s" % str(image_path))

    if RUN_LOCAL:
        plt.show()

    """
    HG = np.array(HACK_GUESSES)
    for i in range(10):
        lBound = i / 10.0
        uBound = lBound + 0.1
        print ("%0.1f - %0.1f = %d" % (lBound, uBound, ((lBound <= HG) & (HG < uBound)).sum()))
    plt.hist(HACK_GUESSES)
    plt.show()
    """

    print ("Average test correct: %.3f" % (np.mean(np.array(allCorrect))))
    print ("Average AUC: %.3f " % (np.mean(np.array(allAUC))))

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
    Xs, Ys = files.convertToInputs(mra, labels, pad=(SIZE-1)//2)
    print ("%d samples" % len(Xs))
    runKFold(Xs, Ys)
    # prediction = generatePrediction(Xs, Ys, mra)
    # files.writePrediction("data/Normal001-MRA-CNN.mat", "forest", prediction)


if __name__ == '__main__':
    global SIZE, N_EPOCHS, BATCH_SIZE, RUN_LOCAL
    SIZE = 7
    N_EPOCHS = 25
    BATCH_SIZE = 16
    RUN_LOCAL = False

    generateAndWriteResults()
