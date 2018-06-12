import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import RepeatedStratifiedKFold
import tensorflow as tf
from tqdm import tqdm

import classifier
import files
import util
import viz

RANDOM_SEED = 194981
LEARNING_RATE = 0.003 # 0.03
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
    # INNER_DIMEN = 32 # M
    INNER_DIMEN = 50  
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


# Build CAE, train with all brain voxels
SUB_SAMPLE = 40
def trainCAE(scanID):
    epochs = N_EPOCHS    
    batchSize = BATCH_SIZE

    inFeat, lTrain, lTest = files.loadAllInputsUpdated(scanID, classifier.ALL_FEAT, classifier.MORE_FEAT)
    labelled = np.concatenate([lTrain, lTest])
    lX, lY = files.convertToInputs(inFeat, labelled, classifier.PAD, False, False, False)  
    # simpleTSNE(lX, lY)
    
    """
    rs = inFeat.reshape(inFeat.shape[0] * inFeat.shape[1] * inFeat.shape[2], -1)
    print ("FEATURE BOUNDS:")
    print (np.min(rs, axis=0))
    print (np.max(rs, axis=0))
    """

    bm = files.loadBM(scanID)
    brainIndices = np.array(np.where(bm == 1)).T
    brainIndices = brainIndices[::SUB_SAMPLE, :]
    nSamples = brainIndices.shape[0]
    print ("Training on %d sub-volumes" % brainIndices.shape[0])

    xInput, xOutput, innerFeat, isTraining, trainOp, cost = buildCAENetwork9()

    initOp = tf.global_variables_initializer()
    
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
            # trainX, trainY = util.randomShuffle(trainX, trainY)
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
        simpleTSNE(_feat, lY)

    return costs

def checkNetwork(scanID, trainedNetwork):
  pass


def main():
  SCAN_ID = '022'
  trainedNetwork = trainCAE(SCAN_ID)
  checkNetwork(SCAN_ID, trainedNetwork)

  
if __name__ == '__main__':
  main()