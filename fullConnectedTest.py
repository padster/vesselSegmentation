# Unused: Test out our network against a fully connected DNN, i.e. no convolutions.
import classifier
import cnn
import math
import sys

import tensorflow as tf

tf.set_random_seed(0)
RANDOM_SEED = 194981

LEARNING_RATE = 0.001 # 0.03
DROPOUT_RATE = 0.7
BASE_BATCH = 5
N_EPOCHS = 10

BATCH_SIZE = BASE_BATCH * (2 if classifier.FLIP_X else 1) * (2 if classifier.FLIP_Y else 1) * (2 if classifier.FLIP_Z else 1)

def kInit(seed):
    return tf.glorot_normal_initializer(seed=seed)

def buildFCNetwork9(dropoutRate=DROPOUT_RATE, learningRate=LEARNING_RATE, seed=RANDOM_SEED):
    assert classifier.SIZE == 9
    # B x 9 x 9 x 9 x F
    #    => B x 729F => B x sqrt(729F) => B x 2
    xInput = tf.placeholder(tf.float32, shape=[None, classifier.SIZE, classifier.SIZE, classifier.SIZE, classifier.N_FEAT])
    yInput = tf.placeholder(tf.float32, shape=[None, 2])
    isTraining = tf.placeholder(tf.bool)

    # 1) Flatten B x S^3 x F to B x (S^3 x F)
    nFlat = classifier.SIZE * classifier.SIZE * classifier.SIZE * classifier.N_FEAT
    flatLayer = tf.reshape(xInput, [-1, nFlat])

    # 2) Cull to B x sqrt(S^3 x F)
    with tf.name_scope("layer_mid"):
      nMid = (int)(math.ceil(math.sqrt(nFlat)))
      midLayer = tf.layers.dense(inputs=flatLayer, units=nFlat, activation=tf.nn.selu, kernel_initializer=kInit(1), trainable=True)

    # TODO: dropout?
    with tf.name_scope("y_conv"):
        prediction = tf.layers.dense(inputs=midLayer, units=2, kernel_initializer=kInit(2))
        predictedProbs = tf.nn.softmax(prediction)

    with tf.name_scope("cross_entropy"):
        cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            targets=yInput, logits=prediction, pos_weight=cnn.ERROR_WEIGHT_FRAC
        ))

    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learningRate)
        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            trainOp = optimizer.minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(yInput, 1))
    numCorrect = tf.reduce_sum(tf.cast(correct, tf.int32))
    return xInput, yInput, isTraining, trainOp, cost, numCorrect, predictedProbs


def buildFCN():
  return buildFCNetwork9

if __name__ == '__main__':
    classifier.initOptions(sys.argv)

    savePath = None
    cnn.overrideNetwork(buildFCN)
    cnn.N_EPOCHS = N_EPOCHS
    cnn.BATCH_SIZE = BATCH_SIZE

    # classifier.singleBrain('002', cnn.runOne, calcScore=True, writeVolume=False, savePath=savePath)
    classifier.brainsToBrain(['002', '019', '022', '023', '034', '058', '066', '082'], '056', cnn.runOne, calcScore=True, writeVolume=False, savePath=None)

    cnn.overrideNetwork(None)
