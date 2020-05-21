# Utility, to go through a scan, and see which exact labels were classified incorrectly.

import matplotlib.pyplot as plt
import numpy as np
import sys

import classifier
import cnn
import files
import util

def generateErrorForSingleBrain(scanID):
  data, labelsTrain, labelsTest = files.loadAllInputsUpdated(scanID, classifier.ALL_FEAT, classifier.MORE_FEAT, oneFeat=classifier.ONE_FEAT_NAME)

  trainX, trainY = files.convertToInputs(data, labelsTrain, classifier.PAD, classifier.FLIP_X, classifier.FLIP_Y, classifier.FLIP_Z)
  testX,   testY = files.convertToInputs(data,  labelsTest, classifier.PAD, False, False, False)

  print ("%d train samples, %d test" % (len(trainX), len(testX)))
  _, _, scores, predictions = cnn.runOne(trainX, trainY, testX, testY, scanID, None)
  print ("  Results Train -> Test\n  -------\n" + util.formatScores(scores))

  delta = labelsTest[:, 3] - np.array(predictions)
  errorTest = np.copy(labelsTest.astype(np.float16))
  errorTest[:, 3] = labelsTest[:, 3] - np.array(predictions)
  print ("%f -> %f" % (np.min(errorTest[:, 3]), np.max(errorTest[:, 3])))

  print ("Switched: %d train samples, %d test" % (len(testX), len(trainX)))
  _, _, scores, predictions = cnn.runOne(testX, testY, trainX, trainY, scanID, None)
  print ("  Results Test -> Train\n  -------\n" + util.formatScores(scores))

  delta = labelsTrain[:, 3] - np.array(predictions)
  errorTrain = np.copy(labelsTrain.astype(np.float16))
  errorTrain[:, 3] = labelsTrain[:, 3] - np.array(predictions)
  print ("%f -> %f" % (np.min(errorTrain[:, 3]), np.max(errorTrain[:, 3])))

  errorPath  = "%s/%s/Normal%s-MRA_annotationAll_errors.mat" % (files.BASE_PATH, scanID, scanID)
  print (errorTest.shape)
  print (errorTrain.shape)
  allErrors = np.vstack((errorTrain, errorTest))
  print (allErrors.shape)
  files.writePrediction(errorPath, 'coordTable', allErrors)
  plt.hist(allErrors[:, 3].ravel())
  plt.show()

def main():
  generateErrorForSingleBrain('034')

if __name__ == '__main__':
  classifier.initOptions(sys.argv)
  main()
