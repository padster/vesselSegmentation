import classifier
import cnn
import sys

def buildFCN():
  print ("hi")
  return None

if __name__ == '__main__':
    classifier.initOptions(sys.argv)

    savePath = None
    cnn.overrideNetwork(buildFCN)
    classifier.singleBrain('002', cnn.runOne, calcScore=True, writeVolume=False, savePath=savePath)

    cnn.overrideNetwork(None)