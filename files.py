import numpy as np
import scipy.io

CUBE_SZ = 3
PAD = (CUBE_SZ - 1) // 2

def loadMat(path, name):
    return scipy.io.loadmat(path).get(name)

# Loads just MRA intensity values from file
def loadMRA(path='data/Normal001-MRA.mat'):
    return loadMat(path, 'V')
def loadEM(path='data/Normal001-MRA-EM.mat'):
    return loadMat(path, 'vessProb')
def loadJV(path='data/Normal001-MRA-JV.mat'):
    return loadMat(path, 'jVessel')
def loadRF(path='data/Normal001-MRA-RF.mat'):
    return loadMat(path, 'forest')

# Loads the 2d [X Y Z 0/1-label] matrix for vessel annotations
def loadLabels(path='data/Normal001-Labels.mat'):
    return loadMat(path, 'coordTable')

# Given [X Y Z 0/1-label], and intensity, pick out cubes (of size CUBE_SZ) around the centres.
def convertToInputs(mra, labels):
    rows, cols = labels.shape
    assert cols == 4

    Xs, Ys = [], []
    for row in range(rows):
        x, y, z, label = labels[row]
        x -= 1
        y -= 1
        z -= 1
        Xs.append(mra[x-PAD:x+PAD+1, y-PAD:y+PAD+1, z-PAD:z+PAD+1])
        Ys.append(label)
    return np.array(Xs), np.array(Ys)

# Given full volume, split up into cubes the same size as the inputs
def convertEntireVolume(mra):
    nX, nY, nZ = mra.shape

    Xs = []
    for x in range(PAD, nX - PAD):
        for y in range(PAD, nY - PAD):
            for z in range(PAD, nZ - PAD):
                Xs.append(mra[x-PAD:x+PAD+1, y-PAD:y+PAD+1, z-PAD:z+PAD+1])
    return np.array(Xs)

def fillPredictions(result, predictions):
    nX, nY, nZ = result.shape
    predictions = predictions.reshape( (nX - 2*PAD, nY - 2*PAD, nZ - 2*PAD) )
    result[PAD:nX-PAD, PAD:nY-PAD, PAD:nZ-PAD] = predictions
    return result

# Write out 3d matrix of [0 - 1] predictions for each cell.
def writePrediction(path, key, prediction):
    data = {}
    data[key] = prediction
    scipy.io.savemat(path, data)
    print ("Saved to " + path)
