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
def loadLabels(path='data/Normal001-MRA-labels.mat'):
    return loadMat(path, 'coordTable')

# Given [X Y Z 0/1-label], and intensity, pick out cubes (of size CUBE_SZ) around the centres.
def convertToInputs(mra, labels, pad=PAD):
    rows, cols = labels.shape
    assert cols == 4
    sz = 2 * pad + 1

    Xs, Ys = [], []
    for row in range(rows):
        x, y, z, label = labels[row]
        x -= 1
        y -= 1
        z -= 1
        xCells = mra[x-pad:x+pad+1, y-pad:y+pad+1, z-pad:z+pad+1]
        if xCells.shape == (sz, sz, sz):
            Xs.append(xCells)
            Ys.append(label)
        else:
            print("Skipping boundary point %s" % str(labels[row]))
    return np.array(Xs), np.array(Ys)

# Given full volume, split up into cubes the same size as the inputs
def convertEntireVolume(mra, pad=None):
    nX, nY, nZ = mra.shape

    if pad is None:
        pad = PAD

    Xs = []
    for x in range(pad, nX - pad):
        for y in range(pad, nY - pad):
            for z in range(pad, nZ - pad):
                Xs.append(mra[x-pad:x+pad+1, y-pad:y+pad+1, z-pad:z+pad+1])
    return np.array(Xs)

def fillPredictions(result, predictions):
    nX, nY, nZ = result.shape

    if pad is None:
        pad = PAD
        
    predictions = predictions.reshape( (nX - 2*pad, nY - 2*pad, nZ - 2*pad) )
    result[pad:nX-pad, pad:nY-pad, pad:nZ-pad] = predictions
    return result

# Write out 3d matrix of [0 - 1] predictions for each cell.
def writePrediction(path, key, prediction):
    data = {}
    data[key] = prediction
    scipy.io.savemat(path, data)
    print ("Saved to " + path)
