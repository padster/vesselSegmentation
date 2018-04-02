import numpy as np
import scipy.io

def loadMat(path, name):
    return scipy.io.loadmat(path).get(name)

# Loads just MRA intensity values from file
def loadMRA(path='data/Normal001-MRA.mat'):
    return loadMat(path, 'V')
def loadEM(path='data/Normal001-MRA-EM.mat'):
    return loadMat(path, 'vessProb')
def loadJV(path='data/Normal001-MRA-JV.mat'):
    return loadMat(path, 'jVessel')
def loadPC(path='data/Normal001-MRA-PC.mat'):
    return loadMat(path, 'PC')
def loadRF(path='data/Normal001-MRA-RF.mat'):
    return loadMat(path, 'forest')

# Loads the 2d [X Y Z 0/1-label] matrix for vessel annotations
def loadLabels(path='data/Normal001-MRA-labels.mat'):
    return loadMat(path, 'coordTable')

# Given [X Y Z 0/1-label], and intensity, pick out cubes (of size CUBE_SZ) around the centres.
def convertToInputs(data, labels, pad):
    rows, cols = labels.shape
    assert cols == 4
    sz = 2 * pad + 1

    Xs, Ys = [], []
    for row in range(rows):
        x, y, z, label = labels[row]
        x -= 1
        y -= 1
        z -= 1
        xCells = data[x-pad:x+pad+1, y-pad:y+pad+1, z-pad:z+pad+1, :]
        if xCells.shape == (sz, sz, sz, data.shape[3]):
            Xs.append(xCells)
            Ys.append(label)
        else:
            # print("Skipping boundary point %s" % str(labels[row]))
            pass
    return np.array(Xs), np.array(Ys)

# Load intensities, plus optionally other features as second channels
def loadAllInputs(allFeatures):
    print ("Loading volume intensitites...")
    data = loadMRA()
    if allFeatures:
        print ("Loading features...")
        featEM, featJV, featPC = loadEM(), loadJV(), loadPC()
        assert data.shape == featEM.shape
        assert data.shape == featJV.shape
        assert data.shape == featPC.shape
        data = np.stack([data, featEM, featJV, featPC], axis=-1)
    else:
        data = np.stack([data], axis=-1)
    print ("Input data loaded, shape = %s" % (str(data.shape)))
    labels = loadLabels()
    return data, labels


# Given full volume, split up into cubes the same size as the inputs
# HACK - doesn't fit into memory
"""
def convertEntireVolume(data):
    nX, nY, nZ, nChan = data.shape

    Xs = []
    for x in range(pad, nX - pad):
        for y in range(pad, nY - pad):
            for z in range(pad, nZ - pad):
                Xs.append(data[x-pad:x+pad+1, y-pad:y+pad+1, z-pad:z+pad+1, :])
    return np.array(Xs)
"""

# Given full volume, split up into a subset of the cubes the same size as the inputs
def convertVolumeStack(data, pad, x, y):
    nX, nY, nZ, nChan = data.shape

    Xs = []
    # for x in range(x, nX - pad):
        # for y in range(pad, nY - pad):
    for z in range(pad, nZ - pad):
        Xs.append(data[x-pad:x+pad+1, y-pad:y+pad+1, z-pad:z+pad+1, :])
    return np.array(Xs)


def fillPredictions(result, predictions, pad):
    nX, nY, nZ = result.shape

    predictions = predictions.reshape( (nX - 2*pad, nY - 2*pad, nZ - 2*pad) )
    result[pad:nX-pad, pad:nY-pad, pad:nZ-pad] = predictions
    return result

# Write out 3d matrix of [0 - 1] predictions for each cell.
def writePrediction(path, key, prediction):
    data = {}
    data[key] = prediction
    scipy.io.savemat(path, data)
    print ("Saved to " + path)
