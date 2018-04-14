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
def loadCNN(path='data/Normal001-MRA-CNN.mat'):
    return loadMat(path, 'cnn')

def loadFeat(path):
    mat = scipy.io.loadmat(path)
    return mat.get('volMRA'), mat.get('EM'), mat.get('JV'), mat.get('PC')

# Loads the 2d [X Y Z 0/1-label] matrix for vessel annotations
def loadLabels(path='data/Normal001-MRA-labels.mat'):
    return loadMat(path, 'coordTable')

# Given [X Y Z 0/1-label], and intensity, pick out cubes (of size CUBE_SZ) around the centres.
def convertToInputs(data, labels, pad, flipX, flipY, flipZ):
    rows, cols = labels.shape
    assert cols == 4
    sz = 2 * pad + 1

    xR = [1, -1] if flipX else [1]
    yR = [1, -1] if flipY else [1]
    zR = [1, -1] if flipZ else [1]

    Xs, Ys = [], []
    for row in range(rows):
        x, y, z, label = labels[row]
        x, y, z = x - 1, y - 1, z - 1
        xCells = data[x-pad:x+pad+1, y-pad:y+pad+1, z-pad:z+pad+1, :]
        if xCells.shape == (sz, sz, sz, data.shape[3]):
            for xr in xR:
                for yr in yR:
                    for zr in zR:
                        Xs.append(xCells[::xr, ::yr, ::zr])
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

def loadAllInputsUpdated(scanID, allFeatures):
    fsPath = 'data/%s/Normal%s-MRA-FS.mat' % (scanID, scanID)
    print ("Loading data for scan %s" % (scanID))
    data, featEM, featJV, featPC = loadFeat(fsPath)
    assert data.shape == featEM.shape
    assert data.shape == featJV.shape
    assert data.shape == featPC.shape

    if allFeatures:
        data = np.stack([data, featEM, featJV, featPC], axis=-1)
    else:
        data = np.stack([data], axis=-1)
    print ("Input data loaded, shape = %s" % (str(data.shape)))

    lTrainPath = 'data/%s/Normal%s-MRA_annotationVess_training_C.mat' % (scanID, scanID)
    lTestPath  = 'data/%s/Normal%s-MRA_annotationVess_testing_C.mat'  % (scanID, scanID)
    return data, loadLabels(lTrainPath), loadLabels(lTestPath)

def convertScanToXY(scanID, allFeatures, pad, flipX, flipY, flipZ, merge):
    data, labelsTrain, labelsTest = loadAllInputsUpdated(scanID, allFeatures)
    if merge:
        labels = np.vstack((labelsTrain, labelsTest))
        return convertToInputs(data, labels, pad, flipX, flipY, flipZ)
    else:
        trainX, trainY = convertToInputs(data, labelsTrain, pad, flipX, flipY, flipZ)
        testX, testY = convertToInputs(data, labelsTest, pad, flipX, flipY, flipZ)
        return trainX, trainY, testX, testY

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
