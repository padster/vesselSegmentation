# import libtiff
import numpy as np
import scipy.io

from tifffile import TiffFile, imsave

import util

from model import PADDED_VOLUMES, SubVolume, transformParamsToID, SubVolumeFromTransformID

CAST_TYPES = True
USE_PREPROC = True

###
### MAT files
###


BASE_PATH = "D:/projects/vessels/inputs"
# BASE_PATH = "/home/ubuntu/data/inputs"
# BASE_PATH = "data/inputs"

# HACK: Set by caller
CNN_FEAT = False
CNN_FEAT_PATH = "data/multiV/04_25/"

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
    print ("Loading cnn from " + path)
    return loadMat(path, 'cnn')
def loadCRF(path='data/Normal001-MRA-CRF.mat'):
    return loadMat(path, 'crf')

def loadFeat(path):
    # print ("Loading features from " + path)
    mat = scipy.io.loadmat(path)
    if CAST_TYPES:
        return mat.get('volMRA').astype(np.float32), mat.get('EM').astype(np.float32), mat.get('JV').astype(np.float32), mat.get('PC').astype(np.float32)
    else:
        return mat.get('volMRA'), mat.get('EM'), mat.get('JV'), mat.get('PC')

def loadBM(scanID):
    path = "D:/projects/vessels/inputs/%s/Normal%s-MRA-BM.mat" % (scanID, scanID)
    print ("Loading Brain Mask from %s" % path)
    mat = scipy.io.loadmat(path)
    return mat.get('BM')

# Loads the 2d [X Y Z 0/1-label] matrix for vessel annotations
def loadLabels(path='data/Normal001-MRA-labels.mat'):
    return loadMat(path, 'coordTable')

# Given [X Y Z 0/1-label], and intensity, pick out cubes (of size CUBE_SZ) around the centres.
def convertToInputs(scanID, data, labels, pad, flipX, flipY, flipZ, flipXY, oneTransID=None):
    rows, cols = labels.shape
    assert cols == 4
    sz = 2 * pad + 1

    transforms = []
    if oneTransID is None:
        # Load all transforms based off flip params:
        # TODO - pull outside this method?
        xR  = [False, True] if flipX  else [False]
        yR  = [False, True] if flipY  else [False]
        zR  = [False, True] if flipZ  else [False]
        xyF = [False, True] if flipXY else [False]
        for xr in xR:
            for yr in yR:
                for zr in zR:
                    for xyf in xyF:
                        transforms.append(transformParamsToID(xyf, xr, yr, zr))
    else:
        # Only load one transform:
        transforms = [oneTransID]

    s = data.shape
    paddedData = np.zeros((s[0] + 2 * pad, s[1] + 2 * pad, s[2] + 2 * pad, s[3]))
    paddedData[pad:pad+s[0], pad:pad+s[1], pad:pad+s[2], :] = data

    Xs, Ys = [], []
    for row in range(rows):
        x, y, z, label = labels[row]
        x, y, z = x - 1, y - 1, z - 1
        # xCells = data[x-pad:x+pad+1, y-pad:y+pad+1, z-pad:z+pad+1, :]
        xCells = paddedData[x:x+2*pad+1, y:y+2*pad+1, z:z+2*pad+1, :]
        if xCells.shape == (sz, sz, sz, data.shape[3]):
            for t in transforms:
                Xs.append(SubVolumeFromTransformID(scanID, x, y, z, pad, t))
                Ys.append(label)
        else:
            print("Skipping boundary point %s" % str(labels[row]))
            pass
    return Xs, np.array(Ys)

def loadAllInputsUpdated(scanID, pad, allFeatures, moreFeatures, oneFeat=None, noTrain=False):
    # fsPath = 'data/%s/Normal%s-MRA-FS.mat' % (scanID, scanID)
    fsPath     = "%s/%s/Normal%s-MRA-FS.mat" % (BASE_PATH, scanID, scanID)
    lTrainPath = "%s/%s/Normal%s-MRA_annotationAll_training_C.mat" % (BASE_PATH, scanID, scanID)
    lTestPath  = "%s/%s/Normal%s-MRA_annotationAll_testing_C.mat" % (BASE_PATH, scanID, scanID)

    if USE_PREPROC:
        if scanID == '084':
            fsPath = "%s/%s/Normal%s-MRA-FS-preproc.mat" % (BASE_PATH, scanID, scanID)
            print ("USING PREPROC")

    print ("Loading data for scan %s" % (scanID))
    data, featEM, featJV, featPC = loadFeat(fsPath)
    assert data.shape == featEM.shape
    assert data.shape == featJV.shape
    assert data.shape == featPC.shape

    allStacks = [data]
    if allFeatures:
        allStacks.extend([featEM, featJV, featPC])

    if oneFeat is not None:
        if oneFeat == 'EM':
            allStacks.extend([featEM])
        elif oneFeat == 'JV':
            allStacks.extend([featJV])
        elif oneFeat == 'PC':
            allStacks.extend([featPC])

    if moreFeatures:
        allStacks.extend(util.genSimpleFeatures(data))

    if CNN_FEAT:
        path = CNN_FEAT_PATH + ("Normal%s-MRA-CNN.mat" % (scanID))
        featCNN = loadCNN(path)
        assert data.shape == featCNN.shape
        allStacks.append(featCNN)

    data = np.stack(allStacks, axis=-1)
    print ("Input data loaded, shape = %s" % (str(data.shape)))

    PADDED_VOLUMES[scanID] = util.padVolume(data, pad)

    if noTrain:
        return data
    else:
        lTrain = loadLabels(lTrainPath)
        lTest = loadLabels(lTestPath)
        print ("Labels: %s train, %s test" % (str(lTrain.shape), str(lTest.shape)))
        return data, lTrain, lTest

def convertScanToXY(scanID, allFeatures, moreFeatures, pad, flipX, flipY, flipZ, flipXY, merge=True, oneFeat=None, oneTransID=None):
    assert merge == True
    data, labelsTrain, labelsTest = loadAllInputsUpdated(scanID, pad, allFeatures, moreFeatures, oneFeat)
    labels = np.vstack((labelsTrain, labelsTest))
    return convertToInputs(scanID, data, labels, pad, flipX, flipY, flipZ, flipXY, oneTransID=oneTransID)
    # Unmerged version, no longer needed - eventually to be removed
    # trainX, trainY = convertToInputs(scanID, data, labelsTrain, pad, flipX, flipY, flipZ, flipXY)
    # testX, testY = convertToInputs(scanID, data, labelsTest, pad, flipX, flipY, flipZ, flipXY)
    # return trainX, trainY, testX, testY


# Given full volume, split up into a subset of the cubes the same size as the inputs
def convertVolumeStack(data, pad, x, y):
    raise Exception("Need to convert this to the new SubVolume model")
    nX, nY, nZ, nChan = data.shape

    Xs = []
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
    data[key] = prediction.astype(np.float16)
    scipy.io.savemat(path, data)
    print ("Saved to " + path)

###
### TIF files
###

# Write prediction as tif file:
def asMatrix(asList, nRows):
    nCols = int(len(asList) / nRows)
    assert nRows * nCols == len(asList), "List wrong size for matrix conversion"
    return [asList[i:i+nCols] for i in range(0, len(asList), nCols)]

def tiffRead(path):
    # First use tifffile to get channel data (not supported by libtiff?)
    shape = None
    with TiffFile(path) as tif:
        shape = tif.asarray().shape
        print("Tiff shape: ", shape)
    nChannels = shape[0] if len(shape) == 4 else 1

    stack = []
    tif = libtiff.TIFF.open(path, mode='r')
    stack = asMatrix([np.array(img) for img in tif.iter_images()], nChannels)
    tif.close()
    return stack

def tiffWrite(path, volume):
    volume = (volume * 65535).astype(np.uint16)
    # tif = libtiff.TIFFFimage(volume, description='')
    # tif.write_file(path, compression='lzw')
    # del tif
    imsave(path, volume)
