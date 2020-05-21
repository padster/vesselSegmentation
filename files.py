# Utility for most of the file loading for this project
from functools import lru_cache
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


BASE_PATH = "H:/projects/vessels/inputs"

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

@lru_cache(maxsize=None)
def loadFeat(path):
    mat = scipy.io.loadmat(path)
    if CAST_TYPES:
        return mat.get('volMRA').astype(np.float32), mat.get('EM').astype(np.float32), mat.get('JV').astype(np.float32), mat.get('PC').astype(np.float32)
    else:
        return mat.get('volMRA'), mat.get('EM'), mat.get('JV'), mat.get('PC')

def loadBM(scanID, maskPad=None):
    path = "%s/%s/Normal%s-MRA-BM.mat" % (BASE_PATH, scanID, scanID)
    print ("Loading Brain Mask from %s" % path)
    mask = scipy.io.loadmat(path).get('BM')
    # Note: masks have false positives around the edges:
    if maskPad is not None:
        mask[ :maskPad, :, :] = 0
        mask[-maskPad:, :, :] = 0
        mask[:,  :maskPad, :] = 0
        mask[:, -maskPad:, :] = 0
    return mask

# Loads the 2d [X Y Z 0/1-label] matrix for vessel annotations.
# NOTE: X/Y/Z values are 1-based.
@lru_cache(maxsize=None)
def loadLabels(path='data/Normal001-MRA-labels.mat'):
    # NOTE: this used an older copy of the labels, now it's possible to load
    # the labels_full.csv table, and select the required rows by scan ID.
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
        # Label X/Y/Z correction from 1-based to 0-based:
        x, y, z = x - 1, y - 1, z - 1
        xCells = paddedData[x:x+2*pad+1, y:y+2*pad+1, z:z+2*pad+1, :]

        if xCells.shape == (sz, sz, sz, data.shape[3]):
            for t in transforms:
                Xs.append(SubVolumeFromTransformID(scanID, x, y, z, pad, t))
                Ys.append(label)
        else:
            print("Skipping boundary point %s" % str(labels[row]))
            pass
    return Xs, np.array(Ys)

# Load all input features, and sparse labels, for a given scan.
def loadAllInputsUpdated(scanID, pad, allFeatures, moreFeatures, oneFeat=None, noTrain=False):
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
        # **NOTE**: labels here are using matlab's 1-based indexing
        # Need to subtract 1 if wanting to use python's 0-based.
        lTrain = loadLabels(lTrainPath)
        lTest = loadLabels(lTestPath)
        print ("Labels: %s train, %s test" % (str(lTrain.shape), str(lTest.shape)))
        return data, lTrain, lTest

# Loads train and test labels for each scan, converting them to X and Y for training.
def convertScanToXY(scanID, allFeatures, moreFeatures, pad, flipX, flipY, flipZ, flipXY, merge=True, oneFeat=None, oneTransID=None):
    assert merge == True
    data, labelsTrain, labelsTest = loadAllInputsUpdated(scanID, pad, allFeatures, moreFeatures, oneFeat)
    labels = np.vstack((labelsTrain, labelsTest))
    return convertToInputs(scanID, data, labels, pad, flipX, flipY, flipZ, flipXY, oneTransID=oneTransID)

# Given full volume, split up into a subset of the cubes the same size as the inputs
def convertVolumeStack(scanID, pad, x, y, zFr, zTo):
    assert scanID in PADDED_VOLUMES, "Volume stack not from cached scan"

    Xs = []
    for z in range(zFr, zTo):
        Xs.append(SubVolume(scanID, x, y, z, pad, False, False, False, False))
    return np.array(Xs)

# Write out 3d volume of predictions as a matlab matrix.
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

def tiffWrite(path, volume):
    volume = (volume * 65535).astype(np.uint16)
    imsave(path, volume)
