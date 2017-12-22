import numpy as np
import scipy.io

CUBE_SZ = 3
PAD = (CUBE_SZ - 1) // 2

# Loads just MRA intensity values from file
def loadMRA(path='data/Normal001-MRA.mat'):
    return scipy.io.loadmat(path).get('V')

# Loads the 2d [X Y Z 0/1-label] matrix for vessel annotations
def loadLabels(path='data/Normal001-Labels.mat'):
    return scipy.io.loadmat(path).get('coordTable')

# Given [X Y Z 0/1-label], and intensity, pick out cubes (of size CUBE_SZ) around the centres.
def convertToInputs(mra, labels):
    rows, cols = labels.shape
    assert cols == 4

    Xs, Ys = [], []
    for row in range(rows):
        x, y, z, label = labels[row]
        Xs.append(mra[x-PAD:x+PAD+1, y-PAD:y+PAD+1, z-PAD:z+PAD+1])
        Ys.append(label)
    return np.array(Xs), np.array(Ys)
