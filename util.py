# General utilities for use throughout the code:
import numpy as np
import datetime
from sklearn.metrics import roc_auc_score
from scipy import ndimage

from model import SubVolume, extractSubVolume

# IDs of MIDAS scans that have manual annotations
SCAN_IDS = ['002', '019', '022', '023', '034', '056', '058', '066', '082']

# 5 Metrics measured by genScores:
METRICS = ['Accuracy', 'Sensitivity', 'Specificity', 'Dice score', 'ROC AUC']

# Given input and corresponding output, apply the same random shuffle to each.
def randomShuffle(X, Y):
	assert X.shape[0] == Y.shape[0]
	p = np.random.permutation(X.shape[0])
	return X[p], Y[p]

# Textual representation of today, for easier results writing.
def todayStr():
	return datetime.datetime.today().strftime('%Y-%m-%d')

# Convert 4d: (R x X x Y x Z) into 2d: (R x XYZ)
def flatCube(data):
	s = data.shape
	return data.reshape((s[0], s[1] * s[2] * s[3]))

# Apply all transforms to the given subvolumes.
# R x (xyz) => 16R x (xyz)
def allRotations(subvolumes):
	allSV = []
	for sv in subvolumes:
		for fXY in [False, True]:
			for rX in [False, True]:
				for rY in [False, True]:
					for rZ in [False, True]:
						allSV.append(SubVolume(sv.s, sv.x, sv.y, sv.z, sv.p, fXY, rX, rY, rZ))
	return allSV

# Given multiple predictions for different transforms, merge to one prediction.
def combinePredictions(predictions):
	return np.mean(predictions, axis=1) # mean seemed best
	# pi = np.prod(predictions, axis=1)
	# mpi = np.prod(1 - predictions, axis=1)
	# return pi / (pi + mpi)

# Convert a collection of subvolumes into a single input tensor.
def subVolumesToTensor(subvolumes):
	result = []
	for sv in subvolumes:
		data = extractSubVolume(sv)
		if data.shape[0] == 0 or data.shape[1] == 0 or data.shape[2] == 0:
			print ("BROKEN")
			print (sv)
		else:
			result.append(data)
	return np.array(result)

# Add a padding of zeros around a volume, to avoid border issues.
def padVolume(volume, pad):
	s = volume.shape
	paddedVolume = np.zeros((s[0] + 2 * pad, s[1] + 2 * pad, s[2] + 2 * pad, s[3]))
	paddedVolume[pad:pad+s[0], pad:pad+s[1], pad:pad+s[2], :] = volume
	return paddedVolume

# Convert a single N x (0 or 1) column into a oneshot N x 2 tensor
def oneshotY(y):
	return np.column_stack((1 - y, y))

# Convert Nx3 (xyz) to NxSxSxS, each row centered on the input
def xyzRowsToVolumes(features, xyz, pad):
	sz, n, f = 2 * pad + 1, xyz.shape[0], features.shape[3]
	vols = np.zeros((n, sz, sz, sz, f))
	target = (sz, sz, sz, f)
	for i in range(n):
		x, y, z = xyz[i]
		feat = features[x-pad:x+pad+1, y-pad:y+pad+1, z-pad:z+pad+1, :]
		if feat.shape != target:
			feat = np.pad(feat, [(0, a - b) for a, b in zip(target, feat.shape)], mode='constant', constant_values=0)
		vols[i] = feat
	return vols

# Generate some simple fake features, by blurring and sharpening the volume
def genBlurSharpen(volume, sigma):
	blurredVolume = ndimage.gaussian_filter(volume, sigma)
	filterBlurredVolume = ndimage.gaussian_filter(blurredVolume, 1)
	alpha = 30
	sharpened = blurredVolume + alpha * (blurredVolume - filterBlurredVolume)
	return [blurredVolume, sharpened]

# Generate simple fake features, using simple image volume algorithms
def genSimpleFeatures(volume):
	return [
		ndimage.prewitt(volume),
		ndimage.sobel(volume)
	] + \
	  genBlurSharpen(volume, 2.0) + \
	  genBlurSharpen(volume, 5.0)

# Given true and predicted Y, generate the scores we care about
def genScores(trueY, predicted, T1=0.5, T2=None):
	assert len(trueY) == len(predicted)
	T2 = T1 if T2 is None else T2
	trueY = (np.array(trueY) > T1)
	predY = (np.array(predicted) > T2)
	TP = np.sum(  predY	 &   trueY )
	FP = np.sum(  predY	 & (~trueY))
	TN = np.sum((~predY) & (~trueY))
	FN = np.sum((~predY) &   trueY )
	return [
		(TP + TN) / (TP + TN + FP + FN), # Accuracy
		(TP) / (TP + FN), # Sensitivity
		(TN) / (TN + FP), # Specificity
		(TP + TP) / (TP + TP + FP + FN), # Dice
		roc_auc_score(trueY, predicted)
	]

# Format the above in a readable manner
def formatScores(scores):
	return """Accuracy   : %3.3f%%
Sensitivity: %3.3f%%
Specificity: %3.3f%%
Dice score : %1.5f
ROC AUC    : %1.5f""" % (scores[0]*100, scores[1]*100, scores[2]*100, scores[3], scores[4])

# Slightly more readable table format.
def formatTable(colVals, rowVals, tableVals):
	result = ", "
	for cv in colVals:
		result += "%.2f, " % cv
	result += "\n"
	for i, rv in enumerate(rowVals):
		result += "%.2f, " % rv
		for j in range(len(colVals)):
			result += "%.3f, " % tableVals[i, j]
		result += "\n"
	return result

# Given a mask and volume, zero out any voxels outside the mask.
def applyBrainMask(data, mask):
	return data * mask

# Given a collection of 0s and 1s, return slice start & end that covers all the 1s.
def maskBounds(zColumn):
	isBrain = np.argwhere(zColumn == 1)
	if len(isBrain) == 0:
		return -1, -1
	else:
		return isBrain[0, 0], isBrain[-1, 0] + 1
