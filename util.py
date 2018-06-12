import numpy as np
import datetime
from sklearn.metrics import roc_auc_score
from scipy import ndimage

def randomShuffle(X, Y):
	assert X.shape[0] == Y.shape[0]
	p = np.random.permutation(X.shape[0])
	return X[p], Y[p]

def todayStr():
	return datetime.datetime.today().strftime('%Y-%m-%d')

# Convert 4d: (R x X x Y x Z) into 2d: (R x XYZ)
def flatCube(data):
	s = data.shape
	return data.reshape((s[0], s[1] * s[2] * s[3]))

# R x (xyz) => 8R x (xyz)
def allRotations(data):
	allData = []
	for i in range(data.shape[0]):
		for xR in [1, -1]:
			for yR in [1, -1]:
				for zR in [1, -1]:
					allData.append(data[i, ::xR, ::yR, ::zR])
	return np.array(allData)

def combinePredictions(predictions):
	# TODO - better ways?
	return np.mean(predictions, axis=1)
	# pi = np.prod(predictions, axis=1)
	# mpi = np.prod(1 - predictions, axis=1)
	# return pi / (pi + mpi)

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

# TODO - improve?
def genBlurSharpen(volume, sigma):
	blurredVolume = ndimage.gaussian_filter(volume, sigma)
	filterBlurredVolume = ndimage.gaussian_filter(blurredVolume, 1)
	alpha = 30
	sharpened = blurredVolume + alpha * (blurredVolume - filterBlurredVolume)
	return [blurredVolume, sharpened]

def genSimpleFeatures(volume):
	return [
		ndimage.prewitt(volume),
		ndimage.sobel(volume)
	] + \
	  genBlurSharpen(volume, 2.0) + \
	  genBlurSharpen(volume, 5.0)

# Given true and predicted Y, generate the scores we care about
def genScores(trueY, predicted):
	assert len(trueY) == len(predicted)
	trueY = (np.array(trueY) > 0.5)
	predY = (np.array(predicted) > 0.5)
	TP = np.sum(  predY	 &   trueY )
	FP = np.sum(  predY	 & (~trueY))
	TN = np.sum((~predY) & (~trueY))
	FN = np.sum((~predY) &   trueY )
	return [
		(TP + TN) / (TP + TN + FP + FN), # Accuracy
		(TP) / (TP + FN), # Sensitivity
		(TN) / (TN + FP), # Specificity
		(TP + TP) / (TP + TP + FP + FN), # F1
		roc_auc_score(trueY, predicted)
	]

def formatScores(scores):
	return """Accuracy:	%3.3f%%
Sensitivity: %3.3f%%
Specificity: %3.3f%%
F1 score:	%1.5f
ROC AUC:	 %1.5f""" % (scores[0]*100, scores[1]*100, scores[2]*100, scores[3], scores[4])


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
