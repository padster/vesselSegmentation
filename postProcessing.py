import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.morphology as morph
import skfmm

from tqdm import tqdm

import files
import util
import viz


import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, unary_from_softmax

SCAN_ID = "019"
FS_FILE = "data/%s/Normal%s-MRA-FS.mat" % (SCAN_ID, SCAN_ID)
CNN_FILE = "data/%s/Normal%s-MRA-CNN%s.mat" % (SCAN_ID, SCAN_ID, "-trans" if SCAN_ID == "023" else "")
CRF_FILE = "data/%s/Normal%s-MRA-CRF.mat" % (SCAN_ID, SCAN_ID)

def makeCompat(a, b):
  return np.array([[a, 1], [1, -b]], dtype=np.float32)


def process2D(rawImg, sxy, compat):
  softmax = np.stack([1 - rawImg, rawImg])
  unary = unary_from_softmax(softmax)
  unary = np.ascontiguousarray(unary)

  d = dcrf.DenseCRF2D(rawImg.shape[0], rawImg.shape[1], 2)  # width, height, nlabels
  d.setUnaryEnergy(unary)

  # This potential penalizes small pieces of segmentation that are
  # spatially isolated -- enforces more spatially consistent segmentations
  # feats = create_pairwise_gaussian(sdims=(20, 20), shape=rawImg.shape[:2])
  # d.addPairwiseEnergy(feats, compat=3,
                      # kernel=dcrf.DIAG_KERNEL,
                      # normalization=dcrf.NORMALIZE_SYMMETRIC)
  d.addPairwiseGaussian(sxy=sxy, compat=compat)

  Q = d.inference(5)
  # print (Q)
  cleanImg = np.argmax(Q, axis=0).reshape((rawImg.shape[0], rawImg.shape[1]))
  return cleanImg


def process3D(rawVolume, sxyz, compat, nIter=5):
  shape = rawVolume.shape

  softmax = np.stack([1 - rawVolume, rawVolume])
  unary = unary_from_softmax(softmax)
  unary = np.ascontiguousarray(unary)

  nLabels = 2
  d = dcrf.DenseCRF(np.prod(shape), nLabels)
  d.setUnaryEnergy(unary)

  feats = create_pairwise_gaussian(sdims=(sxyz, sxyz, sxyz), shape=shape)
  d.addPairwiseEnergy(feats, compat=compat) # kernel=dcrf.FULL_KERNEL)

  Q = d.inference(nIter)
  cleanVolume = np.argmax(Q, axis=0).reshape(rawVolume.shape)
  return cleanVolume

def gradeMatrix(scanID, volume):
  lTrainPath = 'data/%s/Normal%s-MRA_annotationVess_training_C.mat' % (scanID, scanID)
  lTestPath  = 'data/%s/Normal%s-MRA_annotationVess_testing_C.mat'  % (scanID, scanID)
  trainLabels, testLabels = files.loadLabels(lTrainPath), files.loadLabels(lTestPath)
  allLabels = np.concatenate([trainLabels, testLabels], axis=0)
  xyz, answers = allLabels[:, :3] - 1, allLabels[:, 3]

  # plt.hist(volume.flatten())
  # plt.show()

  volumePredictions = np.array([volume[xyz[i, 0], xyz[i, 1], xyz[i, 2]] for i in range(xyz.shape[0])])
  scores = util.genScores(answers, volumePredictions)
  # print ("  Results\n  -------\n" + util.formatScores(scores))
  return scores

def _pairwiseChanges(a, b):
  c01 = np.sum( (a <= 0.5) & (b  > 0.5) )
  c10 = np.sum( (a  > 0.5) & (b <= 0.5) )
  return c01, c10, c01 + c10

def doCRF():
  Z = 40

  mra, _, _, _ = files.loadFeat(FS_FILE)
  # files.tiffWrite("%s-mra.tif" % SCAN_ID, mra)
  rawData = files.loadCNN(CNN_FILE)
  # files.tiffWrite("%s-input.tif" % SCAN_ID, rawData)
  print ("Processing...")
  # sxyz, compat = 0.5, 1.0
  # cleanData = process3D(rawData, sxyz=1, compat=makeCompat(a=0, b=0))
  # files.tiffWrite("%s-output-0-0.tif" % SCAN_ID, cleanData)
  # print ("Done 1/3")
  # cleanData = process3D(rawData, sxyz=1, compat=makeCompat(a=0, b=0.5))
  # files.tiffWrite("%s-output-0-p5.tif" % SCAN_ID, cleanData)
  # print ("Done 2/3")
  cleanData = process3D(rawData, sxyz=1, compat=makeCompat(a=0.5, b=0.5))
  # files.tiffWrite("%s-output-p5-p5.tif" % SCAN_ID, cleanData)
  print ("Done 3/3")
  # print ("Done! Final shape is ", cleanData.shape)
  scores = gradeMatrix(SCAN_ID, cleanData)
  print (scores[0:3])
  print (_pairwiseChanges(rawData, cleanData))

  SC = 0.4
  DPI = 300
  """
  ax = viz.clean_subplots(1, 2, figsize=(1200 * SC, 900 * SC), dpi=DPI)
  ax[0][0].imshow(mra[:, :, Z])
  ax[0][1].imshow(rawData[:, :, Z])
  # plt.savefig('input.png', dpi=DPI)

  # cleanData = postProcess(rawData)

  sxys = [1, 3, 6] # 10, 30, 100]
  compats = [10, 13, 16]#, 4, 8, 16]

  plt.tight_layout()
  ax2 = viz.clean_subplots(len(sxys), len(compats), figsize=(3600 * SC, 3600 * SC), dpi=DPI)
  for i, sxy in enumerate(sxys):
    for j, compat in enumerate(compats):
      if i == len(sxys) - 1:
        ax2[i][j].set_xlabel("compat = %.2f" % compat)
        ax2[i][j].get_xaxis().set_visible(True)
        ax2[i][j].get_xaxis().set_ticks([])
      if j == 0:
        ax2[i][j].set_ylabel("sxy = %.2f" % sxy)
        ax2[i][j].get_yaxis().set_visible(True)
        ax2[i][j].get_yaxis().set_ticks([])
      ax2[i][j].imshow(process2D(rawData[:, :, Z], sxy, compat))
  # ax1, ax2 = ax[0][0], ax[0][1]

  # ax1.imshow(rawData[:, :, Z])
  # ax2.imshow(cleanData[:, :, Z])
  # plt.savefig('result.png', dpi=DPI)
  plt.show()
  """


  """
  print ("Processing...")
  sxys = [0.5, 1.0, 2.0] # 10, 30, 100]
  # compats = [0.5, 1.0, 4.0, 7.0]
  bCompats = [0.0, 0.5, 1.0, 1.5]
  sz = (len(sxys), len(bCompats))
  accy, sens, spec, diff01, diff10 = np.zeros(sz), np.zeros(sz), np.zeros(sz), np.zeros(sz), np.zeros(sz)
  for i, sxyz in tqdm(enumerate(sxys)):
    for j, bc in tqdm(enumerate(bCompats)):
      cleanData = process3D(rawData, sxyz=sxyz, compat=makeCompat(a=0, b=bc))
      scores = gradeMatrix(SCAN_ID, cleanData)
      accy[i, j] = scores[0]
      sens[i, j] = scores[1]
      spec[i, j] = scores[2]
      diff01[i, j], diff10[i, j], _ = _pairwiseChanges(rawData, cleanData)

  # ax = viz.clean_subplots(1, 3, figsize=(1200 * SC, 900 * SC), dpi=DPI)
  # ax[0][0].imshow((accy - 0.66) * 3, vmin=0, vmax=1, cmap='hot')
  # ax[0][1].imshow((sens - 0.66) * 3, vmin=0, vmax=1, cmap='hot')
  # ax[0][2].imshow((spec - 0.66) * 3, vmin=0, vmax=1, cmap='hot')
  # plt.contoru(accy)
  # plt.contour(sens)
  # plt.contour(spec)
  # plt.show()

  print ("\n\nAccuracy:")
  print (util.formatTable(bCompats, sxys, accy))
  print ("\n\nSensitivity:")
  print (util.formatTable(bCompats, sxys, sens))
  print ("\n\nSpecificity:")
  print (util.formatTable(bCompats, sxys, spec))
  print ("\n\nChanges 0->1:")
  print (util.formatTable(bCompats, sxys, diff01))
  print ("\n\nChanges 1->0:")
  print (util.formatTable(bCompats, sxys, diff10))
  # """

  # files.writePrediction(CRF_FILE, 'crf', cleanData)


"""
SKELETON CODE
"""

def doSkeleton(scanID):
  data, lTrain, lTest = files.loadAllInputsUpdated(scanID, False, False)
  mraData = data[:, :, :, 0]
  # mraPath = path = "D:/projects/vessels/inputs/%s/Normal%s-MRA-FS.mat" % (scanID, scanID)
  # mraData, _, _, _ = files.loadFeat(mraPath)
  cnnPath = path = "D:/projects/vessels/inputs/%s/Normal%s-MRA-CNN.mat" % (scanID, scanID)
  cnnData = files.loadCNN(cnnPath)
  mask = files.loadBM(scanID)
  cnnClean = util.applyBrainMask(cnnData, mask)
  smoothed = ndi.gaussian_filter(cnnClean, 2)
  smoothFilt = smoothed > 0.12
  skeleton = morph.skeletonize_3d(smoothFilt)
  smoothSkel = ndi.gaussian_filter(skeleton, 0.5)
  cnnWithSkel = cnnClean * ndi.gaussian_filter(skeleton, 2.5)

  f, ax = viz.clean_subplots(2, 4)
  Z = 35
  ax[0][0].imshow(mraData[:, :, Z])
  ax[0][1].imshow(cnnData[:, :, Z])
  ax[0][2].imshow(cnnClean[:, :, Z])
  ax[0][3].imshow(smoothed[:, :, Z])
  ax[1][0].imshow(smoothFilt[:, :, Z])
  ax[1][1].imshow(skeleton[:, :, Z])
  ax[1][2].imshow(smoothSkel[:, :, Z])
  ax[1][3].imshow(cnnWithSkel[:, :, Z])
  plt.show(block=False)
  """
  print ("Saving...")
  smoothPathOut = "D:/projects/vessels/inputs/%s/20180617-%s-CNN-smooth2.tif" % (scanID, scanID)
  smoothFiltPathOut = "D:/projects/vessels/inputs/%s/20180617-%s-CNN-smoothFilt.tif" % (scanID, scanID)
  skeletonPathOut = "D:/projects/vessels/inputs/%s/20180617-%s-CNN-skeleton.tif" % (scanID, scanID)
  smoothSkelPathOut = "D:/projects/vessels/inputs/%s/20180617-%s-CNN-skeleton-smooth.tif" % (scanID, scanID)
  cnnWithSkelPathOut = "D:/projects/vessels/inputs/%s/20180617-%s-CNN-raw-with-skel.tif" % (scanID, scanID)
  files.tiffWrite(smoothPathOut, smoothed)
  files.tiffWrite(smoothFiltPathOut, smoothFilt)
  files.tiffWrite(skeletonPathOut, skeleton)
  files.tiffWrite(smoothSkelPathOut, smoothSkel)
  files.tiffWrite(cnnWithSkelPathOut, cnnWithSkel)
  """

  skeletonAsBoundary = np.ones(skeleton.shape)
  skeletonAsBoundary[skeleton > 0] = 0
  print ("Calculating distance...")
  distanceToSkeleton = skfmm.distance(skeletonAsBoundary)
  print ("Done!")

  onDist, offDist = [], []
  onCNN, offCNN = [], []
  for labels in [lTrain, lTest]:
    print (labels.shape)
    for i in range(labels.shape[0]):
      x, y, z, l = labels[i, :]
      x, y, z = x - 1, y - 1, z - 1
      if l == 1:
        onDist.append(distanceToSkeleton[x, y, z])
        onCNN.append(cnnData[x, y, z])
      else:
        offDist.append(distanceToSkeleton[x, y, z])
        offCNN.append(cnnData[x, y, z])
  print ("%d on, %d off" % (len(onDist), len(offDist)))
  onDist, offDist = np.array(onDist), np.array(offDist)
  onCNN, offCNN = np.array(onCNN), np.array(offCNN)

  """
  f, ax = viz.clean_subplots(1, 1, axes=True)
  ax = ax[0][0]
  bins = 50
  ax.hist(offDist, bins, alpha=0.5, color='r', label='non-vessel', log=True, range=(0, 80))
  ax.hist(onDist, bins, alpha=0.5, color='b', label='vessel', log=True, range=(0, 80))
  ax.set_xlabel('Distance')
  f.suptitle('Counts of distance to skeleton, by vessel type')
  plt.legend()
  plt.show()
  """

  f, ax = viz.clean_subplots(1, 1, axes=True, pad=0.1)
  ax = ax[0][0]
  ax.scatter(offDist, offCNN, alpha=0.5, c='r', label='non-vessel')
  ax.scatter(onDist, onCNN, alpha=0.5, c='b', label='vessel')
  ax.set_xlabel('Distance')
  ax.set_ylabel('CNN Value')
  f.suptitle('%s: Skeleton Distance vs. CNN prediction, by vessel type' % scanID)
  plt.legend()
  plt.show()



def main():
  # doCRF()
  doSkeleton('022')


if __name__ == '__main__':
  main()