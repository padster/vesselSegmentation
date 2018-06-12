import numpy as np
import matplotlib.pyplot as plt
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

def main():
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



if __name__ == '__main__':
  main()