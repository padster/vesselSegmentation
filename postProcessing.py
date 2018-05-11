import numpy as np
import matplotlib.pyplot as plt

import files
import viz

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, unary_from_softmax

SCAN_ID = "022"
FS_FILE = "data/%s/Normal%s-MRA-FS.mat" % (SCAN_ID, SCAN_ID)
CNN_FILE = "data/%s/Normal%s-MRA-CNN.mat" % (SCAN_ID, SCAN_ID)


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


def postProcess(rawData):
  cleanData = np.copy(rawData)

  # HACK
  Z = 30
  rawImg = rawData[:, :, Z]
  softmax = np.stack([1 - rawImg, rawImg])
  print (softmax.shape)

  unary = unary_from_softmax(softmax)
  print (unary.shape)

  unary = np.ascontiguousarray(unary)
  print (unary.shape)

  d = dcrf.DenseCRF2D(rawImg.shape[0], rawImg.shape[1], 2)  # width, height, nlabels
  d.setUnaryEnergy(unary)

  # This potential penalizes small pieces of segmentation that are
  # spatially isolated -- enforces more spatially consistent segmentations
  # feats = create_pairwise_gaussian(sdims=(20, 20), shape=rawImg.shape[:2])
  # d.addPairwiseEnergy(feats, compat=3,
                      # kernel=dcrf.DIAG_KERNEL,
                      # normalization=dcrf.NORMALIZE_SYMMETRIC)
  d.addPairwiseGaussian(sxy=30, compat=1)

  Q = d.inference(n_iterations=300)
  cleanImg = np.argmax(Q, axis=0).reshape((rawImg.shape[0], rawImg.shape[1]))

  
  cleanData[:, :, Z] = cleanImg
  return cleanData

def main():
  Z = 40
  mra, _, _, _ = files.loadFeat(FS_FILE)
  rawData = files.loadCNN(CNN_FILE)

  DPI = 300
  ax = viz.clean_subplots(1, 2, figsize=(1200, 900), dpi=DPI)
  ax[0][0].imshow(mra[:, :, Z])
  ax[0][1].imshow(rawData[:, :, Z])
  plt.savefig('input.png', dpi=DPI)

  # cleanData = postProcess(rawData)

  sxys = [1, 10, 100] # 10, 30, 100]
  compats = [1, 3, 5]#, 4, 8, 16]

  plt.tight_layout()
  ax2 = viz.clean_subplots(len(sxys), len(compats), figsize=(3600, 3600), dpi=DPI)
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
  plt.savefig('result.png', dpi=DPI)
  # plt.show()


if __name__ == '__main__':
  main()