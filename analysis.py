# Some utility code for inspecting agreement between methods and features.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE

from tqdm import tqdm

import files
import viz

def intersectOverUnion(dataA, dataB):
    return np.sum(np.logical_and(dataA, dataB)) * 1.0 / np.sum(np.logical_or(dataA, dataB))

def checkEMJVAgreement():
    emData = files.loadEM()
    jvData = files.loadJV()
    labels = files.loadLabels()

    rows, cols = labels.shape
    assert cols == 4

    emCorrect, jvCorrect, bothCorrect, eitherCorrect = 0, 0, 0, 0
    emV, jvV, laV = [], [], []

    emContTable = np.zeros((2, 2), dtype=np.int)
    jvContTable = np.zeros((2, 2), dtype=np.int)

    for row in range(rows):
        x, y, z, label = labels[row]
        emYes = (emData[x][y][z] < 0.5)
        jvYes = (jvData[x][y][z] < 0.5)
        laYes = (label > 0.5)
        emRight = (emYes == laYes)
        jvRight = (jvYes == laYes)
        if emRight:
            emCorrect += 1
        if jvRight:
            jvCorrect += 1
        if emRight and jvRight:
            bothCorrect += 1
        if emRight or jvRight:
            eitherCorrect += 1
        emContTable[1 * emYes][1 * laYes] += 1
        jvContTable[1 * jvYes][1 * laYes] += 1
        emV.append(emData[x][y][z])
        jvV.append(jvData[x][y][z])
        laV.append(label)

    print ("EM Contingency table:")
    print (emContTable)
    print ("JV Contingency table:")
    print (jvContTable)

    plt.plot(emV, laV, 'o')
    plt.title("EM prediction vs label")
    plt.xlabel("EM prediction")
    plt.ylabel("Label")
    plt.show()
    plt.plot(jvV, laV, 'o')
    plt.title("JV prediction vs label")
    plt.xlabel("JV prediction")
    plt.ylabel("Label")
    plt.show()

    print ("EM: %d / %d" % (emCorrect, rows))
    print ("JV: %d / %d" % (jvCorrect, rows))
    print ("Both: %d / %d" % (bothCorrect, rows))
    print ("Eith: %d / %d" % (eitherCorrect, rows))


def rfAnalysis():
    print ("\nRF Data:")
    # run some basic analysis on written results
    emData = files.loadEM().flatten()
    jvData = files.loadJV().flatten()
    rfData = 1.0 - files.loadRF().flatten()
    print ("Pearson R coeff for RF-EM: %f" % (np.corrcoef(rfData, emData)[0, 1]))
    print ("Pearson R coeff for RF-JV: %f" % (np.corrcoef(rfData, jvData)[0, 1]))
    emPre, jvPre = np.copy(emData), np.copy(jvData)
    emData = (emData > 0.5)
    jvData = (jvData > 0.5)
    print ("AUC of RF for EM: %f" % roc_auc_score(emData, rfData))
    print ("AUC of RF for JV: %f" % roc_auc_score(jvData, rfData))
    rfData = (rfData > 0.5)
    print ("Agreement EM-RF: %f" % (np.sum(emData == rfData) / len(rfData)))
    print ("Agreement JV-RF: %f" % (np.sum(jvData == rfData) / len(rfData)))
    print ("Agreement EM-JV: %f" % (np.sum(emData == jvData) / len(rfData)))
    print ("IOU EM-RF:  %f" % intersectOverUnion(~emData, ~rfData)) # HACK - this is broken...
    print ("IOU JV-RF:  %f" % intersectOverUnion(~jvData, ~rfData))
    print ("IOU EM-JV:  %f" % intersectOverUnion(~emData, ~jvData))
    return emPre, jvPre


def cnnAnalysis(emData, jvData):
    print ("\nCNN Data:")
    # run some basic analysis on written results
    cnnData = files.loadCNN().flatten()
    print ("Pearson R coeff for CNN-EM: %f" % (np.corrcoef(cnnData, emData)[0, 1]))
    print ("Pearson R coeff for CNN-JV: %f" % (np.corrcoef(cnnData, jvData)[0, 1]))
    plt.scatter(cnnData[::30], jvData[::30])
    plt.show()
    emData = (emData > 0.5)
    jvData = (jvData > 0.5)
    print ("AUC of CNN for EM: %f" % roc_auc_score(emData, cnnData))
    print ("AUC of CNN for JV: %f" % roc_auc_score(jvData, cnnData))
    cnnData = (cnnData > 0.5)
    print ("Agreement EM-CNN: %f" % (np.sum(emData == cnnData) / len(cnnData)))
    print ("Agreement JV-CNN: %f" % (np.sum(jvData == cnnData) / len(cnnData)))
    print ("Agreement EM-JV : %f" % (np.sum(emData == jvData) / len(cnnData)))
    print ("IOU EM-CNN:  %f" % intersectOverUnion(~emData, ~cnnData)) # HACK - this is broken...
    print ("IOU JV-CNN:  %f" % intersectOverUnion(~jvData, ~cnnData))
    print ("IOU EM-JV :  %f" % intersectOverUnion(~emData,  ~jvData))

def show_tsne(ax, Xs, perplexity, learnRate, colors):
    fitted = TSNE(
        n_components=2, perplexity=perplexity, learning_rate=learnRate
    ).fit_transform(Xs)
    ax.scatter(fitted[:, 0], fitted[:, 1], color=colors, marker='x')


# Options for TSNE parameters to explore.
# PERPLEXITY_OPTIONS = [10, 22, 33, 50, 80]
# LEARN_RATE_OPTIONS = [10, 30, 100, 300, 1000]
PERPLEXITY_OPTIONS = [10]
LEARN_RATE_OPTIONS = [300]

def simpleTSNE():
    SZ = 5
    PAD = SZ // 2
    mra = files.loadMRA()
    labels = files.loadLabels()
    Xs, Ys = files.convertToInputs(mra, labels, pad=PAD)
    print ("%d samples" % len(Xs))
    xPoints = Xs.reshape(Xs.shape[0], SZ * SZ * SZ)

    colours = []
    for y in Ys:
        if y < 0.5:
            colours.append('#ff0000')
        else:
            colours.append('#00ff00')

    print (Xs.shape)

    f, ax = viz.clean_subplots(len(PERPLEXITY_OPTIONS), len(LEARN_RATE_OPTIONS))
    for i in tqdm(range(len(PERPLEXITY_OPTIONS))):
        perplexity = PERPLEXITY_OPTIONS[i]
        ax[i][0].get_yaxis().set_visible(True)
        ax[i][0].set_ylabel("Perplexity = %d" % perplexity)
        for j in tqdm(range(len(LEARN_RATE_OPTIONS))):
            learnRate = LEARN_RATE_OPTIONS[j]
            if i == 0:
                ax[0][j].set_title("Learn rate = %d" % learnRate)
            show_tsne(ax[i][j], xPoints, perplexity, learnRate, colours)
    plt.show()


def rawHistograms(ids):
    n = len(ids)
    f, ax = viz.clean_subplots(n, 1, axes=True)

    for i in range(n):
        scanID = ids[i]
        title = "Scan %s" % scanID
        # path = 'data/%s/Normal%s-MRA-FS.mat' % (scanID, scanID)
        path = 'data/tmp/002-%s.mat' % scanID
        # data, _, _, _ = files.loadFeat(path)
        data = files.loadCNN(path)

        data = data.flatten()
        ax[i][0].hist(data, log=True, range=(0, 1), bins=50)
        cp = ax[i][0].twinx()
        cp.set_ylabel(title)
        cp.set_yticklabels([])
        if (i == n - 1):
            ax[i][0].set_xlabel("Intensity")
    f.suptitle("Voxel count by intensity")
    plt.show()


def main():
    # checkEMJVAgreement()
    # emData, jvData = rfAnalysis()
    # emData = files.loadEM().flatten()
    # jvData = files.loadJV().flatten()
    # cnnAnalysis(emData, jvData)
    # simpleTSNE()

    # rawHistograms(['002', '019', '022', '023'])
    rawHistograms(['weighted', 'unweighted'])

if __name__ == '__main__':
    main()
