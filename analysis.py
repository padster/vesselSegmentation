import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

import files

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
    # run some basic analysis on written results
    emData = files.loadEM().flatten()
    jvData = files.loadJV().flatten()
    rfData = files.loadRF().flatten()
    emData = (emData < 0.5)
    jvData = (jvData < 0.5)
    print ("AUC of RF for EM: %f" % roc_auc_score(emData, rfData))
    print ("AUC of RF for JV: %f" % roc_auc_score(jvData, rfData))
    rfData = (rfData > 0.5)
    print ("Agreement EM-RF: %f" % (np.sum(emData == rfData) / len(rfData)))
    print ("Agreement JV-RF: %f" % (np.sum(jvData == rfData) / len(rfData)))
    print ("Agreement EM-JV: %f" % (np.sum(emData == jvData) / len(rfData)))
    print ("IOU EM-RF:  %f" % intersectOverUnion(emData, rfData))
    print ("IOU JV-RF:  %f" % intersectOverUnion(jvData, rfData))
    print ("IOU EM-JV:  %f" % intersectOverUnion(emData, jvData))



def main():
    checkEMJVAgreement()
    # rfAnalysis()

if __name__ == '__main__':
    main()
