import numpy as np
import matplotlib.pyplot as plt

import files

def checkEMJVAgreement():
    emData = files.loadRF() # loadEM()
    jvData = files.loadJV()
    labels = files.loadLabels()

    rows, cols = labels.shape
    assert cols == 4

    emCorrect, jvCorrect, bothCorrect, eitherCorrect = 0, 0, 0, 0
    emV, jvV, laV = [], [], []

    for row in range(rows):
        x, y, z, label = labels[row]
        emRight = (emData[x-1][y-1][z-1] < 0.5) == (label > 0.5)
        jvRight = (jvData[x-1][y-1][z-1] < 0.5) == (label > 0.5)
        if emRight:
            emCorrect += 1
        if jvRight:
            jvCorrect += 1
        if emRight and jvRight:
            bothCorrect += 1
        if emRight or jvRight:
            eitherCorrect += 1
        emV.append(emData[x-1][y-1][z-1])
        jvV.append(jvData[x-1][y-1][z-1])
        laV.append(label)

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




def main():
    checkEMJVAgreement()

if __name__ == '__main__':
    main()
