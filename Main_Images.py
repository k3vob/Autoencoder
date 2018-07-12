import os

import numpy as np

from Model import Autoencoder

projectDir = os.path.dirname(os.path.realpath(__file__))

trainImages = np.loadtxt(projectDir + "/Data/Images/fashion-mnist_train.csv", delimiter=',', skiprows=1)[:, 1:]
testImages = np.loadtxt(projectDir + "/Data/Images/fashion-mnist_test.csv", delimiter=',', skiprows=1)[:, 1:]
numTrain = len(trainImages)
numTest = len(testImages)

learningRate = 0.001
batchSize = 250
numEpochs = 100
tied = True

imgW = imgH = 28
encoderDims = [
    imgW * imgH,
    (imgW // 2) * (imgH // 2),
    (imgW // 3) * (imgH // 3),
    (imgW // 4) * (imgH // 4)
]

codeW = codeH = int(np.sqrt(encoderDims[-1]))

ae = Autoencoder(encoderDims, tied)

for epoch in range(numEpochs):
    epochLoss = 0
    for batch in range(numTrain // batchSize):
        batchInput = trainImages[batch * batchSize: (batch + 1) * batchSize]
        ae.setBatch(batchInput, learningRate)
        batchLoss = ae.run(['loss'], train=True)
        epochLoss += batchLoss

    print("EPOCH:", epoch + 1)
    print("LOSS: ", epochLoss / (numTrain // batchSize), "\n")
