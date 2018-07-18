import os
import random

import matplotlib.pyplot as plt
import numpy as np

from Model import Autoencoder

projectDir = os.path.dirname(os.path.realpath(__file__))

learningRate = 0.001
batchSize = 250
numEpochs = 10000
tied = False
denoise = True
step = 50
newModel = True

# trainImages = np.loadtxt(projectDir + "/Data/Images/fashion-mnist_train.csv", delimiter=',', skiprows=1)[:, 1:]
# testImages = np.loadtxt(projectDir + "/Data/Images/fashion-mnist_test.csv", delimiter=',', skiprows=1)[:, 1:]
trainImages = np.loadtxt(projectDir + "/Data/MNIST/emnist-mnist-train.csv", delimiter=',', skiprows=1)[:, 1:]
testImages = np.loadtxt(projectDir + "/Data/MNIST/emnist-mnist-test.csv", delimiter=',', skiprows=1)[:, 1:]
numTrain = len(trainImages)
numTest = len(testImages)
numTrainBatches = numTrain // batchSize
numTestBatches = numTest // batchSize

imgW = imgH = 28
encoderDims = [
    imgW * imgH,
    (imgW // 2) * (imgH // 2),
    (imgW // 3) * (imgH // 3),
    (imgW // 4) * (imgH // 4)     # Last dim must be a square number
]

codeW = codeH = int(np.sqrt(encoderDims[-1]))

ae = Autoencoder(encoderDims, tied, denoise)

bestLoss = 9999
if not newModel:
    ae.restore()

for epoch in range(numEpochs):
    epochLoss = 0
    for batch in range(numTrainBatches):
        batchInput = trainImages[batch * batchSize: (batch + 1) * batchSize]
        ae.setBatch(batchInput, learningRate)
        batchLoss = ae.run(['loss'], train=True)
        epochLoss += batchLoss

    epochLoss /= numTrainBatches
    print("EPOCH:", epoch + 1)
    print("LOSS: ", epochLoss, "\n")

    if epochLoss < bestLoss:
        ae.save()

    if epoch % step == 0:
        testLoss = 0
        for batch in range(numTestBatches):
            batchInput = testImages[batch * batchSize: (batch + 1) * batchSize]
            ae.setBatch(batchInput)
            batchLoss = ae.run(['loss'])
            testLoss += batchLoss

        testLoss /= numTestBatches
        print("TEST LOSS: ", testLoss, "\n")

        rand = random.randint(0, numTest - 1)
        ae.setBatch([testImages[rand]])
        original, compressed, reconstructed = \
            ae.run(['noisyInput', 'encoded', 'decoded'], train=False)
        plt.imshow(original.reshape(imgW, imgH), cmap='Greys')
        plt.savefig(projectDir + "/Data/Images/originalTest.png")
        plt.clf()
        plt.imshow(compressed.reshape(codeW, codeH), cmap='Greys')
        plt.savefig(projectDir + "/Data/Images/compressedTest.png")
        plt.clf()
        plt.imshow(reconstructed.reshape(imgW, imgH), cmap='Greys')
        plt.savefig(projectDir + "/Data/Images/reconstructedTest.png")
        plt.clf()
