import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Model import Autoencoder

numEpochs = 1
batchSize = 10

projectDir = os.path.dirname(os.path.realpath(__file__))

# userId, movieId, rating,
# allratings = np.loadtxt(projectDir + "/Data/Movies/ratings.csv", delimiter=',', skiprows=1)[:, :-1]
allratings = np.loadtxt(projectDir + "/Data/Movies/ratings_small.csv", delimiter=',', skiprows=1)[:, :-1]

df = pd.DataFrame(allratings).pivot_table(index=0, columns=1, values=2, fill_value=0)

trainDF = []
testDF = []

rows = df.as_matrix()
for row in rows:
    train, test = [], []
    nonzero = list(np.nonzero(row)[0])
    rand = random.sample(nonzero, 10)

    for i in range(len(row)):
        if i in rand:
            train.append(0.0)
            test.append(row[i])
        else:
            train.append(row[i])
            test.append(0.0)

    trainDF.append(train)
    testDF.append(test)

trainDF = pd.DataFrame(trainDF)
testDF = pd.DataFrame(testDF)

num = trainDF.shape[0]

# encoderDims = [
#     df.shape[1],
#     df.shape[1] // 2,
#     df.shape[1] // 4,
#     df.shape[1] // 6,
# ]

encoderDims = [df.shape[1], 500]

ae = Autoencoder(encoderDims)

learningRate = 0.01


def runEpoch(data, train):
    epochLoss = 0
    for batch in range(num // batchSize):
        batchInput = np.array(data.iloc[batch * batchSize: (batch + 1) * batchSize])
        ae.setBatch(batchInput, learningRate)
        batchLoss = ae.run(['loss'], train)
        epochLoss += batchLoss
    epochLoss /= (num // batchSize)
    print("TRAIN", end=" ") if train else print("TEST", end="  ")
    print("LOSS:", epochLoss)
    return epochLoss


trainLosses, testLosses = [], []
for epoch in range(numEpochs):
    print("EPOCH", epoch + 1)
    trainLosses.append(runEpoch(trainDF, train=True))
    testLosses.append(runEpoch(testDF, train=False))
    print("")

    plt.plot(trainLosses, label='Training')
    plt.plot(testLosses, label='Test')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(projectDir + "/Data/Movies/Losses.png")
    plt.clf()
