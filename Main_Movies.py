import os

import numpy as np
import pandas as pd

from Model import Autoencoder

numEpochs = 1000
batchSize = 1

projectDir = os.path.dirname(os.path.realpath(__file__))

# userId, movieId, rating,
# allratings = np.loadtxt(projectDir + "/Data/Movies/ratings.csv", delimiter=',', skiprows=1)[:, :-1]
allratings = np.loadtxt(projectDir + "/Data/Movies/ratings_small.csv", delimiter=',', skiprows=1)[:, :-1]

df = pd.DataFrame(allratings).pivot_table(index=0, columns=1, values=2, fill_value=0)

trainDF = df[:int(df.shape[0] * 0.8)]
testDF = df[int(df.shape[0] * 0.8):]

numTrain = trainDF.shape[0]
numTest = testDF.shape[0]

encoderDims = [
    df.shape[1],
    df.shape[1] // 2,
    df.shape[1] // 4,
    df.shape[1] // 6,
]

# encoderDims = [
#     df.shape[1],
#     128,
#     128,
#     128
# ]

ae = Autoencoder(encoderDims)

learningRate = 0.0001
bestLoss = 0.29
ae.restore()


trainDF = df    # ########################################
numTrain = df.shape[0]


for epoch in range(numEpochs):
    epochLoss = 0
    a, b = [], []
    for batch in range(numTrain // batchSize):
        batchInput = np.array(trainDF.iloc[batch * batchSize: (batch + 1) * batchSize])
        ae.setBatch(batchInput, learningRate)
        batchLoss, original, reconstructed = ae.run(['loss', 'input', 'decoded'], train=True)
        epochLoss += batchLoss
        a.append(np.mean(original))
        b.append(np.mean(reconstructed))
    epochLoss /= (numTrain // batchSize)
    print("EPOCH:", epoch + 1)
    print("LR:   ", learningRate)
    print("LOSS: ", epochLoss, "\n")
    if epochLoss < bestLoss:
        ae.save()

    print(np.mean(a))
    print(np.mean(b), "\n")
