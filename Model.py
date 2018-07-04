import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

projectDir = os.path.dirname(os.path.realpath(__file__))

imgW, imgH = 28, 28
encodingDims = [imgW * imgH, 225, 64]
# encodingDims = [imgW * imgH, imgW * imgH]
decodingDims = list(reversed(encodingDims))

learningRate = 0.01
batchSize = 250
numEpochs = 100000

trainImages = np.loadtxt(projectDir + "/Data/fashion-mnist_train.csv", delimiter=',', skiprows=1)[:, 1:]
testImages = np.loadtxt(projectDir + "/Data/fashion-mnist_test.csv", delimiter=',', skiprows=1)[:, 1:]
numTrain = len(trainImages)
numTest = len(testImages)

encoderWeights, encoderBiases = [], []
decoderWeights, decoderBiases = [], []
for layer in range(len(encodingDims) - 1):
    encoderWeights.append(
        tf.Variable(tf.random_normal([encodingDims[layer], encodingDims[layer + 1]]))
    )
    encoderBiases.append(
        tf.Variable(tf.random_normal([encodingDims[layer + 1]]))
    )
    decoderWeights.append(
        tf.Variable(tf.random_normal([decodingDims[layer], decodingDims[layer + 1]]))
    )
    decoderBiases.append(
        tf.Variable(tf.random_normal([decodingDims[layer + 1]]))
    )

input = tf.placeholder(tf.float32, [None, imgW * imgH])
encoded = input
for layer in range(len(encodingDims) - 1):
    encoded = tf.add(tf.matmul(encoded, encoderWeights[layer]), encoderBiases[layer])
    encoded = tf.nn.sigmoid(encoded)

decoded = encoded
for layer in range(len(decodingDims) - 1):
    decoded = tf.add(tf.matmul(decoded, decoderWeights[layer]), decoderBiases[layer])
    if layer != len(decodingDims) - 2:
        decoded = tf.nn.sigmoid(decoded)

loss = tf.losses.mean_squared_error(labels=input, predictions=decoded)
train = tf.train.AdamOptimizer(learningRate).minimize(loss)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    prevLoss = 99999999999
    for epoch in range(numEpochs):
        epochLoss = 0
        for batch in range(numTrain // batchSize):
            batchInput = trainImages[batch * batchSize: (batch + 1) * batchSize]
            _, batchLoss = session.run([train, loss], {input: batchInput})
            epochLoss += batchLoss

        print("EPOCH:", epoch + 1)
        print("LR:    ", learningRate)
        print("LOSS: ", epochLoss / (numTrain // batchSize), "\n")

        if epoch == 0 or epoch % 50 == 0:
            if epochLoss > prevLoss:
                learningRate = learningRate ** 2
            prevLoss = epochLoss
            rand = random.randint(0, numTrain - 1)
            original, compressed, reconstructed = session.run(
                [input, encoded, decoded], {input: [trainImages[rand]]}
            )
            plt.imshow(original.reshape(imgW, imgH), cmap='Greys')
            plt.savefig(projectDir + "/original.png")
            plt.imshow(compressed.reshape(8, 8), cmap='Greys')
            plt.savefig(projectDir + "/compressed.png")
            plt.imshow(reconstructed.reshape(imgW, imgH), cmap='Greys')
            plt.savefig(projectDir + "/reconstructed.png")
