import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

projectDir = os.path.dirname(os.path.realpath(__file__))

learningRate = 0.01
batchSize = 250
numEpochs = 100000

trainImages = np.loadtxt(projectDir + "/Data/fashion-mnist_train.csv", delimiter=',', skiprows=1)[:, 1:]
testImages = np.loadtxt(projectDir + "/Data/fashion-mnist_test.csv", delimiter=',', skiprows=1)[:, 1:]
numTrain = len(trainImages)
numTest = len(testImages)

imgW = imgH = 28
encoderDims = [
    imgW * imgH,
    (imgW // 2) * (imgH // 2),
    (imgW // 3) * (imgH // 3),
    (imgW // 4) * (imgH // 4)
]
codeW = codeH = int(np.sqrt(encoderDims[-1]))

encoderWeights, encoderBiases = [], []
for layer in range(len(encoderDims) - 1):
    encoderWeights.append(
        tf.Variable(tf.random_normal([encoderDims[layer], encoderDims[layer + 1]]))
    )
    encoderBiases.append(
        tf.Variable(tf.random_normal([encoderDims[layer + 1]]))
    )

input = tf.placeholder(tf.float32, [None, imgW * imgH])
encoded = input
for layer in range(len(encoderDims) - 1):
    # encoded = tf.add(tf.matmul(encoded, encoderWeights[layer]), encoderBiases[layer])
    encoded = tf.matmul(encoded, encoderWeights[layer])
    encoded = tf.nn.sigmoid(encoded)

decoded = encoded
for layer in reversed(range(len(encoderDims) - 1)):
    # decoded = tf.add(tf.matmul(decoded, tf.transpose(encoderWeights[layer])), tf.transpose(encoderBiases[layer]))
    decoded = tf.matmul(decoded, tf.transpose(encoderWeights[layer]))
    if layer != len(encoderDims) - 2:
        decoded = tf.nn.sigmoid(decoded)

loss = tf.losses.mean_squared_error(labels=input, predictions=decoded)
train = tf.train.AdamOptimizer(learningRate).minimize(loss)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for epoch in range(numEpochs):
        epochLoss = 0
        for batch in range(numTrain // batchSize):
            batchInput = trainImages[batch * batchSize: (batch + 1) * batchSize]
            _, batchLoss = session.run([train, loss], {input: batchInput})
            epochLoss += batchLoss

        print("EPOCH:", epoch + 1)
        print("LR:   ", learningRate)
        print("LOSS: ", epochLoss / (numTrain // batchSize), "\n")

        if epoch == 0 or epoch % 50 == 0:
            rand = random.randint(0, numTrain - 1)
            original, compressed, reconstructed = session.run(
                [input, encoded, decoded], {input: [trainImages[rand]]}
            )
            plt.imshow(original.reshape(imgW, imgH), cmap='Greys')
            plt.savefig(projectDir + "/original.png")
            plt.clf()
            plt.imshow(compressed.reshape(codeW, codeH), cmap='Greys')
            plt.savefig(projectDir + "/compressed.png")
            plt.clf()
            plt.imshow(reconstructed.reshape(imgW, imgH), cmap='Greys')
            plt.savefig(projectDir + "/reconstructed.png")
            plt.clf()
