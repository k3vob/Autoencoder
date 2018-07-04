import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

projectDir = os.path.dirname(os.path.realpath(__file__))

imgW, imgH = 28, 28
encodingDims = [imgW * imgH, 225, 64]
decodingDims = list(reversed(encodingDims))

learningRate = 0.01
batchSize = 250
numEpochs = 1000

trainImages = np.loadtxt(projectDir + "/Data/fashion-mnist_train.csv", delimiter=',', skiprows=1)[:, 1:]
testImages = np.loadtxt(projectDir + "/Data/fashion-mnist_test.csv", delimiter=',', skiprows=1)[:, 1:]
numTrain = len(trainImages)
numTest = len(testImages)
# plt.imshow(trainImages[0].reshape(imgW, imgH), cmap='Greys')
# plt.show()

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
    # TRY RELU ACTIVATIONS INSTEAD ##################################################

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
        print("LOSS: ", epochLoss, "\n")
