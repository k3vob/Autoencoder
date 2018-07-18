import os

import tensorflow as tf


class Autoencoder:

    def __init__(self, encoderDims, scarceInput=False, tiedWeights=False, denoise=False):
        self.encoderDims = encoderDims
        self.decoderDims = list(reversed(encoderDims))
        self.scarceInput = scarceInput
        self.tiedWeights = tiedWeights
        self.denoise = denoise          # Only works for greyscale image data

        self.input = tf.placeholder(tf.float32, [None, encoderDims[0]])
        self.learningRate = tf.placeholder(tf.float32, [])

        self.activationFunction = tf.nn.sigmoid                 # Allow to be specified by user
        # self.activationFunction = tf.tanh
        # self.activationFunction = tf.nn.selu
        self.lossFunction = tf.losses.mean_squared_error        # Allow to be specified by user
        self.SGD = tf.train.AdamOptimizer(self.learningRate)    # Allow to be specified by user

        if self.denoise:
            self.__addNoise()
        self.__buildNetwork()           # Constructs Encoder & Decoder
        self.__buildTensorFlowGraph()   # Creates sequential TensorFlow operations

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())     # Initialise weights & biases
        self.saver = tf.train.Saver()
        self.session.graph.finalize()                           # Avoids memory leaks through duplicating graph nodes

    def __addNoise(self):
        # Create a tensor of random numbers with unit variance
        # Then sets pixels to black where values of random tensor > 1
        # (i.e. all values outside the std dev -> ~32% of pixels)
        random = tf.random_normal(tf.shape(self.input))
        mask = tf.greater(random, 1.0)
        self.noisyInput = tf.where(mask, tf.ones_like(self.input) * 255, self.input)

    def __buildNetwork(self):
        # Lists of weights and biases per layer of encoder and decoder
        self.encoderWeights, self.encoderBiases = [], []
        self.decoderWeights, self.decoderBiases = [], []
        for layer in range(len(self.encoderDims) - 1):
            self.encoderWeights.append(
                tf.Variable(tf.random_normal([self.encoderDims[layer], self.encoderDims[layer + 1]]))
            )
            self.encoderBiases.append(
                tf.Variable(tf.zeros([self.encoderDims[layer + 1]]))
            )
            # if layer != len(self.decoderDims) - 2:  # NO BIAS IN OUTPUT LAYER ##################################
            self.decoderBiases.append(
                tf.Variable(tf.zeros([self.decoderDims[layer + 1]]))
            )
            if not self.tiedWeights:
                self.decoderWeights.append(
                    tf.Variable(tf.random_normal([self.decoderDims[layer], self.decoderDims[layer + 1]]))
                )
        if self.tiedWeights:
            self.decoderWeights = [tf.transpose(i) for i in reversed(self.encoderWeights)]

    def __buildTensorFlowGraph(self):
        self.encoded = self.encode()        # Encoded/compressed data
        self.decoded = self.decode()        # Decoded/reconstructed data
        self.loss = self.__calculateLoss()
        self.train = self.SGD.minimize(self.loss)

    def encode(self):
        if self.denoise:
            encoded = self.noisyInput
        else:
            encoded = self.input
        for layer in range(len(self.encoderDims) - 1):
            encoded = tf.matmul(encoded, self.encoderWeights[layer])
            encoded = tf.add(encoded, self.encoderBiases[layer])
            if layer != len(self.encoderDims) - 2:
                encoded = self.activationFunction(encoded)
        return encoded

    def decode(self):
        decoded = self.encoded
        for layer in range(len(self.decoderDims) - 1):
            decoded = tf.matmul(decoded, self.decoderWeights[layer])
            decoded = tf.add(decoded, self.decoderBiases[layer])    # Bias in final layer????
            if layer != len(self.decoderDims) - 2:                  # Keep output layer linear
                decoded = self.activationFunction(decoded)
        return decoded

    def __calculateLoss(self):
        if self.scarceInput:
            nonZeros = tf.where(tf.greater(self.input, 0))            # Only calculates RMSE on
            return tf.sqrt(                                           # non-zero input values
                self.lossFunction(
                    labels=tf.gather(self.input, nonZeros),
                    predictions=tf.gather(self.decoded, nonZeros)
                )
            )
        else:
            return self.lossFunction(labels=self.input, predictions=self.decoded)

    def setBatch(self, input, learningRate=0.0):
        self.batchDict = {
            self.input: input,
            self.learningRate: learningRate
        }

    def run(self, operations=None, train=False):
        # Returns values of specified list of operations
        # Trains network's parameters if specified
        if not type(operations) is list:
            operations = [operations]

        if train:
            ops = [self.train]
        else:
            ops = []

        if operations is not None:
            for op in operations:
                if op == 'input':
                    ops.append(self.input)
                if op == 'noisyInput':
                    ops.append(self.noisyInput)
                if op == 'encoded':
                    ops.append(self.encoded)
                if op == 'decoded':
                    ops.append(self.decoded)
                if op == 'loss':
                    ops.append(self.loss)

        if (train and len(ops) == 2) or (not train and len(ops) == 1):
            return self.session.run(ops, self.batchDict)[-1]
        elif train:
            return self.session.run(ops, self.batchDict)[1:]
        else:
            return self.session.run(ops, self.batchDict)

    def save(self, modelName="Autoencoder"):
        modelName += '.ckpt'
        dir = os.path.dirname(os.path.realpath(__file__)) + '/SavedModels/'
        self.saver.save(self.session, dir + modelName)

    def restore(self, modelName="Autoencoder"):
        modelName += '.ckpt'
        dir = os.path.dirname(os.path.realpath(__file__)) + '/SavedModels/'
        self.saver.restore(self.session, dir + modelName)

    def kill(self):
        self.session.close()
