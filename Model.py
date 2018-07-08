import tensorflow as tf


class Autoencoder:

    def __init__(self, encoderDims):
        self.encoderDims = encoderDims
        self.decoderDims = list(reversed(encoderDims))
        self.input = tf.placeholder(tf.float32, [None, encoderDims[0]])

        self.session = tf.Session()
        self.activationFunction = tf.nn.sigmoid             # Allow to be specified
        self.lossFunction = tf.losses.mean_squared_error    # Allow to be specified
        self.learningRate = tf.placeholder(tf.float32, [])
        self.SGD = tf.train.AdamOptimizer(self.learningRate)

        self.__buildNetwork()
        self.__buildTensorFlowGraph()

        self.session.run(tf.global_variables_initializer())
        self.session.graph.finalize()

    def __buildNetwork(self):
        self.encoderWeights, self.encoderBiases = [], []
        self.decoderWeights, self.decoderBiases = [], []
        for layer in range(len(self.encoderDims) - 1):
            self.encoderWeights.append(
                tf.Variable(tf.random_normal([self.encoderDims[layer], self.encoderDims[layer + 1]]))
            )
            self.encoderBiases.append(
                tf.Variable(tf.random_normal([self.encoderDims[layer + 1]]))
            )
            self.decoderWeights.append(
                tf.Variable(tf.random_normal([self.decoderDims[layer], self.decoderDims[layer + 1]]))
            )
            self.decoderBiases.append(
                tf.Variable(tf.random_normal([self.decoderDims[layer + 1]]))
            )

    def __buildTensorFlowGraph(self):
        self.encoded = self.encode()
        self.decoded = self.decode()
        self.loss = self.__calculateLoss()
        self.train = self.SGD.minimize(self.loss)

    def encode(self):
        encoded = self.input
        for layer in range(len(self.encoderDims) - 1):
            encoded = tf.add(tf.matmul(encoded, self.encoderWeights[layer]), self.encoderBiases[layer])
            encoded = self.activationFunction(encoded)
        return encoded

    def decode(self):
        decoded = self.encoded
        for layer in range(len(self.decoderDims) - 1):
            decoded = tf.add(tf.matmul(decoded, self.decoderWeights[layer]), self.decoderBiases[layer])
            if layer != len(self.decoderDims) - 2:
                decoded = self.activationFunction(decoded)
        return decoded

    def __calculateLoss(self):
        return self.lossFunction(labels=self.input, predictions=self.decoded)

    def setBatch(self, input, learningRate=0.0):
        self.batchDict = {
            self.input: input,
            self.learningRate: learningRate
        }

    def run(self, operations=None, train=False):
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
