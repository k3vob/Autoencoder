import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

projectDir = os.path.dirname(os.path.realpath(__file__))

testImages = np.loadtxt(projectDir + "/Data/MNIST/emnist-mnist-test.csv", delimiter=',', skiprows=1)[:, 1:]
img = testImages[0]

input = tf.placeholder(tf.float32, [784])
random = tf.random_normal(tf.shape(input))
mask = tf.greater(random, 1.0)
output = tf.where(mask, tf.ones_like(input) * 255, input)

with tf.Session() as session:
    out = session.run(output, {input: img})
    plt.imshow(out.reshape(28, 28), cmap='Greys')
    plt.show()
