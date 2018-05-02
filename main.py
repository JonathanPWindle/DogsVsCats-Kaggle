import tensorflow as tf
import config
import cv2
import os
import dataPreProcessing as data
import numpy as np
from matplotlib import pyplot as plt
import modelUtils as model

toloadX = ["./PreProcessed/train/" + i for i in sorted(os.listdir("./PreProcessed/train/")) if "xNormalized" in i]
toloadY = ["./PreProcessed/train/" + i for i in sorted(os.listdir("./PreProcessed/train/")) if "yNormalized" in i]

xTraining = []
yTraining = []

for i, file in enumerate(toloadX):
    loadedX = np.load(file)
    loadedY = np.load(toloadY[i])
    xTraining.append(np.array_split(loadedX, len(loadedX) / 50))
    yTraining.append(np.array_split(loadedY, len(loadedY) / 50))

print(np.array(xTraining).shape)
# print(np.array(yTraining).shape)

# for i, file in enumerate(toloadX):
#     print(file)
#     print(toloadY[i])
#     loadedX = np.load(file)
#     loadedY = np.load(toloadY[i])
#     for j, batch in enumerate(loadedX):
#         plt.imshow(batch)
#         plt.title(loadedY[j])
#         plt.show()

# To hold the image image
x = tf.placeholder(tf.float32, shape=[None, config.IMAGE_SIZE, config.IMAGE_SIZE, config.CHANNELS])
# To hold the output
y = tf.placeholder(tf.float32, shape=[None, 2])

#
# First convolution layer
# Compute 32 features for 5x5 patch
# Weight shape = [5, 5, 3, 32]
#

conv1Weights = model.weightVariable([5, 5, 3, 32])
conv1Bias = model.biasVariable([32])

# Ouput
conv1Output = tf.nn.relu(model.conv2D(x, conv1Weights) + conv1Bias)

# Max Pool
pooling1Output = model.maxPool2x2(conv1Output)


#
# Second convolution layer
# Compute 64 features for 5x5 patch
# Weight shape = [5, 5, 32, 64]
#
conv2Weights = model.weightVariable([5, 5, 32, 64])
conv2Bias = model.biasVariable([64])

conv2Output = tf.nn.relu(model.conv2D(pooling1Output, conv2Weights) + conv2Bias)

pooling2Output = model.maxPool2x2(conv2Output)

#
# Full connected layer
#
fullyCon1Weights = model.weightVariable([16 * 16 * 64, 1024])
fullCon1Bias = model.biasVariable([1024])

pool2Flattened = tf.reshape(pooling2Output, [-1, 16 * 16 * 64])

fullyConOutput = tf.nn.relu(tf.matmul(pool2Flattened, fullyCon1Weights) + fullCon1Bias)

#
# Apply dropout to avoid overfitting
#
keepProb = tf.placeholder(dtype=tf.float32)
fullyConDrop = tf.nn.dropout(fullyConOutput, keepProb)

#
# Readout layer
#

fullyCon2Weights = model.weightVariable([1024, 2])
fullyCon2Bias = model.biasVariable([2])

yConv = tf.matmul(fullyConDrop, fullyCon2Weights) + fullyCon2Bias


crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits= yConv))
trainStep = tf.train.AdamOptimizer(config.RATE).minimize(crossEntropy)
correctPrediction = tf.equal(tf.arg_max(yConv, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

prediction = tf.arg_max(yConv, 1)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(config.EPOCHS):
        randomNum1 = np.random.randint(0, 8)
        randomNum2 = np.random.randint(0, 38)
        randomNum3 = np.random.randint(0, 48)
        if i % config.DISPLAY == 0:
            trainAccuracy = accuracy.eval(feed_dict={x: xTraining[randomNum1][randomNum2], \
                                                     y: yTraining[randomNum1][randomNum2], keepProb: 1.0})
            print("Step %d, training accuracy: %g" % (i, trainAccuracy))
        trainStep.run(feed_dict={x: xTraining[randomNum1][randomNum2], \
                                 y: yTraining[randomNum1][randomNum2], keepProb:0.5})








