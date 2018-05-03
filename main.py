import tensorflow as tf
import config
import pandas as pd
import cv2
import os
import dataPreProcessing as data
import numpy as np
from matplotlib import pyplot as plt
import modelUtils as model

# data.prepareData()

toLoadX = ["./PreProcessed/train/" + i for i in sorted(os.listdir("./PreProcessed/train/")) if "xNormalized" in i]
toLoadY = ["./PreProcessed/train/" + i for i in sorted(os.listdir("./PreProcessed/train/")) if "yNormalized" in i]
toLoadTesting = ["./PreProcessed/test/" + i for i in sorted(os.listdir("./PreProcessed/test/")) if
                 "xTestNormalized" in i]
toLoadValidX = ["./PreProcessed/train/" + i for i in sorted(os.listdir("./PreProcessed/train")) if
                "valXNormalized" in i]
toLoadValidY = ["./PreProcessed/train/" + i for i in sorted(os.listdir("./PreProcessed/train")) if
                "valYNormalized" in i]
# test = np.load("./PreProcessed/test/xTestNormalized0.npy")

# print(test.shape)

xTraining = []
yTraining = []
xValidation = []
yValidation = []
xTesting = []
for i, file in enumerate(toLoadX):
    loadedX = np.load(file)
    loadedY = np.load(toLoadY[i])
    xTraining.append(np.array_split(loadedX, len(loadedX) / 50))
    yTraining.append(np.array_split(loadedY, len(loadedY) / 50))

for i, file in enumerate(toLoadTesting):
    loadedTest = np.load(file)
    xTesting.append(np.array_split(loadedTest, len(loadedTest) / 50))

for i, file in enumerate(toLoadValidX):
    loadedValidX = np.load(file)
    loadedValidY = np.load(toLoadValidY[i])
    xValidation.append(np.array_split(loadedValidX, len(loadedValidX) / 50))
    yValidation.append(np.array_split(loadedValidY, len(loadedValidY) / 50))
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

conv3Weights = model.weightVariable([5, 5, 64, 32])
conv3Bias = model.biasVariable([32])

conv3Output = tf.nn.relu(model.conv2D(conv2Output, conv3Weights) + conv3Bias)

pooling2Output = model.maxPool2x2(conv3Output)

#
# Full connected layer
#
fullyCon1Weights = model.weightVariable([16 * 16 * 32, 1024])
fullCon1Bias = model.biasVariable([1024])

pool2Flattened = tf.reshape(pooling2Output, [-1, 16 * 16 * 32])

fullyConOutput = tf.nn.relu(tf.matmul(pool2Flattened, fullyCon1Weights) + fullCon1Bias)

#
# Apply dropout to avoid overfitting
#
keepProb = tf.placeholder(dtype=tf.float32)
fullyConDrop = tf.nn.dropout(fullyConOutput, keepProb)

fullyCon2Weights = model.weightVariable([1024, 512])
fullyCon2Bias = model.biasVariable([512])

fullyCon2Output = tf.nn.relu(tf.matmul(fullyConDrop, fullyCon2Weights) + fullyCon2Bias)
#
# Readout layer
#

fullyCon3Weights = model.weightVariable([512, 2])
fullyCon3Bias = model.biasVariable([2])

yConv = tf.matmul(fullyCon2Output, fullyCon3Weights) + fullyCon3Bias

crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yConv))
trainStep = tf.train.AdamOptimizer(config.RATE).minimize(crossEntropy)

correctPrediction = tf.equal(tf.arg_max(yConv, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

prediction = tf.nn.softmax(yConv)

# prediction = tf.arg_max(yConv, 1)
# prediction = "Dog" if tf.argmax(yConv, 1) == 0 else "Cat"

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(config.EPOCHS):
        randomNum1 = np.random.randint(0, 8)
        randomNum2 = np.random.randint(0, 38)
        randomNum3 = np.random.randint(0, 48)
        if i % config.DISPLAY == 0:
            # trainAccuracy = accuracy.eval(feed_dict={x: xTraining[randomNum1][randomNum2], \
            #                                          y: yTraining[randomNum1][randomNum2], keepProb: 1.0})
            total = 0
            count = 0
            for j, batches in enumerate(xValidation):
                for k, batch in enumerate(batches):
                    count = count + 1
                    total = total + accuracy.eval(feed_dict={x: batch, y: yValidation[j][k], keepProb: 1.0})

            print("Step %d, training accuracy: %g" % (i, total / count))
            savePath = saver.save(sess, "./Logs/Run2/trainedModel" + str(i) + "/" + str(i) + ".ckpt")

        for j, batches in enumerate(xTraining):
            for k, batch in enumerate(batches):
                trainStep.run(feed_dict={x: batch, \
                                         y: yTraining[j][k], keepProb: 0.5})
    # saver.restore(sess, "./Logs/Run1/trainedModel19900/19900.ckpt")
    #
    # total = 0
    # count = 0
    # for i, batch in enumerate(xValidation):
    #     for j, batches in enumerate(batch):
    #         count = count + 1
    #         total = total + accuracy.eval(feed_dict={x: batches, y: yValidation[i][j], keepProb: 1.0})

    # print(total/count)

    # giles = data.normalizeData(["./Data/giles.png"])
    # pred = "Dog" if prediction.eval(feed_dict={x: giles, keepProb: 1.0}) == 0 else "Cat"
    # img = cv2.imread("./Data/giles.png")
    # img = img[:,:,::-1]
    # plt.imshow(img)
    # plt.title(pred)
    # plt.show()

    predictions = []
    for batches in xTesting:
        print("BATCH")
        for batch in batches:
            # print(prediction.eval(feed_dict={x: batch, keepProb: 1.0}))
            predictions.extend(np.array(prediction.eval(feed_dict={x: batch, keepProb: 1.0})))

    ids = []
    finalPredicts = []
    for i in predictions:
        finalPredicts.append(i[0])

    for i in range(1, len(predictions) + 1):
        ids.append(i)

    data = {"id": ids, "label": finalPredicts}

    df = pd.DataFrame(data=data)
    # df['label'] = df['label'].round(7)
    df.to_csv('Results/results.csv', index=False)
