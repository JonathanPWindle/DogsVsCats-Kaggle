import tensorflow as tf
import config
import cv2
import os
import dataPreProcessing as data
import numpy as np
from matplotlib import pyplot as plt


# data.prepareData()

# xNormalized.append(data.normalizeData(xBatches[0]))

toloadX = ["./PreProcessed/train/" + i for i in sorted(os.listdir("./PreProcessed/train/")) if "xNormalized" in i]
toloadY = ["./PreProcessed/train/" + i for i in sorted(os.listdir("./PreProcessed/train/")) if "yNormalized" in i]

for i, file in enumerate(toloadX):
    print(file)
    print(toloadY[i])
    loadedX = np.load(file)
    loadedY = np.load(toloadY[i])
    for j, batch in enumerate(loadedX):
        # print(loadedY[j])
        plt.imshow(batch)
        plt.title(loadedY[j])
        plt.show()
#
# print(loaded.shape)

# for batch in xNormalized:
# plt.imshow(loaded[0][1500])
# plt.show()



