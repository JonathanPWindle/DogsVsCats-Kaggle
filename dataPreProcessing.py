import cv2
import config
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

xScaler = MinMaxScaler(feature_range=(0, 1))
rTotal = 0
gTotal = 0
bTotal = 0
rgbMean = 0
rgbStdDev = 0


#
# Read image from a file, apply some process to normalize the image size
# Ensure the image is square, IMAGE_SIZE x IMAGE_SIZE
# Apply solid black border to fill in gaps
# filePath: File path pointing to the image to read
# Returns: RGB representation of the image
#
def readImage(filePath):
    # Read from file path, 1 = read in color
    img = cv2.imread(filePath, 1)

    # if height > width: Determine scaling factor to use
    if img.shape[0] >= img.shape[1]:
        resizeto = (config.IMAGE_SIZE, int(round(config.IMAGE_SIZE * float(img.shape[1]) / img.shape[0])))
    else:
        resizeto = (int(round(config.IMAGE_SIZE * float(img.shape[0]) / img.shape[1])), config.IMAGE_SIZE)

    # Resize the image to not exceed the IMAGE_SIZE, use INTER_CUBIC to interpolate
    img2 = cv2.resize(img, (resizeto[1], resizeto[0]), interpolation=cv2.INTER_CUBIC)
    # Add padding (black border) to the image where needed
    img3 = cv2.copyMakeBorder(img2, 0, config.IMAGE_SIZE - img2.shape[0], 0, config.IMAGE_SIZE - img2.shape[1], \
                              cv2.BORDER_CONSTANT, 0)

    # returns image in an RGB format (read in as BGR, must reverse all to be RGB) this is for matplotlib
    return img3[:, :, ::-1]


#
# Uses each RGB channels mean and std dev to normalise the data (from every channel, take the mean and divide by stddev
#
def normalizeData(images):
    numberOfImages = len(images)
    data = np.ndarray((numberOfImages, config.IMAGE_SIZE, config.IMAGE_SIZE, config.CHANNELS), dtype=np.float32)

    for i, imageFile in enumerate(images):
        image = readImage(imageFile)
        imageData = np.array(image, dtype=np.float32)

        imageData[:, :, 2] = (imageData[:, :, 2].astype(float) - config.B_AVERAGE) / config.B_STDDEV
        imageData[:, :, 1] = (imageData[:, :, 1].astype(float) - config.G_AVERAGE) / config.G_STDDEV
        imageData[:, :, 0] = (imageData[:, :, 0].astype(float) - config.R_AVERAGE) / config.R_STDDEV

        data[i] = imageData

    return data


def shuffleData(images, labels):
    permutation = np.random.permutation(len(labels))
    shuffledImages = images[permutation]
    shuffledLabels = labels[permutation]

    return {"shuffledImages": shuffledImages, "shuffledLabels": shuffledLabels}


def prepareData():
    # Populate arrays with file paths for the data
    trainingImages = [config.TRAIN_DIR + i for i in os.listdir(config.TRAIN_DIR)]
    trainingDogs = [config.TRAIN_DIR + i for i in os.listdir(config.TRAIN_DIR) if "dog" in i]
    trainingCat = [config.TRAIN_DIR + i for i in os.listdir(config.TRAIN_DIR) if "cat" in i]
    testImages = [config.TEST_DIR + str(i) + '.jpg' for i in range(1,12501)]

    # Ensure the ratio of dogs and cats are accurate
    trainingImages = trainingDogs[:config.TRAIN_DOG_SIZE] + trainingCat[:config.TRAIN_CAT_SIZE]
    # Add labels that correspond to trainingImages
    trainingLabels = np.array((['dogs'] * config.TRAIN_DOG_SIZE) + (['cats'] * config.TRAIN_CAT_SIZE))
    # Add test image filepaths
    testImages = testImages[:config.TEST_SIZE]
    testImages = np.array(testImages)
    shuffled = shuffleData(np.array(trainingImages), np.array(trainingLabels))

    xTraining = shuffled["shuffledImages"]
    yTraining = shuffled["shuffledLabels"]
    xBatches = []
    yBatches = []
    valXBatches = []
    valYBatches = []
    testBatches = []


    for i in range(0, len(xTraining), config.ALL_TRAIN_SIZE):
        xBatches.append(xTraining[i:i + config.TRAIN_SIZE])
        yBatches.append(yTraining[i:i + config.TRAIN_SIZE])
        valXBatches.append(xTraining[i + config.TRAIN_SIZE: i + config.TRAIN_SIZE + config.VALID_SIZE])
        valYBatches.append(yTraining[i + config.TRAIN_SIZE: i + config.TRAIN_SIZE + config.VALID_SIZE])

    for i in range(0, len(testImages), config.ALL_TRAIN_SIZE):
        testBatches.append(testImages[i: i + config.ALL_TRAIN_SIZE])
    xBatches = np.array(xBatches)
    yBatches = np.array(yBatches)
    valYBatches = np.array(valYBatches)
    valXBatches = np.array(valXBatches)
    testBatches = np.array(testBatches)

    # These will store the labels after one-hot encoding has been applied
    yEncoded = np.ndarray([len(xBatches), config.TRAIN_SIZE, 2])
    valYEncoded = np.ndarray([len(valXBatches), config.VALID_SIZE, 2])

    # Normalize the batches and save out to numpy arrays for later use
    for i, batch in enumerate(xBatches):
        np.save(config.PRE_PROCESS_DIR + "train/xNormalized" + str(i) + ".npy", normalizeData(batch))
        yBatches[i] = (yBatches[i] != "dogs").astype(np.float32)
        yEncoded[i] = (np.arange(2) == yBatches[i][:, None].astype(np.float32)).astype(np.float32)
        np.save(config.PRE_PROCESS_DIR + "train/yNormalized" + str(i) + ".npy", yEncoded[i])

    for i, batch in enumerate(valXBatches):
        np.save(config.PRE_PROCESS_DIR + "train/valXNormalized" + str(i) + ".npy", normalizeData(batch))
        valYBatches[i] = (valYBatches[i] != "dogs").astype(np.float32)
        valYEncoded[i] = (np.arange(2) == valYBatches[i][:, None].astype(np.float32)).astype(np.float32)
        np.save(config.PRE_PROCESS_DIR + "train/valYNormalized" + str(i) + ".npy", valYEncoded[i])

    for i, batch in enumerate(testBatches):
        np.save(config.PRE_PROCESS_DIR + "test/xTestNormalized" + str(i) + ".npy", normalizeData(batch))


#
# Calculate the mean and std deviation for each channel over the whole data set
#
def calcMeanAndStdDev(batches):
    rAverages = []
    gAverages = []
    bAverages = []

    for batch in batches:
        for path in batch:
            image = readImage(path)
            imageData = np.array(image, dtype=np.float32)
            rAverages.append(np.average(imageData[:,:,2]))
            gAverages.append(np.average(imageData[:,:,1]))
            bAverages.append(np.average(imageData[:,:,0]))

    rAverage = np.average(np.array(rAverages))
    gAverage = np.average(np.array(gAverages))
    bAverage = np.average(np.array(bAverages))
    rStdDev = np.std(np.array(rAverages))
    gStdDev = np.std(np.array(gAverages))
    bStdDev = np.std(np.array(bAverages))

    print(rAverage)
    print(gAverage)
    print(bAverage)
    print(rStdDev)
    print(gStdDev)
    print(bStdDev)


def calculateMean(batch):
    rAverage = 0
    gAverage = 0
    bAverage = 0

    for images in batch:

        for i in images:
            image = readImage(i)
            imageData = np.array(image, dtype=np.float32)

            rTotal = np.sum(imageData[:,:,2])
            gTotal = np.sum(imageData[:,:,1])
            bTotal = np.sum(imageData[:,:,0])


            rAverage = rAverage + (rTotal / (config.IMAGE_SIZE * config.IMAGE_SIZE))
            gAverage = gAverage + (gTotal / (config.IMAGE_SIZE * config.IMAGE_SIZE))
            bAverage = bAverage + (bTotal / (config.IMAGE_SIZE * config.IMAGE_SIZE))

    rAverage = rAverage / config.TRAINING_SET_SIZE
    gAverage = gAverage / config.TRAINING_SET_SIZE
    bAverage = bAverage / config.TRAINING_SET_SIZE
    print(rAverage)
    print(gAverage)
    print(bAverage)
