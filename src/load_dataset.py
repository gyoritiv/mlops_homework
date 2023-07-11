"""This modul is responsible for loading train test split for fashion mnist dataset"""
from keras.datasets import fashion_mnist
from keras.utils import to_categorical


def load_fashion_mnist_dataset():
    """Load dataset using the keras api"""
    return fashion_mnist.load_data()


def load_dataset():
    """Load dataset using the keras api
    Output:
    trainX, testX: Reshaped dataset containing single channel
    trainY, testY: One hot encoded target values
    """
    (trainX, trainY), (testX, testY) = load_fashion_mnist_dataset()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def prep_pixels(train, test):
    """This function is converting the train test split to float type and and normalizing the images"""
    train_norm = train.astype("float32")
    test_norm = test.astype("float32")
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm
