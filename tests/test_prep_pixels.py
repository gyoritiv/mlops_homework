import numpy as np
from mlops.load_dataset import load_dataset, prep_pixels


def test_pixel_prep():
    """Testing if he normalized pixel values are between 0 and 1"""
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    assert np.max(trainX) <= 1
    assert np.max(trainX) >= 0
    assert np.max(testX) <= 1
    assert np.max(testX) >= 0
