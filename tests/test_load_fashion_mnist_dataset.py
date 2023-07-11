from mlops.load_dataset import load_fashion_mnist_dataset


def test_load_data():
    (trainX, trainY), (testX, testY) = load_fashion_mnist_dataset()
    assert trainX is not None
    assert trainY is not None
    assert testX is not None
    assert testY is not None
