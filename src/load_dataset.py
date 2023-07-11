"""This modul is responsible for loading train test split for fashion mnist dataset"""
from keras.datasets import fashion_mnist


def load_fashion_mnist_dataset():
    return fashion_mnist.load_data()
