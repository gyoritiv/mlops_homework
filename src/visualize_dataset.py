"""This modul is responsible for visualize the dataset"""
from load_dataset import load_fashion_mnist_dataset
from matplotlib import pyplot


def visualize_dataset():
    """This function is responsible for visualize the dataset"""
    (trainX, _), (_, _) = load_fashion_mnist_dataset()
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(trainX[i], cmap=pyplot.get_cmap("gray"))
    pyplot.show()


if __name__ == "__main__":
    visualize_dataset()
