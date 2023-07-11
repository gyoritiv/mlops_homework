from load_dataset import load_dataset, prep_pixels
from evaluate_model import evaluate_model


def run():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    scores, histories = evaluate_model(trainX, trainY)


if __name__ == "__main__":
    run()
