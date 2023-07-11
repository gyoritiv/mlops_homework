from mlops.load_dataset import load_dataset, prep_pixels
from mlops.evaluate_model import evaluate_model


def run():
    trainX, trainY, testX, testY = load_dataset()
    rainX, testX = prep_pixels(trainX, testX)
    scores, histories = evaluate_model(trainX, trainY)


if __name__ == "__main__":
    run()
