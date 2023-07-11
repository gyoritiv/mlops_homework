""" This module is responsible for evaluating the cnn model with k-fold cross-validation """
import logging
from sklearn.model_selection import KFold
from models import cnn_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)


def evaluate_model(dataX, dataY, checkpoint_file_path=None, n_folds=5):
    """evaluating the model"""
    scores, histories = list(), list()
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(dataX):
        model, callback_list = cnn_model(checkpoint_file_path)
        trainX, trainY, testX, testY = (
            dataX[train_ix],
            dataY[train_ix],
            dataX[test_ix],
            dataY[test_ix],
        )
        history = model.fit(
            trainX,
            trainY,
            epochs=10,
            batch_size=32,
            callbacks=callback_list,
            validation_data=(testX, testY),
            verbose=0,
        )
        _, acc = model.evaluate(testX, testY, verbose=0)
        logging.info(f"> {(acc * 100.0)}")
        scores.append(acc)
        histories.append(history)
    return scores, histories