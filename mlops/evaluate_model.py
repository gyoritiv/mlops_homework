""" This module is responsible for evaluating the cnn model with k-fold cross-validation """
import time
import logging
from pathlib import Path
from mlops.models import cnn_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)


def evaluate_model(trainX, trainY, testX, testY, checkpoint_file_path=None):
    """evaluating the model"""
    model, callback_list = cnn_model(checkpoint_file_path)
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
    logging.info(f"Accuracy: > {(acc * 100.0)}")
    save_cnn_model(model)
    return history, acc


def save_cnn_model(cnn_model):
    cnn_model_path = str(
        Path(__file__).parent.parent.absolute() / "mnist_model" / f"{int(time.time())}"
    )
    logging.info("Path for saving model: " + cnn_model_path)
    cnn_model.save(filepath=cnn_model_path, save_format="tf")
