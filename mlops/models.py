""" This modul is creating convolutional neural network model """
import logging
from pathlib import Path
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)


def setup_default_model():
    """Setup default model properties and return the model"""
    model = Sequential()
    model.add(
        Conv2D(
            32, (3, 3), activation="relu", kernel_initializer="he_uniform", input_shape=(28, 28, 1)
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(10, activation="softmax"))
    return model


def cnn_model(checkpoint_path: str):
    """Return cnn model"""
    model = setup_default_model()
    checkpoint_path = setup_checkpoint_path(checkpoint_path)
    if Path(checkpoint_path).exists():
        logging.info(f"Checkpoint already exist, loading weights from path: {checkpoint_path}")
        model.load_weights(checkpoint_path)
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    checkpoint = save_checkpoint(checkpoint_path)
    return model, checkpoint


def load_weights(model, checkpoint_path: str):
    """Load weights for the ml model

    Args:
        model (Sequential): keras model
        checkpoint_path (str): Path of the checkpoint file.

    Returns:
        model (Sequential): keras model containing the weights coming from hdf file"""
    return model.load_weights(checkpoint_path)


def save_checkpoint(checkpoint_path: str):
    """Saving checkpoint to checkpoint path based on best weights

    Args:
        checkpoint_path (str): Path of the checkpoint file.

    Returns:
        list: checkpoint and earlystop callback list for the fiting"""
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
    )
    earlystop = EarlyStopping(monitor="val_accuracy", patience=4)
    return [checkpoint, earlystop]


def setup_checkpoint_path(path: str):
    """This function is responsible for setting up proper checkpoint file
    Args:
        path: checkpoint file path or None

    Returns:
        Default checkpoint folder path in repo root if path is None else returns the given filepath
    """
    if path is None:
        path = str(Path(__file__).parent.parent.absolute() / "checkpoint" / "weights.best.hdf5")
        logging.info(f"Path for checkpoint not provided, using the default path: {path}")
        return path
    else:
        return path
