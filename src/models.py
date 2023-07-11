""" This modul is creating convolutional neural network model """
from pathlib import Path
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def cnn_model(checkpoint_path: str):
    """Return cnn model"""
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
    opt = SGD(lr=0.01, momentum=0.9)
    checkpoint = save_checkpoint(checkpoint_path)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model, checkpoint


def save_checkpoint(checkpoint_path):
    """Saving checkpoint to checkpoint path based on best weights"""
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
    )
    earlystop = EarlyStopping(monitor="val_accuracy", patience=5)
    return [checkpoint, earlystop]


def setup_checkpoint_path(path: str):
    """This function is responsible for setting up proper checkpoint file
    Args:
        path: checkpoint file path or None

    Returns:
        Default checkpoint folder path in repo root if path is None else returns the given filepath
    """
    if path is None:
        return str(Path(__file__).parent.absolute() / "checkpoint" / "weights.best.hdf5")
    else:
        return path
