import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, Conv1D, MaxPooling1D, Conv1DTranspose, Dropout
from tensorflow.keras.models import Sequential


def normalizeSpecSet(specSet: np.ndarray) -> np.ndarray:
    """
    Normalizing Specset to 0.0 -> 1.0 range, for each spectrum individually
    :param specSet: (N x M) array of N spectra with M wavenumbers
    :return: normalized specset
    """
    for i in range(specSet.shape[0]):
        intens: np.ndarray = specSet[i, :]
        intens -= intens.min()
        if intens.max() != 0:
            intens /= intens.max()
        specSet[i, :] = intens
    return specSet


def prepareSpecSet(specSet: np.ndarray, transpose: bool = True, addDimension: bool = False):
    if transpose:
        specSet = specSet.transpose()

    specSet = normalizeSpecSet(specSet)
    if addDimension:
        specSet = specSet.reshape(specSet.shape[0], specSet.shape[1], 1)

    specSet = tf.cast(specSet, tf.float32)
    return specSet


def getConvReconstructor() -> Sequential:
    from globals import SPECLENGTH
    model: Sequential = Sequential()
    model.add(InputLayer(input_shape=(SPECLENGTH, 1)))
    model.add(Conv1D(32, 4, padding='same', activation="relu"))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Conv1D(32, 4, activation="relu", padding="same"))
    model.add(MaxPooling1D(2, padding="same"))

    model.add(Conv1DTranspose(32, 4, activation="relu", padding="same"))
    model.add(Conv1DTranspose(32, 4, activation='relu', padding="same"))
    model.add(Conv1D(1, 1, activation='relu', padding='same'))
    model.compile(optimizer='adam', loss='mse')
    return model


def getDenseReconstructor(dropout: float = 0.0) -> Sequential:
    from globals import SPECLENGTH

    rec: Sequential = Sequential()
    rec.add(InputLayer(input_shape=(SPECLENGTH)))
    if dropout > 0.0:
        rec.add(Dropout(dropout))
    rec.add(Dense(128, activation="relu"))
    if dropout > 0.0:
        rec.add(Dropout(dropout))
    rec.add(Dense(SPECLENGTH, activation="relu"))

    rec.compile(optimizer='adam', loss='mse')
    return rec
