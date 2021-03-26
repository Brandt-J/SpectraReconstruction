import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from kerastuner.tuners import RandomSearch
import numpy as np


def preprocessSpec(intensities: np.ndarray) -> np.ndarray:
    intensities -= intensities.min()
    if intensities.max() != 0.0:
        intensities /= intensities.max()
    return intensities


def prepareSpecSet(specSet: np.ndarray, transpose: bool = True):
    if transpose:
        for i in range(specSet.shape[1]):
            specSet[:, i] = preprocessSpec(specSet[:, i])
    else:
        for i in range(specSet.shape[0]):
            specSet[i, :] = preprocessSpec(specSet[i, :])

    if transpose:
        specSet = specSet.transpose()
    specSet = tf.cast(specSet, tf.float32)
    return specSet


def optimizeRec(X_train, y_train, X_test, y_test):
    tuner = RandomSearch(
        getReconstructor,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='ATR_Denoising_Search')

    tuner.search(x=X_train,
                 y=y_train,
                 epochs=5,
                 validation_data=(X_test, y_test))
    return tuner


def getReconstructor(hp=None) -> Sequential:
    from globals import SPECLENGTH
    if hp is None:
        latentDims = 128
        numLayers = 2
    else:
        latentDims: int = hp.Int("n_latentDims", 32, 128, 32)
        numLayers: int = hp.Int("n_layers", 1, 3)
    layerDims: np.ndarray = np.linspace(np.log(latentDims), np.log(SPECLENGTH), numLayers+1, endpoint=False)
    layerDimsEnc: np.ndarray = np.uint16(np.round(np.exp(layerDims[1:])))
    layerDimsDec: np.ndarray = layerDimsEnc[::-1]

    rec: Sequential = Sequential()
    for i in range(numLayers):
        rec.add(Dense(layerDimsEnc[i], activation="relu"))

    rec.add(Dense(latentDims, activation="relu"))

    for i in range(numLayers):
        rec.add(Dense(layerDimsDec[i], activation="relu"))

    rec.add(Dense(SPECLENGTH, activation="relu"))

    rec.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return rec

