import numpy as np
import tensorflow as tf
from kerastuner.tuners import RandomSearch
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from sklearn import preprocessing


def prepareSpecSet(specSet: np.ndarray, transpose: bool = True):
    specSet = preprocessing.minmax_scale(specSet, feature_range=(0.0, 1.0))
    if transpose:
        specSet = specSet.transpose()
    specSet = tf.cast(specSet, tf.float32)
    return specSet


def optimizeRec(X_train, y_train, X_test, y_test):
    tuner = RandomSearch(
        getReconstructor,
        objective='val_accuracy',
        max_trials=100,
        executions_per_trial=2,
        directory='NetworkSearch')

    tuner.search(x=X_train,
                 y=y_train,
                 epochs=3,
                 validation_data=(X_test, y_test))
    return tuner


def getReconstructor(hp=None) -> Sequential:
    from globals import SPECLENGTH
    if hp is None:
        latentDims: int = 128
        numLayers: int = 0
        regularization: int = 0
        regPower: float = 1e-5
    else:
        latentDims: int = hp.Int("n_latentDims", 32, 128, 32)
        numLayers: int = hp.Int("n_layers", 0, 3)
        regularization: int = hp.Int("regularization", 0, 2)
        regPower: float = hp.Choice("regPower", values=[1e-2, 1e-3, 1e-4, 1e-5])

    layerDims: np.ndarray = np.linspace(np.log(latentDims), np.log(SPECLENGTH), numLayers+1, endpoint=False)
    layerDimsEnc: np.ndarray = np.uint16(np.round(np.exp(layerDims[1:])))
    layerDimsDec: np.ndarray = layerDimsEnc[::-1]

    rec: Sequential = Sequential()
    for i in range(numLayers):
        rec.add(getDenseLayer(layerDimsEnc[i], regularization=regularization, regPower=regPower))
        # rec.add(Dropout(0.2))

    rec.add(getDenseLayer(latentDims, regularization=regularization, regPower=regPower))

    for i in range(numLayers):
        rec.add(getDenseLayer(layerDimsDec[i], regularization=regularization, regPower=regPower))
        # rec.add(Dropout(0.2))

    rec.add(getDenseLayer(SPECLENGTH, regularization=regularization, regPower=regPower))

    rec.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return rec


def getDenseLayer(numNodes: int, regularization: int, regPower: float = 1e-3) -> Dense:
    if regularization == 0:  # no regularization
        dense = Dense(numNodes, activation='relu')
    elif regularization == 1:  # L1
        dense = Dense(numNodes, activation='relu', activity_regularizer=regularizers.l1(regPower))
    else:  # L2
        dense = Dense(numNodes, activation='relu', activity_regularizer=regularizers.l2(regPower))
    return dense

