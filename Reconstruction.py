import numpy as np
import tensorflow as tf
from kerastuner.tuners import RandomSearch
from tensorflow.keras.layers import InputLayer, Conv1D, UpSampling1D, MaxPooling1D, Conv1DTranspose, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from sklearn import preprocessing


def prepareSpecSet(specSet: np.ndarray, transpose: bool = True):
    specSet = preprocessing.minmax_scale(specSet, feature_range=(0.0, 1.0))
    if transpose:
        specSet = specSet.transpose()

    specSet = normalize(specSet)

    # scaler: StandardScaler = StandardScaler()
    # specSet = scaler.fit_transform(specSet)

    if addDimension:
        specSet = specSet.reshape(specSet.shape[0], specSet.shape[1], 1)

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


def getReconstructor() -> Sequential:
    from globals import SPECLENGTH
    model: Sequential = Sequential()
    model.add(InputLayer(input_shape=(SPECLENGTH, 1)))
    model.add(Conv1D(64, 16, padding='same', activation="relu"))
    model.add(MaxPooling1D(2, padding='same'))  # activation??
    model.add(Conv1D(32, 8, activation="relu", padding="same"))
    model.add(MaxPooling1D(2, padding="same"))

    model.add(Conv1DTranspose(32, 3, activation="relu", padding="same"))
    model.add(UpSampling1D(2))
    model.add(Conv1DTranspose(64, 16, activation='relu', padding="same"))
    model.add(UpSampling1D(2))
    model.add(Conv1D(1, 1, activation='relu', padding='same'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

