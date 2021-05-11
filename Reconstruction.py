import numpy as np
import tensorflow as tf
from typing import List, TYPE_CHECKING
from tensorflow.keras.layers import InputLayer, Dense, Conv1D, MaxPooling1D, Conv1DTranspose, Dropout
from tensorflow.keras.models import Sequential, Model

from globals import SPECLENGTH
if TYPE_CHECKING:
    from tensorflow.python.framework.ops import EagerTensor


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


def prepareSpecSet(specSet: np.ndarray, transpose: bool = True, addDimension: bool = False, normalize: bool = True):
    if transpose:
        specSet = specSet.transpose()

    if normalize:
        specSet = normalizeSpecSet(specSet)

    if addDimension:
        specSet = specSet.reshape(specSet.shape[0], specSet.shape[1], 1)

    specSet = tf.cast(specSet, tf.float32)
    return specSet


def getConvReconstructor() -> 'Reconstructor':
    model: Reconstructor = Reconstructor()
    model.encoder.add(InputLayer(input_shape=(SPECLENGTH, 1)))
    model.encoder.add(Conv1D(32, 4, padding='same', activation="relu"))
    model.encoder.add(MaxPooling1D(2, padding='same'))
    model.encoder.add(Conv1D(32, 4, activation="relu", padding="same"))
    model.encoder.add(MaxPooling1D(2, padding="same"))

    model.decoder.add(Conv1DTranspose(32, 4, activation="relu", padding="same"))
    model.decoder.add(Conv1DTranspose(32, 4, activation='relu', padding="same"))
    model.decoder.add(Conv1D(1, 1, activation='relu', padding='same'))
    model.compile(optimizer='adam', loss='mse')
    return model


def getDenseReconstructor(dropout: float = 0.0) -> 'Reconstructor':
    rec: Reconstructor = Reconstructor()
    rec.encoder.add(InputLayer(input_shape=(SPECLENGTH)))
    if dropout > 0.0:
        rec.encoder.add(Dropout(dropout))
    rec.encoder.add(Dense(128, activation="relu"))
    rec.decoder.add(InputLayer(128))
    if dropout:
        rec.decoder.add(Dropout(dropout))
    rec.decoder.add(Dense(SPECLENGTH, activation="relu"))
    rec.compile(optimizer='adam', loss='mse')
    return rec


class Reconstructor(Model):
    def __init__(self):
        super(Reconstructor, self).__init__()
        self.encoder: Sequential = Sequential()
        self.decoder: Sequential = Sequential()
        self._encodedTrainData: np.ndarray = None

    def call(self, inputs, training=None, mask=None):
        if training:
            self._encodedTrainData = None
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def calculateEncodedTrainingData(self, inputs) -> None:
        self._encodedTrainData = self.encoder(inputs).numpy()

    def getPoorlyRepresentedIndices(self, inputs: 'EagerTensor', maxDist: float = 0.1) -> np.ndarray:
        """
        Calculates distance of encoded inputs to encoded training data. It returns indices of samples
        having a distance > the specified maxDist that is condsidered save for good reconstruction.
        An a-priori of finding a good maxDist is necessary...
        :param inputs: Eager Tensor of inputs for network inference
        :param maxDist: Maximum Distance value (in encoded space dimension)
        :return: array of indices with training data further away than maxDist
        """
        if self._encodedTrainData is None:
            print("Call 'calculateEncodedTrainingData' with the training data first.")
            return None

        encodedInputs: np.ndarray = self.encoder(inputs).numpy()
        invalidIndices: List[int] = []
        for i in range(encodedInputs.shape[0]):
            distances = np.linalg.norm(self._encodedTrainData - encodedInputs[i, :], axis=1)
            avgMinDist = np.mean(np.sort(distances)[:5])
            if avgMinDist > maxDist:
                invalidIndices.append(i)
        return np.array(invalidIndices)
