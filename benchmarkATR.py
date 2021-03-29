import time
import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, Input, Flatten, MaxPooling1D, Conv1DTranspose, InputLayer, UpSampling1D

import outGraphs as out
import distort
import importData as io
from Reconstruction import prepareSpecSet, optimizeRec, getReconstructor
from functions import reduceSpecsToNWavenumbers
from globals import SPECLENGTH

noiseLevel = 0.15
t0 = time.time()
numTrainSpectra, numTestSpectra = 60, 60
numVariationsTrain, numVariationsTest = 500, 500

experimentTitle = "Neural Net Denoising"
print(experimentTitle)

specNames, spectra = io.load_specCSVs_from_directory("ATR Spectra", maxSpectra=numTrainSpectra+numTestSpectra)
wavenums = spectra[:, 0].copy()
spectra = reduceSpecsToNWavenumbers(spectra, SPECLENGTH)
print(f'loading and remapping spectra took {round(time.time()-t0)} seconds')

specs: np.ndarray = spectra[:, 1:]
trainSpectra: np.ndarray = np.tile(specs[:, :numTrainSpectra], (1, numVariationsTrain))

testSpectra = np.tile(specs[:, numTrainSpectra:], (1, numVariationsTest))

numSpecsTotal = len(trainSpectra) + len(testSpectra)

t0 = time.time()
np.random.seed(42)
noisyTrainSpectra = distort.add_noise(trainSpectra, level=noiseLevel, seed=0)
noisyTestSpectra = distort.add_noise(testSpectra, level=noiseLevel, seed=numSpecsTotal)
for i in range(3):
    noisyTrainSpectra = distort.add_distortions(noisyTrainSpectra, level=noiseLevel*5, seed=i * numSpecsTotal)
    noisyTrainSpectra = distort.add_ghost_peaks(noisyTrainSpectra, level=noiseLevel*5, seed=i * numSpecsTotal)
    noisyTestSpectra = distort.add_distortions(noisyTestSpectra, level=noiseLevel*5, seed=2*i * numSpecsTotal)
    noisyTestSpectra = distort.add_ghost_peaks(noisyTestSpectra, level=noiseLevel*5, seed=2*i * numSpecsTotal)

np.save("noisyTrain.npy", noisyTrainSpectra)
np.save("noisyTest.npy", noisyTestSpectra)
# noisyTrainSpectra = np.load("noisyTrain.npy")
# noisyTestSpectra = np.load("noisyTest.npy")
print(f'Distorting spectra took {round(time.time()-t0, 2)} seconds')

addDimension = True
trainSpectra = prepareSpecSet(trainSpectra, addDimension=addDimension)
testSpectra = prepareSpecSet(testSpectra, addDimension=addDimension)
noisyTrainSpectra = prepareSpecSet(noisyTrainSpectra, addDimension=addDimension)
noisyTestSpectra = prepareSpecSet(noisyTestSpectra, addDimension=addDimension)

rec = getReconstructor()
rec.summary()
history = rec.fit(noisyTrainSpectra, trainSpectra,
                  epochs=100, validation_data=(noisyTestSpectra, testSpectra),
                  batch_size=32, shuffle=True)

reconstructedSpecs = rec.call(noisyTestSpectra)
histplot = out.getHistPlot(history.history, title=experimentTitle)
specPlot, boxPlot = out.getSpectraComparisons(testSpectra, noisyTestSpectra, reconstructedSpecs,
                                              includeSavGol=False,
                                              wavenumbers=wavenums,
                                              title=experimentTitle)
# #
# # # corrPlot = out.getCorrelationPCAPlot(noisyTestSpectra.numpy(), reconstructedSpecs.numpy(),
# #                                      testSpectra.numpy(), noisyTrainSpectra.numpy())
