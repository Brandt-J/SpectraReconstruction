import time
import numpy as np

import outGraphs as out
import distort
import importData as io
from Reconstruction import prepareSpecSet, optimizeRec, getReconstructor
from functions import reduceSpecsToNWavenumbers
from globals import SPECLENGTH

noiseLevel = 0.15
t0 = time.time()
numTrainSpectra, numTestSpectra = 50, 20
numVariationsTrain, numVariationsTest = 5000, 100

experimentTitle = "Neural Net Denoising"
print(experimentTitle)

specNames, spectra = io.load_specCSVs_from_directory("ATR Spectra", maxSpectra=numTrainSpectra+numTestSpectra)
wavenums = spectra[:, 0].copy()
spectra = reduceSpecsToNWavenumbers(spectra, SPECLENGTH)
print(f'loading and remapping spectra took {round(time.time()-t0)} seconds')

specs: np.ndarray = spectra[:, 1:]
trainSpectra: np.ndarray = np.tile(specs[:, :numTrainSpectra], (1, numVariationsTrain))

testSpectra = np.tile(specs[:, numTrainSpectra:], (1, numVariationsTest))

t0 = time.time()
np.random.seed(42)
noisyTrainSpectra = distort.add_noise(trainSpectra, level=noiseLevel/2)
noisyTestSpectra = distort.add_noise(testSpectra, level=noiseLevel/2)
# for _ in range(3):
#     noisyTrainSpectra = distort.add_distortions(noisyTrainSpectra, level=noiseLevel*5)
#     noisyTrainSpectra = distort.add_ghost_peaks(noisyTrainSpectra, level=noiseLevel*5)
#     noisyTestSpectra = distort.add_distortions(noisyTestSpectra, level=noiseLevel*5)
#     noisyTestSpectra = distort.add_ghost_peaks(noisyTestSpectra, level=noiseLevel*5)
print(f'Distorting spectra took {round(time.time()-t0, 2)} seconds')


trainSpectra = prepareSpecSet(trainSpectra)
testSpectra = prepareSpecSet(testSpectra)
noisyTrainSpectra = prepareSpecSet(noisyTrainSpectra)
noisyTestSpectra = prepareSpecSet(noisyTestSpectra)


tuner = optimizeRec(noisyTrainSpectra, trainSpectra, noisyTestSpectra, testSpectra)

# rec = getReconstructor()
# history = rec.fit(noisyTrainSpectra, trainSpectra,
#                   epochs=10, validation_data=(noisyTestSpectra, testSpectra),
#                   batch_size=32, shuffle=True)
#
# reconstructedSpecs = rec.call(noisyTestSpectra)
# histplot = out.getHistPlot(history.history, title=experimentTitle)
# specPlot, boxPlot = out.getSpectraComparisons(testSpectra, noisyTestSpectra, reconstructedSpecs,
#                                               includeSavGol=True,
#                                               wavenumbers=wavenums,
#                                               title=experimentTitle)
#
# # corrPlot = out.getCorrelationPCAPlot(noisyTestSpectra.numpy(), reconstructedSpecs.numpy(),
# #                                      testSpectra.numpy(), noisyTrainSpectra.numpy())
