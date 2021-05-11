import numpy as np
import random

import distort
import outGraphs as out
from Reconstruction import prepareSpecSet, getDenseReconstructor
from peakConvDeconv import getSpecFromPeaks
from globals import SPECLENGTH

noiseLevel = 0.8
numSpectra = 100
numVariations = 20
fracValid: float = 0.05
experimentTitle = "Peak Area Restoration Test"
print(experimentTitle)

wavenums = np.linspace(200, 3500, SPECLENGTH)
spectra = np.zeros((numSpectra, SPECLENGTH))
peakParams = []
np.random.seed(42)
for i in range(numSpectra):
    centerWidthAreas = []
    numPeaks = np.random.randint(4, 5)
    for _ in range(numPeaks):
        centerWidthAreas.append((np.random.rand() * SPECLENGTH, 3 + np.random.rand() * 5, 1 + np.random.rand()*2))
    spectra[i, :] = getSpecFromPeaks(centerWidthAreas, SPECLENGTH)
    peakParams.append(centerWidthAreas.copy())

peakParams *= numVariations
spectra = np.tile(spectra, (numVariations, 1))

noisySpectra: np.ndarray = distort.add_noise(spectra.transpose(), level=noiseLevel).transpose()

random.seed(42)
numSpecValid: int = int(round(fracValid * numSpectra*numVariations))
valIndices = sorted(random.sample(range(numSpectra*numVariations), numSpecValid))
trainIndices = list(set(range(len(spectra))) - set(valIndices))
testPeakParams = [param for i, param in enumerate(peakParams) if i in valIndices]

trainSpectra: np.ndarray = spectra[trainIndices]
testSpectra: np.ndarray = spectra[valIndices]
noisyTrainSpectra: np.ndarray = noisySpectra[trainIndices]
noisyTestSpectra: np.ndarray = noisySpectra[valIndices]

trainSpectra = prepareSpecSet(trainSpectra, transpose=False, normalize=False)
testSpectra = prepareSpecSet(testSpectra, transpose=False, normalize=False)
noisyTrainSpectra = prepareSpecSet(noisyTrainSpectra, transpose=False, normalize=False)
noisyTestSpectra = prepareSpecSet(noisyTestSpectra, transpose=False, normalize=False)


rec = getDenseReconstructor()
history = rec.fit(noisyTrainSpectra, trainSpectra,
                  epochs=40, validation_data=(noisyTestSpectra, testSpectra),
                  batch_size=32, shuffle=True)
histplot = out.getHistPlot(history.history, annotate=False)
reconstructedSpecs = rec.call(noisyTestSpectra)
specPlot, boxPlot = out.getSpectraComparisons(testSpectra, noisyTestSpectra, reconstructedSpecs,
                                              includeSavGol=True,
                                              wavenumbers=wavenums,
                                              title=experimentTitle)

import time
t0 = time.time()
areaPlot = out.getPeakAreaBoxPlot(testPeakParams, reconstructedSpecs.numpy(), noisyTestSpectra.numpy())
print(f'deconvolution took {round(time.time()-t0, 2)} seconds')
