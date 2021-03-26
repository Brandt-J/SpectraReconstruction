import numpy as np

import distort
from peakConvDeconv import getSpecFromPeaks
import outGraphs as out
from Reconstruction import prepareSpecSet, getReconstructor
from globals import SPECLENGTH

SPECLENGTH, latentDims = 512, 32
noiseLevel = 0.7

numTrainSpectra, numTestSpectra = 150, 10
numVariationsTrain, numVariationsTest = 500, 10
experimentTitle = "Peak Area Restoration Test"
print(experimentTitle)

wavenums = np.linspace(200, 3500, SPECLENGTH)
spectra = np.zeros((numTrainSpectra + numTestSpectra, SPECLENGTH))
testPeakParams = []
np.random.seed(42)
for i in range(numTrainSpectra + numTestSpectra):
    centerWidthAreas = []
    numPeaks = np.random.randint(2, 8)
    for _ in range(numPeaks):
        centerWidthAreas.append((np.random.rand() * SPECLENGTH, 5 + np.random.rand() * 10, 1 + np.random.rand()*2))
    spectra[i, :] = getSpecFromPeaks(centerWidthAreas, SPECLENGTH)
    if i >= numTrainSpectra:
        testPeakParams.append(centerWidthAreas)

testPeakParams *= numVariationsTest


trainSpectra: np.ndarray = np.tile(spectra[:numTrainSpectra, :], (numVariationsTrain, 1))
testSpectra: np.ndarray = np.tile(spectra[numTrainSpectra:, :], (numVariationsTest, 1))
noisyTrainSpectra = distort.add_noise(trainSpectra.transpose(), level=noiseLevel).transpose()
noisyTestSpectra = distort.add_noise(testSpectra.transpose(), level=noiseLevel).transpose()

trainSpectra = prepareSpecSet(trainSpectra, transpose=False)
testSpectra = prepareSpecSet(testSpectra, transpose=False)
noisyTrainSpectra = prepareSpecSet(noisyTrainSpectra, transpose=False)
noisyTestSpectra = prepareSpecSet(noisyTestSpectra, transpose=False)


rec = getReconstructor()
rec.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = rec.fit(noisyTrainSpectra, trainSpectra,
                  epochs=100, validation_data=(noisyTestSpectra, testSpectra),
                  batch_size=32, shuffle=True)
histplot = out.getHistPlot(history.history, annotate=False)
reconstructedSpecs = rec.call(noisyTestSpectra)
specPlot, boxPlot = out.getSpectraComparisons(testSpectra, noisyTestSpectra, reconstructedSpecs,
                                              includeSavGol=True,
                                              wavenumbers=wavenums,
                                              title=experimentTitle)
# t0 = time.time()
# areaPlot = out.getPeakAreaBoxPlot(testPeakParams, reconstructedSpecs.numpy(), noisyTestSpectra.numpy())
# print(f'generating area plot for {len(testPeakParams)} spectra took {round(time.time()-t0, 2)} seconds')
