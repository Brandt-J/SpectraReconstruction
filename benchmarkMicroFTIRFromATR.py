import time
import random
import numpy as np

import importData as io
import distort
import outGraphs as out
from Reconstruction import prepareSpecSet, getDenseReconstructor
from functions import reduceSpecsToNWavenumbers
from globals import SPECLENGTH

noiseLevel = 0.2
t0 = time.time()
numDifferentSpectra = 150
numVariations = 100

experimentTitle = 'Trained on ATR spectra, applied to µFTIR spectra'
print(experimentTitle)

specNames, spectra = io.load_specCSVs_from_directory("ATR Spectra", maxSpectra=numDifferentSpectra)
wavenums = spectra[:, 0].copy()
spectra = reduceSpecsToNWavenumbers(spectra, SPECLENGTH)
print(f'loading and remapping ATR spectra took {round(time.time()-t0)} seconds')

specs: np.ndarray = spectra[:, 1:]
trainSpectra = np.tile(specs, (1, numVariations))

t0 = time.time()
np.random.seed(42)
for i in range(3):
    noisyTrainSpectra = distort.add_noise(trainSpectra, level=noiseLevel*(i+1)*0.5)
    noisyTrainSpectra = distort.add_distortions(noisyTrainSpectra, level=noiseLevel*(i+1)*2)
    noisyTrainSpectra = distort.add_ghost_peaks(noisyTrainSpectra, level=noiseLevel*(i+1)*2)
print(f'Distorting spectra took {round(time.time()-t0, 2)} seconds')

t0 = time.time()
noisySpecs, cleanSpecs, specNames, wavenumbers = io.load_microFTIR_spectra(SPECLENGTH, maxCorr=0.5)
print(f'loading and remapping µFTIR spectra took {round(time.time()-t0)} seconds')

trainSpectra = prepareSpecSet(trainSpectra)
testSpectra = prepareSpecSet(cleanSpecs, transpose=False)
noisyTrainSpectra = prepareSpecSet(noisyTrainSpectra)
noisyTestSpectra = prepareSpecSet(noisySpecs, transpose=False)

rec = getDenseReconstructor(dropout=0.5)
history = rec.fit(noisyTrainSpectra, trainSpectra,
                  epochs=10, validation_data=(noisyTestSpectra, testSpectra),
                  batch_size=32, shuffle=True)
histplot = out.getHistPlot(history.history, title=experimentTitle)

reconstructedSpecs = rec.call(noisyTestSpectra)
specPlot, boxPlot = out.getSpectraComparisons(testSpectra, noisyTestSpectra, reconstructedSpecs,
                                              includeSavGol=False,
                                              wavenumbers=wavenums,
                                              title=experimentTitle)
