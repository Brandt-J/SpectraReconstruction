import time
import numpy as np
import os

import outGraphs as out
import distort
import importData as io
from Reconstruction import prepareSpecSet, Reconstructor, getDenseReconstructor
from functions import reduceSpecsToNWavenumbers
from globals import SPECLENGTH

os.chdir(os.path.dirname(os.getcwd()))

noiseLevel: float = 0.2
numTrainSpectra, numTestSpectra = 60, 40
numVariationsTrain, numVariationsTest = 100, 100

experimentTitle = f"Synthetically distorted ATR FTIR Spectra"
print(experimentTitle)

t0 = time.time()
specNames, spectra = io.load_specCSVs_from_directory("ATR Spectra", maxSpectra=numTestSpectra+numTrainSpectra)
wavenums = spectra[:, 0].copy()
spectra = reduceSpecsToNWavenumbers(spectra, SPECLENGTH)
specs: np.ndarray = spectra[:, 1:]

print(f'loading and remapping spectra took {round(time.time()-t0)} seconds')

trainSpectra: np.ndarray = np.tile(specs[:, :numTrainSpectra], (1, numVariationsTrain))
testSpectra = np.tile(specs[:, numTrainSpectra:], (1, numVariationsTest))

t0 = time.time()
numSpecsTotal = len(trainSpectra) + len(testSpectra)

noisyTrainSpectra = distort.add_noise(trainSpectra, level=noiseLevel, seed=0)
noisyTestSpectra = distort.add_noise(testSpectra, level=noiseLevel, seed=numSpecsTotal)
for i in range(3):
    noisyTrainSpectra = distort.add_distortions(noisyTrainSpectra, level=noiseLevel*2, seed=i * numSpecsTotal)
    noisyTestSpectra = distort.add_ghost_peaks(noisyTestSpectra, level=noiseLevel*2, seed=2*i * numSpecsTotal)
    noisyTestSpectra = distort.add_distortions(noisyTestSpectra, level=noiseLevel*2, seed=2*i * numSpecsTotal)
    noisyTrainSpectra = distort.add_ghost_peaks(noisyTrainSpectra, level=noiseLevel*2, seed=i * numSpecsTotal)
print(f'Distorting spectra took {round(time.time()-t0, 2)} seconds')

trainSpectra = prepareSpecSet(trainSpectra, addDimension=False)
testSpectra = prepareSpecSet(testSpectra, addDimension=False)
noisyTrainSpectra = prepareSpecSet(noisyTrainSpectra, addDimension=False)
noisyTestSpectra = prepareSpecSet(noisyTestSpectra, addDimension=False)

rec: Reconstructor = getDenseReconstructor(dropout=0.0)

t0 = time.time()
history = rec.fit(noisyTrainSpectra, trainSpectra,
                  epochs=10,
                  validation_data=(noisyTestSpectra, testSpectra),
                  batch_size=32, shuffle=True)
print(f"Training took {round(time.time()-t0, 2)} seconds.")

t0 = time.time()
reconstructedSpecs = rec.call(noisyTestSpectra)
print(f'reconstruction took {round(time.time()-t0, 2)} seconds')
histPLot = out.getHistPlot(history.history, annotate=True)
specPlot, boxPlot = out.getSpectraComparisons(testSpectra, noisyTestSpectra, reconstructedSpecs,
                                              includeSavGol=False,
                                              randomIndSeed=9,
                                              wavenumbers=wavenums,
                                              title=experimentTitle)
