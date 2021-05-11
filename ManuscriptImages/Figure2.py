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

noiseLevel: float = 0.4
numTrainSpectra, numTestSpectra = 90, 20
numVariationsTrain, numVariationsTest = 500, 200

experimentTitle = f"Neural Net Denoising"
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
histPLot = out.getHistPlot(history.history, annotate=True)
t0 = time.time()
reconstructedSpecs = rec.call(noisyTestSpectra)
print(f'reconstruction took {round(time.time()-t0, 2)} seconds')

specPlot, boxPlot = out.getSpectraComparisons(testSpectra, noisyTestSpectra, reconstructedSpecs,
                                              includeSavGol=True,
                                              randomIndSeed=None,
                                              wavenumbers=wavenums,
                                              title=experimentTitle)
