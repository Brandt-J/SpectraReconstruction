import time
import os
import numpy as np
import random
from typing import List

import outGraphs as out
import distort
import importData as io
from Reconstruction import prepareSpecSet, Reconstructor, getDenseReconstructor
from functions import reduceSpecsToNWavenumbers
from globals import SPECLENGTH

os.chdir(os.path.dirname(os.getcwd()))

noiseLevel: float = 0.15
fracValid: float = 0.2
specTypesTotal: int = 100
numVariations: int = 100

t0 = time.time()
specNames, spectra = io.load_specCSVs_from_directory("ATR Spectra", maxSpectra=specTypesTotal)
wavenums = spectra[:, 0].copy()
spectra = reduceSpecsToNWavenumbers(spectra, SPECLENGTH)
specs: np.ndarray = spectra[:, 1:]
print(f'loading and remapping spectra took {round(time.time()-t0)} seconds')

randomShuffle = False
if randomShuffle:
    specs = np.tile(specs, (1, numVariations))
    numSpecs = specs.shape[1]
    valIndices = random.sample(range(numSpecs), int(round(numSpecs * fracValid)))
    trainIndices = [i for i in range(numSpecs) if i not in valIndices]
    trainSpectra: np.ndarray = specs[:, trainIndices]
    testSpectra: np.ndarray = specs[:, valIndices]
    allSpecNames = specNames*numVariations
    testNames: List[str] = [allSpecNames[i] for i in valIndices]
else:
    numTestSpectra = int(round(fracValid * specTypesTotal))
    numTrainSpectra = specTypesTotal - numTestSpectra
    trainSpectra: np.ndarray = np.tile(specs[:, :numTrainSpectra], (1, numVariations))
    testSpectra = np.tile(specs[:, numTrainSpectra:], (1, numVariations))
    testNames: List[str] = specNames[numTrainSpectra:] * numVariations

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

rec: Reconstructor = getDenseReconstructor(dropout=0.0 if randomShuffle else 0.00)

t0 = time.time()
history = rec.fit(noisyTrainSpectra, trainSpectra,
                  epochs=10,
                  validation_data=(noisyTestSpectra, testSpectra),
                  batch_size=32, shuffle=True)
print(f"Training took {round(time.time()-t0, 2)} seconds.")

t0 = time.time()
reconstructedSpecs = rec.call(noisyTestSpectra)
print(f'reconstruction took {round(time.time()-t0, 2)} seconds')

noisyTestSpectraEncoded = rec.encoder(noisyTestSpectra)
noisyTrainSpectraEncoded = rec.encoder(noisyTrainSpectra)
corrPlot = out.getCorrelationPCAPlot(noisyTestSpectraEncoded, reconstructedSpecs, testSpectra, noisyTrainSpectraEncoded)
distPlot = out.getCorrelationToTrainDistancePlot(noisyTestSpectra, noisyTestSpectraEncoded, reconstructedSpecs, testSpectra, noisyTrainSpectraEncoded)
