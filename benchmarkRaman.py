import time
import numpy as np
import random
from typing import List
from copy import copy

import outGraphs as out
import distort
import importData as io
from Reconstruction import prepareSpecSet, Reconstructor, getConvReconstructor, getDenseReconstructor
from functions import reduceSpecsToNWavenumbers
from globals import SPECLENGTH

noiseLevel: float = 0.2
useConvNetwork: bool = False
fracValid: float = 0.2
randomShuffle: bool = True
specTypesTotal: int = 120
numVariations: int = 100

experimentTitle = f"Removal of fluorescence contributions"
print(experimentTitle)

t0 = time.time()
specNames, spectra = io.load_specCSVs_from_directory("ATR Spectra", maxSpectra=specTypesTotal)
wavenums = spectra[:, 0].copy()
spectra = reduceSpecsToNWavenumbers(spectra, SPECLENGTH)
specs: np.ndarray = spectra[:, 1:]
dbSpecs: np.ndarray = specs.copy()
dbNames: List[str] = copy(specNames)

print(f'loading and remapping spectra took {round(time.time()-t0)} seconds')

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

levelRange = (1.0, 5.0)
noisyTrainSpectra = distort.add_fluorescence(noisyTrainSpectra, levelRange=levelRange, seed=0)
noisyTestSpectra = distort.add_fluorescence(noisyTestSpectra, levelRange=levelRange, seed=numSpecsTotal)

noisyTrainSpectra = distort.add_cosmic_ray_peaks(noisyTrainSpectra, numRange=(2, 5), seed=0)
noisyTestSpectra = distort.add_cosmic_ray_peaks(noisyTestSpectra, numRange=(2, 5), seed=numSpecsTotal)


print(f'Distorting spectra took {round(time.time()-t0, 2)} seconds')

trainSpectra = prepareSpecSet(trainSpectra, addDimension=useConvNetwork)
testSpectra = prepareSpecSet(testSpectra, addDimension=useConvNetwork)
noisyTrainSpectra = prepareSpecSet(noisyTrainSpectra, addDimension=useConvNetwork)
noisyTestSpectra = prepareSpecSet(noisyTestSpectra, addDimension=useConvNetwork)

if useConvNetwork:
    rec: Reconstructor = getConvReconstructor()
else:
    rec: Reconstructor = getDenseReconstructor(dropout=0.0 if randomShuffle else 0.15)

t0 = time.time()
history = rec.fit(noisyTrainSpectra, trainSpectra,
                  epochs=20,
                  validation_data=(noisyTestSpectra, testSpectra),
                  batch_size=32, shuffle=True)
print(f"Training took {round(time.time()-t0, 2)} seconds.")

t0 = time.time()
reconstructedSpecs = rec.call(noisyTestSpectra)
print(f'reconstruction took {round(time.time()-t0, 2)} seconds')

histplot = out.getHistPlot(history.history, title=experimentTitle, annotate=False)
specPlot, boxPlot = out.getSpectraComparisons(testSpectra, noisyTestSpectra, reconstructedSpecs,
                                              includeSavGol=False,
                                              wavenumbers=wavenums,
                                              title=experimentTitle)
