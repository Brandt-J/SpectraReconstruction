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

noiseLevel: float = 0.2  # Level to
useConvNetwork: bool = False
fracValid: float = 0.18
randomShuffle: bool = False
specTypesTotal: int = 110
numVariations: int = 5000

experimentTitle = f"Spectra Reconstruction, using {'convolutional' if useConvNetwork else 'dense'} network"
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
# noisyTrainSpectra = distort.distort_to_max_correlation(trainSpectra, maxCorr=0.4, seed=0)
# noisyTestSpectra = distort.distort_to_max_correlation(testSpectra, maxCorr=0.4, seed=numSpecsTotal)

noisyTrainSpectra = distort.add_noise(trainSpectra, level=noiseLevel, seed=0)
noisyTestSpectra = distort.add_noise(testSpectra, level=noiseLevel, seed=numSpecsTotal)
# for i in range(3):
#     noisyTrainSpectra = distort.add_distortions(noisyTrainSpectra, level=noiseLevel*2, seed=i * numSpecsTotal)
#     noisyTestSpectra = distort.add_ghost_peaks(noisyTestSpectra, level=noiseLevel*2, seed=2*i * numSpecsTotal)
#     noisyTestSpectra = distort.add_distortions(noisyTestSpectra, level=noiseLevel*2, seed=2*i * numSpecsTotal)
#     noisyTrainSpectra = distort.add_ghost_peaks(noisyTrainSpectra, level=noiseLevel*2, seed=i * numSpecsTotal)

np.save("noisyTrain.npy", noisyTrainSpectra)
np.save("noisyTest.npy", noisyTestSpectra)
noisyTrainSpectra = np.load("noisyTrain.npy")
noisyTestSpectra = np.load("noisyTest.npy")
print(f'Distorting spectra took {round(time.time()-t0, 2)} seconds')

trainSpectra = prepareSpecSet(trainSpectra, addDimension=useConvNetwork)
testSpectra = prepareSpecSet(testSpectra, addDimension=useConvNetwork)
noisyTrainSpectra = prepareSpecSet(noisyTrainSpectra, addDimension=useConvNetwork)
noisyTestSpectra = prepareSpecSet(noisyTestSpectra, addDimension=useConvNetwork)

if useConvNetwork:
    rec: Reconstructor = getConvReconstructor()
else:
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

histplot = out.getHistPlot(history.history, title=experimentTitle, annotate=False)
specPlot, boxPlot = out.getSpectraComparisons(testSpectra, noisyTestSpectra, reconstructedSpecs,
                                              includeSavGol=True,
                                              wavenumbers=wavenums,
                                              title=experimentTitle)

# predictedNames, report = out.getSpecCorrelation(reconstructedSpecs, testNames, dbSpecs, dbNames)
#
# noisyTestSpectraEncoded = rec.encoder(noisyTestSpectra)
# noisyTrainSpectraEncoded = rec.encoder(noisyTrainSpectra)
# corrPlot = out.getCorrelationPCAPlot(noisyTestSpectraEncoded, reconstructedSpecs, testSpectra, noisyTrainSpectraEncoded)
# distPlot = out.getCorrelationToTrainDistancePlot(noisyTestSpectra, noisyTestSpectraEncoded, reconstructedSpecs, testSpectra, noisyTrainSpectraEncoded)
