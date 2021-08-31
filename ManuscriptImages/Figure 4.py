import time
import numpy as np
import random
import os

os.chdir(os.path.dirname(os.getcwd()))

import outGraphs as out
import distort
import importData as io
from Reconstruction import prepareSpecSet, Reconstructor, getDenseReconstructor
from functions import reduceSpecsToNWavenumbers
from globals import SPECLENGTH

noiseLevel: float = 0.2
fracValid: float = 0.2
numVariations: int = 1000

experimentTitle = f"Removal of fluorescence, interference and cosmic rays"
print(experimentTitle)

t0 = time.time()
spectra = io.load_reference_Raman_spectra()
wavenums = spectra[:, 0].copy()
spectra = reduceSpecsToNWavenumbers(spectra, SPECLENGTH)
specs: np.ndarray = spectra[:, 1:]
print(f'loading and remapping spectra took {round(time.time()-t0)} seconds')

specs = np.tile(specs, (1, numVariations))
numSpecs = specs.shape[1]
random.seed(42)
valIndices = random.sample(range(numSpecs), int(round(numSpecs * fracValid)))
trainIndices = [i for i in range(numSpecs) if i not in valIndices]
trainSpectra: np.ndarray = specs[:, trainIndices]
testSpectra: np.ndarray = specs[:, valIndices]

t0 = time.time()
numSpecsTotal = len(trainSpectra) + len(testSpectra)

noisyTrainSpectra = distort.add_noise(trainSpectra, level=noiseLevel, seed=0, ramanMode=True)
noisyTestSpectra = distort.add_noise(testSpectra, level=noiseLevel, seed=numSpecsTotal, ramanMode=True)

noisyTrainSpectra = distort.add_periodic_interferences_raman(noisyTrainSpectra, seed=0)
noisyTestSpectra = distort.add_periodic_interferences_raman(noisyTestSpectra, seed=numSpecsTotal)

levelRange = (0.5, 1.5)
noisyTrainSpectra = distort.add_fluorescence(noisyTrainSpectra, levelRange=levelRange, seed=0)
noisyTestSpectra = distort.add_fluorescence(noisyTestSpectra, levelRange=levelRange, seed=numSpecsTotal)

noisyTrainSpectra = distort.add_cosmic_ray_peaks(noisyTrainSpectra, numRange=(0, 2), seed=0)
noisyTestSpectra = distort.add_cosmic_ray_peaks(noisyTestSpectra, numRange=(0, 2), seed=numSpecsTotal)


print(f'Distorting spectra took {round(time.time()-t0, 2)} seconds')

trainSpectra = prepareSpecSet(trainSpectra, addDimension=False)
testSpectra = prepareSpecSet(testSpectra, addDimension=False)
noisyTrainSpectra = prepareSpecSet(noisyTrainSpectra, addDimension=False)
noisyTestSpectra = prepareSpecSet(noisyTestSpectra, addDimension=False)


rec: Reconstructor = getDenseReconstructor(dropout=0.0)

t0 = time.time()
history = rec.fit(noisyTrainSpectra, trainSpectra,
                  epochs=50,
                  validation_data=(noisyTestSpectra, testSpectra),
                  batch_size=32, shuffle=True)
print(f"Training took {round(time.time()-t0, 2)} seconds.")

t0 = time.time()
reconstructedSpecs = rec.call(noisyTestSpectra)
print(f'reconstruction took {round(time.time()-t0, 2)} seconds')

# histplot = out.getHistPlot(history.history, title=experimentTitle, annotate=False)
specPlot, boxPlot = out.getSpectraComparisons(testSpectra, noisyTestSpectra, reconstructedSpecs,
                                              includeSavGol=False,
                                              wavenumbers=wavenums,
                                              title=experimentTitle,
                                              randomIndSeed=4)
