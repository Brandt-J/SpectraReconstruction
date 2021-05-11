import time
import numpy as np
import os
import random

import outGraphs as out
import distort
import importData as io
from Reconstruction import prepareSpecSet, Reconstructor, getDenseReconstructor
from functions import reduceSpecsToNWavenumbers
from globals import SPECLENGTH

os.chdir(os.path.dirname(os.getcwd()))

titles = ["TestSpectra known to network",
          "TestSpectra unknown to network\nNo Dropout",
          "TestSpectra unknown to network\n15 % Dropout"]

fracValid: float = 0.4
numVariations: int = 100
specTypesTotal: int = 100
noiseLevel: float = 0.3


for i, experimentTitle in enumerate(titles, start=1):
    print(experimentTitle)
    randomShuffle = True if i == 1 else False
    print('rand shuffle', randomShuffle)
    t0 = time.time()
    specNames, spectra = io.load_specCSVs_from_directory("ATR Spectra", maxSpectra=specTypesTotal)
    wavenums = spectra[:, 0].copy()
    spectra = reduceSpecsToNWavenumbers(spectra, SPECLENGTH)
    specs: np.ndarray = spectra[:, 1:]
    dbSpecs: np.ndarray = specs.copy()

    print(f'loading and remapping spectra took {round(time.time() - t0)} seconds')

    if randomShuffle:
        specs = np.tile(specs, (1, numVariations))
        numSpecs = specs.shape[1]
        valIndices = random.sample(range(numSpecs), int(round(numSpecs * fracValid)))
        trainIndices = [i for i in range(numSpecs) if i not in valIndices]
        trainSpectra: np.ndarray = specs[:, trainIndices]
        testSpectra: np.ndarray = specs[:, valIndices]
    else:
        numTestSpectra = int(round(fracValid * specTypesTotal))
        numTrainSpectra = specTypesTotal - numTestSpectra
        trainSpectra: np.ndarray = np.tile(specs[:, :numTrainSpectra], (1, numVariations))
        testSpectra = np.tile(specs[:, numTrainSpectra:], (1, numVariations))

    t0 = time.time()
    numSpecsTotal = len(trainSpectra) + len(testSpectra)

    noisyTrainSpectra = distort.add_noise(trainSpectra, level=noiseLevel, seed=0)
    noisyTestSpectra = distort.add_noise(testSpectra, level=noiseLevel, seed=numSpecsTotal)
    for j in range(3):
        noisyTrainSpectra = distort.add_distortions(noisyTrainSpectra, level=noiseLevel*2, seed=j * numSpecsTotal)
        noisyTestSpectra = distort.add_ghost_peaks(noisyTestSpectra, level=noiseLevel*2, seed=2*j * numSpecsTotal)
        noisyTestSpectra = distort.add_distortions(noisyTestSpectra, level=noiseLevel*2, seed=2*j * numSpecsTotal)
        noisyTrainSpectra = distort.add_ghost_peaks(noisyTrainSpectra, level=noiseLevel*2, seed=j * numSpecsTotal)

    np.save("noisyTrain.npy", noisyTrainSpectra)
    np.save("noisyTest.npy", noisyTestSpectra)
    noisyTrainSpectra = np.load("noisyTrain.npy")
    noisyTestSpectra = np.load("noisyTest.npy")
    print(f'Distorting spectra took {round(time.time() - t0, 2)} seconds')

    trainSpectra = prepareSpecSet(trainSpectra, addDimension=False)
    testSpectra = prepareSpecSet(testSpectra, addDimension=False)
    noisyTrainSpectra = prepareSpecSet(noisyTrainSpectra, addDimension=False)
    noisyTestSpectra = prepareSpecSet(noisyTestSpectra, addDimension=False)

    dropout = 0.15 if i == 3 else 0.0
    print('dropout', dropout)
    rec: Reconstructor = getDenseReconstructor(dropout=dropout)

    t0 = time.time()
    history = rec.fit(noisyTrainSpectra, trainSpectra,
                      epochs=20,
                      validation_data=(noisyTestSpectra, testSpectra),
                      batch_size=32, shuffle=True)
    print(f"Training took {round(time.time()-t0, 2)} seconds.")

    t0 = time.time()
    reconstructedSpecs = rec.call(noisyTestSpectra)
    print(f'reconstruction took {round(time.time()-t0, 2)} seconds')
    histPLot = out.getHistPlot(history.history, annotate=False, title=experimentTitle)
    histPLot.get_axes()[0].set_ylim(0.007, 0.020)

