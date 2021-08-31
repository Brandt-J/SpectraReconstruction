import numpy as np
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

import distort
import importData as io
from functions import reduceSpecsToNWavenumbers
from globals import SPECLENGTH
from Reconstruction import normalizeSpecSet

os.chdir(os.path.dirname(os.getcwd()))

noiseLevel: float = 0.4
numTrainSpectra, numTestSpectra = 90, 20
numVariationsTrain, numVariationsTest = 500, 200

experimentTitle = f"Savitzky-Golay-Benchmarking"
print(experimentTitle)

specNames, spectra = io.load_specCSVs_from_directory("ATR Spectra", maxSpectra=numTestSpectra+numTrainSpectra)
wavenums = spectra[:, 0].copy()
spectra = reduceSpecsToNWavenumbers(spectra, SPECLENGTH)

specs: np.ndarray = spectra[:, 1:]
trainSpectra: np.ndarray = np.tile(specs[:, :numTrainSpectra], (1, numVariationsTrain))
testSpectra = np.tile(specs[:, numTrainSpectra:], (1, numVariationsTest))

cleanSpectra: np.ndarray = testSpectra.transpose()
noisySpectra = distort.add_noise(cleanSpectra, level=noiseLevel, seed=42)

cleanSpectra = normalizeSpecSet(cleanSpectra)
noisySpectra = normalizeSpecSet(noisySpectra)

lengths: np.ndarray = np.arange(7, 53, 4)
orders: np.ndarray = np.array([1, 2, 3, 4, 5, 6])
numSpecs: int = cleanSpectra.shape[0]
maxCorr = 0
corrMap = np.zeros((len(orders), len(lengths)))
for i, window_length in enumerate(lengths):
    for j, order in enumerate(orders):
        if order >= window_length:
            print(f"Skipping impossible parameter combination: window-length {window_length}, order: {order}")
            continue

        corrs: np.ndarray = np.zeros(numSpecs)
        for k in range(numSpecs):
            noisy: np.ndarray = noisySpectra[k, :]
            savgol = savgol_filter(noisy, window_length=window_length, polyorder=order)
            corrs[k] = np.corrcoef(cleanSpectra[k, :], savgol)[0, 1] * 100

        meanCorr = np.mean(corrs)
        corrMap[j, i] = meanCorr
        if meanCorr > maxCorr:
            maxCorr = meanCorr
            print("############ NEW MAX CORR:")
            print(f"w-length: {window_length}, order: {order}, mean corr: {round(meanCorr)}, std corr: {round(np.std(corrs))}")


corrFig: plt.Figure = plt.figure()
corrAx: plt.Axes = corrFig.add_subplot()
plotObj = corrAx.imshow(corrMap, cmap='jet')
corrAx.set_yticks(np.arange(len(orders)))
corrAx.set_yticklabels(orders)
corrAx.set_ylabel("Polynomial order", fontsize=12)
corrAx.set_xticks(np.arange(len(lengths)))
corrAx.set_xticklabels(lengths)
corrAx.set_xlabel("Window size", fontsize=12)

corrAx.set_title("Savitzky-Golay Benchmark", fontsize=14)
cb = corrFig.colorbar(plotObj)
cb.set_label("Correlation Denoised -> Target", fontsize=12)

corrFig.show()
