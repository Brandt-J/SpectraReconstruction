import time
import numpy as np
from typing import List
from scipy.signal import savgol_filter

import distort
import importData as io
from functions import reduceSpecsToNWavenumbers
from globals import SPECLENGTH

noiseLevel: float = 0.4
numSpectraTypes = 120
numVariations = 100

experimentTitle = f"Savitzky-Golay Denoising"
print(experimentTitle)

t0 = time.time()
specNames, spectra = io.load_specCSVs_from_directory("ATR Spectra", maxSpectra=numSpectraTypes)
wavenums = spectra[:, 0].copy()
spectra = reduceSpecsToNWavenumbers(spectra, SPECLENGTH)
cleanSpectra: np.ndarray = spectra[:, 1:].transpose()
cleanSpectra = np.tile(cleanSpectra, (numVariations, 1))
print(f'loading and remapping spectra took {round(time.time()-t0)} seconds')

# normalize to 0...1
for i in range(cleanSpectra.shape[0]):
    spec: np.ndarray = cleanSpectra[i, :]
    cleanSpectra[i, :] = (spec - spec.min()) / (spec.max() - spec.min())

noisySpectra = distort.add_noise(cleanSpectra, level=noiseLevel, seed=0)


lenghts: List[int] = [7, 11, 19, 21, 27, 35, 51]
orders: List[int] = [2, 3, 4]

numSpecs: int = cleanSpectra.shape[0]

maxCorr = 0

for window_length in lenghts:
    for order in orders:
        if order >= window_length:
            print(f"Skipping impossible parameter combination: window-length {window_length}, order: {order}")
            continue

        corrs: np.ndarray = np.zeros(numSpecs)
        for i in range(numSpecs):
            noisy: np.ndarray = noisySpectra[i, :]
            savgol = savgol_filter(noisy, window_length=window_length, polyorder=order)
            corrs[i] = np.corrcoef(cleanSpectra[i, :], savgol)[0, 1] * 100

        meanCorr = np.mean(corrs)
        if meanCorr > maxCorr:
            maxCorr = meanCorr
            print("############ NEW MAX CORR:")

        print(f"w-length: {window_length}, order: {order}, mean corr: {round(meanCorr)}, std corr: {round(np.std(corrs))}")


import matplotlib.pyplot as plt
worstInd, bestInd = np.argmin(corrs), np.argmax(corrs)

plt.subplot(121)
plt.plot(cleanSpectra[worstInd, :])
plt.plot(noisySpectra[worstInd, :]+0.2)
plt.plot(savgol_filter(noisySpectra[worstInd, :], window_length=window_length, polyorder=order)+0.4)
plt.title(f"Correlation is {corrs[worstInd]} %")

plt.subplot(122)
plt.plot(cleanSpectra[bestInd, :])
plt.plot(noisySpectra[bestInd, :]+0.2)
plt.plot(savgol_filter(noisySpectra[bestInd, :], window_length=window_length, polyorder=order)+0.4)
plt.title(f"Correlation is {corrs[bestInd]} %")