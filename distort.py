"""
SPECTRA PROCESSING
Copyright (C) 2020 Josef Brandt, University of Gothenborg.
<josef.brandt@gu.se>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program, see COPYING.
If not, see <https://www.gnu.org/licenses/>.
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import gaussian
from typing import Tuple


def append_n_distorted_copies(spectra: np.ndarray, n: int, level: float = 0.3, seed: int = 42,
                              plot: bool = False) -> np.ndarray:
    """
    Appends n copies with distortions of the given spectra set to the original set.
    :param spectra: (N, M) array, M-1 spectra with N wavenumbers, wavenumbers in first column
    :param n: int, number of variations to create
    :param level: Max height of added distortion, relative to normalized intensity
    :param seed: Random seed
    :param plot: Whether or not to plot a random selection of spectra
    :return: the altered spectra, shape (N, (M-1)*(n+1) + 1) array
    """
    numSpectra: int = spectra.shape[1] - 1
    finalSpectra: np.ndarray = np.zeros((spectra.shape[0], numSpectra * (n + 1) + 1))
    finalSpectra[:, :spectra.shape[1]] = spectra
    iterationSeed = seed
    if plot:
        np.random.seed(seed)
        plotIterations = np.random.randint(0, n, 5)
        plotIndices = np.random.randint(0, numSpectra, 3)
        plt.subplot(2, 3, 1)
        for offset, j in enumerate(plotIndices):
            plt.plot(spectra[:, 0], spectra[:, j+1] + 0.1*offset)
        plotNumber = 2

    for i in range(n):
        newSpecs: np.ndarray = spectra.copy()
        np.random.seed(iterationSeed)

        iterationSeed += 1
        newSpecs = add_noise(newSpecs, level=0.1, seed=iterationSeed)
        newSpecs = add_distortions(newSpecs, level=level, seed=iterationSeed)  # amplify distortions
        if np.random.rand() > 0.4:
            newSpecs = add_ghost_peaks(newSpecs, level=level, seed=iterationSeed)

        start, stop = (i+1) * numSpectra + 1, (i+2) * numSpectra + 1
        finalSpectra[:, start:stop] = newSpecs[:, 1:]
        if plot and i in plotIterations:
            plt.subplot(2, 3, plotNumber)
            for offset, j in enumerate(plotIndices):
                plt.plot(newSpecs[:, 0], newSpecs[:, j + 1] + 0.1 * offset)
            plotNumber += 1

    if plot:
        plt.tight_layout()
        plt.show(block=True)

    return finalSpectra


def distort_to_max_correlation(spectra: np.ndarray, maxCorr: float, seed: int = 42) -> np.ndarray:
    """
    Adds random distortions to each spectrum, until its correlation to the original spectrum is less then maxCorr.
    :param spectra: (N, M) array, M spectra with N wavenumbers, no wavenumbers
    :param maxCorr: The maximum correlation to input spectra
    :param seed: Random seed
    :return: the altered spectra, shape (N, M) array
    """
    origSpecs: np.ndarray = spectra.copy()
    specs: np.ndarray = spectra.copy()
    for i in range(spectra.shape[1]):
        corr = np.corrcoef(origSpecs[:, i], specs[:, i])[0, 1]
        while corr > maxCorr:
            specs[:, i] = add_noise(specs[:, i][:, np.newaxis], seed=seed)[:, 0]
            specs[:, i] = add_distortions(specs[:, i][:, np.newaxis], seed=seed)[:, 0]
            specs[:, i] = add_ghost_peaks(specs[:, i][:, np.newaxis], seed=seed)[:, 0]
            corr = np.corrcoef(origSpecs[:, i], specs[:, i])[0, 1]
            seed += 1

    return specs


def add_distortions(spectra: np.ndarray, level: float = 0.1, seed: int = 42) -> np.ndarray:
    """
    Adds random distortions with max height of "level" to the set of spectra.
    :param spectra: (N, M) array, M spectra with N wavenumbers, no wavenumbers
    :param level: Max height of added distortion, relative to normalized intensity
    :param seed: Random seed
    :return: the altered spectra, shape (N, M) array
    """
    spectra: np.ndarray = spectra.copy()
    xaxis = np.arange(spectra.shape[0])
    for i in range(spectra.shape[1]):
        seed += 1
        np.random.seed(seed)

        intensities: np.ndarray = spectra[:, i]
        # for each, first normalize, then add the distortion, then scale back up to orig dimensions
        minVal, maxVal = intensities.min(), intensities.max()
        intensities -= minVal
        intensities /= (maxVal - minVal)

        # Bend Baseline
        randIntens = min([np.random.rand() * level, 0.9])
        distortion = _generateSinDistortion(xaxis, (1e-4, 1e-4+0.05))
        intensities = (1 - randIntens) * intensities + randIntens * distortion

        intensities *= (maxVal - minVal)
        intensities += minVal

        spectra[:, i] = intensities

    return spectra


def _generateSinDistortion(xaxis: np.ndarray, frequencyRange: Tuple[float, float],
                           numModes: Tuple[int, int] = (1, 3), left: bool = True) -> np.ndarray:
    """
    Generates a sinusoidal distortion
    :param xaxis:
    :param frequencyRange: Min and Max of frequency to use
    :param numModes: Min and Max number of modes to create
    :param left: If the sin distortions are stronger on the left, or on the right.
    :return:
    """
    randFreq = frequencyRange[0] + np.random.rand() * (frequencyRange[1]-frequencyRange[0])
    randOffset = np.random.rand() * 1000
    distortion = np.sin(xaxis * randFreq + randOffset)
    if numModes[0] == numModes[1]:
        modeRange = [numModes[0]]
    else:
        modeRange = range(np.random.randint(1, 3))
    for _ in modeRange:
        distortion *= np.random.rand() * np.sin(xaxis * randFreq + randOffset / 2)
    distortion -= distortion.min()
    distortion /= distortion.max()

    steep = float(np.random.rand()) + 1.0
    center = float(np.random.rand()) * 0.4 + 0.2
    factor = _sigmoid(xaxis, steepness=steep, center=center)
    if left:
        factor = factor[::-1]
    distortion = distortion * factor
    return distortion


def add_fluorescence(spectra: np.ndarray, levelRange: Tuple[float, float] = (1, 5), seed: int = 42) -> np.ndarray:
    """
    Adds a broad fluorescence_peak contribution, so that the fluoresence is stronger by a factor within the "levelRange" than
    the original signal
    :param spectra: (N, M) array, M spectra with N wavenumbers, no wavenumbers
    :param levelRange: (minimum, maximum) Factor of what the fluorescence_peak is stronger than the signal
    :param seed: random seed to use
    """
    spectra = spectra.copy()
    np.random.seed(seed)
    for i in range(spectra.shape[1]):
        curSpec: np.ndarray = spectra[:, i]
        curSpec = (curSpec - curSpec.min()) / (curSpec.max() - curSpec.min())

        gaussWidth = int(np.random.rand() * spectra.shape[0] * 0.4 + spectra.shape[0] * 0.6)  # 0.8 - 1.0 times spec length
        gaussStd = 0.3 * gaussWidth
        
        fluorescence_peak: np.ndarray = gaussian(gaussWidth, gaussStd)
        fluorescence_peak = (fluorescence_peak - fluorescence_peak.min()) / (fluorescence_peak.max() - fluorescence_peak.min())
        latestPossibleStart = spectra.shape[0] - gaussWidth
        fluorStart = np.random.randint(0, latestPossibleStart)
        # fluorStart: int = int((spectra.shape[0] - gaussWidth) / 2)
        fluorEnd: int = fluorStart + gaussWidth
        if fluorEnd >= spectra.shape[0]:
            diff = spectra.shape[0] - fluorEnd - 1
            fluorEnd -= diff
            fluorescence_peak = fluorescence_peak[:-diff]

        fluorescence = np.zeros_like(curSpec)
        gaussIntens = levelRange[0] + np.random.rand() * (levelRange[1] - levelRange[0])
        fluorescence[fluorStart:fluorEnd] = fluorescence_peak * gaussIntens

        spectra[:, i] = curSpec + fluorescence

    return spectra


def add_periodic_interferences_raman(spectra: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Adds periodic intereferences at the right hand side of the spectra, as sometimes also accuring in Raman.
    :param spectra: (N, M) array of M spectra with N wavenumbers, no wavenumbers in array
    :param seed: random seed to use
    """
    np.random.seed(seed)
    newSpecs: np.ndarray = spectra.copy()
    xaxis = np.arange(spectra.shape[0])
    for i in range(spectra.shape[1]):
        interf: np.ndarray = _generateSinDistortion(xaxis, (0.05, 0.1), numModes=(1, 1), left=False)
        randLevel = 0.1 + np.random.rand() * 0.2
        curSpec = newSpecs[:, i]
        curSpec = (curSpec - curSpec.min()) / (curSpec.max() - curSpec.min())
        newSpecs[:, i] = curSpec * (1 - randLevel) + randLevel * interf

    return newSpecs


def add_cosmic_ray_peaks(spectra: np.ndarray, numRange: Tuple[int, int], seed: int = 42) -> np.ndarray:
    """
    Adds a broad fluorescence_peak contribution, so that the fluoresence is stronger by a factor within the "levelRange" than
    the original signal
    :param spectra: (N, M) array, M spectra with N wavenumbers, no wavenumbers
    :param numRange: (min, max) Number of cosmic ray peaks to add
    :param seed: random seed to use
    """
    spectra = spectra.copy()
    np.random.seed(seed)
    halfPeakWidth: int = 2
    maxHeightFactor: float = 1.0
    specLength = spectra.shape[0]
    for i in range(spectra.shape[1]):
        numCC = np.random.randint(numRange[0], numRange[1])
        curSpec = spectra[:, i]
        for _ in range(numCC):
            center = np.random.randint(halfPeakWidth, specLength-halfPeakWidth)
            height = np.random.rand() * curSpec.max() * maxHeightFactor
            startHeight, endHeight = curSpec[center-halfPeakWidth], curSpec[center] + height
            curSpec[center-halfPeakWidth:center] += np.linspace(startHeight, endHeight, halfPeakWidth)
            curSpec[center:center+halfPeakWidth] += np.linspace(endHeight, startHeight, halfPeakWidth)
        spectra[:, i] = curSpec

    return spectra

def add_ghost_peaks(spectra: np.ndarray, level: float = 0.1, seed: int = 42) -> np.ndarray:
    spectra: np.ndarray = spectra.copy()

    minDistortWidth, maxDistortWidth = round(spectra.shape[0] * 0.6), round(spectra.shape[0] * 0.9)
    minDistortStd, maxDistortStd = 20, 40

    for i in range(spectra.shape[1]):
        seed += 1
        np.random.seed(seed)
        intensities = spectra[:, i]
        # for each, first normalize, then add the distortion, then scale back up to orig dimensions
        minVal, maxVal = intensities.min(), intensities.max()
        intensities -= minVal
        intensities /= (maxVal - minVal)

        # Add fake peaks
        gaussSize: int = int(round(np.random.rand() * (maxDistortWidth - minDistortWidth) + minDistortWidth))
        gaussStd: float = np.random.rand() * (maxDistortStd - minDistortStd) + minDistortStd
        randGauss = np.array(gaussian(gaussSize, gaussStd) * np.random.rand() * level)

        start = int(round(np.random.rand() * (len(intensities) - gaussSize)))
        intensities[start:start + gaussSize] += randGauss

        intensities *= (maxVal - minVal)
        intensities += minVal

        spectra[:, i] = intensities

    return spectra


def add_noise(spectra: np.ndarray, level: float = 0.1, seed: int = 42, ramanMode: bool = False) -> np.ndarray:
    """
    Adds random noise to the spectra..
    :param spectra: (N, M) array, M spectra with N wavenumbers, no wavenumbers included
    :param level: Level of noise
    :param seed: random seed
    :param ramanMode: If True, the noise level increases towards higher wavenumbers, resembling typical behavior of
    CCD Raman sensors.
    :return: new Spectra (N, M) array
    """
    np.random.seed(seed)
    spectra = spectra.copy()
    numWavenums: int = spectra.shape[0]

    if ramanMode:
        noiseLevelProfile: np.ndarray = np.linspace(level, 2*level, numWavenums)
        signalLevelProfile: np.ndarray = 1 - noiseLevelProfile
        for i in range(spectra.shape[1]):
            randomNoise = np.random.rand(numWavenums)
            spectra[:, i] = signalLevelProfile*spectra[:, i] + noiseLevelProfile*randomNoise
    else:
        for i in range(spectra.shape[1]):
            randomNoise = np.random.rand(numWavenums)
            spectra[:, i] = (1-level)*spectra[:, i] + level*randomNoise
    return spectra


def _sigmoid(xaxis: np.ndarray, steepness: float = 1, center: float = 0.5) -> np.ndarray:
    """
    Calculates an inverted sigmoid function to the provided x-axis. It goes from 1.0 at lowest x-values to 0.0 at highest
    x-value.
    :param xaxis: The x-axis to use
    :param steepness: Higher value produce a steeper step
    :param center: At which fraction of the x-axis the curve is at 0.5.
    :return: inverted sigmoid matching the x-axis
    """
    # Normalize and scale the x-axis
    xaxis = np.float64(xaxis.copy())
    xaxis = xaxis - np.min(xaxis)
    xaxis /= xaxis.max()
    xaxis *= steepness * 10
    xaxis -= steepness * 10 * center

    sigm: np.ndarray = 1 / (1 + np.exp(-xaxis))  # Calculate the logistic sigmoid function

    # Normalize the sigmoid
    sigm -= sigm.min()
    sigm /= sigm.max()
    return sigm
