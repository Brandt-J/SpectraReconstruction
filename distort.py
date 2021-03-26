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


import numpy as np
from scipy.signal import gaussian
import matplotlib.pyplot as plt
import numba


@numba.njit()
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


@numba.njit()
def add_distortions(spectra: np.ndarray, level: float = 0.1, seed: int = 42) -> np.ndarray:
    """
    Adds random distortions with max height of "level" to the set of spectra.
    :param spectra: (N, M) array, M spectra with N wavenumbers, no wavenumbers
    :param level: Max height of added distortion, relative to normalized intensity
    :param seed: Random seed
    :return: the altered spectra, shape (N, M) array
    """
    spectra: np.ndarray = spectra.copy()
    for i in range(spectra.shape[1]):
        seed += 1
        np.random.seed(seed)

        intensities: np.ndarray = spectra[:, i]
        # for each, first normalize, then add the distortion, then scale back up to orig dimensions
        minVal, maxVal = intensities.min(), intensities.max()
        intensities -= minVal
        intensities /= (maxVal - minVal)

        # Bend Baseline
        randIntens = np.random.rand() * level
        randFreq = 5e-5 + np.random.rand() * 4e-3
        randOffset = np.random.rand() * 1000
        distortion = np.sin(spectra[:, 0] * randFreq + randOffset)
        for j in range(np.random.randint(1, 5)):
            power = np.random.randint(1, 5)
            distortion += 0.5 * np.random.rand() * np.sin(spectra[:, 0] * randFreq * (j+3) + (j+1) * randOffset) ** power

        distortion -= distortion.min()
        distortion /= distortion.max()
        # Have distortion only on the left-hand side of spectra (that's, where they usually occur)
        steep = float(np.random.rand()) + 1.0
        center = float(np.random.rand()) * 0.4 + 0.2
        distortion *= invsigmoid(spectra[:, 0], steepness=steep, center=center)
        intensities = (1 - randIntens) * intensities + randIntens * distortion

        intensities *= (maxVal - minVal)
        intensities += minVal

        spectra[:, i] = intensities

    return spectra


def add_ghost_peaks(spectra: np.ndarray, level: float = 0.1, seed: int = 42) -> np.ndarray:
    spectra: np.ndarray = spectra.copy()

    minDistortWidth, maxDistortWidth = round(spectra.shape[0] * 0.6), round(spectra.shape[0] * 0.9)
    minDistortStd, maxDistortStd = 20, 40

    for i in range(spectra.shape[1]):
        seed += 1
        np.random.seed(seed)
        intensities: numba.types.float64[:] = spectra[:, i]
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


@numba.njit()
def add_noise(spectra: np.ndarray, level: float = 0.1, seed: int = 42) -> np.ndarray:
    """
    Adds random noise to the spectra..
    :param spectra: (N, M) array, M spectra with N wavenumbers, no wavenumbers included
    :param level: Level of noise
    :param seed: random seed
    :return: new Spectra (N, M) array
    """
    np.random.seed(seed)
    spectra = spectra.copy()
    numWavenums: int = spectra.shape[0]

    for i in range(spectra.shape[1]-1):
        randomNoise = np.random.rand(numWavenums)
        spectra[:, i] = (1-level)*spectra[:, i] + level*randomNoise
    return spectra


@numba.njit()
def invsigmoid(xaxis: np.ndarray, steepness: float = 1, center: float = 0.5) -> np.ndarray:
    """
    Calculates an inverted sigmoid function to the provided x-axis. It goes from 1.0 at lowest x-values to 0.0 at highest
    x-value.
    :param xaxis: The x-axis to use
    :param steepness: Higher value produce a steeper step
    :param center: At which fraction of the x-axis the curve is at 0.5.
    :return: inverted sigmoid matching the x-axis
    """
    # Normalize and scale the x-axis
    xaxis = xaxis.copy()
    xaxis -= xaxis.min()
    xaxis /= xaxis.max()
    xaxis *= steepness * 10
    xaxis -= steepness * 10 * center

    sigm: np.ndarray = 1 / (1 + np.exp(-xaxis))  # Calculate the logistic sigmoid function

    # Normalize the sigmoid
    sigm -= sigm.min()
    sigm /= sigm.max()

    sigm = sigm[::-1]  # Invert the sigmoid
    return sigm
