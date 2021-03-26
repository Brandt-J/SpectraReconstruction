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


from typing import List, Tuple, Dict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
from sklearn.preprocessing import StandardScaler


def getNMostDifferentSpectra(assignments: List[str], spectra: np.ndarray, n: int) -> Tuple[List[str], np.ndarray]:
    """
    Takes a set of spectra and returns the indices of the n spectra that are furthest apart from each other,
    in terms of the first two PCA components.
    :param assignments: List of M-1 spectra
    :param spectra: (NxM) array of spec of M-1 spectra with N wavenumbers (wavenumbers in first column).
    :param n: Desired number of spectra to keep
    :return: Tuple[shortened AssignmentList, shortened Spectra array]
    """
    maxIndex = np.argmin(np.abs(spectra[:, 0] - 2000))  # only go up to 2000 cm-1, above it's so unspecific...
    intensities = spectra[:maxIndex, 1:]
    intensities = StandardScaler().fit_transform(intensities.transpose())
    indices: List[int] = []
    pca: PCA = PCA(n_components=2, random_state=42)
    princComps: np.ndarray = pca.fit_transform(intensities)
    validIndices = np.where(np.logical_and(princComps[:, 0] < 10, princComps[:, 1] < 10))[0]
    princComps: np.ndarray = princComps[validIndices, :]
    centers = k_means(princComps, n, random_state=42)[0]

    for i in range(n):
        distances = np.linalg.norm(princComps-centers[i, :], axis=1)
        indices.append(validIndices[int(np.argmin(distances))])

    # FOR DEBUG DISPLAY
    # import matplotlib.pyplot as plt
    # plt.scatter(princComps[:, 0], princComps[:, 1], color='lightgray')
    # plt.scatter(princComps[indices, 0], princComps[indices, 1], color='black')
    # plt.xlabel('PC 1')
    # plt.ylabel('PC 2')
    # plt.title(f'Chosing {n} out of {princComps.shape[0]} spectra')
    # plt.show(block=True)

    assignments = [assignments[i] for i in indices]
    indices = [0] + [i + 1 for i in indices]
    spectra = spectra[:, indices]
    assert len(assignments) == spectra.shape[1] - 1
    return assignments, spectra


def mapSpectrasetsToSameWavenumbers(set1: np.ndarray, set2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matches two sets of spectra to each other, so that both have the same wavenumber axis (the shorter set is used)
    :param set1: np.ndarray shape(N, M) with N wavenumbers and M-1 spectra, wavenumbers in first column
    :param set2: np.ndarray shape(K, L) with K wavenumbers and L-1 spectra, wavenumbers in first column
    :return: Tuple of set1 and set2, with same wavenumber axis.
    """
    shorterWavenums, longerWavenums = set1[:, 0], set2[:, 0]
    set1IsShorter: bool = True
    if len(shorterWavenums) > len(longerWavenums):
        shorterWavenums, longerWavenums = longerWavenums, shorterWavenums
        set1IsShorter = False

    newSet1: np.ndarray = set1 if set1IsShorter else set2
    newSet2: np.ndarray = np.zeros((set1.shape[0], set2.shape[1])) if set1IsShorter else np.zeros((set2.shape[0], set1.shape[1]))
    newSet2[:, 0] = shorterWavenums

    for i, num in enumerate(shorterWavenums):
        correspondInd = np.argmin(np.abs(longerWavenums - num))
        newSet2[i, :] = set2[correspondInd, :] if set1IsShorter else set1[correspondInd, :]

    returnTuple: Tuple[np.ndarray, np.ndarray] = (newSet1, newSet2) if set1IsShorter else (newSet2, newSet1)
    assert returnTuple[0].shape[1] == set1.shape[1] and returnTuple[1].shape[1] == set2.shape[1]
    assert returnTuple[0].shape[0] == returnTuple[1].shape[0] == len(shorterWavenums)
    return returnTuple


def remapSpecArrayToWavenumbers(spectra: np.ndarray, wavenumbers: np.ndarray) -> np.ndarray:
    """
    Takes a spectrum array and maps it to the currently present spectra.
    :param spectra: (N, M) shape array of M-1 spectra with wavenumbs in first column
    :param wavenumbers: The wavenumbers to map to. If None, the wavenumbers of the currently present spectra set
    is used.
    :return: shape (L, M) shape spectrum with new wavenumber axis
    """
    newSpecs: np.ndarray = np.zeros((len(wavenumbers), spectra.shape[1]))
    newSpecs[:, 0] = wavenumbers
    for i in range(spectra.shape[1]-1):
        newSpecs[:, i+1] = remapSpectrumToWavenumbers(spectra[:, [0, i+1]], wavenumbers)[:, 1]
    return newSpecs


def remapSpectrumToWavenumbers(spectrum: np.ndarray, wavenumbers: np.ndarray = None) -> np.ndarray:
    """
    Takes a spectrum array and maps it to the currently present spectra.
    :param spectrum: (N, 2) shape spectrum with wavenumbs in first column
    :param wavenumbers: The wavenumbers to map to. If None, the wavenumbers of the currently present spectra set
    is used.
    :return: shape (M, 2) shape spectrum with new wavenumber axis
    """
    newSpec = np.zeros((len(wavenumbers), 2))
    newSpec[:, 0] = wavenumbers
    for i in range(len(wavenumbers)):
        clostestIndex = np.argmin(np.abs(spectrum[:, 0] - wavenumbers[i]))
        newSpec[i, 1] = spectrum[clostestIndex, 1]
    return newSpec


def reduceSpecsToNWavenumbers(spectra: np.ndarray, n: int) -> np.ndarray:
    """
    Reduces the present spectra to only have n wavenumbers
    :param spectra: shape (NxM) spectra array of M-1 spectra with N wavenums (wavenums in first column)
    :param n: desired number of wavenumbers
    :return:
    """
    curWavenums: np.ndarray = spectra[:, 0]
    newSpecs: np.ndarray = np.zeros((n, spectra.shape[1]))
    newWavenums: np.ndarray = np.linspace(curWavenums.min(), curWavenums.max(), n)
    newSpecs[:, 0] = newWavenums

    for i, wavenum in enumerate(newWavenums):
        closestIndex = np.argmin(np.abs(curWavenums - wavenum))
        newSpecs[i, 1:] = spectra[closestIndex, 1:]

    return newSpecs

