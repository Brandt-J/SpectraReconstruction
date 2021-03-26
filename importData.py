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
import os
import numpy as np
from imblearn import over_sampling, under_sampling
from typing import List, Tuple

from functions import getNMostDifferentSpectra, reduceSpecsToNWavenumbers, remapSpectrumToWavenumbers, remapSpecArrayToWavenumbers


def load_microFTIR_spectra(specLength: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the Spectra from the MicroFTIR Spectra directory.
    :param specLength: Number of wavenumbers for the spectra
    :return: (NoisyPolymers, CleanPolymers, polymNames, SoilSpectra, wavenumbers)
    """
    path: str = "MicroFTIRSpectra"
    allSpecs: np.ndarray = np.load(os.path.join(path, "polymers.npy"))
    allSpecs = reduceSpecsToNWavenumbers(allSpecs, specLength)
    soilSpecs: np.ndarray = np.load(os.path.join(path, "soil.npy"))
    wavenumbers = allSpecs[:, 0].copy()
    soilSpecs = remapSpecArrayToWavenumbers(soilSpecs, wavenumbers).transpose()

    polyms_noisy = np.transpose(allSpecs[:, 1:])
    polymNames = np.genfromtxt(os.path.join(path, "polymerNames.txt"), dtype=str)

    # sampler = over_sampling.SMOTE(random_state=42)
    # sampler = over_sampling.ADASYN(random_state=42)
    # sampler = over_sampling.RandomOverSampler(random_state=42)
    # sampler = under_sampling.ClusterCentroids(random_state=42)
    # print(f'balancing with: {sampler}, num specs before: {len(polyms_noisy)}')
    # polyms_noisy, polymNames = sampler.fit_resample(polyms_noisy, polymNames)
    # print(f'num specs after balancing: {len(polyms_noisy)}')
    # allSpecs = allSpecs.transpose()

    uniqueAssignments: List[str] = list(np.unique(polymNames))
    uniqueSpectra: List[np.ndarray] = []
    for assignment in uniqueAssignments:
        cleanSpec = np.loadtxt(os.path.join(path, assignment+".txt"), delimiter=',')
        cleanSpec = remapSpectrumToWavenumbers(cleanSpec, wavenumbers)
        uniqueSpectra.append(cleanSpec)
    Polyms_Clean: np.ndarray = np.zeros_like(polyms_noisy)

    for i, assignment in enumerate(polymNames):
        specIndex = uniqueAssignments.index(assignment)
        Polyms_Clean[i, :] = uniqueSpectra[specIndex][:, 1]

    return polyms_noisy, Polyms_Clean, polymNames, soilSpecs, wavenumbers


def load_specCSVs_from_directory(path: str, fixName: str = None, maxSpectra=1e6) -> Tuple[List[str], np.ndarray]:
    """
    Reads Spectra from CSV viles in path. If given, a fix name is assigned to each spectrum
    :param path: Directory path
    :param fixName: If None, each spectrum has the filename as name, otherwise the indicated fixName
    :param maxSpectra: Max number of spectra to take.
    :return: Tuple[Assignment List, spectra array]
    """
    spectra: np.ndarray = None
    names: list = []
    for file in os.listdir(path):
        if file.lower().endswith('.csv'):
            curSpec: list = []
            specName = fixName if fixName is not None else file.lower().split('.csv')[0]
            names.append(specName)

            with open(os.path.join(path, file), 'r') as fp:
                if spectra is None:
                    wavenumbers = []
                    # for index, row in enumerate(reader):
                    for line in fp.readlines():
                        wavenum, intensity = get_numbers_from_line(line)
                        curSpec.append(intensity)
                        wavenumbers.append(wavenum)

                    spectra = np.array(wavenumbers)
                else:
                    tmpSpec = []
                    tmpWavenumber = []
                    for line in fp.readlines():
                        wavenum, intensity = get_numbers_from_line(line)
                        tmpSpec.append(intensity)
                        tmpWavenumber.append(wavenum)

                    tmpSpec = np.array(tmpSpec)
                    tmpWavenumber = np.array(tmpWavenumber)
                    for number in spectra[:, 0]:
                        index = np.argmin(np.abs(tmpWavenumber - number))
                        curSpec.append(tmpSpec[index])

                if len(spectra.shape) == 1:
                    spectra = np.append(spectra[:, np.newaxis], np.array(curSpec)[:, np.newaxis], axis=1)
                else:
                    spectra = np.append(spectra, np.array(curSpec)[:, np.newaxis], axis=1)

    numSpectra = spectra.shape[1] - 1
    if numSpectra > maxSpectra:
        names, spectra = getNMostDifferentSpectra(names, spectra, maxSpectra)

    return names, spectra


def get_numbers_from_line(line: str) -> Tuple[float, float]:
    """
    Takes a line from a csv or txt document and checks for delimiter and decimal separator to yield exactly
    two float numbers
    :param line:
    :return: the two float numbers as Tuple
    """
    origline = line.strip()
    try:
        line = origline.split(';')
        assert len(line) == 2
        numbers: Tuple[float, float] = float(line[0].replace(',', '.')), float(line[1].replace(',', '.'))
    except AssertionError:
        line = origline.split(',')
        assert len(line) == 2
        numbers: Tuple[float, float] = float(line[0]), float(line[1])
    except AssertionError as e:
        print(e)
        raise
    return numbers
