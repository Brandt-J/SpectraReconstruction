import numpy as np
from scipy.signal import gaussian
from scipy.optimize import least_squares
from typing import List, Tuple
from functools import lru_cache


def getSpecFromPeaks(posWidthAreas: List[Tuple[float, float, float]], specLength: int) -> np.ndarray:
    """
    Creates a fake spectrum, using the provided list of tuples of: (PeakPosition, PeakWidth, PeakArea).
    :param posWidthAreas: 
    :param specLength: 
    :return: 
    """
    spec: np.ndarray = np.zeros(specLength)
    for pos, width, area in posWidthAreas:
        spec += getGaussOfProfile(specLength, pos, width, area)
    return spec


def recoverPeakAreas(intensities: np.ndarray, posWidthAreas: List[Tuple[float, float, float]]) -> List[float]:
    """
    Recovers the actual PeakAreas from a (noisy) spectrum, given by its intensities
    :param intensities: spec Intensities
    :param posWidthAreas: List of (peakPosition, PeakWidth, PeakArea) that was used for creating the spectrum
    :return:
    """
    errFunc = lambda x: getError(x, intensities, posWidthAreas)
    x0 = [i[2] for i in posWidthAreas]
    opt = least_squares(errFunc, np.array(x0), bounds=(np.array([0] * len(posWidthAreas)),
                                                       np.array([np.inf] * len(posWidthAreas))), method='dogbox')
    return list(opt.x)


@lru_cache(maxsize=1000)
def getGaussOfProfile(specLength: int, pos: float, width: float, area: float) -> np.ndarray:
    sampleLength = int(specLength * 3)
    peak: np.ndarray = gaussian(sampleLength, std=width)  # returns a gauss peak centered in spectrum
    peak /= peak.max()  # the gaussian returns it ALMOST normalized, this is just for the last few percent fractions..
    center = np.argmax(peak)
    start = int(round(center - pos))
    end = start + specLength
    fullPeak: np.ndarray = np.zeros(specLength)
    fullPeak += peak[start:end]
    fullPeak *= area
    return fullPeak


def getError(curAreas: np.ndarray, spec: np.ndarray, posWidthAreas: List[Tuple[float, float, float]]) -> float:
    newPosWithAreas = []
    for i in range(len(posWidthAreas)):
        newPosWithAreas.append((posWidthAreas[i][0], posWidthAreas[i][1], curAreas[i]))
    expectedSpec: np.ndarray = getSpecFromPeaks(newPosWithAreas, len(spec))
    return float(np.sum(np.abs(spec - expectedSpec)))
