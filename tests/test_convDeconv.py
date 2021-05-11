import unittest

from peakConvDeconv import *
from outGraphs import getDeconvolutionAccuracies


class TestConvolutionDeconvolution(unittest.TestCase):
    def test_getGauss(self):
        for center in [20, 50, 80]:
            for area in [0.5, 1.0, 3.0]:
                peak: np.ndarray = getGaussOfProfile(100, center, 10, area)
                self.assertEqual(peak.max(), area)
                self.assertEqual(round(np.argmax(peak)), center)

    def test_recoverPeakAreas(self):
        centers = [30, 20, 70]
        widths = [5, 5, 7]
        areas = [1, 3, 2]
        # Note: The first two Peaks overlap quite strongly, so the second one is a shoulder to the first one..
        centerWidthAreas = list(zip(centers, widths, areas))
        spec = getSpecFromPeaks(centerWidthAreas, 100)

        recoveredAreas: List[float] = recoverPeakAreas(spec, centerWidthAreas)
        for area1, area2 in zip(areas, recoveredAreas):
            self.assertAlmostEqual(area1, area2, places=4)

        np.random.seed(42)
        noisySpec = spec + np.random.rand(len(spec)) * 0.02
        recoveredAreas: List[float] = recoverPeakAreas(noisySpec, centerWidthAreas)
        for area1, area2 in zip(areas, recoveredAreas):
            self.assertAlmostEqual(area1, area2, places=1)

    def test_getDeconvolutionAccuracies(self) -> None:
        centers = [30, 20, 70]
        widths = [5, 5, 7]
        areas = [1, 3, 2]
        # Note: The first two Peaks overlap quite strongly, so the second one is a shoulder to the first one..
        centerWidthAreas = list(zip(centers, widths, areas))
        spec = getSpecFromPeaks(centerWidthAreas, 100)
        accuracies = getDeconvolutionAccuracies([centerWidthAreas], spec[np.newaxis, :])
        self.assertEqual(accuracies, [100, 100, 100])

        areas = [area*2 for area in areas]
        centerWidthAreas = list(zip(centers, widths, areas))
        accuracies = getDeconvolutionAccuracies([centerWidthAreas], spec[np.newaxis, :])
        for acc in accuracies:
            self.assertAlmostEqual(acc, 50, places=3)


