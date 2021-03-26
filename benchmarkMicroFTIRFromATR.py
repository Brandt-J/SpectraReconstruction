import numpy as np
import time

from functions import reduceSpecsToNWavenumbers
import distort
import importData as io
import outGraphs as out
from Reconstruction import prepareSpecSet, Reconstructor


specLength, latentDims = 512, 128
noiseLevel = 0.15
t0 = time.time()
numTrainSpectra, numTestSpectra = 90, 20
numVariationsTrain, numVariationsTest = 5000, 100

# experimentTitle = f'Training with {numTrainSpectra} spectra and {numVariationsTrain} variations,\n' \
#                   f'Validation with {numTestSpectra} spectra and {numVariationsTest} variations.\n' \
#                   f'SpecLength: {specLength}, numLatentDimentions: {latentDims}'
experimentTitle = 'Trained on ATR spectra, applied to µFTIR spectra'
print(experimentTitle)

specNames, spectra = io.load_specCSVs_from_directory("ATR Spectra", maxSpectra=numTrainSpectra+numTestSpectra)
wavenums = spectra[:, 0].copy()
spectra = reduceSpecsToNWavenumbers(spectra, specLength)
print(f'loading and remapping spectra took {round(time.time()-t0)} seconds')

specs: np.ndarray = spectra[:, 1:]
trainSpectra: np.ndarray = np.tile(specs[:, :numTrainSpectra], (1, numVariationsTrain))

testSpectra = np.tile(specs[:, numTrainSpectra:], (1, numVariationsTest))
# testSpectra = np.hstack((testSpectra, np.random.rand(512, 50)))

t0 = time.time()
np.random.seed(42)
# noisyTrainSpectra = distort.add_noise(trainSpectra, level=noiseLevel/2)
# noisyTestSpectra = distort.add_noise(testSpectra, level=noiseLevel/2)
# for _ in range(3):
#     noisyTrainSpectra = distort.add_distortions(noisyTrainSpectra, level=noiseLevel*5)
#     noisyTrainSpectra = distort.add_ghost_peaks(noisyTrainSpectra, level=noiseLevel*5)
#     noisyTestSpectra = distort.add_distortions(noisyTestSpectra, level=noiseLevel*5)
#     noisyTestSpectra = distort.add_ghost_peaks(noisyTestSpectra, level=noiseLevel*5)
# print(f'Distorting spectra took {round(time.time()-t0, 2)} seconds')
#
# np.save("noisyTrain.npy", noisyTrainSpectra)
# np.save("noisyTest.npy", noisyTestSpectra)
noisyTrainSpectra = np.load("noisyTrain.npy")
noisyTestSpectra = np.load("noisyTest.npy")

trainSpectra = prepareSpecSet(trainSpectra)
testSpectra = prepareSpecSet(testSpectra)
noisyTrainSpectra = prepareSpecSet(noisyTrainSpectra)
noisyTestSpectra = prepareSpecSet(noisyTestSpectra)

rec = Reconstructor(specLength=specLength, latentDims=latentDims)
rec.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = rec.fit(noisyTrainSpectra, trainSpectra,
                  epochs=10, validation_data=(noisyTestSpectra, testSpectra),
                  batch_size=32, shuffle=True)
histplot = out.getHistPlot(history.history, title=experimentTitle)


noisySpecs, cleanSpecs, specNames, soilSpecs, wavenumbers = io.load_microFTIR_spectra(specLength)
noisySpecs = prepareSpecSet(noisySpecs, transpose=False)
cleanSpecs = prepareSpecSet(cleanSpecs, transpose=False)

reconstructedSpecs = rec.call(noisySpecs)
specPlot, boxPlot = out.getSpectraComparisons(cleanSpecs, noisySpecs, reconstructedSpecs,
                                              includeSavGol=False,
                                              wavenumbers=wavenums,
                                              title=experimentTitle)

corrPlot = out.getCorrelationPCAPlot(noisySpecs.numpy(), reconstructedSpecs.numpy(),
                                     cleanSpecs.numpy(), noisyTrainSpectra.numpy())
