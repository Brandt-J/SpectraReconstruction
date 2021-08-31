import os.path
import random
import time
from typing import List

import numpy as np

import importData as io
import outGraphs as out
from Reconstruction import prepareSpecSet, getDenseReconstructor
from globals import SPECLENGTH
from functions import remapSpecArrayToWavenumbers

t0 = time.time()
#
noisySpecs, cleanSpecs, specNames, wavenumbers = io.load_microFTIR_spectra(SPECLENGTH, maxCorr=0.8)
numSpecs = noisySpecs.shape[0]
print(f'loading and remapping {numSpecs} spectra took {round(time.time()-t0)} seconds')
experimentTitle = 'MicroFTIR Spectra'

np.random.seed(42)
# validationIndices = [i for i in range(numSpecs) if specNames[i] in ['PP', 'PS']]
fracValid = 0.010
validationIndices: list = random.sample(range(numSpecs), round(numSpecs * fracValid))
valIndSet = set(validationIndices)
trainIndices: List[int] = [i for i in range(numSpecs) if i not in valIndSet]

trainSpectra = prepareSpecSet(cleanSpecs[trainIndices, :], transpose=False)
noisyTrainSpectra = prepareSpecSet(noisySpecs[trainIndices, :], transpose=False)

testSpectra = prepareSpecSet(cleanSpecs[validationIndices, :], transpose=False)
noisyTestSpectra = prepareSpecSet(noisySpecs[validationIndices, :], transpose=False)
print(f'{len(trainSpectra)} Specs for Training, {len(testSpectra)} Specs for Testing')
rec = getDenseReconstructor()
rec.compile(optimizer='adam', loss='mse')
history = rec.fit(noisyTrainSpectra, trainSpectra,
                  epochs=200, validation_data=(noisyTestSpectra, testSpectra),
                  batch_size=32, shuffle=True)
#
#
folder: str = r"C:\Users\xbrjos\Desktop\tempMP\tests\Acquisition Resolution Comparison\PET_PS_PVC\spectra"
realSpecs: np.ndarray = np.load(os.path.join(folder, "Spectra Series 0 - Copy.npy"))
realSpecs = remapSpecArrayToWavenumbers(realSpecs, wavenumbers)
realSpecs = prepareSpecSet(realSpecs[:, 1:], transpose=True)

reconstructed = np.array(rec.call(realSpecs))
reconstructed = np.hstack((wavenumbers[:, np.newaxis], reconstructed.transpose()))
np.save(os.path.join(folder, "Spectra Series 0.npy"), reconstructed)
# histplot = out.getHistPlot(history.history, title=experimentTitle, annotate=False, marker=None)
# reconstructedSpecs = rec.call(noisyTestSpectra)
# specPlot, boxPlot = out.getSpectraComparisons(testSpectra, noisyTestSpectra, reconstructedSpecs,
#                                               title=experimentTitle, includeSavGol=False, wavenumbers=wavenumbers)
#
# corrPlot = out.getCorrelationPCAPlot(noisyTestSpectra.numpy(), reconstructedSpecs.numpy(),
#                                      testSpectra.numpy(), noisyTrainSpectra.numpy())
