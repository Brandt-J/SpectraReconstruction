import time
import numpy as np
import os
from typing import List
import random

import outGraphs as out
import importData as io
from Reconstruction import prepareSpecSet, Reconstructor, getDenseReconstructor
from globals import SPECLENGTH

os.chdir(os.path.dirname(os.getcwd()))

t0 = time.time()

noisySpecs, cleanSpecs, specNames, wavenumbers = io.load_microFTIR_spectra(SPECLENGTH, maxCorr=0.5)
numSpecs = noisySpecs.shape[0]
print(f'loading and remapping {numSpecs} spectra took {round(time.time()-t0)} seconds')
experimentTitle = 'MicroFTIR Spectra'

np.random.seed(42)
random.seed(42)
fracValid = 0.50
validationIndices: list = random.sample(range(numSpecs), round(numSpecs * fracValid))
valIndSet = set(validationIndices)
trainIndices: List[int] = [i for i in range(numSpecs) if i not in valIndSet]

trainSpectra = prepareSpecSet(cleanSpecs[trainIndices, :], transpose=False)
noisyTrainSpectra = prepareSpecSet(noisySpecs[trainIndices, :], transpose=False)

testSpectra = prepareSpecSet(cleanSpecs[validationIndices, :], transpose=False)
noisyTestSpectra = prepareSpecSet(noisySpecs[validationIndices, :], transpose=False)
print(f'{len(trainSpectra)} Specs for Training, {len(testSpectra)} Specs for Testing')

rec: Reconstructor = getDenseReconstructor(dropout=0.0)

t0 = time.time()
history = rec.fit(noisyTrainSpectra, trainSpectra,
                  epochs=200,
                  validation_data=(noisyTestSpectra, testSpectra),
                  batch_size=32, shuffle=True)
print(f"Training took {round(time.time()-t0, 2)} seconds.")

t0 = time.time()
reconstructedSpecs = rec.call(noisyTestSpectra)
print(f'reconstruction took {round(time.time()-t0, 2)} seconds')
histPLot = out.getHistPlot(history.history, annotate=False)
specPlot, boxPlot = out.getSpectraComparisons(testSpectra, noisyTestSpectra, reconstructedSpecs,
                                              includeSavGol=False,
                                              randomIndSeed=1,
                                              wavenumbers=wavenumbers,
                                              title=experimentTitle)
