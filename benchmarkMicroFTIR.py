import numpy as np
import time
from typing import List
import random
from tensorflow.keras.models import load_model

import importData as io
import outGraphs as out
from Reconstruction import prepareSpecSet, Reconstructor

specLength, latentDims = 512, 128
t0 = time.time()

noisySpecs, cleanSpecs, specNames, soilSpecs, wavenumbers = io.load_microFTIR_spectra(specLength)
print(f'loading and remapping spectra took {round(time.time()-t0)} seconds')
experimentTitle = 'MicroFTIR Spectra'

np.random.seed(42)
numSpecs = noisySpecs.shape[0]
# validationIndices = [i for i in range(numSpecs) if specNames[i] in ['PP', 'PS']]
fracValid = 0.5
validationIndices: list = random.sample(range(numSpecs), round(numSpecs * fracValid))
valIndSet = set(validationIndices)
trainIndices: List[int] = [i for i in range(numSpecs) if i not in valIndSet]

trainSpectra = prepareSpecSet(cleanSpecs[trainIndices, :], transpose=False)
noisyTrainSpectra = prepareSpecSet(noisySpecs[trainIndices, :], transpose=False)

testSpectra = prepareSpecSet(cleanSpecs[validationIndices, :], transpose=False)
noisyTestSpectra = prepareSpecSet(noisySpecs[validationIndices, :], transpose=False)


# rec = load_model("micro_Reconstructor")
rec = Reconstructor(specLength=specLength, latentDims=latentDims)
rec.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = rec.fit(noisyTrainSpectra, trainSpectra,
                  epochs=200, validation_data=(noisyTestSpectra, testSpectra),
                  batch_size=32, shuffle=True)

histplot = out.getHistPlot(history.history, title=experimentTitle, annotate=False, marker=None)
reconstructedSpecs = rec.call(noisyTestSpectra)
specPlot, boxPlot = out.getSpectraComparisons(testSpectra, noisyTestSpectra, reconstructedSpecs,
                                              title=experimentTitle, includeSavGol=False, wavenumbers=wavenumbers)

corrPlot = out.getCorrelationPCAPlot(noisyTestSpectra.numpy(), reconstructedSpecs.numpy(),
                                     testSpectra.numpy(), noisyTrainSpectra.numpy())
