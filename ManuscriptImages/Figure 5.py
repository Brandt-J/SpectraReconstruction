import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
import os
from sklearn.decomposition import PCA

import importData as io
from Reconstruction import prepareSpecSet

"""
TO USE THAT SCRIPT:
First train the generators for PS and PP using the "/Spectra Generation/gan.py" script. Train one for PP and one for PS.
Save the generators in the directories "/Spectra Generation/PP Generator" and "/Spectra Generation/PS Generator" 
Use 1,000 - 2,000 epochs for training. Then Run this script for plotting..
"""

specLength = 512
latent_dim = 64

folderName = os.path.basename(os.getcwd())
os.chdir(os.path.dirname(os.getcwd()))
noisySpecs, cleanSpecs, specNames, wavenumbers = io.load_microFTIR_spectra(specLength)
indicesPS = [i for i, name in enumerate(specNames) if name == 'PS']
indicesPP = [i for i, name in enumerate(specNames) if name == 'PP']

specsPS = noisySpecs[indicesPS]
specsPP = noisySpecs[indicesPP]

noisySpecs = noisySpecs[indicesPS]
dataset = prepareSpecSet(noisySpecs, transpose=False, addDimension=True)


numSynth = 300
generatorPS = load_model("Spectra Generation/PS Generator")
random_latent_vectors = tf.random.normal(shape=(numSynth, latent_dim))
generated_specs_ps = generatorPS(random_latent_vectors)
generated_specs_ps = generated_specs_ps.numpy()
generated_specs_ps = generated_specs_ps.reshape((generated_specs_ps.shape[0], generated_specs_ps.shape[1]))

generatorPP = load_model("Spectra Generation/PP Generator")
random_latent_vectors = tf.random.normal(shape=(numSynth, latent_dim))
generated_specs_pp = generatorPP(random_latent_vectors)
generated_specs_pp = generated_specs_pp.numpy()
generated_specs_pp = generated_specs_pp.reshape((generated_specs_pp.shape[0], generated_specs_pp.shape[1]))


def normalize(arr: np.ndarray) -> np.ndarray:
    arr -= arr.min()
    arr /= arr.max()
    return arr


numPP, numPS = specsPP.shape[0], specsPS.shape[0]
for i in range(numPP):
    specsPP[i, :] = normalize(specsPP[i, :])

for i in range(numPS):
    specsPS[i, :] = normalize(specsPS[i, :])

for i in range(numSynth):
    generated_specs_pp[i, :] = normalize(generated_specs_pp[i, :])
    generated_specs_ps[i, :] = normalize(generated_specs_ps[i, :])

mpl.use("Qt5Agg")
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131)
offset = 0
for i in range(5):
    offset -= 0.5
    if i == 0:
        ax1.plot(wavenumbers, specsPP[i+10, :] + offset, color='blue', label='Real PP')
    else:
        ax1.plot(wavenumbers, specsPP[i + 10, :] + offset, color='blue')

offset -= 1.0
for i in range(5):
    offset -= 0.5
    if i == 0:
        ax1.plot(wavenumbers, specsPS[i + 10, :] + offset, color='red', label='Real PS')
    else:
        ax1.plot(wavenumbers, specsPS[i + 10, :] + offset, color='red')

ax1.set_title("Real Spectra")
ax1.legend()

ax2 = fig.add_subplot(132)
offset = 0
for i in range(5):
    offset -= 0.5
    if i == 0:
        ax2.plot(wavenumbers, generated_specs_pp[i+10, :] + offset, color='blue', label='Generated PP')
    else:
        ax2.plot(wavenumbers, generated_specs_pp[i + 10, :] + offset, color='blue')

offset -= 1.0
for i in range(5):
    offset -= 0.5
    if i == 0:
        ax2.plot(wavenumbers, generated_specs_ps[i + 10, :] + offset, color='red', label='Generated PS')
    else:
        ax2.plot(wavenumbers, generated_specs_ps[i + 10, :] + offset, color='red')

ax2.set_title("Generated Spectra")
ax2.legend()

for ax in [ax1, ax2]:
    ax.set_xlabel("Wavenumbers (cm-1)", fontsize=12)
    ax.set_yticks([])


trainPlusGenerated = np.vstack((specsPP, specsPS, generated_specs_pp, generated_specs_ps))
pca: PCA = PCA(n_components=2, random_state=42)
princComps: np.ndarray = pca.fit_transform(trainPlusGenerated)

pcaAx: plt.Axes = fig.add_subplot(133)
pcaAx.scatter(princComps[:numPP, 0], princComps[:numPP, 1], color='blue', s=4, label='PP Real')
start, stop = numPP, numPP+numPS
pcaAx.scatter(princComps[start:stop, 0], princComps[start:stop, 1], color='red', s=4, label='PS Real')
start, stop = numPP+numPS, numPP+numPS+numSynth
pcaAx.scatter(princComps[start:stop, 0], princComps[start:stop, 1], color='blue', alpha=0.1, label='PP generated')
start, stop = numPP+numPS+numSynth, numPP+numPS+numSynth+numSynth
pcaAx.scatter(princComps[start:stop, 0], princComps[start:stop, 1], color='red', alpha=0.1, label='PS generated')
pcaAx.set_xlabel("PC 1", fontsize=12)
pcaAx.set_ylabel("PC 2", fontsize=12)
pcaAx.set_title("PCA Plot")
pcaAx.legend()

fig.tight_layout()
