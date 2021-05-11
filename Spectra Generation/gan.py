import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
import os
from sklearn.decomposition import PCA

import importData as io
from Reconstruction import prepareSpecSet

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


discriminator = keras.Sequential(
    [
        keras.Input(shape=(specLength, 1)),
        layers.Conv1D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(latent_dim, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
discriminator.summary()

inputLen = int(specLength / 8)  # because we do the convTranspose (i.e., double size) 3 times..
generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(inputLen * latent_dim),
        layers.Reshape((inputLen, latent_dim)),
        layers.Conv1DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1DTranspose(512, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(1, kernel_size=5, padding="same", activation="sigmoid"),
    ],
    name="generator",
)
generator.summary()


class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, wavenumbers: np.ndarray, latent_dim=latent_dim):
        self.num_img = 6
        self.wavenumbers: np.ndarray = wavenumbers
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 20 == 0:
            random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
            generated_specs = self.model.generator(random_latent_vectors)
            generated_specs = generated_specs.numpy()
            generated_specs = generated_specs.reshape((generated_specs.shape[0], generated_specs.shape[1]))
            np.save(os.path.join(folderName, "generationOutput", f"generatedSpecsEpoch_{epoch+1}.npy"), generated_specs)

            fig: plt.Figure = plt.figure(figsize=(10, 5), dpi=300)
            for i, spec in enumerate(generated_specs):
                ax = fig.add_subplot(2, 3, i+1)
                ax.set_title(f"Random Spec {i+1} after epoch {epoch+1}")
                ax.plot(self.wavenumbers, spec)

            fig.tight_layout()
            fig.savefig(os.path.join(folderName, "generationOutput", f"Random Specs after epoch {epoch+1}.png"))


epochs = 1000

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

gan.fit(dataset, epochs=epochs, callbacks=[GANMonitor(wavenumbers=wavenumbers, latent_dim=latent_dim)])

"""
The following line is important to save out the generator for later use!
"""
gan.generator.save("PP Generator")


"""
The following is only for plotting and required trained and saved generators.
"""
# numSynth = 300
# generatorPS = load_model("Spectra Generation/PS Generator")
# random_latent_vectors = tf.random.normal(shape=(numSynth, latent_dim))
# generated_specs_ps = generatorPS(random_latent_vectors)
# generated_specs_ps = generated_specs_ps.numpy()
# generated_specs_ps = generated_specs_ps.reshape((generated_specs_ps.shape[0], generated_specs_ps.shape[1]))
#
# generatorPP = load_model("Spectra Generation/PP Generator")
# random_latent_vectors = tf.random.normal(shape=(numSynth, latent_dim))
# generated_specs_pp = generatorPP(random_latent_vectors)
# generated_specs_pp = generated_specs_pp.numpy()
# generated_specs_pp = generated_specs_pp.reshape((generated_specs_pp.shape[0], generated_specs_pp.shape[1]))
#
#
# def normalize(arr: np.ndarray) -> np.ndarray:
#     arr -= arr.min()
#     arr /= arr.max()
#     return arr
#
#
# numPP, numPS = specsPP.shape[0], specsPS.shape[0]
# for i in range(numPP):
#     specsPP[i, :] = normalize(specsPP[i, :])
#
# for i in range(numPS):
#     specsPS[i, :] = normalize(specsPS[i, :])
#
# for i in range(numSynth):
#     generated_specs_pp[i, :] = normalize(generated_specs_pp[i, :])
#     generated_specs_ps[i, :] = normalize(generated_specs_ps[i, :])
#
# mpl.use("Qt5Agg")
# fig = plt.figure(figsize=(12, 4))
# ax1 = fig.add_subplot(131)
# offset = 0
# for i in range(5):
#     offset -= 0.5
#     if i == 0:
#         ax1.plot(wavenumbers, specsPP[i+10, :] + offset, color='blue', label='Real PP')
#     else:
#         ax1.plot(wavenumbers, specsPP[i + 10, :] + offset, color='blue')
#
# offset -= 1.0
# for i in range(5):
#     offset -= 0.5
#     if i == 0:
#         ax1.plot(wavenumbers, specsPS[i + 10, :] + offset, color='red', label='Real PS')
#     else:
#         ax1.plot(wavenumbers, specsPS[i + 10, :] + offset, color='red')
#
# ax1.set_title("Real Spectra")
# ax1.legend()
#
# ax2 = fig.add_subplot(132)
# offset = 0
# for i in range(5):
#     offset -= 0.5
#     if i == 0:
#         ax2.plot(wavenumbers, generated_specs_pp[i+10, :] + offset, color='blue', label='Generated PP')
#     else:
#         ax2.plot(wavenumbers, generated_specs_pp[i + 10, :] + offset, color='blue')
#
# offset -= 1.0
# for i in range(5):
#     offset -= 0.5
#     if i == 0:
#         ax2.plot(wavenumbers, generated_specs_ps[i + 10, :] + offset, color='red', label='Generated PS')
#     else:
#         ax2.plot(wavenumbers, generated_specs_ps[i + 10, :] + offset, color='red')
#
# ax2.set_title("Generated Spectra")
# ax2.legend()
#
# for ax in [ax1, ax2]:
#     ax.set_xlabel("Wavenumbers (cm-1)", fontsize=12)
#     ax.set_yticks([])
#
#
# trainPlusGenerated = np.vstack((specsPP, specsPS, generated_specs_pp, generated_specs_ps))
# pca: PCA = PCA(n_components=2, random_state=42)
# princComps: np.ndarray = pca.fit_transform(trainPlusGenerated)
#
# pcaAx: plt.Axes = fig.add_subplot(133)
# pcaAx.scatter(princComps[:numPP, 0], princComps[:numPP, 1], color='blue', s=4, label='PP Real')
# start, stop = numPP, numPP+numPS
# pcaAx.scatter(princComps[start:stop, 0], princComps[start:stop, 1], color='red', s=4, label='PS Real')
# start, stop = numPP+numPS, numPP+numPS+numSynth
# pcaAx.scatter(princComps[start:stop, 0], princComps[start:stop, 1], color='blue', alpha=0.1, label='PP generated')
# start, stop = numPP+numPS+numSynth, numPP+numPS+numSynth+numSynth
# pcaAx.scatter(princComps[start:stop, 0], princComps[start:stop, 1], color='red', alpha=0.1, label='PS generated')
# pcaAx.set_xlabel("PC 1", fontsize=12)
# pcaAx.set_ylabel("PC 2", fontsize=12)
# pcaAx.set_title("PCA Plot")
# pcaAx.legend()
#
# fig.tight_layout()
