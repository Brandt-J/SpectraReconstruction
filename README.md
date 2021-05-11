### <b>SpecReconstruction</b>

Code for using an Autoencoder network for reconstructing FTIR spectra. The repository comes with the following datasets:
 - ATR Spectra from the siMPle library, found in the folder "ATR Spectra"
 - µFTIR Spectra acquired in lab, found in the folder "MicroFTIRSpectra"


There are four different benchmark protocols that can be used to explore the autoencoder's performance.
 - benchmarkATR.py -> Benchmarking on the ATR spectra
 - benchmarkMicroFTIR.py -> Benchmarking on the µFTIR Spectra
 - benchmarMicroFTIRFromATR.py -> Forcing poor selection of training spectra. Training on ATR Spectra, inference on µFTIR spectra
 - benchmarkQuantitative.py -> Running peak area restoration test

The "ManuscriptImages" - Folder includes the code to generate the figures that were used in the publication.

In addition, the "Spectra Generation" folder includes code for training and using a Generative Adversarial Network for training spectra synthesis.
