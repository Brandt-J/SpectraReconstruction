import random
from typing import List, Dict, Tuple, TYPE_CHECKING
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.python.framework.ops import EagerTensor

from peakConvDeconv import recoverPeakAreas


def getHistPlot(history: Dict[str, List], title: str = '', annotate: bool = True,
                marker: str = 'o') -> plt.Figure:
    histPlot: plt.Figure = plt.figure()
    histAx: plt.Axes = histPlot.add_subplot()
    if marker is None:
        histAx.plot(history["loss"], label='training')
        histAx.plot(history["val_loss"], label='validation')
    else:
        histAx.plot(history["loss"], label='training', marker=marker)
        histAx.plot(history["val_loss"], label='validation', marker=marker)
    histAx.set_ylabel("Loss")
    histAx.set_xlabel("Epochs")

    xAxis = np.arange(len(history["loss"]))
    if annotate:
        histAx.set_xticks(xAxis)
        for i, (train, val) in enumerate(zip(history["loss"], history["val_loss"])):
            histAx.annotate(f'{round(train, 3)}', xy=(xAxis[i], train), textcoords='data')
            histAx.annotate(f'{round(val, 3)}', xy=(xAxis[i], val), textcoords='data')

    histAx.legend()
    histAx.set_title(title)
    return histPlot


def getSpectraComparisons(origSpecs: 'EagerTensor', noisySpecs: 'EagerTensor', recSpecs: 'EagerTensor',
                          wavenumbers: np.ndarray, title: str = '', randomIndSeed: int = 42,
                          includeSavGol: bool = True) -> Tuple[plt.Figure, plt.Figure]:
    """
    Used for creating an overview of the spectra reconstruction
    :param origSpecs: Tensor of original spectra
    :param noisySpecs: Tensor of noisy (or distorted) spectra
    :param recSpecs: Tensor of recunstructed spectra
    :param wavenumbers: Wavenumbers to use for the plots
    :param title: Title to give the spectra overview
    :param randomIndSeed: if True, random spectra are used for plotting. If an integer is provided, it is used as
                          seed for the random index selection. If False,
    :param includeSavGol:
    :return:
    """

    origSpecs: np.ndarray = tensor_to_npy2D(origSpecs)
    noisySpecs: np.ndarray = tensor_to_npy2D(noisySpecs)
    recSpecs: np.ndarray = tensor_to_npy2D(recSpecs)

    wavenumbers = np.linspace(wavenumbers[0], wavenumbers[-1], origSpecs.shape[1])
    plotIndices = []
    corrs = np.zeros((len(recSpecs), 2))
    fig: plt.Figure = plt.figure(figsize=(14, 7))
    for step in ["step1", "step2"]:
        if step == "step2" and plotIndices == []:
            if randomIndSeed is not None:
                random.seed(randomIndSeed if type(randomIndSeed) == int else 42)
                plotIndices = random.sample(range(corrs.shape[0]), 4)
            else:
                indGoodNN = np.argwhere(corrs[:, 0] > 90).flatten()
                indBadNN = np.argwhere(corrs[:, 0] < 50).flatten()
                plotIndices = list(indGoodNN[-2:]) + list(indBadNN[-2:])

        for i in range(len(recSpecs)):
            orig = origSpecs[i]
            noisy = noisySpecs[i]
            reconst = recSpecs[i]

            if step == "step1":
                corrNN = np.corrcoef(orig, reconst)[0, 1] * 100
                if np.isnan(corrNN):
                    corrNN = 0
                corrs[i, 0] = corrNN
                savgol = savgol_filter(noisy, window_length=21, polyorder=4)
                corrSavGol = np.corrcoef(orig, savgol)[0, 1] * 100
                corrs[i, 1] = corrSavGol

            elif step == "step2":
                if i in plotIndices:
                    orig -= orig.min()
                    orig /= orig.max()
                    noisy -= noisy.min()
                    noisy /= noisy.max()
                    reconst -= reconst.min()
                    reconst /= reconst.max()

                    corrNN = corrs[i, 0]
                    if np.isnan(corrNN):
                        corrNN = 0.0
                    plotNumber = plotIndices.index(i) + 1
                    ax: plt.Axes = fig.add_subplot(2, 2, plotNumber)
                    ax.plot(wavenumbers, noisy, color='blue')
                    ax.plot(wavenumbers, orig - 1, color='orange')
                    ax.plot(wavenumbers, reconst - 2, color='green')
                    if includeSavGol:
                        savgol = savgol_filter(noisy, window_length=21, polyorder=4)
                        savgol -= savgol.min()
                        savgol /= savgol.max()
                        corrSavGol = corrs[i, 1]
                        ax.plot(wavenumbers, savgol - 3, color='red')
                        if np.isnan(corrSavGol):
                            corrSavGol = 0.0
                        ax.set_title(f"Neural net: {round(corrNN)} % Correlation\nSavGol Filter: {round(corrSavGol)} % Correlation", fontsize=13)
                    else:
                        ax.set_title(f"Neural net: {round(corrNN)} % Correlation", fontsize=13)
                    ax.set_xlabel("Wavenumbers (cm-1)", fontsize=12)
                    ax.set_yticks([])
                    plotNumber += 1

    lines = tuple(ax.lines)
    if includeSavGol:
        fig.legend(lines, ('Input', 'Target', 'Neural Net', 'Savitzky-Golay'), fontsize=12)
    else:
        fig.legend(lines, ('Input', 'Target', 'Neural Net'), fontsize=12)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()

    if includeSavGol:
        summary = title + f'\nmean NN: {round(np.mean(corrs[:, 0]))}, mean savgol: {round(np.mean(corrs[:, 1]))}'
    else:
        summary = title + f'\nmean NN: {round(np.mean(corrs[:, 0]))}'
    boxfig: plt.Figure = plt.figure(figsize=(4, 5))
    box_ax: plt.Axes = boxfig.add_subplot()
    if includeSavGol:
        box_ax.boxplot(corrs, labels=['Neuronal\nNet', 'Savitzky-\nGolay'], widths=[0.6, 0.6], showfliers=False)
    else:
        box_ax.boxplot(corrs[:, 0], labels=['Neuronal\nNet'], widths=[0.6], showfliers=True)
    box_ax.set_title(summary, fontsize=12)
    box_ax.set_ylabel('Pearson Correlation (%)')
    box_ax.set_ylim(min([0, corrs.min()]), 100)
    boxfig.tight_layout()

    return fig, boxfig


def getCorrelationPCAPlot(noisyTest: 'EagerTensor', reconstructed: 'EagerTensor',
                          true: 'EagerTensor', noisyTrain: 'EagerTensor') -> plt.Figure:
    noisyTest: np.ndarray = tensor_to_npy2D(noisyTest)
    reconstructed: np.ndarray = tensor_to_npy2D(reconstructed)
    true: np.ndarray = tensor_to_npy2D(true)
    noisyTrain: np.ndarray = tensor_to_npy2D(noisyTrain)
    corrs: List[float] = []
    for specTrue, specReconst in zip(true, reconstructed):
        corrs.append(np.corrcoef(specTrue, specReconst)[0, 1])
    corrs: np.ndarray = np.array(corrs)

    numNoisyTest, numNoisyTrain = noisyTest.shape[0], noisyTrain.shape[0]
    noisyTestPlusnoisyTrain: np.ndarray = np.vstack((noisyTest, noisyTrain))
    standardScaler: StandardScaler = StandardScaler()
    standardScaler.fit(noisyTestPlusnoisyTrain)
    pca: PCA = PCA(n_components=3, random_state=42)
    princComps: np.ndarray = pca.fit_transform(noisyTestPlusnoisyTrain)
    princComps -= princComps.min()
    princComps /= princComps.max()

    fig: plt.Figure = plt.figure()
    ax1: plt.Axes = fig.add_subplot(projection='3d')
    plot = ax1.scatter(princComps[:numNoisyTest, 0], princComps[:numNoisyTest, 1], princComps[:numNoisyTest, 2], c=corrs, alpha=0.5)
    ax1.set_title('PCA Map of Testing data', fontsize=14)
    cb = fig.colorbar(plot)
    cb.set_label("Correlation Reconstruction -> Target", fontsize=12)

    fig.tight_layout()
    return fig


def getCorrelationToTrainDistancePlot(noisy: 'EagerTensor', noisyEncoded: 'EagerTensor', reconstructed: 'EagerTensor',
                                      true: 'EagerTensor', trainEncoded: 'EagerTensor', numClosestPoints: int = 5) -> plt.Figure:
    noisy: np.ndarray = tensor_to_npy2D(noisy)
    noisyEncoded: np.ndarray = tensor_to_npy2D(noisyEncoded)
    reconstructed: np.ndarray = tensor_to_npy2D(reconstructed)
    true: np.ndarray = tensor_to_npy2D(true)
    trainEncoded: np.ndarray = tensor_to_npy2D(trainEncoded)
    corrs: np.ndarray = np.zeros(noisy.shape[0])
    for i, (specTrue, specReconst) in enumerate(zip(true, reconstructed)):
        corrs[i] = np.corrcoef(specTrue, specReconst)[0, 1]

    minDistances: np.ndarray = np.zeros_like(corrs)
    plotDistances: np.ndarray = np.zeros_like(corrs)
    origCorrs: np.ndarray = np.zeros_like(corrs)

    for i in range(noisyEncoded.shape[0]):
        distances = np.linalg.norm(trainEncoded - noisyEncoded[i, :], axis=1)
        avgMinDist = np.mean(np.sort(distances)[:numClosestPoints])
        minDistances[i] = avgMinDist
        plotDistances[i] = (1000*avgMinDist**2 + corrs[i]**2)**0.5
        origCorrs[i] = np.corrcoef(noisy[i, :], true[i, :])[0, 1]

    fig = plt.figure()
    ax = fig.add_subplot()
    plot = ax.scatter(minDistances, corrs, c=origCorrs, alpha=1.0, label='testing data')
    ax.set_xlabel(f"Average Distance to {numClosestPoints} closest Training Point", fontsize=12)
    ax.set_ylabel("Correlation Reconstruction -> Target", fontsize=12)
    ax.set_title("Distance of Testing to Training data", fontsize=14)
    cb = fig.colorbar(plot)
    cb.set_label("Correlation Input -> Target", fontsize=12)
    fig.tight_layout()
    return fig


def getPeakAreaBoxPlot(peakParams: List[List[Tuple[float, float, float]]],
                       reconstSpecs: np.ndarray, noisySpecs: np.ndarray) -> plt.Figure:
    areaAccuraciesNN: List[float] = getDeconvolutionAccuracies(peakParams, reconstSpecs)
    savGolSpecs = np.zeros_like(noisySpecs)
    for i in range(noisySpecs.shape[0]):
        savGolSpecs[i, :] = savgol_filter(noisySpecs[i, :], window_length=21, polyorder=4)
    areaAccuraciesSG: List[float] = getDeconvolutionAccuracies(peakParams, savGolSpecs)

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    ax.boxplot(np.vstack((areaAccuraciesNN, areaAccuraciesSG)).transpose(),
               labels=['Neuronal\nNet', 'Savitzky-\nGolay'], widths=[0.6, 0.6], showfliers=False)
    ax.set_ylabel("Accuracy of recovered peak area")
    ax.set_title(f"Mean Area Accuracies:\n"
                 f"{round(np.mean(areaAccuraciesNN))} % Neural Net, "
                 f"{round(np.mean(areaAccuraciesSG))} % SG Filter")
    return fig


def getDeconvolutionAccuracies(peakParamsList: List[List[Tuple[float, float, float]]], reconstSpecs: np.ndarray) -> List[float]:
    areaAccuracies: List[float] = []
    for peakParams, reconstSpec in zip(peakParamsList, reconstSpecs):
        recoveredAreas = recoverPeakAreas(reconstSpec, peakParams)
        origAreas = [i[2] for i in peakParams]
        for recovered, orig in zip(recoveredAreas, origAreas):
            error = abs(recovered - orig) / orig
            areaAccuracies.append(100 - (error * 100))

    return areaAccuracies


def tensor_to_npy2D(tensor: EagerTensor) -> np.ndarray:
    """
    Converts a tensor into 2D numpy array
    :param tensor:
    :return: (NxM) nparray of N sampes with M features
    """
    if type(tensor) == np.ndarray:
        arr: np.ndarray = tensor
    elif type(tensor) == EagerTensor:
        arr: np.ndarray = tensor.numpy()

    if len(arr.shape) == 3:
        arr = arr.reshape((arr.shape[0], arr.shape[1]))
    return arr


def getSpecCorrelation(reconstructedSpecs: 'EagerTensor', origNames: List[str], dbSpecs: np.ndarray, dbNames: List[str]):
    """
    Gets Quality of Spectra Reconstruction by running a correlation to database spectra.
    :param reconstructedSpecs: Eager Tensor of reconstructed Spectra
    :param origNames: Expected Names for each reconstructed Spectrum
    :param dbSpecs: (NxM) array of database Specs (M spectra with N wavenumbers)
    :param dbNames: Names of spectra in database
    :return:
    """
    specs: np.ndarray = tensor_to_npy2D(reconstructedSpecs)
    dbSpecs = dbSpecs.copy().transpose()
    predictedNames: List[str] = []
    for i in range(specs.shape[0]):
        predictedName: str = getPredictionForSpec(specs[i, :], dbSpecs, dbNames)
        predictedNames.append(predictedName)

    report = classification_report(origNames, predictedNames)
    return predictedNames, report


def getPredictionForSpec(intensities: np.ndarray, dbSpectra: np.ndarray, dbNames: List[str], thresh: float = 0.00) -> str:
    assert dbSpectra.shape[0] == len(dbNames)
    assert dbSpectra.shape[1] == len(intensities)
    numDBSpecs: int = len(dbNames)
    corrs: np.ndarray = np.zeros(numDBSpecs)
    for i in range(numDBSpecs):
        corrs[i] = np.corrcoef(intensities, dbSpectra[i, :])[0, 1]
    maxCorr = corrs.max()
    if maxCorr < thresh:
        assignment = 'unknown'
    else:
        assignment = dbNames[np.argmax(corrs)]
    return assignment
