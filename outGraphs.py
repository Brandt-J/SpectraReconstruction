import random
from typing import List, Dict, Tuple, TYPE_CHECKING
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
if TYPE_CHECKING:
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
    histAx.set_yscale("log")
    histAx.legend()
    histAx.set_title(title)
    return histPlot


def getSpectraComparisons(origSpecs: 'EagerTensor', noisySpecs: 'EagerTensor', recSpecs: 'EagerTensor',
                          wavenumbers: np.ndarray, title: str = '',
                          includeSavGol: bool = True) -> Tuple[plt.Figure, plt.Figure]:

    origSpecs: np.ndarray = tensor_to_npy2D(origSpecs)
    noisySpecs: np.ndarray = tensor_to_npy2D(noisySpecs)
    recSpecs: np.ndarray = tensor_to_npy2D(recSpecs)

    wavenumbers = np.linspace(wavenumbers[0], wavenumbers[-1], origSpecs.shape[1])
    plotIndices = []
    corrs = np.zeros((len(recSpecs), 2))
    fig: plt.Figure = plt.figure(figsize=(14, 7))
    for step in ["step1", "step2"]:
        if step == "step2" and plotIndices == []:
            plotIndices = random.sample(range(corrs.shape[0]), 4)
            # diffCorrs = list(corrs[:, 0] - corrs[:, 1])
            # if includeSavGol:
            #     validDiffs = [diff for i, diff in enumerate(diffCorrs) if corrs[i, 0] < 60]
            # else:
            #     validDiffs = diffCorrs
            # sortedDiffs = sorted(validDiffs)
            # plotIndices.append(diffCorrs.index(sortedDiffs[-1]))
            # ind = -2
            # while True:
            #     nextInd = diffCorrs.index(sortedDiffs[ind])
            #     if nextInd in plotIndices:
            #         ind -= 1
            #     else:
            #         plotIndices.append(nextInd)
            #         break
            #
            # plotIndices.append(diffCorrs.index(sortedDiffs[0]))
            # ind = 1
            # while True:
            #     nextInd = diffCorrs.index(sortedDiffs[ind])
            #     if nextInd in plotIndices:
            #         ind += 1
            #     else:
            #         plotIndices.append(nextInd)
            #         break

        for i in range(len(recSpecs)):
            orig = origSpecs[i]
            noisy = noisySpecs[i]
            reconst = recSpecs[i]

            if step == "step1":
                corrNN = np.corrcoef(orig, reconst)[0, 1] * 100
                if np.isnan(corrNN):
                    corrNN = 0
                corrs[i, 0] = corrNN
                # savgol = savgol_filter(noisy, window_length=21, polyorder=4)
                # corrSavGol = np.round(np.corrcoef(orig, savgol)[0, 1] * 100)
                # corrs[i, 1] = corrSavGol

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
        summary = title + f'\nmean NN: {round(np.mean(corrs[:, 0]))}, mean savgol: {round(np.mean(corrs[:, 1]), 2)}'
    else:
        summary = title + f'\nmean NN: {round(np.mean(corrs[:, 0]))}'
    boxfig = plt.figure(figsize=(4, 5))
    box_ax: plt.Axes = boxfig.add_subplot()
    if includeSavGol:
        box_ax.boxplot(corrs, labels=['Neuronal\nNet', 'Savitzky-\nGolay'], widths=[0.6, 0.6], showfliers=False)
    else:
        box_ax.boxplot(corrs[:, 0], labels=['Neuronal\nNet'], widths=[0.6], showfliers=True)
    box_ax.set_title(summary, fontsize=12)
    box_ax.set_ylabel('Pearson Correlation (%)')
    box_ax.set_ylim(min([0, corrs.min()]), 100)

    return fig, boxfig


def getCorrelationPCAPlot(noisy: np.ndarray, reconstructed: np.ndarray,
                          true: np.ndarray, train: np.ndarray) -> plt.Figure:
    corrs: List[float] = []
    for specTrue, specReconst in zip(true, reconstructed):
        corrs.append(np.corrcoef(specTrue, specReconst)[0, 1] * 100)
    corrs: np.ndarray = np.array(corrs)
    maxPointsNoisy, maxPointsTrain = 200, 10000
    if noisy.shape[0] > maxPointsNoisy:
        sortedCorrs = np.argsort(corrs)
        maxPointsNoisy = round(maxPointsNoisy / 2)
        indices = np.append(sortedCorrs[:maxPointsNoisy], sortedCorrs[-maxPointsNoisy:])  # take half of the best, half of the worst..
        noisy = noisy[indices, :]
        corrs = corrs[indices]

    if train.shape[0] > maxPointsTrain:
        randIndices = random.sample(range(train.shape[0]), maxPointsTrain)
        train = train[randIndices, :]

    numNoisy, numTrain = noisy.shape[0], train.shape[0]
    noisyPlusTrain: np.ndarray = np.vstack((noisy, train))
    standardScaler: StandardScaler = StandardScaler()
    standardScaler.fit(noisyPlusTrain)
    pca: PCA = PCA(n_components=2, random_state=42)
    princComps: np.ndarray = pca.fit_transform(noisyPlusTrain)
    distances: List[float] = []

    princComps[:, 0] -= princComps[:, 0].min()
    princComps[:, 0] /= princComps[:, 0].max()
    princComps[:, 1] -= princComps[:, 1].min()
    princComps[:, 1] /= princComps[:, 1].max()
    heatMapSize = 30
    heatmap = np.zeros((heatMapSize, heatMapSize))
    for i in range(numTrain):
        x = int(min([round(princComps[numNoisy+i, 0] * heatMapSize), heatMapSize-1]))
        y = int(min([round(princComps[numNoisy+i, 1] * heatMapSize), heatMapSize-1]))
        heatmap[x, y] += 1

    heatmap = cv2.blur(heatmap, ksize=(5, 5))

    for i in range(numNoisy):
        x = int(min([round(princComps[i, 0] * heatMapSize), heatMapSize-1]))
        y = int(min([round(princComps[i, 1] * heatMapSize), heatMapSize-1]))
        distances.append(heatmap[x, y])

    fig: plt.Figure = plt.figure()
    ax1: plt.Axes = fig.add_subplot(121)
    ax1.imshow(heatmap, cmap='gray', alpha=0.75)
    goodBadBorder = int(numNoisy/2)
    ax1.scatter(princComps[:goodBadBorder, 0]*heatMapSize, princComps[:goodBadBorder, 1]*heatMapSize, color='red', s=16, alpha=1.0)
    ax1.scatter(princComps[goodBadBorder:numNoisy, 0] * heatMapSize, princComps[goodBadBorder:numNoisy, 1] * heatMapSize,
                color='green', s=16, alpha=1.0)
    ax1.set_xlim(0, heatMapSize-1)
    ax1.set_ylim(0, heatMapSize-1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('PCA Map of training and validation')

    ax2: plt.Axes = fig.add_subplot(122)
    ax2.scatter(distances[:goodBadBorder], corrs[:goodBadBorder], color='red')
    ax2.scatter(distances[goodBadBorder:numNoisy], corrs[goodBadBorder:numNoisy], color='green')
    ax2.set_xlabel("training data coverage (a.u.)")
    ax2.set_ylabel("prediction quality")
    fig.tight_layout()
    return fig


def getPeakAreaBoxPlot(peakParams: list, reconstSpecs: np.ndarray, noisySpecs: np.ndarray) -> plt.Figure:
    areaAccuraciesNN: List[float] = getAccuracies(peakParams, reconstSpecs)
    savGolSpecs = np.zeros_like(noisySpecs)
    for i in range(noisySpecs.shape[0]):
        savGolSpecs[i, :] = savgol_filter(noisySpecs[i, :], window_length=21, polyorder=4)
    areaAccuraciesSG: List[float] = getAccuracies(peakParams, savGolSpecs)

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    ax.boxplot(np.vstack((areaAccuraciesNN, areaAccuraciesSG)).transpose(),
               labels=['Neuronal\nNet', 'Savitzky-\nGolay'], widths=[0.6, 0.6], showfliers=False)
    ax.set_ylabel("Accuracy of recovered peak area")
    ax.set_title(f"Mean Area Accuracies:\n"
                 f"{round(np.mean(areaAccuraciesNN))} % Neural Net, "
                 f"{round(np.mean(areaAccuraciesSG))} % SG Filter")
    return fig


def getAccuracies(peakParams: list, reconstSpecs: np.ndarray) -> List[float]:
    areaAccuracies: List[float] = []
    peakInd = 0
    for peakParams, reconstSpec in zip(peakParams, reconstSpecs):
        recoveredAreas = recoverPeakAreas(reconstSpec, peakParams)
        origAreas = [i[2] for i in peakParams]
        for recovered, orig in zip(recoveredAreas, origAreas):
            error = abs(recovered - orig) / orig
            areaAccuracies.append(100 - (error * 100))
            peakInd += 1

    return areaAccuracies


def tensor_to_npy2D(tensor: 'EagerTensor') -> np.ndarray:
    """
    Converts a tensor into 2D numpy array
    :param tensor:
    :return: (NxM) nparray of N sampes with M features
    """
    arr: np.ndarray = tensor.numpy()
    if len(arr.shape) == 3:
        arr = arr.reshape((arr.shape[0], arr.shape[1]))
    return arr
