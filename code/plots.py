"""
Plotting functions
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt


def plot_spectra(freqs, spectra):
    # compute mean and std
    psd_mean = np.mean(spectra, axis=0)
    psd_std = np.std(spectra, axis=0)

    # plot
    fig, ax = plt.subplots(figsize=[6,4])
    ax.loglog(freqs, psd_mean)
    ax.fill_between(freqs, psd_mean-psd_std, psd_mean+psd_std, color='grey', alpha=0.5)
    ax.set(xlabel="frequency (Hz)", ylabel='power (\u03BCV\u00b2/Hz)')
    ax.set_title("")
    plt.show()