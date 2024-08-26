"""
Utility functions for TFR analysis. i.e. time-frequency representations of 
power (spectrograms).

Functions:
----------
plot_tfr : Plot spectrogram
compute_tfr : Compute spectrogam of power using multitaper method.
zscore_tfr : Normalize spectrogam by z-scoring power at each frequency.
subtract_baseline : Subtract baseline from signals.
crop_tfr : Crop spectrogam to specified time range.
downsample_tfr : decimate spectrogam in time.

"""

# Imports
import numpy as np


def plot_tfr(time, freqs, tfr, fname_out=None, title=None,
             norm_type='log', vmin=None, vmax=None, fig=None, ax=None,
             cax=None, cbar_label=None, annotate_zero=False, log_yscale=False):
    """
    Plot time-frequency representation (TFR)

    Parameters
    ----------
    time : 1D array
        Time vector.
    freqs : 1D array
        Frequency vector.
    tfr : 2D array
        Time-frequency representation of power (spectrogram).
    fname_out : str, optional
        File name to save figure. The default is None.
    title : str, optional
        Title of plot. The default is None.
    norm_type : str, optional
        Type of normalization for color scale. Options are 'linear', 'log',
        'centered', and 'two_slope'. The default is 'log'.
    vmin, vmax : float, optional
        Minimum/maximum value for color scale. The default is None, which
        sets the min/max to the min/max of the TFR.
    fig : matplotlib figure, optional
        Figure to plot on. The default is None, which creates a new figure.
    ax : matplotlib axis, optional
        Axis to plot on. The default is None, which creates a new axis.
    cax : matplotlib axis, optional
        Axis to plot colorbar on. The default is None.
    cbar_label : str, optional
        Label for colorbar. The default is None.
    annotate_zero : bool, optional
        Whether to annotate zero on the time axis. The default is False.
    log_yscale : bool, optional
        Whether to use a log scale for the y-axis. The default is False.

    Returns
    -------
    None.
    """

    # imports
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, LogNorm, CenteredNorm, TwoSlopeNorm


    # Define a color map and normalization of values
    if vmin is None:
        vmin = np.nanmin(tfr)
    if vmax is None:
        vmax = np.nanmax(tfr)

    if norm_type == 'linear':
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = 'hot'
    elif norm_type == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = 'hot'
    elif norm_type == 'centered':
        norm = CenteredNorm(vcenter=0)
        cmap = 'coolwarm'
    elif norm_type == 'two_slope':
        norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        cmap = 'coolwarm'
    else:
        print("norm_type must be 'linear', 'log', 'centered', or 'two_slope'")
    
    # create figure
    if (ax is None) & (fig is None):
        fig, ax = plt.subplots(constrained_layout=True)
    elif (ax is None) | (fig is None):
        raise ValueError('Both fig and ax must be provided if one is provided.')

    # plot tfr
    ax.pcolor(time, freqs, tfr, cmap=cmap, norm=norm)

    # set labels and scale
    if log_yscale is True:
        ax.set(yscale='log')
        ax.set_yticks([10, 100])
        ax.set_yticklabels(['10','100'])

    # set title
    if not title is None:
        ax.set_title(title)

    # label axes
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    # add colorbar
    if cax is None:
        cbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    else:
        cbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    if not cbar_label is None:
        cbar.set_label(cbar_label)

    # annotate zero
    if annotate_zero:
        ax.axvline(0, color='k', linestyle='--', linewidth=2)

    # add grid
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.8)

    # save fig
    if not fname_out is None:
        plt.savefig(fname_out)


def compute_tfr(lfp, fs, freqs, freq_spacing='lin', time_window_length=0.5, 
                freq_bandwidth=4, n_jobs=-1, decim=1, output='power', 
                verbose=False):
    """
    Compute time-frequency representation (TFR) of LFP data.

    Parameters
    ----------
    lfp : 3d array
        LFP data (trials x channels x samples).
    fs : int
        Sampling frequency.
    freqs : 1d array
        Frequency vector (start, stop, n_freqs).
    """

    # imports
    from mne.time_frequency import tfr_array_multitaper

    # define hyperparameters
    if freq_spacing == 'lin':
        freq = np.linspace(*freqs)
    elif freq_spacing == 'log':
        freq = np.logspace(*np.log10(freqs[:2]), freqs[2])
    n_cycles = freq * time_window_length # set n_cycles based on fixed time window length
    time_bandwidth =  time_window_length * freq_bandwidth # must be >= 2

    # TF decomposition using multitapers
    tfr = tfr_array_multitaper(lfp, sfreq=fs, freqs=freq, n_cycles=n_cycles, 
                                time_bandwidth=time_bandwidth, output=output, 
                                n_jobs=n_jobs, decim=decim, verbose=verbose)

    return tfr, freq


def zscore_tfr(tfr):
    """
    Normalize time-frequency representation (TFR) by z-scoring each frequency.
    TFR should be 2D (frequency x time).

    Parameters
    ----------
    tfr : 2D array
        Time-frequency representation of power (spectrogram).

    Returns
    -------
    tfr_norm : 2D array
        Z-score normalized TFR.
    """
    
    # initialize 
    tfr_norm = np.zeros(tfr.shape)
    
    # z-score normalize 
    for i_freq in range(tfr.shape[0]):
        tfr_norm[i_freq] = (tfr[i_freq] - np.mean(tfr[i_freq])) / np.std(tfr[i_freq])
        
    return tfr_norm


def subtract_baseline(signals, time, t_baseline):
    """
    Subtract baseline from signals. Baseline is defined as the mean of the
    signal between t_baseline[0] and t_baseline[1]. Signals should be 2D
    (signals x time).

    Parameters
    ----------
    signals : 2D array
        Signals to be baseline corrected.
    time : 1D array
        Time vector.
    t_baseline : 1D array
        Time range for baseline (t_start, t_stop).

    Returns
    -------
    signals_bl : 2D array
        Baseline corrected signals.
    """
    
    # initialize
    signals_bl = np.zeros_like(signals)
    
    # subtract baseline from each signal
    for ii in range(len(signals)):
        mask_bl = ((time>t_baseline[0]) & (time<t_baseline[1]))
        bl = np.mean(signals[ii, mask_bl])
        signals_bl[ii] = signals[ii] - bl
    
    return signals_bl


def crop_tfr(tfr, time, time_range):
    """
    Crop time-frequency representation (TFR) to time_range.
    TFR can be mulitdimensional (time must be last dimension).

    Parameters
    ----------
    tfr : array
        Time-frequency representation of power (spectrogram).
    time : 1D array
        Associated time vector (length should be equal to that of
        the last dimension of tfr).
    time_range : 1D array
        Time range to crop (t_start, t_stop).

    Returns
    -------
    tfr, time : array, array
        Cropped TFR and time vector.
    """
    
    tfr = tfr[..., (time>time_range[0]) & (time<time_range[1])]
    time = time[(time>time_range[0]) & (time<time_range[1])]
    
    return tfr, time


def downsample_tfr(tfr, time, n):
    """
    Downsample time-frequency representation (TFR) to n time bins.
    TFR can be mulitdimensional (time must be last dimension)

    Parameters
    ----------
    tfr : array
        Time-frequency representation of power (spectrogram).
    time : 1D array
        Associated time vector (length should be equal to that of 
        the last dimension of tfr).
    n : int
        Desired number of time bins after downsampling.

    Returns
    ------- 
    tfr, time : array, array
        Downsampled TFR and time vector.
    """

    # determine step size for downsampling and counnt number of samples
    n_samples = len(time)
    step = int(np.floor(tfr.shape[-1]/n))

    # downsample
    tfr = tfr[..., np.arange(0, n_samples-1, step)] 
    time = time[np.arange(0, n_samples-1, step)] 
    
    return tfr, time