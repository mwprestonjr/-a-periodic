"""
Utility functions
"""

# imports
import numpy as np


def find_probes_in_region(session, region):
    probe_ids = session.probes.index.values
    has_region = np.zeros_like(probe_ids).astype(bool)

    for i_probe, probe_id in enumerate(probe_ids):
        regions = session.channels[session.channels.probe_id == probe_id].ecephys_structure_acronym.unique()
        has_region[i_probe] = region in regions

    ids = probe_ids[has_region]
    names = session.probes.description.values[has_region]

    return ids, names


def align_lfp(lfp, event_times, t_window=[-1,1], dt=0.001):
    """
    Modified from AllenSDK example code:
    https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_lfp_analysis.html

    Aligns LFP data to stimulus presentation times.

    Parameters
    ----------
    lfp : xarray.core.dataarray.DataArray
        LFP data to be aligned. Must have a time coordinate.
    event_times : array_like
        Array of shape (n_trials,) of event times (in seconds) to align to.
    t_window : array_like, optional
        Array of shape (2,) of the time window (in seconds) to be extracted around
        each event. Default is [-1,1].
    dt : float, optional
        Time resolution (in seconds) of the aligned LFP data. Default is 0.001.

    Returns
    -------
    aligned_lfp : array_like
        LFP data aligned to events. (n_trials, n_channels, n_timepoints)
    trial_window : array_like
        Associated time-vector for aligned LFP data. (n_timepoints,)
    """

    # imports
    import pandas as pd

    # determine indices of time window around stimulus presentation
    trial_window = np.arange(t_window[0], t_window[1], dt)
    time_selection = np.concatenate([trial_window + t for t in event_times])
    inds = pd.MultiIndex.from_product((np.arange(len(event_times)), trial_window), 
                                    names=('presentation_id', 'time_from_presentation_onset'))

    # epoch LFP data around stimulus presentation
    ds = lfp.sel(time = time_selection, method='nearest').to_dataset(name='aligned_lfp')
    ds = ds.assign(time=inds).unstack('time')

    # reshape data (n_trials, n_channels, n_timepoints) and convert to numpy
    aligned_lfp = ds['aligned_lfp'].values
    aligned_lfp = np.swapaxes(aligned_lfp, 0, 1)

    return aligned_lfp, trial_window

