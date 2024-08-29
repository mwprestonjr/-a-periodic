"""
Utility functions for handling data and objects from the AllenSDK.

Functions:
----------
- get_lfp_epochs
- find_probes_in_region
- align_lfp

"""

# imports
import numpy as np


def get_lfp_epochs(session_data, brain_structure, fs=1250):
    """
    Load LFP data for a given session and brain structure. Align LFP data to
    natural movie events.
    
    NOTE: this is written for the Functional Connectivity dataset.

    Parameters
    ----------
    session_data : AllenSDK session object
        AllenSDK session object.
    brain_structure : str
        brain structure to filter LFP data.
    fs : int, optional
        sampling frequency. The default is 1250.

    Returns
    -------
    lfp_epochs : xarray.core.dataarray.DataArray
        LFP data aligned to natural movie events.
    time : array
        time vector.
    """
    
    # imports
    import xarray as xr

    # identify probes in ROI
    print(f"Finding LFP data for {brain_structure}...")
    probe_ids, _ = find_probes_in_region(session_data, brain_structure)

    # check data, print status
    print(f"  {len(probe_ids)} probe(s) found")
    if len(probe_ids)==0:
        print('Check settings! No LFP data found')
        return np.nan, np.nan
    
    # import LFP data
    lfp_list = []
    for probe_id in probe_ids:
        # get LFP for probe
        lfp_i = session_data.get_lfp(probe_id)

        # get LFP for ROI
        chan_ids = session_data.channels[(session_data.channels.probe_id==probe_id) & \
            (session_data.channels.ecephys_structure_acronym==brain_structure)].index.values
        lfp_list.append(lfp_i.sel(channel=slice(np.min(chan_ids), np.max(chan_ids))))
    lfp = xr.concat(lfp_list, "channel")
    
    # check data, print status
    print(f"  {len(lfp['channel'])} channel(s) found")
    if len(lfp['channel'])==0:
        print('Check settings! No LFP data found')
        return np.nan, np.nan
    
    # load stimulus info
    print("Aligning LFP to natural movie events...")
    stim_table = session_data.stimulus_presentations
    stim_times = stim_table.loc[((stim_table['stimulus_name'] == 'natural_movie_one_more_repeats') & \
                                 (stim_table['frame'] == 0)), 'start_time'].values
    
    # check data, print status
    print(f"  {len(stim_times)} events identified")
    if len(stim_times)==0:
        print('Check settings! No events found')
        return np.nan, np.nan

    # align
    lfp_epochs, time = align_lfp(lfp, stim_times, t_window=[0, 30], dt=1/fs)
                                
    return lfp_epochs, time


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


def load_project_cache():
    # imports
    from utils import set_data_root()
    
    # Set file location based on platform.
    data_root = set_data_root()

    # load project cache
    manifest_path = os.path.join(data_root, "allen-brain-observatory/visual-coding-neuropixels/ecephys-cache/manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    
    return cache
