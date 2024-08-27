"""
Utility functions
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
    print(f"Finding data for {brain_structure}...")
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

# Function to get spikes within a time window
def get_spikes_in_window(spike_times, start_time, stop_time):
    """
    Extract spike times that fall within a specified time window.

    Parameters:
    spike_times (array-like): Array of spike times.
    start_time (float): Start time of the window.
    stop_time (float): End time of the window.

    Returns:
    array: Spike times that occur within the specified window.
    """
    return spike_times[(spike_times >= start_time) & (spike_times < stop_time)]

# Function for burst detection (credit: Benjamin W Corrigan et al, Neuron, 2022)
def maxInterval(spiketrain, max_begin_ISI=0.04, max_end_ISI=0.041, min_IBI=0.1, min_burst_duration=0.01,
                min_spikes_in_burst=3):
    allBurstData = {}
    '''
    Phase 1 - Burst Detection
    Here a burst is defined as starting when two consecutive spikes have an
    ISI less than max_begin_ISI apart. The end of the burst is given when two
    spikes have an ISI greater than max_end_ISI.
    Find ISIs closer than max_begin_ISI and end with max_end_ISI.
    The last spike of the previous burst will be used to calculate the IBI.
    For the first burst, there is no previous IBI.
    '''
    inBurst = False
    burstNum = 0
    currentBurst = []
    for n in range(1, len(spiketrain)):
        ISI = spiketrain[n] - spiketrain[n - 1]
        if inBurst:
            if ISI > max_end_ISI:  # end the burst
                currentBurst = np.append(currentBurst, spiketrain[n - 1])
                allBurstData[burstNum] = currentBurst
                currentBurst = []
                burstNum += 1
                inBurst = False
            elif (ISI < max_end_ISI) & (n == len(spiketrain) - 1):
                currentBurst = np.append(currentBurst, spiketrain[n])
                allBurstData[burstNum] = currentBurst
                burstNum += 1
            else:
                currentBurst = np.append(currentBurst, spiketrain[n - 1])
        else:
            if ISI < max_begin_ISI:
                currentBurst = np.append(currentBurst, spiketrain[n - 1])
                inBurst = True
    # Calculate IBIs
    IBI = []
    for b in range(1, burstNum):
        prevBurstEnd = allBurstData[b - 1][-1]
        currBurstBeg = allBurstData[b][0]
        IBI = np.append(IBI, (currBurstBeg - prevBurstEnd))

    '''
    Phase 2 - Merging of Bursts
    Here we see if any pair of bursts have an IBI less than min_IBI; if so,
    we then merge the bursts. We specifically need to check when say three
    bursts are merged into one.
    '''
    tmp = allBurstData
    allBurstData = {}
    burstNum = 0
    for b in range(1, len(tmp)):
        prevBurst = tmp[b - 1]
        currBurst = tmp[b]
        if IBI[b - 1] < min_IBI:
            prevBurst = np.append(prevBurst, currBurst)
        allBurstData[burstNum] = prevBurst
        burstNum += 1
    if burstNum >= 2:
        allBurstData[burstNum] = currBurst

    '''
    Phase 3 - Quality Control
    Remove small bursts less than min_bursts_duration or having too few
    spikes less than min_spikes_in_bursts. In this phase we have the
    possibility of deleting all spikes.
    '''
    tooShort = 0
    tmp = allBurstData
    allBurstData = {}
    burstNum = 0
    if len(tmp) > 1:
        for b in range(len(tmp)):
            currBurst = tmp[b]
            if len(currBurst) <= min_spikes_in_burst:
                tooShort +=1
            elif currBurst[-1] - currBurst[0] <= min_burst_duration:
                tooShort += 1
            else:
                allBurstData[burstNum] = currBurst
                burstNum += 1

    return allBurstData, tooShort

def collect_burst_times(trial_spikes, trial_times):
    """
    Collect burst start and stop times for each unit in each trial.
    
    :param trial_spikes: Dictionary of spikes for each trial and unit
    :param trial_times: DataFrame with start and stop times for each trial
    :return: Dictionary of burst times for each trial and unit
    """
    burst_times = {}
    
    for trial, units in trial_spikes.items():
        burst_times[trial] = {}
        for unit, spikes in units.items():
            allBurstData, _ = maxInterval(np.array(spikes))
            
            if allBurstData:  # If bursts were detected
                burst_starts = [burst[0] for burst in allBurstData.values()]
                burst_stops = [burst[-1] for burst in allBurstData.values()]
                burst_times[trial][unit] = list(zip(burst_starts, burst_stops))
            else:
                burst_times[trial][unit] = []
    
    return burst_times
