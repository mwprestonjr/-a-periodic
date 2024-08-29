"""
Spike analysis utility functions.

FUNCTIONS:
----------
- get_trial_stimuli: Processes movie presentations into trial-based stimuli data.
- get_spikes_in_window: Extracts spike times within a specified time window.
- get_trial_spikes: Extracts spike times for each trial and unit in a specified brain region.
- maxInterval: Detects bursts in a spike train using the MaxInterval method.
- get_burst_times: Collects burst start and stop times for each unit in each trial.
- get_burst_counts: Calculates burst counts for each time bin across all trials and units.
- get_session_bursts: Processes a session for a specific brain region, extracting burst information.

EXAMPLE USAGE:
-------------

spike_df, burst_df, network_burst_df = get_session_bursts(session, region_acronym, FRAMES_PER_TRIAL, TOTAL_TRIALS, BIN_DURATION, OVERLAP_THRESHOLD, WINDOW_SIZE)
    Returns:
    spike_df: DataFrame containing spike counts for each bin in each trial.
    burst_df: DataFrame containing burst counts for each bin in each trial.
    network_burst_df: DataFrame containing whether network burst happened and proportion of units bursting for each bin in each trial.


"""



# imports
import os
import numpy as np
import pandas as pd
from collections import deque
from settings import SESSIONS, BRAIN_STRUCTURES, FRAMES_PER_TRIAL, TOTAL_TRIALS, BIN_DURATION, OVERLAP_THRESHOLD, WINDOW_SIZE

def trial_stimuli(movie_presentations, FRAMES_PER_TRIAL, TOTAL_TRIALS):
    """
    Process movie presentations into trial-based stimuli data.

    Parameters:
    movie_presentations (pd.DataFrame): DataFrame containing movie presentation data.
    FRAMES_PER_TRIAL (int): Number of frames in each trial.
    TOTAL_TRIALS (int): Total number of trials to process.

    Returns:
    tuple: A tuple containing two DataFrames:
        - stimuli_frames_df: Detailed frame-by-frame stimuli data for each trial.
        - trials_df: Summary data for each trial, including start and stop times.
    """
        
    # Create an empty list to store our trial data
    trial_data = []
    
    for trial in range(TOTAL_TRIALS):
        trial_frames = movie_presentations.iloc[trial * FRAMES_PER_TRIAL : (trial + 1) * FRAMES_PER_TRIAL]
        
        for frame, (_, row) in enumerate(trial_frames.iterrows()):
            trial_data.append({
                'trial_number': trial + 1,
                'frame_number': trial * FRAMES_PER_TRIAL + frame + 1,
                'frame_within_trial': frame + 1,
                'start_time': row['start_time'],
                'stop_time': row['stop_time'],
                'stimulus_presentation_id': row['stimulus_condition_id']
            })
    
    # Convert to DataFrame
    stimuli_frames_df = pd.DataFrame(trial_data)
    
    # Calculate the start and end times for each trial
    trials_df = stimuli_frames_df.groupby('trial_number').agg({
        'start_time': 'min',
        'stop_time': 'max'
    }).reset_index()
    
    # Calculate the duration of each trial
    trials_df['trial_duration'] = trials_df['stop_time'] - trials_df['start_time']
    
    return stimuli_frames_df, trials_df

# Function to get spikes within a time window
def get_spikes_in_window(spike_times, start_time, stop_time):
    """
    Extract spike times within a specified time window.

    Parameters:
    spike_times (array-like): Array of spike timestamps.
    start_time (float): Start time of the window.
    stop_time (float): End time of the window.

    Returns:
    array-like: Spike times that fall within the specified time window.
    """
    
    return spike_times[(spike_times >= start_time) & (spike_times < stop_time)]

def get_trial_spikes(session, trials_df, region_acronym):
    """
    Extract spike times for each trial and unit in a specified brain region.

    Parameters:
    session (object): Session object containing spike times and unit information.
    trials_df (pd.DataFrame): DataFrame containing trial information, including start and stop times.
    region_acronym (str): Acronym of the brain region to process.

    Returns:
    dict: A nested dictionary structure where:
        - The outer key is the trial number.
        - The inner key is the unit ID.
        - The value is an array of spike times for that unit in that trial.
    """
    
    # Filter units based on the specified region
    region_units = session.units[session.units.ecephys_structure_acronym == region_acronym]
    
    # Create a dictionary to store spikes for each trial
    trial_spikes = {trial: {} for trial in trials_df['trial_number']}
    
    # Iterate through each unit in the specified region
    for unit_id, unit in region_units.iterrows():
        # Check if the unit_id exists in session.spike_times
        if unit_id not in session.spike_times:
            print(f"Warning: Unit ID {unit_id} not found in session.spike_times. Skipping.")
            continue
        
        unit_spike_times = session.spike_times[unit_id]
        
        # Iterate through each trial
        for _, trial in trials_df.iterrows():
            trial_number = trial['trial_number']
            start_time = trial['start_time']
            stop_time = trial['stop_time']
            
            # Get spikes for this unit in this trial
            spikes_in_trial = get_spikes_in_window(unit_spike_times, start_time, stop_time)
            
            # Store the spikes only if there are any
            if len(spikes_in_trial) > 0:
                trial_spikes[trial_number][unit_id] = spikes_in_trial
    
    return trial_spikes

def maxInterval(spiketrain, max_begin_ISI, max_end_ISI, min_IBI, min_burst_duration,
                min_spikes_in_burst, pre_burst_silence=0.1):
    
    allBurstData = {}
    '''
    Phase 1 - Burst Detection
    Here a burst is defined as starting when two consecutive spikes have an
    ISI less than max_begin_ISI apart, and there's at least pre_burst_silence
    of no spikes before the burst. The end of the burst is given when two
    spikes have an ISI greater than max_end_ISI.
    '''
    inBurst = False
    burstNum = 0
    currentBurst = []
    last_spike_time = -np.inf  # Initialize to negative infinity

    for n in range(1, len(spiketrain)):
        ISI = spiketrain[n] - spiketrain[n - 1]
        time_since_last_spike = spiketrain[n - 1] - last_spike_time

        if inBurst:
            if ISI > max_end_ISI:  # end the burst
                currentBurst = np.append(currentBurst, spiketrain[n - 1])
                allBurstData[burstNum] = currentBurst
                currentBurst = []
                burstNum += 1
                inBurst = False
                last_spike_time = spiketrain[n - 1]
            elif (ISI < max_end_ISI) & (n == len(spiketrain) - 1):
                currentBurst = np.append(currentBurst, spiketrain[n])
                allBurstData[burstNum] = currentBurst
                burstNum += 1
            else:
                currentBurst = np.append(currentBurst, spiketrain[n - 1])
        else:
            if ISI < max_begin_ISI and time_since_last_spike >= pre_burst_silence:
                currentBurst = np.append(currentBurst, spiketrain[n - 1])
                inBurst = True
            else:
                last_spike_time = spiketrain[n - 1]

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

def get_burst_times(trial_spikes, trial_times, burst_params):
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
            allBurstData, _ = maxInterval(np.array(spikes), burst_params['max_begin_ISI'], burst_params['max_end_ISI'], burst_params['min_IBI'], burst_params['min_burst_duration'], burst_params['min_spikes_in_burst'], burst_params['pre_burst_silence'])
            
            if allBurstData:  # If bursts were detected
                burst_starts = [burst[0] for burst in allBurstData.values()]
                burst_stops = [burst[-1] for burst in allBurstData.values()]
                burst_times[trial][unit] = list(zip(burst_starts, burst_stops))
            else:
                burst_times[trial][unit] = []
    
    return burst_times

def get_burst_counts(burst_times, trials_df, bin_duration):
    """
    Calculate burst counts for each time bin across all trials and units.

    Parameters:
    burst_times (dict): A nested dictionary structure where:
        - The outer key is the trial number
        - The inner key is the unit ID
        - The value is a list of tuples, each tuple containing (burst_start, burst_end) times
    trials_df (pd.DataFrame): A DataFrame containing trial information, with columns:
        - 'trial_number': The trial identifier
        - 'start_time': The start time of the trial
        - 'stop_time': The end time of the trial
    bin_duration (float): The duration of each time bin in seconds

    Returns:
    pd.DataFrame: A DataFrame with columns:
        - 'trial': The trial number
        - 'bin': The bin number within the trial
        - 'burst_count': The number of bursts active in that bin
    """

    # Initialize an empty list to store burst data for all trials
    burst_data = []
    
    # Iterate through each trial in the burst_times dictionary
    for trial_number, trial_bursts in burst_times.items():
        # Get the trial information from trials_df
        trial_info = trials_df[trials_df['trial_number'] == trial_number].iloc[0]
        trial_start = trial_info['start_time']
        trial_end = trial_info['stop_time']
        trial_duration = trial_end - trial_start
        
        # Calculate the number of bins for this trial
        n_bins = int(np.floor(trial_duration / bin_duration))  # Use floor to avoid extra bin
        
        # Initialize an array to count bursts in each bin
        burst_counts = np.zeros(n_bins)
        
        # Iterate through each unit's bursts in this trial
        for unit_id, bursts in trial_bursts.items():
            # Process each burst for this unit
            for burst_start, burst_end in bursts:
                # Check if the burst is within the trial time
                if trial_start <= burst_start < trial_end:
                    # Calculate which bins this burst spans
                    start_bin = max(0, int((burst_start - trial_start) / bin_duration))
                    end_bin = min(n_bins - 1, int((burst_end - trial_start) / bin_duration))
                    # Increment the count for all bins this burst spans
                    burst_counts[start_bin:end_bin+1] += 1
        
        # Create a dictionary entry for each bin in this trial
        for bin_number, burst_count in enumerate(burst_counts):
            burst_data.append({
                'trial': trial_number,
                'bin': bin_number,
                'burst_count': burst_count
            })
    
    # Convert the list of dictionaries to a DataFrame and return it
    return pd.DataFrame(burst_data)


def get_spike_counts(trial_spikes, trials_df, bin_duration):
    """
    Calculate spike counts for each time bin across all trials and units.
    
    Parameters:
    trial_spikes (dict): A nested dictionary structure where:
        - The outer key is the trial number
        - The inner key is the unit ID
        - The value is a list of spike times
    trials_df (pd.DataFrame): A DataFrame containing trial information, with columns:
        - 'trial_number': The trial identifier
        - 'start_time': The start time of the trial
        - 'stop_time': The end time of the trial
    bin_duration (float): The duration of each time bin in seconds
    
    Returns:
    pd.DataFrame: A DataFrame with columns:
        - 'trial': The trial number
        - 'bin': The bin number within the trial
        - 'spike_count': The number of spikes in that bin
    """
    # Initialize an empty list to store spike data for all trials
    spike_data = []
    
    # Iterate through each trial in the trial_spikes dictionary
    for trial_number, trial_spikes_data in trial_spikes.items():
        # Get the trial information from trials_df
        trial_info = trials_df[trials_df['trial_number'] == trial_number].iloc[0]
        trial_start = trial_info['start_time']
        trial_end = trial_info['stop_time']
        trial_duration = trial_end - trial_start
        
        # Calculate the number of bins for this trial
        n_bins = int(np.floor(trial_duration / bin_duration))  # Use floor to avoid extra bin
        
        # Initialize an array to count spikes in each bin
        spike_counts = np.zeros(n_bins)
        
        # Iterate through each unit's spikes in this trial
        for unit_id, spikes in trial_spikes_data.items():
            # Process each spike for this unit
            for spike_time in spikes:
                # Check if the spike is within the trial time
                if trial_start <= spike_time < trial_end:
                    # Calculate which bin this spike belongs to
                    bin_number = int((spike_time - trial_start) / bin_duration)
                    if 0 <= bin_number < n_bins:
                        spike_counts[bin_number] += 1
        
        # Create a dictionary entry for each bin in this trial
        for bin_number, spike_count in enumerate(spike_counts):
            spike_data.append({
                'trial': trial_number,
                'bin': bin_number,
                'spike_count': spike_count
            })
    
    # Convert the list of dictionaries to a DataFrame and return it
    return pd.DataFrame(spike_data)

def get_network_burst_counts(burst_times, trials_df, bin_duration, overlap_threshold, window_size):
    """
    Calculate network burst counts using a sliding window approach.
    
    Parameters:
    burst_times (dict): A nested dictionary structure where:
        - The outer key is the trial number
        - The inner key is the unit ID
        - The value is a list of tuples, each tuple containing (burst_start, burst_end) times
    trials_df (pd.DataFrame): A DataFrame containing trial information, with columns:
        - 'trial_number': The trial identifier
        - 'start_time': The start time of the trial
        - 'stop_time': The end time of the trial
    bin_duration (float): The duration of each time bin in seconds
    overlap_threshold (float): The proportion of units that must be bursting simultaneously to count as a network burst
    window_size (int): The number of bins to consider for the sliding window
    
    Returns:
    pd.DataFrame: A DataFrame with columns:
        - 'trial': The trial number
        - 'bin': The bin number within the trial
        - 'network_burst_count': The number of distinct network bursts in that bin
        - 'proportion_bursting': The proportion of units bursting in that bin
        - 'num_bursting_units': The number of units bursting in that bin
    """
    network_burst_data = []
    
    for trial_number, trial_bursts in burst_times.items():
        trial_info = trials_df[trials_df['trial_number'] == trial_number].iloc[0]
        trial_start = trial_info['start_time']
        trial_end = trial_info['stop_time']
        trial_duration = trial_end - trial_start
        
        n_bins = int(np.floor(trial_duration / bin_duration))
        n_units = len(trial_bursts)
        unit_threshold = int(n_units * overlap_threshold)
        
        # Create a 2D array to track bursting status of each unit in each bin
        burst_activity = np.zeros((n_units, n_bins), dtype=bool)
        
        for unit_idx, (unit_id, bursts) in enumerate(trial_bursts.items()):
            for burst_start, burst_end in bursts:
                if trial_start <= burst_start < trial_end:
                    start_bin = min(n_bins - 1, max(0, int((burst_start - trial_start) / bin_duration)))
                    end_bin = min(n_bins - 1, max(0, int((burst_end - trial_start) / bin_duration)))
                    burst_activity[unit_idx, start_bin:end_bin+1] = True
        
        # Use a sliding window to detect network bursts
        network_bursts = np.zeros(n_bins, dtype=int)
        num_bursting_units = np.sum(burst_activity, axis=0)
        proportion_bursting = num_bursting_units / n_units
        
        for bin_number in range(n_bins):
            window_start = max(0, bin_number - window_size + 1)
            window_end = bin_number + 1
            window_proportion = np.mean(proportion_bursting[window_start:window_end])
            
            if window_proportion >= overlap_threshold:
                network_bursts[bin_number] = 1
        
        # Compile the data
        for bin_number in range(n_bins):
            network_burst_data.append({
                'trial': trial_number,
                'bin': bin_number,
                'network_burst_count': network_bursts[bin_number],
                'proportion_bursting': proportion_bursting[bin_number],
                'num_bursting_units': num_bursting_units[bin_number]
            })
    
    return pd.DataFrame(network_burst_data)


def get_session_bursts(session, region_acronym, FRAMES_PER_TRIAL, TOTAL_TRIALS, BIN_DURATION, OVERLAP_THRESHOLD, WINDOW_SIZE):
    """
    Process a session for a specific brain region, extracting burst information.

    Parameters:
    session (object): Session object containing unit and stimulus information.
    region_acronym (str): Acronym of the brain region to process.
    FRAMES_PER_TRIAL (int): Number of frames in each trial.
    TOTAL_TRIALS (int): Total number of trials to process.
    BIN_DURATION (float): Duration of each bin for burst counting.

    Returns:
    spike_df: DataFrame containing spike counts for each bin in each trial.
    burst_df: DataFrame containing burst counts for each bin in each trial.
    network_burst_df: DataFrame containing whether network burst happened and proportion of units bursting for each bin in each trial.
    """
    # Load units in region for this session
    region_units = session.units[session.units.ecephys_structure_acronym == region_acronym]
    units = session.units
    
    # Load stimuli for this session
    stimulus_presentations = session.stimulus_presentations
    movie_presentations = stimulus_presentations[stimulus_presentations['stimulus_name'].isin(['natural_movie_one_more_repeats'])]
    
    print(f'Working with session: {session.ecephys_session_id}')
    print(f'Extracting bursts for region: {region_acronym}')
    print(f'Total units in session: {units.shape[0]}')
    print(f'Total units in {region_acronym}: {region_units.shape[0]}')
    
    # Get trial stimuli time stamps
    stimuli_frames_df, trials_df = get_trial_stimuli(movie_presentations, FRAMES_PER_TRIAL, TOTAL_TRIALS)
    print(f'Extracted stimuli timestamps per trial')
    
    # Get timestamps of all spikes within each trial
    trial_spikes = get_trial_spikes(session, trials_df, region_acronym)
    print(f'Extracted spike timestamps within trial')
    
    # Define Burst Parameters
    burst_params = {}
    if region_acronym == 'LGd':
        burst_params['max_begin_ISI'] = 0.04
        burst_params['max_end_ISI'] = 0.1
        burst_params['min_IBI'] = 0.1
        burst_params['min_burst_duration'] = 0.01
        burst_params['min_spikes_in_burst'] = 3
    else:
        burst_params['max_begin_ISI'] = 0.17 
        burst_params['max_end_ISI'] = 0.3 
        burst_params['min_IBI'] = 0.2
        burst_params['min_burst_duration'] = 0.01
        burst_params['min_spikes_in_burst'] = 3
    
     # Extract spike counts across all units across all trials
    spike_df = get_spike_counts(trial_spikes, trials_df, BIN_DURATION)
    
    # Extract burst start and stop times across trials
    burst_times = get_burst_times(trial_spikes, trials_df, burst_params)
    
    # Extract burst counts across all units across all trials
    burst_df = get_burst_counts(burst_times, trials_df, BIN_DURATION)

    # Extract network bursts across all bins across all trials
    network_burst_df = get_network_burst_counts(burst_times, trials_df, BIN_DURATION, OVERLAP_THRESHOLD, WINDOW_SIZE)
    
    return  spike_df, burst_df, network_burst_df
