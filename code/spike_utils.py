"""
Spike analysis utility functions:

Functions include:
- get_spikes_in_window
- maxInterval
- collect_burst_times

"""

# imports
import numpy as np


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