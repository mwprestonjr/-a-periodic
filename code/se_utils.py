""" Spectral Events """


def events_to_pd(events):
    """
    Function to convert events dictionary (Spectral Events toolbox) to pandas dataframe

    """
     
    import pandas as pd
    import numpy as np

    event_df_list = list()
    for trial_idx, trial_events in enumerate(events):
        for event_idx, event_dict in enumerate(trial_events):
            event_dict["trial"] = trial_idx
            event_df_list.append(event_dict)

    event_df = pd.DataFrame(event_df_list).groupby("trial").agg(np.mean).reset_index()

    return event_df


def extract_se(lfp, foi, event_band=[2, 20], fs=1250, n_cycles=5.0, n_jobs=10):
    
    import sys
    sys.path.append('SpectralEvents')
    import spectralevents as se
    from mne.time_frequency import tfr_array_morlet
    import pandas as pd
    import numpy as np

    """
    Function to extract the Spectral Events for a current session and region

    Parameters
    ----------
    lfp : 3d array
        LFP data (trials x channels x samples).
    freqs : tuple
        frequency range of the TFR (lower bound, upper bound, step)
    event_band : list
        range of frequencies for which events should be identified

    fs : float
        Sampling rate (for Allen LFP 1250)
    n_cycles : float
        number of cycles for the Morlet wavelet TFR
    n_jobs : int
        number of jobs for parallelization (-1 if no parallelization)


    Returns
    ----------
    se_df : pandas data frame containing peak frequency, peak power, and duration of events in each trial

    """

    freqs = np.arange(*foi)
    tfr = tfr_array_morlet(
        lfp,
        fs,
        freqs,
        n_cycles=n_cycles,
        zero_mean=True,
        use_fft=True,
        decim=1,
        output="power",
        n_jobs=n_jobs,
    )

    # Change LFP rate for SE function - first cut 30s movies into 1s trials
    l_epoch = 1/foi[2]
    tfr_epoch = np.zeros(
        (tfr.shape[0], tfr.shape[1], len(freqs), int(tfr.shape[-1] / int(fs*l_epoch)), int(fs*l_epoch))
    )
    for trial_idx in range(tfr.shape[0]):
        for chan in range(tfr.shape[1]):
            for f in range(len(freqs)):
                tfr_epoch[trial_idx][chan][f] = tfr[trial_idx][chan][f].reshape(
                    int(tfr.shape[-1] / fs), fs
                )

    # convert to channels x epochs x freqs x samples
    tfr_epoch_se = np.zeros(
        (tfr.shape[1], int(tfr_epoch.shape[-2] * tfr.shape[0]), len(freqs), fs)
    )
    trial_ids = np.zeros((int(tfr_epoch.shape[-2] * tfr.shape[0]),))
    bin_ids = np.zeros((int(tfr_epoch.shape[-2] * tfr.shape[0]),))
    for chan in range(tfr.shape[1]):
        c = 0
        for film_idx in range(tfr.shape[0]):
            for bin_id in range(tfr_epoch.shape[-2]):
                trial_ids[c] = film_idx
                bin_ids[c] = bin_id
                tfr_epoch_se[chan, c, :, :] = tfr_epoch[film_idx, chan, :, bin_id, :]
                c += 1

    # store trial and bin info
    sweep = (trial_ids > 29).astype(int)
    trial_df = pd.Series(
        np.concatenate((trial_ids[trial_ids <= 29], trial_ids[trial_ids > 29] - 30)),
        name="trial",
    )
    sweep_df = pd.Series(sweep, name="sweep")
    bin_df = pd.Series(bin_ids, name="bin")

    ## Find spectral events
    times = np.linspace(0, 1, fs)
    tfr_avg = tfr_epoch_se.mean(axis=0)
    events = se.find_events(
        tfr=tfr_avg, times=times, freqs=freqs, event_band=event_band, threshold_FOM=6
    )

    # convert to data frame
    events_df = events_to_pd(events)

    ## Create dataframe with trial info and events
    se_df = pd.DataFrame(
        {
            "trial": np.arange(tfr_avg.shape[0]),
            "Peak Frequency": 0,
            "Event Duration": 0,
            "Normalized Peak Power": 0,
        }
    )
    se_df.set_index("trial", inplace=True)
    events_df.set_index("trial", inplace=True)
    se_df.update(
        events_df[["Peak Frequency", "Event Duration", "Normalized Peak Power"]]
    )
    se_df.index.rename("", inplace=True)

    se_df["sweep"] = sweep_df
    se_df["trial"] = trial_df
    se_df["bin"] = bin_df

    return se_df
