"""
Extract LFP and spike features
"""

# imports - general
import os
import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# imports - custom
import sys
sys.path.append("code")
from settings import *
from utils import set_data_root, apply_specparam, compute_flattened_spectra
from allen_utils import get_lfp_epochs
from tfr_utils import compute_tfr
from spike_utils import get_session_bursts


def main():
    # Set file location based on platform.
    data_root = set_data_root()

    # load project cache
    manifest_path = os.path.join(data_root, "allen-brain-observatory/visual-coding-neuropixels/ecephys-cache/manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    
    # initiailize results
    df_list = []
    
    # loop through sessions of interest
    for session_id in SESSIONS:
        # load session data
        session_data = cache.get_session_data(session_id)
        
        # loop through regions of interest
        for brain_structure in BRAIN_STRUCTURES:
            # extract LFP features -----------------------------------------------------
            # get LFP events
            lfp_epochs, time = get_lfp_epochs(session_data, fs=FS_LFP,
                                              brain_structure=brain_structure)

            # compute tfr
            tfr_all, tfr_freqs = compute_tfr(lfp_epochs, FS_LFP, FREQS, decim=FS_LFP,
                                             freq_bandwidth=FREQ_BANDWIDTH,
                                             time_window_length=TIME_WINDOW_LENGTH)
            tfr = np.mean(tfr_all, axis=1) # average over channels
            
            # parameterize spectra, compute aperiodic exponent, and flattened spectra
            sgm = apply_specparam(tfr, tfr_freqs, SPECPARAM_SETTINGS, N_JOBS)
            exponent = sgm.get_params('aperiodic', 'exponent')
            tfr_flat = compute_flattened_spectra(sgm)
            
            """
            <EXTRACT LFP FEATURES>
            """

            # extract Spike Features ----------------------------------------------------
            burst_df, spike_df = get_session_bursts(session_data, brain_structure, 
                                                    FRAMES_PER_TRIAL, TOTAL_TRIALS, 
                                                    BIN_DURATION)
            
            """
            <EXTRACT SPIKE FEATURES>
            """
            
            # store results --------------------------------------------------------=----
            df = pd.DataFrame({
                'session_id'     : [session_id]*1800,
                'brain_structure': [brain_structure]*1800,
                'sweep'          : np.repeat(np.arange(2), 900),
                'trial'          : np.repeat(np.arange(30), 60),
                'bin'            : np.tile(np.arange(30), 60)})
            df['exponent'] = exponent
            df['burst_count'] = burst_df['burst_count']
            df['spike_count'] = spike_df['spike_count']
            
            """
            <ADD FEATURES TO DF>
            """
            
            df_list.append(df)
            # break # TEMP!            
        break # TEMP!
    
    # save results
    results = pd.concat(df_list)
    results.to_csv('data/feature_df.csv', index=False)

    
if __name__ == '__main__':
    main()
