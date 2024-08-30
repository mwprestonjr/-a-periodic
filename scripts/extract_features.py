"""
Extract LFP and spike features
"""

# suppress cumbersome AllenSDK warnings
import warnings
warnings.filterwarnings("ignore")

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
from se_utils import extract_se


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
            # store results --------------------------------------------------------=----
            df = pd.DataFrame({
                'session_id'     : [session_id]*1800,
                'brain_structure': [brain_structure]*1800,
                'sweep'          : np.repeat(np.arange(2), 900),
                'trial'          : np.repeat(np.arange(30), 60),
                'bin'            : np.tile(np.arange(30), 60)})
            
            # extract LFP features -----------------------------------------------------
            # get LFP events
            lfp_epochs, time = get_lfp_epochs(session_data, fs=FS_LFP,
                                              brain_structure=brain_structure)

            # compute tfr
            tfr, tfr_freqs = compute_tfr(lfp_epochs, FS_LFP, FREQS, decim=1,
                                             method='morlet', n_morlet_cycle=N_CYCLES, n_jobs=N_JOBS)
            
            
            tfr = np.mean(tfr, axis=1) # average over channels
            
            tfr = tfr.reshape(tfr.shape[0],tfr.shape[1],int(tfr.shape[-1] / FS_LFP), FS_LFP)
            
            # spectral events
            se_df = extract_se(tfr,FREQS,event_band=EVENT_BAND)
                        
            for feature_in, feature in zip(["Peak Frequency", "Event Duration", "Normalized Peak Power"],
                                            ["peak_frequency", "event_duration", "normalized_peak_power"]):
                df[feature] = se_df[feature_in]
            
            # decim TFR for spec param
            tfr = tfr.mean(axis=3)
            
            # parameterize spectra, compute aperiodic exponent and total power, and 
            sgm = apply_specparam(tfr, tfr_freqs, SPECPARAM_SETTINGS, N_JOBS)
            exponent = sgm.get_params('aperiodic', 'exponent')
            df['exponent'] = exponent            
            df['total_power'] = np.ravel(np.mean(tfr, axis=1))
            
            # flattened spectra
            tfr_flat = compute_flattened_spectra(sgm)
            
            # find frequency with max power
            avg_spec = tfr_flat.mean(axis=0)        
            df['periodic_pow'] = tfr_flat[:,np.argmax(avg_spec)]

            # extract Spike Features ----------------------------------------------------
            spike_df = get_session_bursts(session_data, brain_structure, FRAMES_PER_TRIAL, 
                                          TOTAL_TRIALS, BIN_DURATION, OVERLAP_THRESHOLD, 
                                          WINDOW_SIZE)
            for feature in spike_df.columns[2:]:
                df[feature] = spike_df[feature]

            # store results
            df_list.append(df)
            # break # TEMP!         
        break # TEMP!
    
    # save results
    results = pd.concat(df_list)
    results.to_csv('data/feature_df.csv', index=False)

    
if __name__ == '__main__':
    main()
