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

# imports - custom
import sys
sys.path.append("code")
from settings import *
from utils import set_data_root, apply_specparam, compute_flattened_spectra
from allen_utils import load_project_cache, get_lfp_epochs
from tfr_utils import compute_tfr
from spike_utils import get_session_bursts
from se_utils import extract_se
from time_utils import get_start_time, print_time_elapsed

def main():
    # start timer
    t_start = get_start_time()
    
    # create ouput dir
    for folder in ['tfr', 'features']:
        if not os.path.exists(f"scratch/{folder}"):
            os.makedirs(f"scratch/{folder}")
    
    # load project cache
    cache = load_project_cache()

    # initiailize results
    df_list = []
    
    # loop through sessions of interest
    for session_id in SESSIONS:
        # load session data
        session_data = cache.get_session_data(session_id)
        
        # loop through regions of interest
        for brain_structure in BRAIN_STRUCTURES:
            # display progress - time it
            t_start_s = get_start_time()
            print(f"\nAnalyzing session: {session_id}, region: {brain_structure} -----------------")
            
            # store results --------------------------------------------------------=----

            df = pd.DataFrame({
                'session_id'     : [session_id]*1800,
                'brain_structure': [brain_structure]*1800,
                'sweep'          : np.repeat(np.arange(2), 900),
                'trial'          : np.repeat(np.arange(30), 60),
                'bin'            : np.tile(np.arange(30), 60)})
            
            # extract LFP features 
            lfp_epochs, time = get_lfp_epochs(session_data, fs=FS_LFP,
                                              brain_structure=brain_structure)

            # compute tfr
            print('Extracting LFP features...')
            print('  Computing spectrogram')
            tfr, tfr_freqs = compute_tfr(lfp_epochs, FS_LFP, FREQS, method='morlet', 
                                         n_morlet_cycle=N_CYCLES, n_jobs=N_JOBS)
            
            tfr = np.mean(tfr, axis=1) # average over channels
            
            # extract spectral events
            print('  Computing spectral events')
            tfr = tfr.reshape(tfr.shape[0],tfr.shape[1],int(tfr.shape[-1] / FS_LFP), FS_LFP)
            se_df = extract_se(tfr, FREQS, event_band=EVENT_BAND, fs=FS_LFP, n_jobs=N_JOBS)
                        
            for feature_in, feature in zip(["Peak Frequency", "Event Duration", "Normalized Peak Power", "Peak Time"],
                                            ["peak_frequency", "event_duration", "normalized_peak_power", "peak_time"]):
                df[feature] = se_df[feature_in]
            
            # parameterize spectra, compute aperiodic exponent and total power
            print('  Parameterizing spectra')
            tfr = tfr.mean(axis=3) # average over 1 second bins
            
            sgm = apply_specparam(tfr, tfr_freqs, SPECPARAM_SETTINGS, N_JOBS)
            exponent = sgm.get_params('aperiodic', 'exponent')
            df['exponent'] = exponent            
            df['total_power'] = np.ravel(np.mean(tfr, axis=1))
            
            # flattened spectra
            print('  Flattening spectra')
            tfr_flat = compute_flattened_spectra(sgm)
            np.save(f'scratch/tfr/{session_id}_{brain_structure}.npy', tfr_flat)
            
            # find maximum power
            # find frequency with max power
            avg_spec = tfr_flat.mean(axis=0)        
            df['periodic_pow'] = tfr_flat[:,np.argmax(avg_spec)]
            
            del tfr_flat
        
            # extract spike and behavior features ---------------------------------------
            print('Extracting spike and behavior features...')
            behavior_df = get_session_bursts(session_data, brain_structure, FRAMES_PER_TRIAL, 
                                    TOTAL_TRIALS, BIN_DURATION, OVERLAP_THRESHOLD, WINDOW_SIZE)
            for feature in behavior_df.columns[2:]:
                df[feature] = behavior_df[feature]

            # save intermediateresults
            df.to_csv(f'scratch/features/{session_id}_{brain_structure}.csv', index=False)
            
            # store results
            df_list.append(df)
            
            # display progress
            print_time_elapsed(t_start_s, "Session/structure complete in: ")
            
            #break # TEMP!         
        #break # TEMP!
    
    # save results
    results = pd.concat(df_list)
    results.to_csv('scratch/feature_df.csv', index=False)
                               
    # display progress
    print_time_elapsed(t_start, "\n\nTotal analysis time: ")

if __name__ == '__main__':
    main()
