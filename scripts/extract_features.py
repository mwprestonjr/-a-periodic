"""
Extract LFP and spike features
"""

# imports - general
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# imports - custom
import sys
sys.path.append("../code")
from settings import SESSIONS, BRAIN_STRUCTURES, FS_LFP, MOVIE_DURATION, FREQS, FREQ_BANDWIDTH, TIME_WINDOW_LENGTH, SPECPARAM_SETTINGS, N_JOBS
from utils import set_data_root, compute_exponents
from allen_utils import get_lfp_epochs
from tfr_utils import compute_tfr

# settings


def main():
    # Set file location based on platform.
    data_root = set_data_root()

    # load project cache
    manifest_path = os.path.join(data_root, "allen-brain-observatory/visual-coding-neuropixels/ecephys-cache/manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    
    # Initiailize results dataframe
    """
    <INITIALIZE RESULTS DATAFRAME>
    """
    
    # loop through sessions of interest
    for session_id in SESSIONS:
        # load session data
        session_data = cache.get_session_data(session_id)
        
        # loop through regions of interest
        for brain_structure in BRAIN_STRUCTURES:
            # Extract LFP features -----------------------------------------------------
            # Get LFP events
            lfp_epochs, time = get_lfp_epochs(session_data, brain_structure='VISp', fs=FS_LFP)

            # compute tfr
            tfr_all, tfr_freqs = compute_tfr(lfp_epochs, FS_LFP, FREQS, freq_bandwidth=FREQ_BANDWIDTH,
                                             time_window_length=TIME_WINDOW_LENGTH, decim=FS_LFP) # decim to 1 Hz
            tfr = np.mean(tfr_all, axis=1) # average over channels
            
            # compute aperiodic exponent
            exponent = compute_exponents(tfr, tfr_freqs, SPECPARAM_SETTINGS, N_JOBS)
            
            """
            <EXTRACT LFP FEATURES>
            """

            
            # Extract Spike Features ----------------------------------------------------
            """
            <EXTRACT SPIKE FEATURES>
            """
    
    # Save results
    """
    <SAVE RESULTS>
    """

if __name__ == '__main__':
    main()