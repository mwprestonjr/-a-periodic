"""
settings
"""

# dataset details
FS_LFP = 1250 # approximate sampling frequency
FRAMES_PER_TRIAL = 900
TOTAL_TRIALS = 60

# data of interest
SESSIONS = [766640955, 768515987, 778240327, 821695405, 835479236, 839068429, 839557629] #ids (NOT SESSION_IDS)
BRAIN_STRUCTURES = ['LGd', 'VISp', 'VISl']

# general settings
BIN_DURATION = 0.5 # Seconds

# SpecParam hyperparameters
SPECPARAM_SETTINGS = {
    'peak_width_limits' :   [2, 20], # default : (0.5, 12.0) - recommends at least frequency resolution * 2
    'min_peak_height'   :   0, # default : 0
    'max_n_peaks'       :   4, # default : inf
    'peak_threshold'    :   3, # default : 2.0
    'aperiodic_mode'    :   'knee'} # 'fixed' or 'knee'
