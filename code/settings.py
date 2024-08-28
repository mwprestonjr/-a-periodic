"""
Settings and dataset details

"""

# dataset details
FS_LFP = 1250 # approximate sampling frequency for LFP
MOVIE_DURATION = 30 # natural movies (Functional Connectivity)
FRAMES_PER_TRIAL = 900 # natural movies (Functional Connectivity)
TOTAL_TRIALS = 60 # natural movies (Functional Connectivity)

# data of interest
SESSIONS = [766640955, 768515987, 778240327, 821695405, 835479236, 839068429, 839557629] #ids (NOT SESSION_IDS)
BRAIN_STRUCTURES = ['LGd', 'VISp', 'VISl']

# general settings
BIN_DURATION = 0.5 # Seconds

# spectrogram hyperparameters
FREQS = [2, 100, 128] # [start, stop, n_freqs] (Hz)
FREQ_BANDWIDTH = 4 # frequency smoothing (Hz)
TIME_WINDOW_LENGTH = 1 # spectrogram bin size (s)

# SpecParam hyperparameters
SPECPARAM_SETTINGS = {
    'peak_width_limits' :   [2, 20], # default : (0.5, 12.0) - recommends at least frequency resolution * 2
    'min_peak_height'   :   0, # default : 0
    'max_n_peaks'       :   4, # default : inf
    'peak_threshold'    :   3, # default : 2.0
    'aperiodic_mode'    :   'knee'} # 'fixed' or 'knee'
N_JOBS = -1 # number of jobs for parallel processing