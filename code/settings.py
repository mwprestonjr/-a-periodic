"""
Settings and dataset details

"""

# dataset details
FS_LFP = 1250 # approximate sampling frequency for LFP
MOVIE_DURATION = 30 # natural movies (Functional Connectivity)
FRAMES_PER_TRIAL = 900 # natural movies (Functional Connectivity)
TOTAL_TRIALS = 60 # natural movies (Functional Connectivity)

# data of interest
SESSIONS = [766640955, 768515987,  835479236,  839557629] #ids (NOT SESSION_IDS)
# SESSIONS = [821695405, 778240327, 839068429] # These sessions have fewer than 10 LGd Units
BRAIN_STRUCTURES = ['LGd', 'VISp', 'VISl']

# general settings
BIN_DURATION = 0.5 # Seconds

# network burst parameters
OVERLAP_THRESHOLD = 0.02 # percentage of units whose burst times must overlap to count as a "network burst"
WINDOW_SIZE = 1 # Rolling bin window for overlapping bursts

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

# SpectralEvents hyperparameters
EVENT_BAND = [2, 20] # where to look for events
FREQS_SE = FREQS # TEMP - should refactor and compute tfr only once in extract_features()
N_CYCLES = 2 # TEMP - should refactor and compute tfr only once in extract_features()
