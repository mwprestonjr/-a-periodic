"""
Utility functions.

Functions:
----------
- set_data_root
- compute_exponents

"""

# imports
import numpy as np


def set_data_root():
    # Set file location based on platform. 
    import platform
    platstring = platform.platform()
    if ('Darwin' in platstring) or ('macOS' in platstring):
        # macOS 
        data_root = "/Volumes/Brain2024/"
    elif 'Windows'  in platstring:
        # Windows (replace with the drive letter of USB drive)
        data_root = "E:/"
    elif ('amzn' in platstring):
        # then on Code Ocean
        data_root = "/data/"
    else:
        # then your own linux platform
        # EDIT location where you mounted hard drive
        data_root = "/media/$USERNAME/Brain2024/"
        
    return data_root


def apply_specparam(spectra, freqs, specparam_settings, n_jobs=-1):
    """
    Apply spectral parameterization to 3D array of power spectra.

    Parameters
    ----------
    spectra : 3d array
        Power spectra, with dimensions [n_spectra, n_freqs, n_times].
    freqs : 1d array
        Frequency values for the power spectra.
    specparam_settings : dict
        Settings for the spectral parameterization.
    n_jobs : int
        Number of parallel jobs to run.

    Returns
    -------
    exponent : 2d array
        Aperiodic exponents for each power spectrum. Dimensions are 
        [n_spectra, n_times].
    """
    
    # imports
    from specparam import SpectralGroupModel
    from specparam.objs import fit_models_3d, combine_model_objs
    
    spectra_rs = np.swapaxes(spectra, 1, 2) # make freq dimension last
    fg = SpectralGroupModel(**specparam_settings)
    fgs = fit_models_3d(fg, freqs, spectra_rs, n_jobs=n_jobs)
    fgs = combine_model_objs(fgs)
    
    return fgs
