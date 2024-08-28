"""
Utility functions.

Functions:
----------
- set_data_root
- compute_exponents

"""

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


def compute_exponents(spectra, freqs, specparam_settings, n_jobs=-1):
    """
    Compute the aperiodic exponent for each power spectrum in a 3D array.

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
    
    spectra_rs = np.swapaxes(spectra, 1, 2) # make freq dimension last
    fg = SpectralGroupModel(**specparam_settings)
    fgs = fit_models_3d(fg, freqs, spectra_rs, n_jobs=n_jobs)
    fgs = combine_model_objs(fgs)
    exponent = fgs.get_params('aperiodic', 'exponent')
    # exponent = np.reshape(exponent, [spectra.shape[0], spectra.shape[2]])
    
    return exponent
    
