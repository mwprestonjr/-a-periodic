"""
Utility functions
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