""" Time utility functions """

# imports
from time import time

def get_start_time():

    return time()


def print_time_elapsed(t_start, prefix='Time elapsed: '):
    from time import time

    hour, min, sec = hour_min_sec(time() - t_start)
    print(f"{prefix}{hour} hour, {min} min, and {sec :0.1f} s")


def hour_min_sec(duration):
    """
    Convert duration in seconds to hours, minutes, and seconds.

    Parameters
    ----------
    duration : float
        Duration in seconds.

    Returns
    -------
    hours, mins, secs : int
        Duration in hours, minutes, and seconds.
    """

    hours = int(duration // 3600)
    mins = int(duration % 3600 // 60)
    secs = int(duration % 60)
    
    return hours, mins, secs