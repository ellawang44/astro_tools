import numpy as np

# define constants
_c = 299792.458 # speed of light in km s^-1

def polyfit(x, y, x_out, deg=1):
    '''Fits a polynomial to input data after shifting data to 0.

    Parameters
    ----------
    x : List[Real] or 1darray
        x values to fit over.
    y : List[Real] or 1darray
        y values to fit over.
    x_out : List[Real]
        Output x values. 
    deg : Int, optional
        Degree of fitted polynomial. 1 by default.

    Returns
    -------
    y_out : 1darray
        Output y values at the input x_out values. 
    '''

    # convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # fit
    center_x = np.mean(x)
    fit = np.polyfit(x - center_x, y, deg=deg)
    y_out = np.polyval(fit, x_out - center_x)
    return y_out

def cut_wavelength(wavelength, center=670.9659, upper=10, lower=10):
    """Cuts the wavelength returns the values between center - lower and center + upper. Useful for plotting mostly because many functions return a cut line profile but not cut wavelength.
    Parameters
    ----------
    wavelength : List[Real] or 1darray
        Input wavelengths. Needs to be monotonically increasing.
    center : Real, optional
        The center of the wavelengths where the cut should be taken, in the same units as the wavelength. 
    upper : Positive Real, optional
        The amount to go above the center when taking the cut, in the same units as the wavelength.
    lower : Positive Real, optional
        The amount to go below the center when taking the cut, in the same units as the wavelength.
    Returns
    -------
    wl_cut : 2darray
        Cut wavelengths.
    """

    wavelength = np.array(wavelength)

    low = center - lower
    high = center + upper
    wl_cut = wavelength[(low <= wavelength) & (high >= wavelength)]
    return wl_cut

def cut(wavelength, line_profile, center=670.9659, upper=10, lower=10):
    """Cuts the wavelength and line profile and returns the values between center - lower and center + upper.
    Parameters
    ----------
    wavelength : List[Real] or 1darray
        Input wavelengths. Needs to be monotonically increasing.
    line_profile : List[Real] or 1darray
        Input line profile.
    center : Real, optional
        The center of the wavelengths where the cut should be taken, in the same units as the wavelength. 
    upper : Positive Real, optional
        The amount to go above the center when taking the cut, in the same units as the wavelength.
    lower : Positive Real, optional
        The amount to go below the center when taking the cut, in the same units as the wavelength.
    Returns
    -------
    cut_data : 2darray
        Cut wavelengths and line profiles.
    """

    wavelength = np.array(wavelength)
    line_profile = np.array(line_profile)

    low = center - lower
    high = center + upper
    mask = (low <= wavelength) & (high >= wavelength)
    wl_cut = wavelength[mask]
    line_cut = line_profile[mask]
    cut_data = np.array([wl_cut, line_cut])
    return cut_data

def wl_to_vr(wl, center=670.9659):
    '''Converts wavelengths to radial velocity, works for errors too. wl should be in nm. Returns vr in km/s.
    '''

    if isinstance(wl, float):
        return wl*_c/center
    else:
        return np.array(wl)*_c/center

def vr_to_wl(vr, center=670.9659):
    '''Converts wavelengths to radial velocity, works for errors too. vr should be in km/s. Returns wl in nm.
    '''

    if isinstance(vr, float):
        return vr*center/_c
    else:
        return np.array(vr)*center/_c