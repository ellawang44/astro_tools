import numpy as np
from scipy.stats.mstats import theilslopes

# define constants
_c = 299792.458 # speed of light in km s^-1


class SpecAnalysis:
    '''Analyse astronomy spectra.
    '''

    def __init__(self, wavelength, flux, flux_err=None):
        '''
        Parameters
        ----------
        wavelength : List[Real] or 1darray
            Input wavelengths. Needs to be monotonically increasing.
        flux : List[Real] or 1darray
            Input flux profile.
        flux_err : List[Real] or 1darray
            Input flux error.
        '''

        # if no error, then set error to 0
        if flux_err is None:
            self.flux_err = np.full(len(flux), 0)
        
        # change to numpy array
        try: 
            self.wl = np.array(wavelength, dtype=float)
            self.flux = np.array(flux, dtype=float)
            self.flux_err = np.array(flux_err, dtype=float)
        except:
            raise ValueError('Could not turn input into numpy arrays.')

        # check monotonic increasing wavelength
        if not ((self.wl[1:] - self.wl[:-1]) > 0).all():
            raise ValueError('Wavelength needs to be strictly increasing')

        # check length of all input equal
        if not len(self.wl) == len(self.flux) == len(self.flux_err):
            raise ValueError(
        'wavelength, flux, flux_err are not the same length')

    def save(self, wl, flux, flux_err):
        '''Save the values given into the class
        '''

        self.wl = wl
        self.flux = flux
        self.flux_err = flux_err

    def mask_region(self, masks, mtype='out'):
        '''Mask (remove) a region of the spectrum. 

        Parameters
        ----------
        masks : list of lists[float]
            The regions which you want to mask.
        mtype : str, optional
            To remove the region in the mask or out. Accepted values: 'in' and 
            'out'.

        Returns
        -------
        wl, flux, flux_err : 3darray
            Masked wavelength, flux, and flux error.
        '''

        # make mask
        mask_full = np.ones(len(self.wl), dtype=bool)
        for lower, upper in masks:
            mask = (lower <= self.wl) & (self.wl <= upper)
            if mtype == 'in':
                mask = ~mask
            mask_full = mask_full & mask

        # apply mask
        self.save(
            self.wl[mask_full], self.flux[mask_full], self.flux_err[mask_full]
        )

        return self.wl, self.flux, self.flux_err

    def cut(self, center=670.9659, upper=10, lower=10, rtype='wl'):
        '''Cuts the wavelength, flux, and flux error and returns the values 
        between center - lower and center + upper.

        Parameters
        ----------
        center : Real, optional
            The center of the wavelengths where the cut should be taken, in the
            same units as the wavelength. 
        upper : Positive Real, optional
            The amount to go above the center when taking the cut, in the same 
            units of nm if rtype=wl, or in km/s if rtype=vr.
        lower : Positive Real, optional
            The amount to go below the center when taking the cut, in the same 
            units of nm if rtype=wl, or in km/s if rtype=vr.
        rtype : str, optional
            The type upper and lower is in. Either wl or vr (wavelength, 
            radial velocity respectively).

        Returns
        -------
        wl, flux, flux_err : 3darray
            Cut wavelength, flux, and flux error.
        '''

        # convert to wavelength
        if rtype == 'vr':
            lower = vr_to_wl(lower, center=center)
            upper = vr_to_wl(upper, center=center) 

        # cut
        low = center - lower
        high = center + upper
        self.mask_region([[low, high]])

        return self.wl, self.flux, self.flux_err

    def sigma_clip(self, func, args=(), sigma_cut=3, iterations=1):
        '''clip outliers based on a sigma cut.

        Parameters
        ----------
        func : callable ``func(self.wl, self.flux, self.flux_err, *args)``.
            The function to fit the spectrum to. 
        args : tuple, optional
            Extra arguments passed to func, 
            i.e., ``func(self.wl, self.flux, self.flux_err, *args)``.
        sigma_cut : float
            The tolerance on sigma clip.
        iterations : int
            The number of times to iterate the sigma clip.

        Returns
        -------
        wl, flux, flux_err : 3darray
            Clipped wavelength, flux, and flux error.
        '''

        for _ in range(iterations):
            flux_fit = func(self.wl, self.flux, self.flux_err, *args)
            diff = np.abs(self.flux - flux_fit)
            sigma = np.std(diff)
            mask = diff < sigma*sigma_cut
            self.save(self.wl[mask], self.flux[mask], self.flux_err[mask])
        return self.wl, self.flux, self.flux_err

    def cont_norm(self, center, mask_step=0.01, sigma_cut=3, iterations=3):
        '''Normalise the continuum for the flux and flux_error. Returns flux and flux_err.
        '''

        # save original values
        wl = self.wl
        flux = self.flux
        flux_err = self.flux_err
        
        # first do a shitty normalisation with the main line removed
        masks = [[center - mask_step, center + mask_step]]
        self.mask_region(masks, mtype='in')
        med = np.median(self.flux)
        flux = flux/med
        flux_err = flux_err/med

        # sigma clip 
        self.save(wl, flux, flux_err)
        median = lambda x,y,z:np.median(y)
        self.sigma_clip(median, sigma_cut=sigma_cut, iterations=iterations)

        # fit
        fit = theilslopes(self.flux, self.wl)
        grad = fit[0]
        intercept = fit[1]

        # remove linear slope
        fit = wl*grad + intercept
        flux = flux/fit
        flux_err = flux_err/fit
        self.save(wl, flux, flux_err)

        return self.wl, self.flux, self.flux_err


def polyfit(x, y, x_out=None, deg=1):
    '''Fits a polynomial to input data after shifting data to 0.

    Parameters
    ----------
    x : List[Real] or 1darray
        x values to fit over.
    y : List[Real] or 1darray
        y values to fit over.
    x_out : List[Real] or None
        Output x values. If None, then return the fit.
    deg : Int, optional
        Degree of fitted polynomial. 1 by default.

    Returns
    -------
    y_out or center, fit: 1darray or tuple(float)
        Output y values at the input x_out values or x mean, grad, intercept.
    '''

    # convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # fit
    center_x = np.mean(x)
    fit = np.polyfit(x - center_x, y, deg=deg)

    # return fit and center
    if x_out is None:
        return center_x, fit[0], fit[1]
    
    # return y evaluated at x_out
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

def wl_to_vr(wl, center=670.9659):
    '''Converts wavelengths to radial velocity, works for errors too.

    Parameters
    ----------
    wl : float or ndarray
        Wavelength to be converted, in nm.
    center : float 
        The wavelength that vr=0 is at.

    Returns
    -------
    vr : float or ndarray
        Radial velocity in km/s.
    '''

    if isinstance(wl, float):
        return wl*_c/center
    else:
        return np.array(wl)*_c/center

def vr_to_wl(vr, center=670.9659):
    '''Converts wavelengths to radial velocity, works for errors too. 
    
    Parameters
    ----------
    vr : float or ndarray
        Radial velocity to be converted, in km/s.
    center : float 
        The wavelength that vr=0 is at.

    Returns
    -------
    wl : float or ndarray
        Wavelengths in nm.
    '''

    if isinstance(vr, float):
        return vr*center/_c
    else:
        return np.array(vr)*center/_c

def vac_to_air(lam):
    '''Convert from vacuum to air wavelengths. 
    From https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    
    Parameters
    ----------
    lam : float or ndarray
        Wavelengths in vacuum in angstroms.
        
    Returns
    -------
    air : float or ndarray
        Wavelengths in air in angstroms.
    '''

    s = 1e4/lam
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    return lam/n

def air_to_vac(lam):
    '''Convert from air to vacuum wavelengths. 
    From https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

    Parameters
    ----------
    lam : float or ndarray
        Wavelengths in air in angstroms.

    Returns
    -------
    vac : float or ndarray
        Wavelengths in vacuum in angstroms.
    '''
    
    s = 1e4/lam
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    return lam*n
