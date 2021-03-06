import numpy as np
from scipy.stats.mstats import theilslopes
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

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
        flux_err : List[Real] or 1darray, optional
            Input flux error. If None, then set to array of 0. 
        '''

        # if no error, then set error to 0
        if flux_err is None:
            flux_err = np.full(len(flux), 0)
        
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

    def save(self, wavelength, flux, flux_err=None):
        '''Save the values given into the class.

        Parameters
        ----------
        wavelength : List[Real] or 1darray
            Input wavelengths. Needs to be monotonically increasing.
        flux : List[Real] or 1darray
            Input flux profile.
        flux_err : List[Real] or 1darray, optional
            Input flux error. If None, then set to array 0.
        '''

        # if no error, then set error to 0
        if flux_err is None:
            flux_err = np.full(len(flux), 0)

        self.wl = wavelength
        self.flux = flux
        self.flux_err = flux_err

    def mask_region(self, masks, rm='out'):
        '''Mask (remove) a region of the spectrum. 

        Parameters
        ----------
        masks : list of lists[float]
            The regions which you want to mask.
        rm : str, optional
            To remove the region in the mask or out. Accepted values: 'in' and 
            'out'.

        Returns
        -------
        wl, flux, flux_err : 3darray
            Masked wavelength, flux, and flux error.
        '''

        # make mask
        mask_full = np.zeros(len(self.wl), dtype=bool)
        for lower, upper in masks:
            mask = (lower <= self.wl) & (self.wl <= upper)
            mask_full = mask_full | mask

        # flip if masking inside
        if rm == 'in':
            mask_full = ~mask_full

        # apply mask
        self.save(
            self.wl[mask_full], self.flux[mask_full], self.flux_err[mask_full]
        )

        return self.wl, self.flux, self.flux_err

    def cut(self, center, upper=10, lower=10, domain='wl'):
        '''Cuts the wavelength, flux, and flux error and returns the values 
        between center - lower and center + upper.

        Parameters
        ----------
        center : Real
            The center of the wavelengths where the cut should be taken, in the
            same units as the wavelength. 
        upper : Positive Real, optional
            The amount to go above the center when taking the cut, in the same 
            units of nm if rtype=wl, or in km/s if rtype=vr.
        lower : Positive Real, optional
            The amount to go below the center when taking the cut, in the same 
            units of nm if rtype=wl, or in km/s if rtype=vr.
        domain : str, optional
            The domain upper and lower is in. Either wl or vr (wavelength, 
            radial velocity respectively).

        Returns
        -------
        wl, flux, flux_err : 3darray
            Cut wavelength, flux, and flux error.
        '''

        # convert to wavelength
        if domain == 'vr':
            lower = vr_to_wl(lower, center=center)
            upper = vr_to_wl(upper, center=center) 

        # cut
        low = center - lower
        high = center + upper
        self.mask_region([[low, high]])

        return self.wl, self.flux, self.flux_err
    
    def sigma_clip(self, func, args=(), sigma_cut=3, iterations=1):
        '''Clip outliers based on a sigma cut.

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
            diff = self.flux - flux_fit
            sigma = np.std(diff)
            mask = np.abs(diff) < sigma*sigma_cut
            self.save(self.wl[mask], self.flux[mask], self.flux_err[mask])
        return self.wl, self.flux, self.flux_err

    def cont_norm(self, center, mask_step=0.01, sigma_cut=3, iterations=3):
        '''Normalise the continuum. Assumes you're normalising a line profile. 
        1. Do basic normalisation. 
        2. Sigma clip lines and outliers. 
        3. Fit theilslope on the clipped spectrum.
        4. Remove fit from line. 
        Only works for a small region with linear continuum.

        Parameters
        ----------
        center : float
            The center of the line. 
        mask_step : float
            Width/2 of the line. 
        sigma_cut : float
            The tolerance on sigma clip.
        iterations : int
            The number of times to iterate the sigma clip.

        Returns
        -------
        wl, flux, flux_err : 3darray
            Normalised wavelength, flux, and flux error.
        '''

        # save original values
        wl = self.wl
        flux = self.flux
        flux_err = self.flux_err

        # first do a shitty normalisation with the main line removed
        masks = [[center - mask_step, center + mask_step]]
        self.mask_region(masks, rm='in')
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

    def gaussian_broaden(center, sigma=0, num=None):
        '''Only works for synthetic spectra because it uses cubicspline. Might 
        be unpredictable for synthetic spectra with more than 1 line or gaps.
        TODO: investigate behaviour on harder to deal with synthetic spectra.
        '''

        # convert to velocity space
        vr = wl_to_v(self.wl, center=center)
        cs = CubicSpline(vr, self.flux)

        # set steps
        if num is None:
            num = int((vr[-1] - vr[0]) / np.min(vr[1:] - vr[:-1])) + 1
        num *= 2
        # set kernel
        g_gen = scipy.stats.norm(0, sigma/2.35482) # convert FWHM to sigma

        # convolve
        tau = np.linspace(vr[0], vr[-1], num)
        convolver = np.array([g_gen.pdf(t) for t in tau])
        convolver /= np.sum(convolver)
        integrand = [cs(vr - t)*convolver[i] for i, t in enumerate(tau)]
        flux_conv = np.sum(integrand, axis = 0)

        self.flux = flux_conv

        return self.wl, self.flux  


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

def convolve(f, g, n, m):
    '''Compute the discrete convolution.

    Parameters
    ----------
    f : Function
        Function 1.
    g : Function
        Function 2.
    n : 1darray
        Shift applied to g.
    m : 1darray
        Discrete values at which the function is evaluated.

    Returns
    -------
    conv : float
        Discrete convolution at n f*g(n).
    '''

    m_shift = m + n.reshape(n.shape[0], -1) # m+n
    return np.sum(f(m)*g(m_shift), axis=1)

def common_range(x_range, shifts):
    '''Compute the common range shared after the shifts are applied.

    Parameters
    ----------
    x_range : Tuple(Float)
        The range over which the functions are defined.
    shifts : List[Float]
        The shifts to apply to the functions.

    Returns
    -------
    left, right : Float, Float
        The shared range (left, right).
    '''

    min_shift = np.min(shifts)
    max_shift = np.max(shifts)

    left = max(x_range[0], x_range[0] - min_shift)
    right = min(x_range[1], x_range[1] - max_shift)

    if left == right:
        raise ValueError('Shifts are too extreme given the x_range, there is no overlap.')

    return left, right

def cross_correlate(f, g, x_range, shifts, num=10000, plot=False):
    '''Compute the cross correlation between two functions. Truncates edges, no
    extrapolation.

    Parameters
    ----------
    f : Function 
        Function 1. 
    g : Function 
        Function 2. 
    x_range : tuple(Float)
        The common range that f and g are defined over. 
        i.e. f : [-2, 2] -> R
        g : [-1, 3] -> R
        Then x_range = (-1, 2)
    shifts : 1darray
        The shifts to apply to the function g.
    num : int
        The number of points to sample the function f and g.
    plot : bool
        Display plots for debugging.

    Returns
    -------
    cc : List[Float]
        The cross correlations of the given shifts.
    '''

    #TODO: x_range could be improved to take into account differences in range
    # of f and g, this doesn't happen a lot in practice though

    left, right = common_range(x_range, shifts)
    m = np.linspace(left, right, num=num)

    if plot:
        # original functions
        x = np.linspace(x_range[0], x_range[1], 1000)
        plt.scatter(x, f(x), label='f', s=4, alpha=0.5)
        plt.scatter(x, g(x), label='g', s=4, alpha=0.5)
        # shift g
        plt.scatter(m, g(m+np.max(shifts)), s=4, alpha=0.5,
                    label=f'g min={np.min(shifts):.2f} shift')
        plt.scatter(m, g(m+np.min(shifts)), s=4, alpha=0.5,
                    label=f'g max={np.max(shifts):.2f} shift')
        # common region
        for line in [left, right]:
            plt.axvline(line, color='black', linestyle='--')
        plt.legend()
        plt.show()

    return convolve(f, g, shifts, m)

def radial_velocity(f, g, x_range, shifts, num=10000, plot=False):
    '''Compute the radial velocity from the max cross correlation.
    f and g must have continuum centered at 0. f(n) = g(n-rv)

    Parameters
    ----------
    f : Function 
        Function 1. 
    g : Function 
        Function 2. 
    x_range : tuple(Float)
        The common range that f and g are defined over. 
        i.e. f : [-2, 2] -> R
        g : [-1, 3] -> R
        Then x_range = (-1, 2)
    shifts : 1darray
        The shifts to apply to the function g.
    num : int
        The number of points to sample the function f and g.
    plot : bool
        Display plots for debugging.

    Returns
    -------
    rv : Float
        The radial velocity of g with respect to f.
    '''

    # cast to numpy array
    shifts = np.array(shifts)

    cc = cross_correlate(f, g, x_range, shifts, num=num, plot=plot)
    return shifts[np.argmax(cc)]
