from astro_tools import *
import numpy as np
from scipy.interpolate import interp1d
import unittest

def assert_array_eq(x, y):
    assert np.array_equal(x, y)


class Test_SpecAnalysis(unittest.TestCase):
    '''Test __init__ from SpecAnalysis.
    '''

    def test_np_array(self):
        #TODO: test all of the valueerrors are raised correctly in the init
        pass


class TestMaskRegion(unittest.TestCase):
    '''Test mask_region from SpecAnalysis. Test that masks are correctly 
    applied for multiple masks. Single masks are tested well in TestCut.
    '''

    def test_multiple_out(self):
        masked = SpecAnalysis(
            [1, 2, 3, 100, 101, 102, 201, 202, 203],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ).mask_region([[1, 3], [200, 203]])
        assert_array_eq(masked, [[1, 2, 3, 201, 202, 203], [1, 2, 3, 7, 8, 9], [0, 0, 0, 0, 0, 0]])

    def test_multiple_in(self):
        masked = SpecAnalysis(
            [1, 2, 3, 100, 101, 102, 201, 202, 203],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ).mask_region([[1, 3], [200, 203]], rm='in')
        assert_array_eq(masked, [[100, 101, 102], [4, 5, 6], [0.4, 0.5, 0.6]])

    def test_empty_in(self):
        masked = SpecAnalysis(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ).mask_region([[1.5, 1.7], [5.6, 7.6]], rm='in')
        assert_array_eq(masked, [[1, 2, 3, 4, 5, 8], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])

    def test_overlap_in(self):
        masked = SpecAnalysis(
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 1, 1, 1, 1, 1]
        ).mask_region([[2, 3], [2, 4]], rm='in')
        assert_array_eq(masked, [[1, 5, 6], [1, 5, 6], [1, 1, 1]])

    def test_overlap_out(self):
        masked = SpecAnalysis(
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 1, 1, 1, 1, 1]
        ).mask_region([[2, 3], [2, 4]], rm='out')
        assert_array_eq(masked, [[2, 3, 4], [2, 3, 4], [1, 1, 1]])


class TestCut(unittest.TestCase):
    '''Test cut from SpecAnalysis.
    '''

    def test_empty(self):
        assert_array_eq(SpecAnalysis([], [], []).cut(center=670.9659), np.array([[], [], []]))

    def test_inc_edge(self):
        cut = SpecAnalysis(
            [1, 2, 3, 4], 
            [4, 5, 6, 7], 
            [0.1, 0.2, 0.1, 0.1]
        ).cut(2, upper=1, lower=1)
        assert_array_eq(cut, [[1, 2, 3], [4, 5, 6], [0.1, 0.2, 0.1]])

    def test_rv(self):
        cut = SpecAnalysis(
            [650, 660, 671, 680, 690],
            [1, 2, 3, 4, 5],
            [0.1, 0.2, 0.2, 0.3, 0.1]
        ).cut(670.9659, lower=100, upper=100, domain='vr')
        assert_array_eq(cut, [[671], [3], [0.2]])

    def test_case1(self):
        cut = SpecAnalysis(
            [660, 669, 671, 680, 690], 
            [1, -3, -6, 0, 2], 
            [0.1, 0.1, 0.2, 0.1, 0.1]
        ).cut(670.9659)
        assert_array_eq(cut, [[669, 671, 680], [-3, -6, 0], [0.1, 0.2, 0.1]])

    def test_case2(self):
        cut = SpecAnalysis(
            [810, 812, 813, 816, 818, 819], 
            [0, 1, 2, 3, 4, 5]
        ).cut(812.8606, upper=1, lower=2)
        assert_array_eq(cut, [[812, 813], [1, 2], [0, 0]])

    def test_case3(self):
        cut = SpecAnalysis(
            [608, 609, 610, 612, 614], 
            [-9, 1, 203, 40, 1], 
            [0.1, 0.1, 0.1, 0.2, 0.1]
        ).cut(610.5298, upper = 3, lower = 0)
        assert_array_eq(cut, [[612], [40], [0.2]])


class TestSigmaClip(unittest.TestCase):
    '''Test sigma_clip from SpecAnalysis.
    '''

    def test_mean(self):
        clip = SpecAnalysis(
            [1, 2, 3, 4, 5, 6], 
            [0.1, 0.2, 0.1, -0.1, 100, -0.2]
        ).sigma_clip(lambda x, y, z: np.mean(y), sigma_cut=2)
        # mean will be more affected by outliers, so std will be lower
        # which means sigma_cut can be higher to remove outlier
        assert_array_eq(clip, [
            [1, 2, 3, 4, 6], 
            [0.1, 0.2, 0.1, -0.1, -0.2], 
            [0, 0, 0, 0, 0]
            ])

    def test_median(self):
        clip = SpecAnalysis(
            [1, 2, 3, 4, 5, 6], 
            [0.1, 0.2, 0.1, -0.1, 100, -0.2], 
            [0.1, 0.1, 0.1, 0.1, 10, 0.1]
        ).sigma_clip(lambda x, y, z: np.median(y), sigma_cut=2)
        # median will be more resistant to outliers, so std will be higher
        # which means sigma_cut needs to be smaller to remove outlier
        assert_array_eq(clip, [
            [1, 2, 3, 4, 6], 
            [0.1, 0.2, 0.1, -0.1, -0.2], 
            [0.1, 0.1, 0.1, 0.1, 0.1]
            ])

    def test_func_iter(self):
        #TODO: implement something that will test a poly function fit
        pass


class TestContNorm(unittest.TestCase):
    '''Test cont_norm from SpecAnalysis.
    '''

    def test_nofluxerr(self):
        norm = SpecAnalysis(
            [1, 2, 3, 4, 5, 6, 7], 
            [1, 1.01, 1.02, 0.03, 1.04, 1.05, 1.06]
        ).cont_norm(4, sigma_cut=1, iterations=1)
        expected = np.array(
            [[1, 2, 3, 4, 5, 6, 7], 
            [1, 1, 1, 0, 1, 1, 1], 
            [0, 0, 0, 0, 0, 0, 0]], dtype=float
        )
        self.assertTrue(np.allclose(norm, expected, atol=1e-1))

    def test_case1(self):
        norm = SpecAnalysis(
            [1, 2, 3, 4, 5, 6, 7], 
            [1, 1.01, 1.02, 0.03, 1.04, 1.05, 1.06], 
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        ).cont_norm(4, sigma_cut=1, iterations=1)
        expected = np.array(
            [[1, 2, 3, 4, 5, 6, 7], 
            [1, 1, 1, 0, 1, 1, 1], 
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=float
        )
        self.assertTrue(np.allclose(norm, expected, atol=1e-1))


class TestGaussianBroaden(unittest.TestCase):
    '''Test gaussian_broaden from SpecAnalysis.
    '''

    def test_case1(self):
        #TODO: write some tests for gaussian broaden
        pass


class TestPolyfit(unittest.TestCase):
    '''Test polyfit function
    '''

    def test_normal(self):
        '''Test that polyfit fits correctly. This is a normal data set which 
        should be fit very easily
        '''
        x = np.arange(0, 10, 1)
        y = np.arange(0, 10, 1)
        y_out = polyfit(x, y, x, deg=1)

        self.assertTrue(np.allclose(y, y_out))

    def test_extreme(self):
        '''Test that polyfit shifts to center and fits correctly. Without shifting
        x to center, normal fitting methods will fail for this input.
        '''

        x = [8000.2, 8000.3, 8000.4, 8000.6]
        y = [100, -100, 100, -100]
        x_out = x
        deg = 3
        y_out = polyfit(x, y, x_out, deg=deg)

        self.assertTrue(np.sum(np.square(np.array(y) - y_out)) < 1e-5)

    def test_out_fit(self):
        x = [1, 2, 3]
        y = [1, 2, 3]
        center_x, grad, inter = polyfit(x, y, deg=1)
        self.assertTrue(np.abs(center_x - 2) < 1e-5)
        self.assertTrue(np.abs(grad - 1) < 1e-5)
        self.assertTrue(np.abs(inter - 2) < 1e-5)


class Test_cut_wavelength(unittest.TestCase):
    '''Test cut_wavelength function
    '''

    def test_empty(self):
        assert_array_eq(cut_wavelength([]), [])

    def test_inc_edge(self):
        assert_array_eq(cut_wavelength([1, 2, 3, 4], center = 2, upper = 1, lower = 1), [1, 2, 3])

    def test_case1(self):
        assert_array_eq(cut_wavelength([660, 670, 680, 690]), [670, 680])

    def test_case2(self):
        assert_array_eq(cut_wavelength([810, 812, 813, 816, 818, 819], center = 812.8606, upper = 1, lower = 2), [812, 813])

    def test_case3(self):
        assert_array_eq(cut_wavelength([608, 609, 610, 612, 614], center = 610.5298, upper = 3, lower = 0), [612])


class TestWlVr(unittest.TestCase):
    '''Test wl_to_vr and vr_to_wl functions.
    '''

    def test_array(self):
        '''Test that converting from wl to vr and back gives same results for 
        array
        '''

        wl = np.array([600, 700, 800])
        center = 600
        vr = wl_to_vr(wl, center=center)
        self.assertTrue(np.allclose(vr_to_wl(vr, center=center), wl))

    def test_single(self):
        '''Test that converting from wl to vr and back gives the same results
        for single inputs.
        '''

        wl = 750.3
        center = 700.93
        vr = wl_to_vr(wl, center=center)
        self.assertTrue(np.abs(vr_to_wl(vr, center=center) - wl) < 1e-5)


class TestAirVac(unittest.TestCase):
    '''Test vac_to_air and air_to_vac functions.
    '''

    def test_array(self):
        '''Test that converting from vacuum to air and back gives the same 
        results for array.
        '''

        wl = np.array([6000, 7000, 8000])
        air = vac_to_air(wl)
        self.assertTrue(np.allclose(air_to_vac(air), wl))

    def test_single(self):
        '''Test that converting from vacuum to air and back gives the same 
        results for single.
        '''
        
        wl = 6839.4
        air = vac_to_air(wl)
        self.assertTrue(np.abs(air_to_vac(air) - wl) < 1e-5)


class TestRadialVelocity(unittest.TestCase):
    '''Test the radial_velocity function and its subfunctions
    '''

    def test_convolve(self):
        '''Test that convolving with multiple shifts is the same as convolving
        with each shift individually.
        '''
        x = np.array([-3, -2, -1, 0, 1, 2, 3])
        y = np.array([1, 1, 0.5, 0, 0.5, 1, 1]) - 1
        f = interp1d(x, y)
        y2 = np.array([1, 1, 1, 0.5, 0, 0.5, 1]) - 1
        g = interp1d(x, y2)
        
        shifts = np.array([-1, 0, 1])
        c_single = [convolve(f, g, np.array([shift]), x[1:-1])[0] for shift in shifts]
        c_array = convolve(f, g, shifts, x[1:-1])
        self.assertTrue(np.allclose(c_single, c_array))

    def test_common_range_error(self):
        #TODO: check that the valueerror is raised correctly.
        pass

    def test_common_range(self):
        '''Test that the common range found.
        '''

        # if both negative and positive shifts
        left, right = common_range((-2, 2), [-1, 0, 1])
        self.assertEqual(left, -1)
        self.assertEqual(right, 1)
        # if all positive shifts
        left, right = common_range((-2, 2), [1, 2])
        self.assertEqual(left, -2)
        self.assertEqual(right, 0)
        # if all negative shifts
        left, right = common_range((-2, 2), [-2, -1])
        self.assertEqual(left, 0)
        self.assertEqual(right, 2)

    def plot_cc_debug(self):
        '''Plot the cross_correlate debugging feature.
        This function will only run if called specifically
        '''
        
        x_range = (-2, 2)
        x = [-2, -1, 0, 1, 2]
        y = np.array([1, 1, 0, 1, 1]) - 1
        f = interp1d(x, y)
        g = interp1d(x, y)
        shifts = np.array([-1, 0, 1])

        _ = cross_correlate(f, g, x_range, shifts, plot=True)

    def test_radial_velocity(self):
        '''Test radial velocity is found correctly.
        '''

        f = lambda x,y: interp1d(x, y)

        # best shift is edge or middle
        x_range = (-3, 3)
        x = [-3, -2, -1, 0, 1, 2, 3]
        shifts = np.array([-1, 0, 1])
        # 0 shift
        y = np.array([1, 1, 1, 0, 1, 1, 1]) - 1
        rv = radial_velocity(f(x, y), f(x, y), x_range, shifts)
        self.assertEqual(rv, 0)
        # 1 shift
        y2 = np.array([1, 1, 1, 1, 0, 1, 1]) - 1
        rv = radial_velocity(f(x, y), f(x, y2), x_range, shifts)
        self.assertEqual(rv, 1)
        # -1 shift
        y2 = np.array([1, 1, 0, 1, 1, 1, 1]) - 1
        rv = radial_velocity(f(x, y), f(x, y2), x_range, shifts)
        self.assertEqual(rv, -1)

        # best shift is not edge value
        y = np.array([1, 1, 0.5, 0, 0.5, 1, 1]) - 1
        shifts = np.array([-2, -1, 0, 1, 2])
        # 1 shift
        y2 = np.array([1, 1, 1, 0.5, 0, 0.5, 1]) - 1
        rv = radial_velocity(f(x, y), f(x, y2), x_range, shifts)
        self.assertEqual(rv, 1)
        # -1 shift
        y2 = np.array([1, 0.5, 0, 0.5, 1, 1, 1]) - 1
        rv = radial_velocity(f(x, y), f(x, y2), x_range, shifts)
        self.assertEqual(rv, -1)

        # best shift is closest tested shift value
        shifts = np.array([-2.2, -1.2, 0.8, 1.8])
        y2 = np.array([1, 1, 1, 0.5, 0, 0.5, 1]) - 1
        rv = radial_velocity(f(x, y), f(x, y2), x_range, shifts)
        self.assertEqual(rv, 0.8)
