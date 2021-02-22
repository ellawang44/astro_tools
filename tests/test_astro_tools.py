from astro_tools import *
import numpy as np
import unittest


def assert_array_eq(x, y):
    assert np.array_equal(x, y)


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

        np.allclose(y, y_out)

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


class Test_cut(unittest.TestCase):
    '''Test cut function.
    '''

    def test_empty(self):
        assert_array_eq(cut([], []), np.array([[], []]))

    def test_inc_edge(self):
        assert_array_eq(cut([1, 2, 3, 4], [4, 5, 6, 7], center = 2, upper = 1, lower = 1), [[1, 2, 3], [4, 5, 6]])

    def test_case1(self):
        assert_array_eq(cut([660, 669, 671, 680, 690], [1, -3, -6, 0, 2]), [[669, 671, 680], [-3, -6, 0]])

    def test_case2(self):
        assert_array_eq(cut([810, 812, 813, 816, 818, 819], [0, 1, 2, 3, 4, 5], center = 812.8606, upper = 1, lower = 2), [[812, 813], [1, 2]])

    def test_case3(self):
        assert_array_eq(cut([608, 609, 610, 612, 614], [-9, 1, 203, 40, 1], center = 610.5298, upper = 3, lower = 0), [[612], [40]])


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
        np.allclose(vr_to_wl(vr, center=center), wl)

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
        np.allclose(air_to_vac(air), wl)

    def test_single(self):
        '''Test that converting from vacuum to air and back gives the same 
        results for single.
        '''
        
        wl = 6839.4
        air = vac_to_air(wl)
        self.assertTrue(np.abs(air_to_vac(air) - wl) < 1e-5)
