from astro_tools import *
import numpy as np
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
