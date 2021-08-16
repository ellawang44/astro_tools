import ffnn
import torch.nn as nn
import unittest

class Test_make_model(unittest.TestCase):
    '''Test ffnn _make_model. It's currently just testing the length of the model is correct, could change this to comparing the blocks of the model in the future.
    '''

    def test_neuron_int(self):
        '''
        '''
        FFNN = ffnn.ffnn(2, 10)
        FFNN._make_model(4, 1)
        self.assertEqual(len(FFNN.model), len([(4, 10), 'relu', (10, 1)]))

    def test_neuron_list(self):
        '''
        '''
        FFNN = ffnn.ffnn(0, [10, 20])
        FFNN._make_model(4, 1)
        self.assertEqual(len(FFNN.model), len([(4, 10), 'relu', (10, 20), 'relu', (20, 1)]))
    
    def test_list(self):
        '''Both neuron and f_act are lists
        '''
        FFNN = ffnn.ffnn(0, [20, 30], [nn.ReLU(), nn.ReLU(), nn.ReLU()])
        FFNN._make_model(4, 1)
        self.assertEqual(len(FFNN.model), len([(4, 20), 'relu', (20, 30), 'relu', (30, 1), 'relu']))

    #TODO: make sure error is raised correctly

class Test_converge(unittest.TestCase):
    '''Test ffnn _converge.
    '''

    def test_len(self):
        '''Test if length is too short.
        '''
        FFNN = ffnn.ffnn(2, 10, n_iter_no_change=10)
        self.assertTrue(FFNN._converge([]))
        self.assertTrue(FFNN._converge([2, 3]))

    def test_not_conv(self):
        '''Test when model has not converged.
        '''
        FFNN = ffnn.ffnn(2, 10, n_iter_no_change=2, tol=1)
        self.assertTrue(FFNN._converge([2, 3]))
        self.assertTrue(FFNN._converge([4, 2, 2.5]))
        self.assertTrue(FFNN._converge([2, 0.5]))

    def test_conv(self):
        '''Test when model has converged.
        '''
        FFNN = ffnn.ffnn(2, 10, n_iter_no_change=2, tol=1)
        self.assertFalse(FFNN._converge([2, 2]))
        self.assertFalse(FFNN._converge([2, 1.5]))
        