import unittest
import numpy as np
import numpy.testing as npt

import os

from robotics.lookup import *

class LookupTableTestCase(unittest.TestCase):

    def test_1d_integration(self):

        lookup_table = LookupTable(
                input_specifications = [
                    {'from': -1, 'to': 1, 'points': 3}],
                output_size = 1)

        lookup_table.populate(function = lambda x: x)

        # Nearest interpolation at grid points
        npt.assert_almost_equal([-1], lookup_table.get_nearest([-1]))
        npt.assert_almost_equal([0], lookup_table.get_nearest([0]))
        npt.assert_almost_equal([1], lookup_table.get_nearest([1]))

        # Nearest interpolation not at grid points
        npt.assert_almost_equal([-1], lookup_table.get_nearest([-0.75]))
        npt.assert_almost_equal([0], lookup_table.get_nearest([-0.25]))
        npt.assert_almost_equal([0], lookup_table.get_nearest([0.25]))
        npt.assert_almost_equal([1], lookup_table.get_nearest([0.75]))

        # Nearest interpolation out of grid bounds
        npt.assert_almost_equal([-1], lookup_table.get_nearest([-10]))
        npt.assert_almost_equal([1], lookup_table.get_nearest([10]))

        # Linear interpolation at grid points
        npt.assert_almost_equal([-1], lookup_table.get_lerp([-1]))
        npt.assert_almost_equal([0], lookup_table.get_lerp([0]))
        npt.assert_almost_equal([1], lookup_table.get_lerp([1]))

        # Linear interpolation not at grid points
        npt.assert_almost_equal([-0.75], lookup_table.get_lerp([-0.75]))
        npt.assert_almost_equal([-0.25], lookup_table.get_lerp([-0.25]))
        npt.assert_almost_equal([0.25], lookup_table.get_lerp([0.25]))
        npt.assert_almost_equal([0.75], lookup_table.get_lerp([0.75]))

        # Linear interpolation out of grid bounds
        npt.assert_almost_equal([-1], lookup_table.get_lerp([-10]))
        npt.assert_almost_equal([1], lookup_table.get_lerp([10]))

    def test_file(self):

        # Make sure the file does not exist
        if os.path.exists('test.npy'): os.remove('test.npy')

        # Saving works as expected
        lookup_table_a = LookupTable(
                input_specifications = [
                    {'from': 0, 'to': 2, 'points': 3}],
                output_size = 1)
        lookup_table_a.populate(function = lambda x: x)
        for i in xrange(3):
            output_vector = lookup_table_a._get([i])
            npt.assert_almost_equal(i, output_vector[0])
        lookup_table_a.save('test')

        # Loading works as expected
        lookup_table_b = LookupTable(
                input_specifications = [
                    {'from': 0, 'to': 2, 'points': 3}],
                output_size = 1)
        lookup_table_b.load('test.npy')
        for i in xrange(3):
            output_vector = lookup_table_b._get([i])
            npt.assert_almost_equal(i, output_vector[0])

        # Rewritting works as expected
        lookup_table_a.populate(function = lambda x: 2 * x)
        for i in xrange(3):
            output_vector = lookup_table_a._get([i])
            npt.assert_almost_equal(2 * i, output_vector[0])
        lookup_table_a.save('test')

        # Reloading works as expected
        lookup_table_b.load('test.npy')
        for i in xrange(3):
            output_vector = lookup_table_b._get([i])
            npt.assert_almost_equal(2 * i, output_vector[0])

        # Loading into a table with more grid points fails
        lookup_table_c = LookupTable(
                input_specifications = [
                    {'from': 0, 'to': 2, 'points': 5}],
                output_size = 1)
        npt.assert_raises(ValueError, lookup_table_c.load, 'test.npy')
        for i in xrange(5):
            output_vector = lookup_table_c._get([i])
            npt.assert_almost_equal(0, output_vector[0])

        # Loading into a table with a larger input size fails
        lookup_table_d = LookupTable(
                input_specifications = [
                    {'from': 0, 'to': 2, 'points': 3},
                    {'from': 0, 'to': 2, 'points': 3}],
                output_size = 1)
        npt.assert_raises(ValueError, lookup_table_d.load, 'test.npy')
        for i in xrange(3):
            for j in xrange(3):
                output_vector = lookup_table_d._get([i, j])
                npt.assert_almost_equal(0, output_vector[0])

        # Loading into a table with a larger output size fails
        lookup_table_e = LookupTable(
                input_specifications = [
                    {'from': 0, 'to': 2, 'points': 3}],
                output_size = 2)
        npt.assert_raises(ValueError, lookup_table_e.load, 'test.npy')
        for i in xrange(3):
            output_vector = lookup_table_e._get([i])
            npt.assert_almost_equal(0, output_vector[0])
            npt.assert_almost_equal(0, output_vector[1])

        # Clean up after ourselves
        if os.path.exists('test.npy'): os.remove('test.npy')
