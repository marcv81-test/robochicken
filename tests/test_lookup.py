import unittest
import numpy as np
import numpy.testing as npt

import os

from robotics.lookup import *

class LookupTableTestCase(unittest.TestCase):

    def test_1d_nearest(self):

        # Lookup table for 1D identity function
        lookup_table = LookupTable(
                input_specifications = [{'from': -1, 'to': 1, 'points': 3}],
                output_size = 1)
        lookup_table.populate(function = lambda x: x)

        # Nearest interpolation at grid points
        npt.assert_almost_equal([-1], lookup_table.get_nearest([-1]))
        npt.assert_almost_equal([0], lookup_table.get_nearest([0]))
        npt.assert_almost_equal([1], lookup_table.get_nearest([1]))

        # Nearest interpolation between grid points
        npt.assert_almost_equal([-1], lookup_table.get_nearest([-0.75]))
        npt.assert_almost_equal([0], lookup_table.get_nearest([-0.25]))
        npt.assert_almost_equal([0], lookup_table.get_nearest([0.25]))
        npt.assert_almost_equal([1], lookup_table.get_nearest([0.75]))

        # Nearest interpolation outside of grid bounds
        npt.assert_almost_equal([-1], lookup_table.get_nearest([-10]))
        npt.assert_almost_equal([1], lookup_table.get_nearest([10]))

    def test_1d_lerp(self):

        # Lookup table for 1D identity function
        lookup_table = LookupTable(
                input_specifications = [{'from': -1, 'to': 1, 'points': 3}],
                output_size = 1)
        lookup_table.populate(function = lambda x: x)

        # Linear interpolation at grid points
        npt.assert_almost_equal([-1], lookup_table.get_lerp([-1]))
        npt.assert_almost_equal([0], lookup_table.get_lerp([0]))
        npt.assert_almost_equal([1], lookup_table.get_lerp([1]))

        # Linear interpolation between grid points
        npt.assert_almost_equal([-0.75], lookup_table.get_lerp([-0.75]))
        npt.assert_almost_equal([-0.25], lookup_table.get_lerp([-0.25]))
        npt.assert_almost_equal([0.25], lookup_table.get_lerp([0.25]))
        npt.assert_almost_equal([0.75], lookup_table.get_lerp([0.75]))

        # Linear interpolation outside of grid bounds
        npt.assert_almost_equal([-1], lookup_table.get_lerp([-10]))
        npt.assert_almost_equal([1], lookup_table.get_lerp([10]))

    def test_2d_lerp(self):

        # Lookup table for a linear 2D function
        def f(x):
            return (x[0] + x[1], x[0] - x[1])
        lookup_table = LookupTable(
                input_specifications = [
                    {'from': 0, 'to': 1, 'points': 2},
                    {'from': 0, 'to': 1, 'points': 2}],
                output_size = 2)
        lookup_table.populate(function = f)

        # Linear interpolation at grid points
        npt.assert_almost_equal([0, 0], lookup_table.get_lerp([0, 0]))
        npt.assert_almost_equal([1, -1], lookup_table.get_lerp([0, 1]))
        npt.assert_almost_equal([1, 1], lookup_table.get_lerp([1, 0]))
        npt.assert_almost_equal([2, 0], lookup_table.get_lerp([1, 1]))

        # Linear interpolation between grid points
        npt.assert_almost_equal([1, -0.5], lookup_table.get_lerp([0.25, 0.75]))
        npt.assert_almost_equal([1, 0.5], lookup_table.get_lerp([0.75, 0.25]))

        # Linear interpolation outside of grid bounds
        npt.assert_almost_equal([0.75, -0.75], lookup_table.get_lerp([-1, 0.75]))
        npt.assert_almost_equal([1.25, 0.75], lookup_table.get_lerp([2, 0.25]))
        npt.assert_almost_equal([0.75, 0.75], lookup_table.get_lerp([0.75, -1]))
        npt.assert_almost_equal([1.25, -0.75], lookup_table.get_lerp([0.25, 2]))
