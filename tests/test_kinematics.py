import unittest
import numpy as np
import numpy.testing as npt

from robotics.kinematics import *

class DisplacementTestCase(unittest.TestCase):

    def test_constructor_side_effects(self):

        # create_translation() does not hijack input vector instance
        vector = np.array((1, 0, 0), np.float_)
        a = Displacement.create_translation(vector)
        npt.assert_equal(False, vector is a._translation)

        # create_rotation() does not clobber input axis
        axis = np.array((1, 0, 0), np.float_)
        b = Displacement.create_rotation(axis, tau / 4)
        npt.assert_almost_equal((1, 0, 0), axis)

    def test_trivialities(self):

        a = Displacement.create_rotation((0, 0, 1), tau / 4)
        b = Displacement.create_translation((1, 0, 0))

        # Pure rotation has null translation vector
        npt.assert_almost_equal((0, 0, 0), a.translation_vector())

        # Pure translation has translation vector it was created with
        npt.assert_almost_equal((1, 0, 0), b.translation_vector())

        # Composition with pure rotation does not change translation vector
        npt.assert_almost_equal((1, 0, 0), b.compose(a).translation_vector())

    def test_composition(self):

        a = Displacement.create_rotation((0, 0, 1), tau / 4)
        b = Displacement.create_translation((1, 0, 0))
        c = Displacement.create_rotation((0, 1, 0), tau / 4)

        # Positive rotation around Z in a right-handed coordinates system
        d = a.compose(b)
        npt.assert_almost_equal((0, 1, 0), d.translation_vector())

        # Combination of positive rotations in a right-handed coordinates system
        f = a.compose(b).compose(c).compose(b)
        npt.assert_almost_equal((0, 1, -1), f.translation_vector())
