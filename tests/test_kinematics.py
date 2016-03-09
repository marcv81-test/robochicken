import unittest
import numpy as np
import numpy.testing as npt

from robotics.kinematics import *

class DisplacementTestCase(unittest.TestCase):

    def test_trivial_translation_vector(self):

        a = Displacement.create_rotation([0, 0, 1], tau / 4)
        b = Displacement.create_translation([1, 0, 0])

        npt.assert_almost_equal([0, 0, 0], a.translation_vector())
        npt.assert_almost_equal([1, 0, 0], b.translation_vector())
        npt.assert_almost_equal([1, 0, 0], b.compose(a).translation_vector())

    def test_translation_vector(self):

        a = Displacement.create_rotation([0, 0, 1], tau / 4)
        b = Displacement.create_translation([1, 0, 0])
        c = Displacement.create_rotation([0, 1, 0], tau / 4)

        # Positive rotation around Z in a right-handed coordinates system
        d = a.compose(b)
        npt.assert_almost_equal([0, 1, 0], d.translation_vector())

        # Combination of positive rotations in a right-handed coordinates system
        f = a.compose(b).compose(c).compose(b)
        npt.assert_almost_equal([0, 1, -1], f.translation_vector())
