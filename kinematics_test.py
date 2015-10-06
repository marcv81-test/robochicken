import numpy as np
import numpy.testing as npt

from kinematics import *

class TestDisplacement:

    def test_trivial_translation_vector(self):

        a = Displacement.create_rotation([0, 0, 1], tau / 2)
        b = Displacement.create_translation([1, 0, 0])

        npt.assert_almost_equal([0, 0, 0], a.translation_vector())
        npt.assert_almost_equal([1, 0, 0], b.translation_vector())
        npt.assert_almost_equal([1, 0, 0], b.compose(a).translation_vector())

    def test_translation_vector(self):

        a = Displacement.create_rotation([0, 0, 1], tau / 2)
        b = Displacement.create_translation([1, 0, 0])

        npt.assert_almost_equal([-1, 0, 0], a.compose(b).translation_vector())

if __name__ == "__main__":
    npt.run_module_suite()
