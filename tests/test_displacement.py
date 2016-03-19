import unittest
import numpy as np
import numpy.testing as npt

from robotics.displacement import *

class RotationTestCase(unittest.TestCase):

    def test_axis_angle(self):

        # Identity rotation has identity matrix
        r_1 = Rotation.axis_angle((1, 0, 0), 0)
        npt.assert_almost_equal(np.identity(3), r_1._matrix)

        # Tau/8 rotation around Z has the expected matrix
        r_2 = Rotation.axis_angle((0, 0, 1), tau / 8)
        npt.assert_almost_equal((
                (np.sqrt(2) / 2, -np.sqrt(2) / 2, 0),
                (np.sqrt(2) / 2,  np.sqrt(2) / 2, 0),
                (0,               0,              1)), r_2._matrix)

    def test_rotate(self):

        # Rotated vector has the expected value
        r = Rotation.axis_angle((0, 0, 1), tau / 4)
        npt.assert_almost_equal((0, 1, 0), r.rotate((1, 0, 0)))
        npt.assert_almost_equal((-1, 0, 0), r.rotate((0, 1, 0)))

    def test_compose(self):

        r_x = Rotation.axis_angle((1, 0, 0), -tau / 4)
        r_y = Rotation.axis_angle((0, 1, 0), tau / 4)
        r_z = Rotation.axis_angle((0, 0, 1), tau / 4)

        # Tau/4 rotation around Z then Y is the same as around -X then Z
        npt.assert_almost_equal(
            r_z.compose(r_y)._matrix,
            r_x.compose(r_z)._matrix)

    def test_invese(self):

        r_0 = Rotation.axis_angle((1, 0, 0), 0)
        r_1 = Rotation.axis_angle((1, 2, 3), tau / 12)

        # Composition with own inverse is identity
        r_2 = r_1.compose(r_1.inverse())
        npt.assert_almost_equal(r_0._matrix, r_2._matrix)

class DisplacementTestCase(unittest.TestCase):

    def test_compose(self):

        r_x = Displacement(rotation = Rotation.axis_angle((1, 0, 0), -tau / 4))
        r_y = Displacement(rotation = Rotation.axis_angle((0, 1, 0), tau / 4))
        r_z = Displacement(rotation = Rotation.axis_angle((0, 0, 1), tau / 4))
        t_x = Displacement(translation = (1, 0, 0))
        t_y = Displacement(translation = (0, 1, 0))

        # Tau/4 rotation around Z then Y is the same as around -X then Z
        npt.assert_almost_equal(
            r_z.compose(r_y).rotation._matrix,
            r_x.compose(r_z).rotation._matrix)

        # Tau/4 rotation around Z then translation along X is the same as
        # translation along Y then tau/4 rotation around Z.
        d_1 = r_z.compose(t_x)
        d_2 = t_y.compose(r_z)
        npt.assert_almost_equal(d_1.translation, d_2.translation)
        npt.assert_almost_equal(d_1.rotation._matrix, d_2.rotation._matrix)

    def test_inverse(self):

        d_0 = Displacement()
        d_1 = Displacement(
                translation = (1, 2, 3),
                rotation = Rotation.axis_angle((3, 2, 1), 3 * tau / 16))

        # Composition with own inverse is identity
        d_2 = d_1.compose(d_1.inverse())
        npt.assert_almost_equal(d_0.translation, d_2.translation)
        npt.assert_almost_equal(d_0.rotation._matrix, d_2.rotation._matrix)
