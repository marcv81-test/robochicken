import unittest
import numpy as np
import numpy.testing as npt

from robotics.displacement import *

class DisplacementTestCase(unittest.TestCase):

    def test_rotation(self):

        # Pure rotation has null translation vector
        r = Displacement.create_rotation((0, 0, 1), tau / 4)
        npt.assert_almost_equal((0, 0, 0), r.translation_vector())

    def test_translation(self):

        # Pure translation has trivial translation vector
        t = Displacement.create_translation((1, 0, 0))
        npt.assert_almost_equal((1, 0, 0), t.translation_vector())

    def test_composition(self):

        r1 = Displacement.create_rotation((0, 0, 1), tau / 4)
        r2 = Displacement.create_rotation((0, 1, 0), tau / 4)
        t = Displacement.create_translation((1, 0, 0))

        # Use your right hand to create a frame of reference: thumb is X,
        # index finger is Y, middle finger is Z. Remember the initial
        # orientation. Imagine a point at (0, 0, 0).

        # Translation vectors and rotation axes are given in the frame of
        # reference created by your right hand at the time you apply them.
        # Translations displace the point. Rotations turn your right hand
        # without moving the point.

        # Positive rotations around an axis: point the right hand thumb to
        # the axis, the rotation follow the curl of the fingers.

        # The displacement translation vector is the position of the point
        # in the frame of reference created by the initial orientation of
        # your right hand.

        # Test #1
        # - Positive rotation around Z
        # - Positive translation along X (now pointing toward initial Y)
        x = r1.compose(t)
        npt.assert_almost_equal((0, 1, 0), x.translation_vector())

        # Test #2
        # - Positive rotation around Z
        # - Positive translation along X (now pointing toward initial Y)
        # - Positive rotation around Y (now pointing toward initial -X)
        # - Positive translation along X (now pointing toward initial -Z)
        y = r1.compose(t).compose(r2).compose(t)
        npt.assert_almost_equal((0, 1, -1), y.translation_vector())

    def test_inverse(self):

        # Identity displacement
        d0 = Displacement.create_translation((0, 0, 0))

        # Non-trivial displacements
        t = Displacement.create_translation((1, 2, 3))
        r = Displacement.create_rotation((3, 2, 1), 3 * tau / 16)
        a = r.compose(t)

        # Composition with own inverse results in identity displacement
        b = a.compose(a.inverse())
        npt.assert_almost_equal(b._translation, d0._translation)
        npt.assert_almost_equal(b._rotation, d0._rotation)
