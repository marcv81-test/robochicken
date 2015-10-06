import math
import numpy as np

tau = 6.28318530718

class Displacement:
    """
    Combination of a rotation and a translation
    representing a rigid body displacement in space.

    This implementation stores the rotation as a 3x3 matrix
    and the translation as a vector applied *after* the rotation.
    """

    def __init__(self):
        """Identity constructor"""
        self._rotation = np.identity(3)
        self._translation = np.zeros(3)

    @staticmethod
    def create_rotation(axis, angle):
        """Factory from rotation axis and angle"""
        axis = axis / np.linalg.norm(axis)
        a = math.cos(angle / 2.0)
        b, c, d = -axis * math.sin(angle / 2.0)
        aa = a * a
        bb = b * b
        cc = c * c
        dd = d * d
        bc = b * c
        ad = a * d
        ac = a * c
        ab = a * b
        bd = b * d
        cd = c * d
        displacement = Displacement()
        displacement._rotation = [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]]
        return displacement

    @staticmethod
    def create_translation(vector):
        """Factory from translation vector"""
        displacement = Displacement()
        displacement._translation = vector
        return displacement

    def compose(self, other):
        """Equivalent displacement to self then other"""
        displacement = Displacement()
        displacement._rotation = np.dot(other._rotation, self._rotation)
        displacement._translation = other._translation + \
                np.dot(other._rotation, self._translation)
        return displacement

    def translation_vector(self):
        """Vector representing only the translation of the displacement"""
        inverse_rotation = np.linalg.inv(self._rotation)
        return np.dot(inverse_rotation, self._translation)
