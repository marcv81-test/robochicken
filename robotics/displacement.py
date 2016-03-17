import numpy as np

tau = 6.28318530718

class Displacement:
    """Combination of a translation and a rotation representing a rigid
    body displacement in space.

    This implementation stores the translation as a vector and the
    rotation as a 3x3 matrix. The translation is applied *before* the
    rotation.
    """

    def __init__(self):
        """Identity constructor"""
        self._translation = np.zeros(3)
        self._rotation = np.identity(3)

    @staticmethod
    def create_translation(vector):
        """Create translation from vector"""
        displacement = Displacement()
        displacement._translation = np.array(vector, np.float_)
        return displacement

    @staticmethod
    def create_rotation(axis, angle):
        """Create rotation from axis and angle"""
        axis = axis / np.linalg.norm(axis)
        a = np.cos(angle / 2.0)
        b, c, d = -axis * np.sin(angle / 2.0)
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
        displacement._rotation = np.array((
                (aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)),
                (2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)),
                (2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc)),
                np.float_)
        return displacement

    def compose(self, other):
        """Equivalent displacement to self then other"""
        displacement = Displacement()
        displacement._translation = self._translation + \
                np.dot(self._rotation, other._translation)
        displacement._rotation = np.dot(self._rotation, other._rotation)
        return displacement

    def translation_vector(self):
        """Translation vector of the displacement"""
        return self._translation
