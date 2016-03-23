import numpy as np

tau = 6.28318530718

class Rotation:
    """This implementation uses 3x3 matrices"""

    def __init__(self):
        """Identity constructor"""
        self._matrix = np.identity(3, np.float_)

    @staticmethod
    def axis_angle(axis, angle):
        """Creates a rotation from an axis and an angle"""
        axis = np.asfarray(axis)
        normalized_axis = axis / np.linalg.norm(axis)
        a = np.cos(angle / 2.0)
        b, c, d = -normalized_axis * np.sin(angle / 2.0)
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
        rotation = Rotation()
        rotation._matrix = np.array((
                (aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)),
                (2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)),
                (2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc)),
                np.float_)
        return rotation

    def copy(self):
        """Clones a rotation"""
        rotation = Rotation()
        rotation._matrix = self._matrix.copy()
        return rotation

    def rotate(self, vector):
        """Rotates a vector"""
        vector = np.asfarray(vector)
        return np.dot(self._matrix, vector)

    def compose(self, other):
        """Equivalent rotation to self then other"""
        rotation = Rotation()
        rotation._matrix = np.dot(self._matrix, other._matrix)
        return rotation

    def inverse(self):
        """Returns the inverse rotation (same axis, opposite angle)"""
        rotation = Rotation()
        rotation._matrix = np.transpose(self._matrix)
        return rotation

class Displacement:
    """Combination of a translation and a rotation representing a rigid
    body displacement in space. The translation is applied *before* the
    rotation.
    """

    def __init__(self, translation = None, rotation = None):
        """Constructor"""
        if translation != None: self.translation = np.array(translation)
        else: self.translation = np.asarray((0, 0, 0), np.float_)
        if rotation != None: self.rotation = rotation.copy()
        else: self.rotation = Rotation()

    def copy(self):
        """Clones a displacement"""
        displacement = Displacement(
                translation = self.translation.copy(),
                rotation = self.rotation.copy())
        return displacement

    def compose(self, other):
        """Equivalent displacement to self then other"""
        displacement = Displacement(
            self.translation + self.rotation.rotate(other.translation),
            self.rotation.compose(other.rotation))
        return displacement

    def inverse(self):
        """Returns the inverse displacement"""
        inverse_rotation = self.rotation.inverse()
        displacement = Displacement(
            inverse_rotation.rotate(-self.translation), inverse_rotation)
        return displacement
