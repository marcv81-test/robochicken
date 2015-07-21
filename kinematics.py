import math
import numpy as np

tau = 6.28318530718

def rotation(axis, theta):
    """Rotation matrix from axis and angle"""
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
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
    return [
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]]

class RigidMotion:
    """Rotation followed by a translation"""

    def __init__(self, rotation = np.identity(3), translation = np.zeros(3)):
        self._rotation = rotation
        self._translation = translation

    def translation_only(self):
        """Equivalent translation if applied without (or before) the rotation"""
        inverse_rotation = np.linalg.inv(self._rotation)
        return np.dot(inverse_rotation, self._translation)

    def compose(self, other):
        """Equivalent rigid motion to applying self then other"""
        rotation = np.dot(other._rotation, self._rotation)
        translation = np.dot(other._rotation, self._translation) + other._translation
        return RigidMotion(rotation, translation)

class JacobianSolver:
    """Numerical solver using a simplified Jacobian inverse technique"""

    def __init__(self, function, input_delta = 0.001):
        self._function = function
        self._input_delta = input_delta

    def jacobian_matrix(self, input_vector, output_vector = None):
        """Jacobian matrix of the function at the input vector"""
        if (output_vector == None):
            output_vector = self._function(input_vector)
        input_size = len(input_vector)
        output_size = len(output_vector)
        matrix = np.zeros((input_size, output_size))
        for i in range(input_size):
            altered_input_vector = list(input_vector)
            altered_input_vector[i] = altered_input_vector[i] + self._input_delta
            altered_output_vector = self._function(altered_input_vector)
            output_delta_vector = altered_output_vector - output_vector
            matrix[i] = output_delta_vector / self._input_delta
        return np.transpose(matrix)

    def converge(self, input_vector, target_output_vector, output_vector = None):
        """Update input vector to converge toward target output vector (unsafe near singularities)"""
        if (output_vector == None):
            output_vector = self._function(input_vector)
        output_error_vector = target_output_vector - output_vector
        jacobian_matrix = self.jacobian_matrix(input_vector, output_vector)
        input_fix_vector = np.linalg.solve(jacobian_matrix, output_error_vector)
        return input_vector + input_fix_vector

class Limb:
    """Robotic arm/leg model, hexapod leg by default"""

    def __init__(self,
            initial_rigid_motion = RigidMotion(),
            lengths = [0.25, 1, 2],
            axes = [[0, 0, 1], [0, 1, 0], [0, 1, 0]],
            angles_limits = [
                [-tau / 8.0, tau / 8.0],
                [-3.0 * tau / 8.0, 3.0 * tau / 8.0],
                [-3.0 * tau / 8.0, -tau / 8.0]]):
        self._initial_rigid_motion = initial_rigid_motion
        self._lengths = lengths
        self._axes = axes
        self._angles_limits = angles_limits
        self._sections_count = len(lengths)
        default_angles = [0] * self._sections_count
        for i in range(self._sections_count):
            min_angle = self._angles_limits[i][0]
            max_angle = self._angles_limits[i][1]
            default_angles[i] = (min_angle + max_angle) / 2.0
        self.forward_kinematics(default_angles)

    def end_point(self, angles):
        """Forward kinematics equation"""
        rigid_motion = self._initial_rigid_motion
        for i in range(self._sections_count):
            rigid_motion = rigid_motion.compose(RigidMotion(
                rotation(self._axes[i], angles[i]),
                [self._lengths[i], 0, 0]))
        return rigid_motion.translation_only()

    def forward_kinematics(self, angles):
        """Update the internal state according to forward kinematics"""
        for i in range(self._sections_count):
            min_angle = self._angles_limits[i][0]
            max_angle = self._angles_limits[i][1]
            if (angles[i] < min_angle or angles[i] > max_angle):
                return
        self._angles = angles
        self._end_point = self.end_point(angles)

    def inverse_kinematics(self, target_end_point):
        """Update the internal state according to inverse kinematics"""
        solver = JacobianSolver(lambda x: self.end_point(x))
        self.forward_kinematics(solver.converge(
            self._angles, target_end_point, output_vector = self._end_point))
