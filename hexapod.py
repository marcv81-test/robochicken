from visual import *

from kinematics import *
from jacobian import *

class HexapodLeg:
    """Legs are made of 3 revolute joints and 3 rigid links"""

    def __init__(self, initial_displacement = Displacement()):
        """Constructor"""
        self._initial_displacement = initial_displacement
        self._sections_count = 3
        self._links_lengths = [0.25, 1, 2]
        self._joints_axes = [[0, 0, 1], [0, 1, 0], [0, 1, 0]]
        self._joints_angles = [0, -tau / 8, tau / 4]
        self._endpoint = self.endpoint(self._joints_angles)
        self._default_endpoint = self._endpoint
        self._target_endpoint = self._endpoint
        self._solver = JacobianInverseSolver(
                function = lambda x: self.endpoint(x),
                max_input_fix_norm = 0.25)

    def endpoint(self, joints_angles):
        """Forward kinematics equation"""
        displacement = self._initial_displacement
        for i in range(self._sections_count):
            displacement = displacement.compose(
                    Displacement.create_rotation(self._joints_axes[i], joints_angles[i]))
            displacement = displacement.compose(
                    Displacement.create_translation([self._links_lengths[i], 0, 0]))
        return displacement.translation_vector()

    def inverse_kinematics(self, target_endpoint):
        """Update the joints angles according to an IK approximation"""
        self._target_endpoint = target_endpoint
        self._joints_angles = self._solver.converge(
                input_vector = self._joints_angles,
                target_output_vector = self._target_endpoint,
                output_vector = self._endpoint)
        self._endpoint = self.endpoint(self._joints_angles)

    def initialize_draw(self):
        """Initialize the visual elements"""
        self._rods = [None] * self._sections_count
        for i in range(self._sections_count):
            self._rods[i] = visual.cylinder(radius = 0.05);
        self._ball = visual.sphere(radius = 0.1, color = visual.color.red)

    def draw(self):
        """Render the visual elements"""
        points = [None] * (self._sections_count + 1)
        displacement = self._initial_displacement
        points[0] = displacement.translation_vector()
        for i in range(self._sections_count):
            displacement = displacement.compose(
                    Displacement.create_rotation(self._joints_axes[i], self._joints_angles[i]))
            displacement = displacement.compose(
                    Displacement.create_translation([self._links_lengths[i], 0, 0]))
            points[i + 1] = displacement.translation_vector()
            self._rods[i].pos = points[i]
            self._rods[i].axis = points[i + 1] - points[i]
        self._ball.pos = self._target_endpoint

class Hexapod:
    """Hexapod with direct legs control"""

    def __init__(self):
        """Constructor"""
        self._legs_count = 6
        self._legs = [None] * self._legs_count
        initial_translation = Displacement.create_translation([1, 0, 0])
        for i in range(self._legs_count):
            angle = tau / (2 * self._legs_count) + i * tau / self._legs_count
            initial_rotation = Displacement.create_rotation([0, 0, 1], angle)
            initial_displacement = initial_rotation.compose(initial_translation)
            self._legs[i] = HexapodLeg(initial_displacement)

    def direct_control(self, x, y, z, angle):
        """Control the legs directly"""
        target_rotation = Displacement.create_rotation([0, 0, 1], angle)
        target_translation = Displacement.create_translation([x, y, z])
        target_displacement = target_translation.compose(target_rotation)
        for i in range(self._legs_count):
            default_displacement = Displacement.create_translation(
                    self._legs[i]._default_endpoint)
            leg_displacement = target_displacement.compose(default_displacement)
            target_endpoint = leg_displacement.translation_vector()
            self._legs[i].inverse_kinematics(target_endpoint)

    def initialize_draw(self):
        """Initialize the visual elements"""
        for i in range(self._legs_count):
            self._legs[i].initialize_draw()

    def draw(self):
        """Render the visual elements"""
        for i in range(self._legs_count):
            self._legs[i].draw()
