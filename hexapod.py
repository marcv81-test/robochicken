from visual import *

from kinematics import *
from jacobian import *

class HexapodLeg:
    """Hexapod leg"""

    def __init__(self, initial_displacement = Displacement()):
        """Constructor"""
        self._initialize_tree(initial_displacement)
        self._joints_angles = [0] * 3
        self._joints_amplitudes = [tau / 8, tau / 4, 3 * tau / 16]
        self._endpoint = self.endpoint(self._joints_angles)
        self._default_endpoint = self._endpoint
        self._target_endpoint = self._endpoint
        self._solver = JacobianInverseSolver(
                function = lambda x: self.endpoint(x),
                max_input_fix_norm = 0.25)

    def endpoint(self, joints_angles):
        """Forward kinematics equation of the tree endpoint"""
        parameters = self._prepare_parameters(joints_angles)
        displacements = self._tree.evaluate(parameters)
        return displacements['tibia'].translation_vector()

    def inverse_kinematics(self, target_endpoint):
        """Update the joints angles according to an IK approximation"""
        self._target_endpoint = target_endpoint
        self._joints_angles = self._solver.converge(
                input_vector = self._joints_angles,
                target_output_vector = self._target_endpoint,
                output_vector = self._endpoint)
        self._limit_joints_angles()
        self._endpoint = self.endpoint(self._joints_angles)

    def initialize_draw(self):
        """Initialize the visual elements"""
        self._tree.initialize_draw()
        self._ball = visual.sphere(radius = 0.1, color = visual.color.red)

    def draw(self):
        """Render the visual elements"""
        parameters = self._prepare_parameters(self._joints_angles)
        self._tree.draw(parameters)
        self._ball.pos = self._target_endpoint

    def _initialize_tree(self, initial_displacement):
        """Initialize the kinematic tree"""
        self._tree = Tree(initial_displacement)
        self._tree.add_node(
                key = 'root_coxa_joint',
                part = RevoluteJoint([0, 0, 1], 0))
        self._tree.add_node(
                key = 'coxa',
                part = RigidLink(0.25),
                parent = 'root_coxa_joint')
        self._tree.add_node(
                key = 'coxa_femur_joint',
                part = RevoluteJoint([0, 1, 0], tau / 8),
                parent = 'coxa')
        self._tree.add_node(
                key = 'femur',
                part = RigidLink(1),
                parent = 'coxa_femur_joint')
        self._tree.add_node(
                key = 'femur_tibia_joint',
                part = RevoluteJoint([0, 1, 0], -tau / 4),
                parent = 'femur')
        self._tree.add_node(
                key = 'decoration',
                part = RigidLink(0.1),
                parent = 'femur')
        self._tree.add_node(
                key = 'tibia',
                part = RigidLink(2),
                parent = 'femur_tibia_joint')

    def _prepare_parameters(self, joints_angles):
        """Prepare tree parameters from joints angles"""
        parameters = self._tree.prepare_parameters()
        parameters['root_coxa_joint']['angle'] = joints_angles[0]
        parameters['coxa_femur_joint']['angle'] = joints_angles[1]
        parameters['femur_tibia_joint']['angle'] = joints_angles[2]
        return parameters

    def _limit_joints_angles(self):
        """Apply mechanical constraints on joints angles"""
        for i in range(3):
            if self._joints_angles[i] > self._joints_amplitudes[i]:
                self._joints_angles[i] = self._joints_amplitudes[i]
            if self._joints_angles[i] < -self._joints_amplitudes[i]:
                self._joints_angles[i] = -self._joints_amplitudes[i]

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
