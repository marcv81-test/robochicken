from visual import *

from kinematics import *
from jacobian import *

class HexapodLeg:
    """Hexapod leg"""

    def __init__(self,
            initial_displacement = Displacement(),
            limited_joints = False,
            algorithm = 'Jacobian Inverse'):
        """Constructor"""

        self._initialize_tree(initial_displacement, limited_joints)
        self._joints_angles = [0] * 3
        self._endpoint = self.endpoint(self._joints_angles)
        self._default_endpoint = self._endpoint
        self._target_endpoint = self._endpoint

        # IK algorithm selection
        if algorithm == 'Damped Least Squares':
            self._solver = DampedLeastSquaresSolver(
                    function = lambda x: self.endpoint(x),
                    constant = 0.8)
        elif algorithm == 'Jacobian Inverse':
            self._solver = JacobianInverseSolver(
                    function = lambda x: self.endpoint(x),
                    max_input_fix = 0.5)

    def endpoint(self, joints_angles):
        """Forward kinematics equation of the tree endpoint"""
        parameters = self._prepare_parameters(joints_angles)
        displacements = self._tree.evaluate(parameters)
        return displacements['tibia'].translation_vector()

    def endpoint_inverse_kinematics(self, target_endpoint):
        """
        Update the joints angles according to an IK approximation
        to attempt to reach a set endpoint position.
        """
        self._target_endpoint = target_endpoint
        self._joints_angles = self._solver.converge(
                input_vector = self._joints_angles,
                target_output_vector = self._target_endpoint,
                output_vector = self._endpoint)
        self._endpoint = self.endpoint(self._joints_angles)

    def displacement_inverse_kinematics(self, target_displacement):
        """
        Update the joints angles according to an IK approximation
        to attempt to reach a set displacement from the default
        endpoint position.
        """
        default_displacement = Displacement.create_translation(
                self._default_endpoint)
        displacement = target_displacement.compose(default_displacement)
        target_endpoint = displacement.translation_vector()
        self.endpoint_inverse_kinematics(target_endpoint)

    def initialize_draw(self):
        """Initialize the visual elements"""
        self._tree.initialize_draw()
        self._ball = visual.sphere(radius = 0.1, color = visual.color.red)

    def draw(self):
        """Render the visual elements"""
        parameters = self._prepare_parameters(self._joints_angles)
        self._tree.draw(parameters)
        self._ball.pos = self._target_endpoint

    def _initialize_tree(self, initial_displacement, limited_joints):
        """Initialize the kinematic tree"""

        # Prepare the joints factory
        joints_axes = {
                'root_coxa_joint': [0, 0, 1],
                'coxa_femur_joint': [0, 1, 0],
                'femur_tibia_joint': [0, 1, 0]
        }
        joints_mount_angles = {
                'root_coxa_joint': 0,
                'coxa_femur_joint': tau / 8,
                'femur_tibia_joint': -tau / 4
        }
        joints_amplitudes = {
                'root_coxa_joint': tau / 8,
                'coxa_femur_joint': tau / 4,
                'femur_tibia_joint': tau / 4
        }
        if limited_joints:
            def create_joint(name):
                return LimitedRevoluteJoint(
                        axis = joints_axes[name],
                        mount_angle = joints_mount_angles[name],
                        amplitude = joints_amplitudes[name])
        else:
            def create_joint(name):
                return RevoluteJoint(
                        axis = joints_axes[name],
                        mount_angle = joints_mount_angles[name])

        # Now build the tree
        self._tree = Tree(initial_displacement)
        self._tree.add_node(
                key = 'root_coxa_joint',
                part = create_joint('root_coxa_joint'))
        self._tree.add_node(
                key = 'coxa',
                part = RigidLink(0.25),
                parent = 'root_coxa_joint')
        self._tree.add_node(
                key = 'coxa_femur_joint',
                part = create_joint('coxa_femur_joint'),
                parent = 'coxa')
        self._tree.add_node(
                key = 'femur',
                part = RigidLink(1),
                parent = 'coxa_femur_joint')
        self._tree.add_node(
                key = 'femur_tibia_joint',
                part = create_joint('femur_tibia_joint'),
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

class Hexapod:
    """Hexapod with direct legs control"""

    def __init__(self,
            displacement = Displacement(),
            limited_joints = False,
            algorithm = 'Jacobian Inverse'):
        """Constructor"""
        self._legs_count = 6
        self._legs = [None] * self._legs_count
        leg_translation = Displacement.create_translation([1, 0, 0])
        for i in range(self._legs_count):
            angle = tau / (2 * self._legs_count) + i * tau / self._legs_count
            leg_rotation = Displacement.create_rotation([0, 0, 1], angle)
            leg_displacement = leg_rotation.compose(leg_translation)
            initial_displacement = displacement.compose(leg_displacement)
            self._legs[i] = HexapodLeg(
                    initial_displacement = initial_displacement,
                    limited_joints = limited_joints,
                    algorithm = algorithm)

    def direct_control(self, x, y, z, angle):
        """Control the legs directly"""
        target_rotation = Displacement.create_rotation([0, 0, 1], angle)
        target_translation = Displacement.create_translation([x, y, z])
        target_displacement = target_translation.compose(target_rotation)
        for i in range(self._legs_count):
            self._legs[i].displacement_inverse_kinematics(target_displacement)

    def initialize_draw(self):
        """Initialize the visual elements"""
        for i in range(self._legs_count):
            self._legs[i].initialize_draw()

    def draw(self):
        """Render the visual elements"""
        for i in range(self._legs_count):
            self._legs[i].draw()
