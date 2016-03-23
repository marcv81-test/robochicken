from visual import *

from robotics.kinematics.tree import *
from robotics.displacement import *
from robotics.jacobian import *
from robotics.lookup import *

class Leg:
    """Multipod leg"""

    def __init__(self, initial_displacement = None):
        """Constructor"""
        if initial_displacement == None:
            self._initialize_tree(Displacement())
        else:
            self._initialize_tree(initial_displacement)
        self._joints_angles = np.asfarray([0] * 3)
        self._endpoint = self.endpoint(self._joints_angles)
        self._default_endpoint = self._endpoint
        self._rotation = initial_displacement.rotation.inverse()

    def endpoint(self, joints_angles):
        """Forward kinematics equation of the tree endpoint"""
        parameters = self._prepare_parameters(joints_angles)
        displacements = self._tree.evaluate(parameters)
        return displacements['tibia'].translation

    def initialize_draw(self):
        """Initialize the visual elements"""
        self._tree.initialize_draw()

    def uninitialize_draw(self):
        """Delete the visual elements"""
        self._tree.uninitialize_draw()

    def draw(self):
        """Render the visual elements"""
        parameters = self._prepare_parameters(self._joints_angles)
        self._tree.draw(parameters)

    def _initialize_tree(self, initial_displacement):
        """Initialize the kinematic tree"""
        self._tree = Tree(initial_displacement)
        self._tree.add_node(
                key = 'root_coxa_joint',
                part = RevoluteJoint(
                    axis = [0, 0, 1],
                    mount_angle = 0))
        self._tree.add_node(
                key = 'coxa',
                part = RigidLink(0.25),
                parent = 'root_coxa_joint')
        self._tree.add_node(
                key = 'coxa_femur_joint',
                part = RevoluteJoint(
                    axis = [0, 1, 0],
                    mount_angle = tau / 8),
                parent = 'coxa')
        self._tree.add_node(
                key = 'femur',
                part = RigidLink(1),
                parent = 'coxa_femur_joint')
        self._tree.add_node(
                key = 'femur_tibia_joint',
                part = RevoluteJoint(
                    axis = [0, 1, 0],
                    mount_angle = -tau / 4),
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

class JacobianSolverLeg(Leg):
    """Base class for legs solving inverse kinematics using the
    Jacobian matrix.
    """

    def __init__(self, **kwargs):
        """Constructor"""
        Leg.__init__(self, **kwargs)

    def endpoint_inverse_kinematics(self, target_offset):
        """Update the joints angles according to an IK approximation to
        attempt to reach a set endpoint position.
        """
        target_endpoint = self._default_endpoint + target_offset
        self._joints_angles = self._solver.converge(
                input_vector = self._joints_angles,
                target_output_vector = target_endpoint,
                output_vector = self._endpoint)
        self._endpoint = self.endpoint(self._joints_angles)

class JacobianInverseSolverLeg(JacobianSolverLeg):
    """Leg solving inverse kinematics using the Jacobian inverse"""

    def __init__(self, **kwargs):
        """Constructor"""
        JacobianLeg.__init__(self, **kwargs)
        self._solver = JacobianInverseSolver(
                function = lambda x: self.endpoint(x),
                max_input_fix = 0.5)

class DampedLeastSquaresSolverLeg(JacobianSolverLeg):
    """Leg solving inverse kinematics using the damped least squares"""

    def __init__(self, **kwargs):
        """Constructor"""
        JacobianLeg.__init__(self, **kwargs)
        self._solver = DampedLeastSquaresSolver(
                function = lambda x: self.endpoint(x),
                constant = 0.8)

class LookupTableLeg(Leg):
    """Leg solving inverse kinematics using a lookup table"""

    def __init__(self, lookup_table, initial_displacement = None):
        """Constructor"""
        Leg.__init__(self, initial_displacement = initial_displacement)
        self._lookup_table = lookup_table

    def endpoint_inverse_kinematics(self, target_offset):
        """Update the joints angles to reach a set endpoint position
        according to the lookup table.
        """
        input_vector = self._rotation.rotate(target_offset)
        self._joints_angles = self._lookup_table.get_lerp(input_vector)
        self._endpoint = self.endpoint(self._joints_angles)

    @staticmethod
    def populate(lookup_table):
        """Populate the lookup table using damped least squares
        iterations.
        """
        def f(input_vector):
            leg = DampedLeastSquaresSolverLeg()
            for _ in xrange(10):
                leg.endpoint_inverse_kinematics(input_vector)
            return leg._joints_angles
        lookup_table.populate(function = f)
