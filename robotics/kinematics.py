import collections
import numpy as np
import visual as v

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

class Tree:
    """Kinematic Tree. Can be used as a kinematic chain.

    Each node has a key, a part, a parent, and a list of children.
    The part, parent, and list of children are stored by key in
    dictionaries. This allows arbitrary node access by key.
    """

    def __init__(self, root_displacement):
        """Constructor"""
        self._parts = { 'root': _Root(root_displacement) }
        self._parents = dict()
        self._children = { 'root': list() }

    def add_node(self, key, part, parent = 'root'):
        """Add a node to the tree"""
        if key in self._parts:
            raise ValueError('Key "' + key + '" aready exists')
        self._parts[key] = part
        self._parents[key] = parent
        self._children[key] = list()
        self._children[parent].append(key)

    def prepare_parameters(self):
        """Prepare empty parameters for evaluate"""
        parameters = dict()
        for key in self._parts:
            parameters[key] = dict()
        return parameters

    def evaluate(self, parameters):
        """Walk the tree and evaluate the displacement at each node
        using forward kinematics.
        """
        displacements = dict()
        todo = collections.deque()
        todo.append('root')
        while todo:
            key = todo.popleft()
            displacement = self._parts[key].displacement(**parameters[key])
            if key != 'root':
                previous_displacement = displacements[self._parents[key]]
                displacement = previous_displacement.compose(displacement)
            displacements[key] = displacement
            for child_key in self._children[key]:
                todo.append(child_key)
        return displacements

    def initialize_draw(self):
        """Initialize the visual parts"""
        for key in self._parts:
            if hasattr(self._parts[key], 'initialize_draw'):
                self._parts[key].initialize_draw()

    def uninitialize_draw(self):
        """Delete the visual parts"""
        for key in self._parts:
            if hasattr(self._parts[key], 'initialize_draw'):
                self._parts[key].uninitialize_draw()

    def draw(self, parameters):
        """Draw the visual parts"""
        displacements = self.evaluate(parameters)
        for key in self._parts:
            if hasattr(self._parts[key], 'draw'):
                displacement_before = displacements[self._parents[key]]
                displacement_after = displacements[key]
                self._parts[key].draw(displacement_before, displacement_after)

class RigidLink:
    """Rigid link part"""

    def __init__(self, length):
        self._displacement = Displacement.create_translation([length, 0, 0])

    def displacement(self):
        return self._displacement

    def initialize_draw(self):
        self._rod = v.cylinder(radius = 0.05)

    def uninitialize_draw(self):
        self._rod.visible = False
        del self._rod

    def draw(self, displacement_before, displacement_after):
        point_before = displacement_before.translation_vector()
        point_after = displacement_after.translation_vector()
        self._rod.pos = point_before
        self._rod.axis = point_after - point_before

class RevoluteJoint:
    """Revolute joint part"""

    def __init__(self, axis, mount_angle):
        self._axis = axis
        self._mount_angle = mount_angle

    def displacement(self, angle):
        angle = angle + self._mount_angle
        return Displacement.create_rotation(self._axis, angle)

class _Root:
    """Part used as the root node of any tree.

    Stores the intial displacement. Shall *not* be used as a regular
    part anywhere else in a tree.
    """

    def __init__(self, displacement):
        self._displacement = displacement

    def displacement(self):
        return self._displacement
