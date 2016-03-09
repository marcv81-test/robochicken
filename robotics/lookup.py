import numpy as np

class LookupTable:
    """
    Instances of this class associate output vectors to input vectors at the
    points of a uniform grid. We can then calculate the output vector at any
    input vector using nearest-neighbor or linear interpolation. The output
    and input vectors can have any size.
    """

    def __init__(self, input_specifications, output_size, epsilon = 1e-9):
        """
        Constructor. Defines the bounds of the grid and the number of its points
        for each component of the input vector.
        """
        self._epsilon = float(epsilon)
        # Input specifications
        self._input_size = len(input_specifications)
        self._input_from = np.array([float(x['from']) for x in input_specifications])
        self._input_to = np.array([float(x['to']) for x in input_specifications])
        self._input_points = np.array([int(x['points']) for x in input_specifications])
        # Output specifications
        self._output_size = int(output_size)
        # Internal table
        shape = list(self._input_points)
        shape.append(self._output_size)
        self._table = np.zeros(shape)

    def save(self, filename):
        """Saves the lookup table"""
        np.save(filename, np.array(self._table))

    def load(self, filename):
        """Loads the lookup table"""
        shape = list(self._input_points)
        shape.append(self._output_size)
        table = np.load(filename)
        if table.shape != tuple(shape):
            raise ValueError('Shapes do not match')
        self._table = table

    def populate(self, function):
        """Populates the lookup table at all the points of the grid"""
        self._iterate_all(lambda input_vector:
                self._set(input_vector, function(self._to_world(input_vector))))

    def get_nearest(self, world_input_vector):
        """Estimates the output vector using nearest-neighbor interpolation"""
        input_vector = self._from_world(world_input_vector)
        input_vector = np.round(input_vector)
        return self._get(input_vector)

    def get_lerp(self, world_input_vector):
        """Estimates the output vector using linear interpolation"""
        input_vector = self._from_world(world_input_vector)
        first_corner = np.floor(input_vector)
        weights = list()
        self._iterate_hypercube(
                function = lambda input_vector: weights.append(self._get(input_vector)),
                input_vector = first_corner)
        distances = list(reversed(input_vector - first_corner))
        return LookupTable._process_lerp(weights, distances)

    def _from_world(self, world_input_vector):
        """
        Converts an input vector from world to internal coordinates. At the points
        of the grid the components of the input vector in internal coordinates are
        integers.
        """
        input_vector = np.array(world_input_vector)
        input_vector = input_vector - self._input_from
        input_vector = np.multiply(input_vector, self._input_points - 1)
        input_vector = np.divide(input_vector, self._input_to - self._input_from)
        # Ensure input vector is within grid bounds
        input_vector = np.maximum(input_vector, self._epsilon)
        input_vector = np.minimum(input_vector, self._input_points - 1 - self._epsilon)
        return input_vector

    def _to_world(self, input_vector):
        """Converts an input vector from internal to world coordinates"""
        world_input_vector = np.array(input_vector)
        world_input_vector = np.multiply(world_input_vector, self._input_to - self._input_from)
        world_input_vector = np.divide(world_input_vector, self._input_points - 1)
        world_input_vector = world_input_vector + self._input_from
        return world_input_vector

    def _get(self, input_vector):
        """
        Gets the output vector at points of the grid. Operates on an input vector
        in internal coordinates.
        """
        return self._table[tuple(input_vector)]

    def _set(self, input_vector, output_vector):
        """
        Sets the output vector at points of the grid. Operates on an input vector
        in internal coordinates.
        """
        self._table[tuple(input_vector)] = output_vector

    def _iterate_all(self, function, input_vector = None, index = 0):
        """
        Calls a function on all the points of the grid. Operates on input vectors
        in internal coordinates.
        """
        if input_vector == None: input_vector = [0] * self._input_size
        if index == self._input_size: function(input_vector)
        else:
            for i in xrange(self._input_points[index]):
                new_input_vector = list(input_vector)
                new_input_vector[index] = i
                self._iterate_all(function, new_input_vector, index + 1)

    def _iterate_hypercube(self, function, input_vector, index = 0):
        """
        Calls a function on all the corners of a hypercube made of adjacent points
        of the grid. The input vector is the lowest corner of the hypercube. Operates
        on input vectors in internal coordinates.
        """
        if index == self._input_size: function(input_vector)
        else:
            for i in xrange(2):
                new_input_vector = list(input_vector)
                new_input_vector[index] += i
                self._iterate_hypercube(function, new_input_vector, index + 1)

    @staticmethod
    def _process_lerp(weights, distances):
        """Processes the linear interpolation weights and distances"""
        if len(weights) == 1: return weights[0]
        else:
            d1 = 1 - distances[0]
            d2 = distances[0]
            new_weights = list()
            while len(weights) > 0:
                w1 = weights.pop(0)
                w2 = weights.pop(0)
                new_weights.append(w1 * d1 + w2 * d2)
            new_distances = distances[1:]
            return LookupTable._process_lerp(new_weights, new_distances)
