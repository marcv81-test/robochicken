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
        Constructor. Defines the bounds of the grid and the number of its
        points for each component of the input vector, and the number of
        components of the output vector.
        """
        # Input
        self._input_size = len(input_specifications)
        self._input_from = np.array(
                tuple(float(x['from']) for x in input_specifications))
        self._input_to = np.array(
                tuple(float(x['to']) for x in input_specifications))
        self._input_points = np.array(
                tuple(int(x['points']) for x in input_specifications))
        self._input_span = self._input_to - self._input_from
        # Output
        self._output_size = int(output_size)
        # Lookup table
        shape = list(self._input_points)
        shape.append(self._output_size)
        self._table = np.zeros(shape)
        # Epsilon
        self._epsilon = float(epsilon)

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
        def f(input_indices):
            input_vector = self._from_indices(input_indices)
            self._set(input_indices, function(input_vector))
        self._iterate_all(function = f)

    def get_nearest(self, input_vector):
        """Estimates the output vector using nearest-neighbor interpolation"""
        input_indices = self._to_indices(input_vector)
        input_indices = np.round(input_indices)
        return self._get(input_indices)

    def get_lerp(self, input_vector):
        """Estimates the output vector using linear interpolation"""
        input_indices = self._to_indices(input_vector)
        first_corner = np.floor(input_indices)
        weights = list()
        def f(input_indices):
            weights.append(self._get(input_indices))
        self._iterate_hypercube(function = f, input_indices = first_corner)
        distances = list(reversed(input_indices - first_corner))
        return LookupTable._process_lerp(weights, distances)

    def _to_indices(self, input_vector):
        """
        Converts an input vector to lookup table indices. The indices may
        not be integers and may need to be rounded. However they are always
        in the open inverval from 0 to self._input_points - 1.
        """
        input_indices = np.array(input_vector)
        input_indices = input_indices - self._input_from
        input_indices = np.multiply(input_indices, self._input_points - 1)
        input_indices = np.divide(input_indices, self._input_span)
        # Ensure indices are within bounds
        minimum = self._epsilon
        maximum = self._input_points - 1 - self._epsilon
        input_indices = np.maximum(input_indices, minimum)
        input_indices = np.minimum(input_indices, maximum)
        return input_indices

    def _from_indices(self, input_indices):
        """Converts lookup table indices to an input vector"""
        input_vector = np.array(input_indices)
        input_vector = np.multiply(input_vector, self._input_span)
        input_vector = np.divide(input_vector, self._input_points - 1)
        input_vector = input_vector + self._input_from
        return input_vector

    def _get(self, input_indices):
        """Gets the output vector at a point of the grid."""
        return self._table[tuple(input_indices)]

    def _set(self, input_indices, output_vector):
        """Sets the output vector at a point of the grid."""
        self._table[tuple(input_indices)] = output_vector

    def _iterate_all(self, function, input_indices = None, index = 0):
        """Calls a function on all the points of the grid."""
        if input_indices == None: input_indices = [0] * self._input_size
        if index == self._input_size: function(input_indices)
        else:
            for i in xrange(self._input_points[index]):
                new_input_indices = list(input_indices)
                new_input_indices[index] = i
                self._iterate_all(function,
                        input_indices = new_input_indices,
                        index = index + 1)

    def _iterate_hypercube(self, function, input_indices, index = 0):
        """
        Calls a function on all the corners of a hypercube made of adjacent
        points of the grid. The input indices are for the lowest corner.
        """
        if index == self._input_size: function(input_indices)
        else:
            for i in xrange(2):
                new_input_indices = list(input_indices)
                new_input_indices[index] += i
                self._iterate_hypercube(function,
                        input_indices = new_input_indices,
                        index = index + 1)

    @staticmethod
    def _process_lerp(weights, distances):
        """Combines the linear interpolation weights and distances"""
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
