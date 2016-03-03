import numpy as np

class LookupTable:

    def __init__(self, input_specifications, output_dimension, epsilon = 1e-3):
        # Input specifications
        self._input_dimension = len(input_specifications)
        self._input_points = np.array([int(item['points']) for item in input_specifications])
        self._input_from = np.array([float(item['from']) for item in input_specifications])
        self._input_to = np.array([float(item['to']) for item in input_specifications])
        # Output specifications
        self._output_dimension = int(output_dimension)
        # Epsilon
        self._epsilon = float(epsilon)
        # Internal table
        shape = list(self._input_points)
        shape.append(self._output_dimension)
        self._table = np.zeros(shape)

    def _get(self, input_vector):
        return self._table[tuple(input_vector)]

    def _set(self, input_vector, output_vector):
        self._table[tuple(input_vector)] = output_vector

    def _from_world(self, world_input_vector):
        """Convert input vector from world to internal coordinates"""
        input_vector = np.array(world_input_vector)
        input_vector = input_vector - self._input_from
        input_vector = np.multiply(input_vector, self._input_points - 1)
        input_vector = np.divide(input_vector, self._input_to - self._input_from)
        input_vector = np.maximum(input_vector, self._epsilon)
        input_vector = np.minimum(input_vector, self._input_points - 1 - self._epsilon)
        return input_vector

    def _to_world(self, input_vector):
        """Convert input vector from internal to world coordinates"""
        world_input_vector = np.array(input_vector)
        world_input_vector = np.multiply(world_input_vector, self._input_to - self._input_from)
        world_input_vector = np.divide(world_input_vector, self._input_points - 1)
        world_input_vector = world_input_vector + self._input_from
        return world_input_vector

    def _iterate_all(self, function, input_vector = None, index = 0):
        if input_vector == None: input_vector = [0] * self._input_dimension
        if index == self._input_dimension: function(input_vector)
        else:
            for i in xrange(self._input_points[index]):
                new_input_vector = list(input_vector)
                new_input_vector[index] = i
                self._iterate_all(function, new_input_vector, index + 1)

    def _iterate_hypercube(self, function, input_vector, index = 0):
        if index == self._input_dimension: function(input_vector)
        else:
            for i in xrange(2):
                new_input_vector = list(input_vector)
                new_input_vector[index] += i
                self._iterate_hypercube(function, new_input_vector, index + 1)

    def populate(self, function):
        self._iterate_all(lambda input_vector:
                self._set(input_vector, function(self._to_world(input_vector))))

    def get_nearest(self, world_input_vector):
        input_vector = self._from_world(world_input_vector)
        input_vector = np.round(input_vector)
        return self._table[tuple(input_vector)]

    def get_lerp(self, world_input_vector):
        input_vector = self._from_world(world_input_vector)
        first_corner = np.floor(input_vector)
        weights = list()
        self._iterate_hypercube(
                function = lambda input_vector: weights.append(self._get(input_vector)),
                input_vector = first_corner)
        distances = list(reversed(input_vector - first_corner))
        return self._lerp_sum(weights, distances)

    def _lerp_sum(self, weights, distances):
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
            return self._lerp_sum(new_weights, new_distances)

    def save(self, filename):
        np.save(filename, np.array(self._table))

    def load(self, filename):
        self._table = np.load(filename)
