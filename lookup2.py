import numpy as np

class LookupTable:

    def __init__(self, input_dimension, output_dimension, resolution):
        self._input_dimension = int(input_dimension)
        self._output_dimension = int(output_dimension)
        self._resolution = int(resolution)
        # Non-trivial attributes
        shape = [self._resolution] * self._input_dimension
        shape.append(output_dimension)
        self._table = np.zeros(shape)

    def _get(self, input_vector):
        return self._table[tuple(input_vector)]

    def _set(self, input_vector, output_vector):
        self._table[tuple(input_vector)] = output_vector

    def _iterate_all(self, function, input_vector = None, index = 0):
        if input_vector == None: input_vector = [0] * self._input_dimension
        if index == self._input_dimension: function(input_vector)
        else:
            for i in xrange(self._resolution):
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
        self._iterate_all(lambda input_vector: self._set(input_vector, function(input_vector)))

    def get_nearest(self, input_vector):
        input_vector = np.round(input_vector)
        return self._table[tuple(input_vector)]

    def get_lerp(self, input_vector):
        first_corner = np.floor(input_vector)
        weights = list()
        self._iterate_hypercube(
                function = lambda input_vector: weights.append(self._get(input_vector)),
                input_vector = first_corner)
        distances = input_vector - first_corner
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
