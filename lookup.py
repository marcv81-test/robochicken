import numpy as np

class LookupTable:
    """Input vector components must be between 0.0 and 1.0"""

    def __init__(self, input_vector_size, output_vector_size, segments_count):
        self._input_vector_size = int(input_vector_size)
        self._output_vector_size = int(output_vector_size)
        self._segments_count = int(segments_count)
        self._indices_count = self._segments_count**self._input_vector_size
        self._table = np.zeros([self._indices_count, output_vector_size])

    def _input_vector_to_index(self, input_vector):
        scaled_vector = np.round((self._segments_count - 1) * np.array(input_vector))
        index = 0
        for i in xrange(self._input_vector_size):
            index *= self._segments_count
            index += int(scaled_vector[i])
        return index

    def _index_to_input_vector(self, index):
        scaled_vector = np.zeros(self._input_vector_size)
        for i in xrange(self._input_vector_size - 1, -1, -1):
            scaled_vector[i] = float(index % self._segments_count)
            index /= self._segments_count
        return scaled_vector / (self._segments_count - 1)

    def populate(self, function):
        for i in xrange(self._indices_count):
            print str(float(i) / self._indices_count * 100) + '%'
            self._table[i] = function(self._index_to_input_vector(i))

    def evaluate(self, input_vector):
        index = self._input_vector_to_index(input_vector)
        return self._table[index]

    def save(self, filename):
        np.save(filename, np.array(self._table))

    def load(self, filename):
        self._table = np.load(filename)
