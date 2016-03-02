import math
import pylab

import numpy as np

from lookup2 import *

def test_function(input_vector):
    x = input_vector[0]
    y = input_vector[1]
    return [math.cos(x) + math.sin(y)]

lookup_table = LookupTable(2, 1, 11)
lookup_table.populate(test_function)

x = np.linspace(0.01, 9.99, 100)
y = np.linspace(0.01, 9.99, 100)

def prepare_plot(function):
    color = pylab.zeros([len(x), len(y)])
    for i in xrange(len(x)):
        for j in xrange(len(y)):
            input_vector = [x[i], y[j]]
            output_vector = function(input_vector)
            color[i, j] = output_vector[0]
    return color

# Plot original function
f = prepare_plot(lambda input_vector: test_function(input_vector))
pylab.subplot(1, 3, 1)
pylab.pcolor(x, y, f)

# Plot nearest interpolation
f_nearest = prepare_plot(lambda input_vector: lookup_table.get_nearest(input_vector))
pylab.subplot(1, 3, 2)
pylab.pcolor(x, y, f_nearest)

# Plot linear interpolation
f_lerp = prepare_plot(lambda input_vector: lookup_table.get_lerp(input_vector))
pylab.subplot(1, 3, 3)
pylab.pcolor(x, y, f_lerp)

pylab.show()