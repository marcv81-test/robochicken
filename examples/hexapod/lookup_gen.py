from robotics.kinematics import *
from robotics.lookup import *
from examples.hexapod.hexapod import *

scene.range = 5
scene.forward = [1, 0, 0]
scene.up = [0, 0, -1]
arrow(axis = [1, 0, 0], color = color.red)
arrow(axis = [0, 1, 0], color = color.green)
arrow(axis = [0, 0, 1], color = color.blue)

def xxx(input_vector):
    hexapod = Hexapod(algorithm = 'Damped Least Squares')
    hexapod.initialize_draw()
    x = input_vector[0]
    y = input_vector[1]
    z = input_vector[2]
    angle = input_vector[3]
    for _ in xrange(5):
        hexapod.direct_control(x, y, z, angle)
        hexapod.draw()
    output_vector = np.zeros(18)
    for i in xrange(6):
        for j in xrange(3):
            output_vector[(3 * i) + j] += hexapod._legs[i]._joints_angles[j]
    hexapod.uninitialize_draw()
    return output_vector

lookup_table = LookupTable(
        input_specifications = [
            {'from': -1.5, 'to': 1.5, 'points': 5},
            {'from': -1.5, 'to': 1.5, 'points': 5},
            {'from': -1.5, 'to': 1.5, 'points': 5},
            {'from': -tau / 8, 'to': tau / 8, 'points': 5}],
        output_size = 18)
lookup_table.populate(lambda x: xxx(x))

lookup_table.save('data')
