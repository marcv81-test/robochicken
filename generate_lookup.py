from kinematics import *
from hexapod import *
from lookup import *

scene.range = 5
scene.forward = [1, 0, 0]
scene.up = [0, 0, -1]
arrow(axis = [1, 0, 0], color = color.red)
arrow(axis = [0, 1, 0], color = color.green)
arrow(axis = [0, 0, 1], color = color.blue)

def xxx(input_vector):
    hexapod = Hexapod(algorithm = 'Damped Least Squares')
    hexapod.initialize_draw()
    x = 3.0 * input_vector[0] - 1.5
    y = 3.0 * input_vector[1] - 1.5
    z = 3.6 * input_vector[2] - 1.8
    angle = (tau / 4) * input_vector[3] - (tau / 8)
    for _ in xrange(20):
        hexapod.direct_control(x, y, z, angle)
        hexapod.draw()
    output_vector = np.zeros(18)
    for i in xrange(6):
        for j in xrange(3):
            output_vector[(3 * i) + j] += hexapod._legs[i]._joints_angles[j]
    hexapod.uninitialize_draw()
    return output_vector


lookup_table = LookupTable(4, 18, 7)
lookup_table.populate(lambda x: xxx(x))

lookup_table.save('data')
