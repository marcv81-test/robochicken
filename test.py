import joystick

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
    for _ in xrange(3):
        hexapod.direct_control(x, y, z, angle)
        hexapod.draw()
    output_vector = np.zeros(18)
    for i in xrange(6):
        for j in xrange(3):
            output_vector[(3 * i) + j] += hexapod._legs[i]._joints_angles[j]
    hexapod.uninitialize_draw()
    return output_vector

class LookupHexapod(Hexapod):

    def __init__(self, lookup_table):
        Hexapod.__init__(self)
        self._lookup_table = lookup_table

    def direct_control(self, x, y, z, angle):
        """Control the legs directly"""
        input_vector = np.zeros(4)
        input_vector[0] = x / 3.0 + 0.5
        input_vector[1] = y / 3.0 + 0.5
        input_vector[2] = z / 3.6 + 0.5
        input_vector[3] = angle / (tau / 4) + 0.5

        output_vector = self._lookup_table.evaluate(input_vector)
        for i in range(self._legs_count):
            for j in xrange(3):
                self._legs[i]._joints_angles[j] = output_vector[(3 * i) + j]

lookup_table = LookupTable(4, 18, 7)
lookup_table.populate(lambda x: xxx(x))

lookup_hexapod = LookupHexapod(lookup_table)
lookup_hexapod.initialize_draw()

joystick = joystick.Joystick("/dev/input/js1")

t = 0.0
dt = 0.04
while True:
    rate(25)
    t += dt
    joystick.update()
    x = -1.5 * joystick.axis_states["ry"]
    y = 1.5 * joystick.axis_states["rx"]
    z = 1.8 * (joystick.axis_states["y"])
    angle = tau / 8 * (joystick.axis_states["x"])
    lookup_hexapod.direct_control(x, y, z, angle)
    lookup_hexapod.draw()