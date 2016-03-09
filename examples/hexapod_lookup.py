from robotics.joystick import *
from robotics.kinematics import *
from robotics.hexapod import *
from robotics.lookup import *

class LookupHexapod(Hexapod):

    def __init__(self, lookup_table):
        Hexapod.__init__(self)
        self._lookup_table = lookup_table

    def direct_control(self, x, y, z, angle):
        """Control the legs directly"""
        input_vector = np.zeros(4)
        input_vector[0] = x * 1.5
        input_vector[1] = y * 1.5
        input_vector[2] = z * 1.5
        input_vector[3] = angle * (tau / 4)

        output_vector = self._lookup_table.get_lerp(input_vector)
        for i in range(self._legs_count):
            for j in xrange(3):
                self._legs[i]._joints_angles[j] = output_vector[(3 * i) + j]

scene.range = 5
scene.forward = [1, 0, 0]
scene.up = [0, 0, -1]
arrow(axis = [1, 0, 0], color = color.red)
arrow(axis = [0, 1, 0], color = color.green)
arrow(axis = [0, 0, 1], color = color.blue)

lookup_table = LookupTable(
        input_specifications = [
            {'from': -1.5, 'to': 1.5, 'points': 5},
            {'from': -1.5, 'to': 1.5, 'points': 5},
            {'from': -1.5, 'to': 1.5, 'points': 5},
            {'from': -tau / 8, 'to': tau / 8, 'points': 5}],
        output_size = 18)
lookup_table.load('data.npy')

lookup_hexapod = LookupHexapod(lookup_table)
lookup_hexapod.initialize_draw()

joystick = Joystick("/dev/input/js1")

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
