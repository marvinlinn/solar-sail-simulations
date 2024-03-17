'''
The following class implements an object type "System", which emulates a collection of celestial objects.

The "SolarSystem" class is a subclass of "System" which automatically insantiates planetary bodies within the solar system.

By Andrew Ji & Marvin Lin
March 2024
'''
import numpy as np
import body as body

class System:

    def __init__(self, name, bodies):
        self.name = name
        self.bodies = bodies

    def add_body(self, body):
        self.bodies.append(body)

    def propogateBodies(self, timestep):
        for celestialBody in self.bodies:
            celestialBody.propogate(timestep)

class SolarSystem(System):
    
    SUN = body.Body("sun", np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]), 1.9891E+30)
    MERCURY = body.Body("sun", np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]), 3.285E+23)
    VENUS = body.Body("sun", np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]), 4.867E+24)
    EARTH = body.Body("earth", np.array([384400000,0,0]), np.array([0,0,0]), np.array([0,0,0]), 5.97219E+24)
    MARS = body.Body("sun", np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]), 6.42E+23)
    JUPITER = body.Body("sun", np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]), 1.898E+27)
    SATURN = body.Body("sun", np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]), 5.683000E+26)
    URANUS = body.Body("sun", np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]), 8.681E+25)
    NEPTUNE = body.Body("sun", np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]), 1.024E+26)
    PLUTO = body.Body("sun", np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]), 1.30900E+22)

    SOLAR_BODIES = np.array([SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO])

    def __init__(self, name):
        self.name = name
        self.bodies = SolarSystem.SOLAR_BODIES