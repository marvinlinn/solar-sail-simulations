'''
The following class implements an object type "System", which emulates a collection of celestial objects.

The "SolarSystem" class is a subclass of "System" which automatically insantiates planetary bodies within the solar system.

By Andrew Ji & Marvin Lin
March 2024
'''
import numpy as np
import functions.body as body
import stateCollection.spiceInterface as spice

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
    
    SUN = body.CelestialBody("sun", '10', spice.computeBody('10'), 1.9891E+30)
    MERCURY = body.CelestialBody("mercury", '1', spice.computeBody('1'), 3.285E+23)
    VENUS = body.CelestialBody("venus", '2', spice.computeBody('2'), 4.867E+24)
    EARTH = body.CelestialBody("earth", '3', spice.computeBody('3'), 5.97219E+24)
    MARS = body.CelestialBody("mars", '4', spice.computeBody('4'), 6.42E+23)
    JUPITER = body.CelestialBody("jupiter", '5', spice.computeBody('5'), 1.898E+27)
    SATURN = body.CelestialBody("saturn", '6', spice.computeBody('6'), 5.683000E+26)
    URANUS = body.CelestialBody("uranus", '7', spice.computeBody('7'), 8.681E+25)
    NEPTUNE = body.CelestialBody("neptune", '8', spice.computeBody('8'), 1.024E+26)
    PLUTO = body.CelestialBody("pluto", '9', spice.computeBody('9'), 1.30900E+22)

    SOLAR_BODIES = np.array([SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, PLUTO])

    def __init__(self, name):
        self.name = name
        self.bodies = SolarSystem.SOLAR_BODIES