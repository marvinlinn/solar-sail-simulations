'''
The following class implements an object type "System", which emulates a collection of celestial objects.

The "SolarSystem" class is a subclass of "System" which automatically insantiates planetary bodies within the solar system.

By Andrew Ji & Marvin Lin
March 2024
'''
import numpy as np
import functions.body as body
import stateCollection.spiceInterface as spice
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class System:

    def __init__(self, name):
        self.name = name
        self.bodies = []
    def __init__(self, name, bodies):
        self.name = name
        self.bodies = bodies

    def add_body(self, body):
        self.bodies.append(body)

    def propogateBodies(self, timestep):
        for celestialBody in self.bodies:
            celestialBody.propogate(timestep)

class SolarSystem(System):
    
    def __init__(self, name):
        self.name = name
        self.size = 10000
        self.timeStep = 0

        timeObj = spice.Time(1, 1, 2000, 1000)

        self.bodies = []

        SUN = body.CelestialBody("sun", '10', self, timeObj, 1.9891E+30, color='yellow')
        MERCURY = body.CelestialBody("mercury", '1', self, timeObj, 3.285E+23, color='grey')
        VENUS = body.CelestialBody("venus", '2', self, timeObj, 4.867E+24, color='orange')
        EARTH = body.CelestialBody("earth", '3', self, timeObj, 5.97219E+24, color='green')
        MARS = body.CelestialBody("mars", '4', self, timeObj, 6.42E+23, color='red')
        '''
        JUPITER = body.CelestialBody("jupiter", '5', self, timeObj, 1.898E+27, color='tan')
        SATURN = body.CelestialBody("saturn", '6', self, timeObj, 5.683000E+26, color='brown')
        URANUS = body.CelestialBody("uranus", '7', self, timeObj, 8.681E+25, color='royalblue')
        NEPTUNE = body.CelestialBody("neptune", '8', self, timeObj, 1.024E+26, color='mediumblue')
        PLUTO = body.CelestialBody("pluto", '9', self, timeObj, 1.30900E+22, color='blue')
        '''
        

    def animateBodies(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')

        # Setting the axes properties
        
        plt.title("Solar Sytem Animation from Jan 1, 2000 for 1000 days")
        def update(frame, bodies = self.bodies):
            self.ax.clear()
            self.ax.set_xlim3d([-3E+8, 3E+8])
            self.ax.set_xlabel('X')

            self.ax.set_ylim3d([-3E+8, 3E+8])
            self.ax.set_ylabel('Y')

            self.ax.set_zlim3d([-3E+8, 3E+8])
            self.ax.set_zlabel('Z')
            for body in bodies:
                body.draw(frame*10)

        numframes = int(4800/10)

        ani = animation.FuncAnimation(self.fig, update, numframes,interval=100/numframes, blit=False)
        #ani.save('solarsystem.gif')
        plt.show()