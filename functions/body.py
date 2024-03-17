'''
The following class implements an object type "Body", which emulates any celestial object

By Andrew Ji & Marvin Lin
March 2024
'''
import numpy as np

class Body:
    def __init__(self, name, position, velocity, acceleration, mass):
        self.name = name
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.mass = mass
        self.locations = np.array([position])

class CelestialBody(Body):
    def __init__(self, name, spkid, position, mass, velocity=None, acceleration=None):
        self.spkid = spkid
        super().__init__(name, position, velocity, acceleration, mass)

class SatelliteBody(Body):

    def __init__(self, name, position, velocity, acceleration, mass):
        super().__init__(name, position, velocity, acceleration, mass)

    # Increments the simulation by the designated timestep
    def propogate(self, timestep):
        newVelocity = self.velocity + self.acceleration * timestep
        newPosition = self.position + ((self.velocity + newVelocity) * timestep)/2
        self.velocity = newVelocity
        self.position = newPosition
        self.locations = np.append(self.locations, [self.position], axis=0)
    
    # Prints out the current state of the object of interest
    def displayState(self):
        print(self.name, "is currently positioned at:", self.position, "with a velocity of:", self.velocity, "and a acceleration of:", self.acceleration)

    # Clears all previous simulated locations of the body
    def clearLocs(self):
        self.locations = np.array([])


    