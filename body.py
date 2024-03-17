import numpy as np

class Body:

    def __init__(self, name, position, velocity, acceleration, mass):
        self.name = name
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.mass = mass
        self.locations = np.array([position])

    #moves the simulation forward!
    def propogate(self, timestep):
        newVelocity = self.velocity + self.acceleration * timestep
        newPosition = self.position + ((self.velocity + newVelocity) * timestep)/2
        self.velocity = newVelocity
        self.position = newPosition
        self.locations = np.append(self.locations, [self.position], axis=0)
    
    def displayState(self):
        print(self.name, "is currently positioned at:", self.position, "with a velocity of:", self.velocity, "and a acceleration of:", self.acceleration)

    #clears alllll locations INCLUDING INITIAL
    def clearLocs(self):
        self.locations = np.array([])


    