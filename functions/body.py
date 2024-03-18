'''
The following class implements an object type "Body", which emulates any celestial object

By Andrew Ji & Marvin Lin
March 2024
'''
import numpy as np
import math
import stateCollection.spiceInterface as spice
import functions.system as system
import functions.utils as utils

class Body:
    def __init__(self, name, position, velocity, acceleration, mass):
        self.name = name
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.mass = mass
        self.locations = np.array([position])

class CelestialBody(Body):
    min_display_size = 10
    display_log_base = 10
    def __init__(self, name, spkid, system, timeObj, mass, color='black', acceleration=None):
        self.spkid = spkid
        self.system = system
        self.color = color
        self.mass = mass
        self.timeStep = 5 #in hours
        position, velocity = spice.requestData(spkid, timeObj, self.timeStep)
        self.display_size = max(
            math.log(self.mass, self.display_log_base)/2,
            self.min_display_size,
        )
        self.system.add_body(self)
        super().__init__(name, position, velocity, acceleration, mass)
        self.locations = position

    def getPositon(self, currStep):
        return self.position[currStep]
    
    def draw(self, currStep):
        self.system.ax.plot(
            self.position[0][currStep],
            self.position[1][currStep],
            self.position[2][currStep],
            marker="o",
            markersize=self.display_size,
            color=self.color
        )

class SatelliteBody(Body):

    def __init__(self, name, position, velocity, acceleration, mass):
        super().__init__(name, position, velocity, acceleration, mass)

    # Increments the simulation by the designated timestep
    def propagate(self, timestep):
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

class SolarSail(SatelliteBody):

    #includes the angles of the solar sail in order to determine solar sail acceleration
    def __init__(self, name, position, velocity, acceleration, yawAngle, pitchAngle=0, rollAngle=0):
        self.mass = 0.01 #10 gram mass
        self.sailArea = 1 #1 sq meter sail area
        self.yawAngle = yawAngle #degrees relative to the velocity vector & rollAngle
        self.pitchAngle = pitchAngle #degrees relative to the velocity vector & rollAngle
        self.rollAngle = rollAngle #degrees, 0 means in the same plane as the orbit of the planets 
        super().__init__(name, position, velocity, acceleration, self.mass)
    
    def propagate(self, timestep, currstep, planetarysys):
        self.determineSolarAccel(currstep, planetarysys)
        super().propogate(timestep)
    
    #currently only considering the yaw Angle
    def determineSolarAccel(self, currstep, planetarysys):
        solarRadiationVector = self.postion - planetarysys.SUN.getPositon(currstep) # distance vector between current sun location and solarsail location
        solarRadiationVectorMag = np.linalg.norm(solarRadiationVector)
        
        sailNormal = (utils.rotate(self.velocity, "yaw", self.yawAngle))/(np.linalg.norm(self.velocity)) # unit vector
        alpha = np.arccos((np.dot(solarRadiationVector, sailNormal))/(solarRadiationVectorMag * 1)) # incident angle of the solar radiation on the sail
        
        radiationAccel = utils.P0 * (np.cos(alpha) ** 2) * ((utils.AU/solarRadiationVector)**2) * self.sailArea / self.mass # acceleration due to the radiation present
        self.acceleration = self.acceleration + (radiationAccel * sailNormal)
     
    def setYaw(self, angle):
        self.yawAngle = angle

    