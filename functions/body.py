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
    def __init__(self, name, position, velocity, acceleration, mass, opacity=1, 
                 path_style='past', trail_length=5e3, show_traj=False):
        self.name = name
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.mass = mass
        self.locations = np.array([position])

        # plot style
        self.opacity = opacity
        self.path_style = path_style
        self.trail_length = trail_length
        self.show_traj = show_traj

class CelestialBody(Body):
    min_display_size = 10
    display_log_base = 10
    def __init__(self, name, spkid, system, timeObj, mass, color='black', 
                 acceleration=None, opacity=1, path_style='past', 
                 trail_length=5e3, show_traj=False):
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
        super().__init__(name, position, velocity, acceleration, mass,
                         opacity, path_style, trail_length, show_traj)
        self.locations = position

    def getPositon(self, currStep):
        return self.position[currStep]
    
    def draw(self, currStep):
        self.system.ax.plot(
            self.locations[0][currStep],
            self.locations[1][currStep],
            self.locations[2][currStep],
            marker="o",
            markersize=self.display_size,
            color=self.color
        )

class SatelliteBody(Body):

    def __init__(self, name, position, velocity, acceleration, mass, opacity=1, 
                 path_style='past', trail_length=5e3, show_traj=False):
        super().__init__(name, position, velocity, acceleration, mass,
                         opacity, path_style, trail_length, show_traj)

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
    def __init__(self, name, position, velocity, acceleration, yawAngle, 
                 coneAngle, pitchAngle=0, rollAngle=0, opacity=.5, 
                 path_style='past', trail_length=5e2, show_traj=False):
        self.mass = 0.01 #10 gram mass
        self.sailArea = 1 #1 sq meter sail area
        
        self.yawAngle = yawAngle #degrees relative to the velocity vector & rollAngle
        self.pitchAngle = pitchAngle #degrees relative to the velocity vector & rollAngle
        self.rollAngle = rollAngle #degrees, 0 means in the same plane as the orbit of the planets
        
        self.coneAngle = coneAngle # function mapping (t,s) -> angle
        self.currStep = 0 # current step in the trajectory

        super().__init__(name, position, velocity, acceleration, self.mass,
                         opacity, path_style, trail_length, show_traj)
    
    def propagate(self, timestep, currstep, planetarysys):
        self.determineSolarAccel(currstep, planetarysys, coneAngle)
        super().propagate(timestep)
    
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

    #TODO: fix traj
    # def getCurrentConeAngle(self,t):
    #     if self.currStep + 1 >= len(self.trajectory[0]):
    #         return self.trajectory[1][self.currStep]
    #     elif self.trajectory[0][self.currStep + 1] < t: #find what step it is on
    #         step = self.currStep + 1
    #         print(step)
    #         while step < len(self.trajectory[0]) and self.trajectory[0][step] < t:
    #             step += 1
    #         self.currStep = step - 1
    #     return self.trajectory[1][self.currStep]


    
