import stateCollection.horizonAPI as horizon
import functions.system as system
import functions.body as body
import functions.utils as utils
import numpy as np

solar_system = system.SolarSystem("solar")
solar_system.animateBodies()
utils.animatebodies(solar_system.bodies, 15)

vect = np.array([0,0,1])
print(vect)
newvect = utils.rotate(vect, "pitch", 90)
print(newvect)
print(utils.rotate(newvect, "yaw", 90))