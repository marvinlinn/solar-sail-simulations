import stateCollection.horizonAPI as horizon
import functions.system as system
import functions.body as body
import functions.utils as utils
import numpy as np
from scipy.integrate import solve_ivp

solar_system = system.SolarSystem("solar")
#solar_system.animateBodies()

print(solar_system.bodies[3].locations.T[0])

sail = body.SolarSail("sail1", 0,0,0,0)
sail.locations = solve_ivp(utils.simpleSailGenerator, [0, 3.2e7], [1.496e11, 0, 0, 30], rtol=1e-8, args=[-0.6])[:2, :]

print(sail.locations)
#utils.animatebodies(np.append(solar_system.bodies), 15)

vect = np.array([0,0,1])
print(vect)
newvect = utils.rotate(vect, "pitch", 90)
print(newvect)
print(utils.rotate(newvect, "yaw", 90))