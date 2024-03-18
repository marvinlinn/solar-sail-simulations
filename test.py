import stateCollection.horizonAPI as horizon
import functions.system as system
import functions.body as body
import functions.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ

solar_system = system.SolarSystem("solar")
#solar_system.animateBodies()
AU = 1.496e11 /1e3  # astronomical unit in km, distance from sun to earth

def testCone(t):
    if True:
        return -0.6

span = np.arange(0, 3.2e7, 1e5)

sail = body.SolarSail("sail1", 0,0,0,0, np.array([[0,1.6e7,3.2e7],[0.6, -0.6, 0]]))
newsailLocs = integ.odeint(utils.npODESailGenerator, np.array([AU, 0, 0, 0, 30, 0]),span, args=(sail,))
saillocs = integ.solve_ivp(utils.npSailGenerator, [0, 3.2e7], np.array([AU, 0, 0, 0, 30, 0]), rtol=1e-8, args=[sail])
sail.locations = np.array(saillocs.y)


print(np.array(newsailLocs))
#print(np.linalg.norm(np.array([1,1,1,1])))
#plt.plot(saillocs.y[0], saillocs.y[1])
#print(len(sail.locations[0]))
#utils.animatebodies(np.array([sail]))
#plt.show()
#utils.animatebodies(np.append(solar_system.bodies), 15)


'''
testing below
'''
trajectory = np.array([[0, 5, 10, 16, 20],[1, 2, 3, 4, 5]])

currStep = 0

def getCurrentConeAngle(trajectory,t,currStep):
    if trajectory[0][currStep + 1] < t and t < trajectory[0][-1]: #find what step it is on
        step = currStep + 1
        while trajectory[0][step] < t:
            step += 1
        currStep = step - 1
    print (currStep)
    return trajectory[1][currStep]

sailtest = body.SolarSail("test", np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]), 0, trajectory)
'''
print(getCurrentConeAngle(trajectory, 1, currStep), currStep)
print(getCurrentConeAngle(trajectory, 2, currStep), currStep)
print(getCurrentConeAngle(trajectory, 11, currStep), currStep)
print(getCurrentConeAngle(trajectory, 1, currStep), currStep)
print(getCurrentConeAngle(trajectory, 17, currStep), currStep)

print(sailtest.getCurrentConeAngle(1), sailtest.currStep)
print(sailtest.getCurrentConeAngle(2), sailtest.currStep)
print(sailtest.getCurrentConeAngle(11), sailtest.currStep)
print(sailtest.getCurrentConeAngle(1), sailtest.currStep)
print(sailtest.getCurrentConeAngle(17), sailtest.currStep)
'''