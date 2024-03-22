import stateCollection.horizonAPI as horizon
import functions.system as system
import functions.body as body
import functions.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
import stateCollection.spiceInterface as spice

#solar_system = system.SolarSystem("solar")
#solar_system.animateBodies()
AU = 1.496e11 /1e3  # astronomical unit in km, distance from sun to earth

def testCone(t):
    if True:
        return -0.6

span = np.arange(0, 3.2e7, 1e5)

#sail = body.SolarSail("sail1", 0,0,0,0, np.array([[0,1.6e7,3.2e7],[0.6, -0.6, 0]]))
#sail2 = body.SolarSail("sail2", 0,0,0,0, np.array([[0,1.6e7,3.2e7],[0.6, 0.6, 0.6]]))
#sail3 = body.SolarSail("sail3", 0,0,0,0, np.array([[0,1.6e7,3.2e7],[-0.6, -0.6, -0.6]]))
#sail2ivp = body.SolarSail("sail2ivp", 0,0,0,0, np.array([[0,1.6e7,3.2e7],[0.6, -0.6, 0.6]]))

#newsailLocs = integ.solve_ivp(utils.npSailODE, [0, 3.2e7], np.array([AU, 0, 0, 0, 30, 0]), rtol=1e-8,t_eval=span, args=[sail])
#newsailLocs2 = integ.solve_ivp(utils.npSailODE,[0, 3.2e7], np.array([AU, 0, 0, 0, 30, 0]), rtol=1e-8,t_eval=span, args=[sail2])
#newsailLocs3 = integ.solve_ivp(utils.npSailODE, [0, 3.2e7], np.array([AU, 0, 0, 0, 30, 0]),rtol=1e-8,t_eval=span, args=[sail3])


#ivpsaillocs = integ.solve_ivp(utils.npSailGenerator, [0, 3.2e7], np.array([AU, 0, 0, 0, 30, 0]), rtol=1e-8, args=[sail2ivp], t_eval=span)
#sail2ivp.locations = ivpsaillocs.y[:3, :]
#print (sail2ivp.locations)
#sail.locations = newsailLocs.y[:3,:]
#sail2.locations = newsailLocs2.y[:3, :]
#sail3.locations = newsailLocs3.y[:3, :]

#print(np.array(newsailLocs).T[:3,:])
#print(np.linalg.norm(np.array([1,1,1,1])))
#plt.plot(saillocs.y[0], saillocs.y[1])
#print(len(sail.locations[0]))
#plt.show()
#utils.animatebodies(np.append(solar_system.bodies), 15)

#sail1 = utils.sailGenerator("sail1", np.array([AU,0,0]), np.array([0,30,0]), np.array([[0, 2.6e6, 5.2e6, 7.8e6, 1.04e7,1.3e7,1.56e7, 1.82e7],[0.6, -0.6, 0]]), [0, 3.2e7], 1e5)
#sail2 = utils.sailGenerator("sail2", np.array([AU,0,0]), np.array([0,30,0]), np.array([[0,1.6e7,3.2e7],[-0.6, -0.6, 0]]), [0, 3.2e7], 1e5)
#sail3 = utils.sailGenerator("sail3", np.array([AU,0,0]), np.array([0,30,0]), np.array([[0,1.6e7,3.2e7],[0.6, 0.6, 0]]), [0, 3.2e7], 1e5)
#utils.animatebodies(np.array([sail1, sail2, sail3]))


"""
for n in range(1000):
    sailorientations = (np.random.random_sample((8,)) * 1.2) - 0.6
    print(sailorientations)
    newSail = utils.sailGenerator(("sail"+ str(n)), np.array([AU,0,0]), np.array([0,30,0]), 
                                  np.array([[0, 2.6e6, 5.2e6, 7.8e6, 1.04e7,1.3e7, 1.56e7, 1.82e7], sailorientations]), [0, 1.82e7], 1e3)
    sailset = np.append(sailset, newSail)  

utils.animatebodies(sailset, 50)
"""

'''
Deliverable 

# 6 different points varied between 0.6, 0, -0.6
testTime = spice.Time(1, 1, 2000, 720) # 2 years
timeSeconds = testTime.lengthSeconds

#planet generation
sys = system.SolarSystem("720 day sys", testTime)
sysbds = sys.bodies
numSteps = len(sysbds[0].locations[0])

#trajectories generation
trajs = np.array([[0,0,0,0,0,0,0]])
possibleOrients = np.array([-0.6,0,0.6])
timeInt = np.array([0, timeSeconds/6, (2*timeSeconds)/6, (3*timeSeconds)/6, (4*timeSeconds)/6, (5*timeSeconds)/6, timeSeconds])

for a in possibleOrients:
    for b in possibleOrients:
        for c in possibleOrients:
            for d in possibleOrients:
                for e in possibleOrients:
                    for f in possibleOrients:
                        trajs = np.append(trajs, [[a,b,c,d,e,f,0]], axis=0)

#sail generation
#init conditions -> earth position, velocity must also be vectorized correctly
initPos = sysbds[3].locations.T[0]
initVelVec = (sysbds[3].locations.T[1]-sysbds[3].locations.T[0])/np.linalg.norm(sysbds[3].locations.T[1]-sysbds[3].locations.T[0]) #velocity vector via linearization between point 0 and 1
initVel = initVelVec * 30
sailset = np.array([])

for n in range(len(trajs)):
    newSail = utils.sailGenerator(("sail"+ str(n)), initPos, initVel, 
                                  np.array([timeInt, trajs[n]]), [0, timeSeconds], numSteps)
    sailset = np.append(sailset, newSail)  

utils.animatebodies(sailset, 50)

#utils.animatebodies(sysbds)                        
print(np.linalg.norm(initPos))

print(initPos)
print(initVelVec)
'''

'''
less iterations hopefully less messy
'''

# 6 different points varied between 0.6, 0, -0.6
testTime = spice.Time(1, 1, 2000, 720) # 2 years
timeSeconds = testTime.lengthSeconds

#planet generation
sys = system.SolarSystem("720 day sys", testTime)
sysbds = sys.bodies
numSteps = len(sysbds[0].locations[0])

#trajectories generation
trajs = np.array([[0,0,0,0]])
possibleOrients = np.array([-0.6,0,0.6])
timeInt = np.array([0, timeSeconds/3, (2*timeSeconds)/3, timeSeconds])

for a in possibleOrients:
    for b in possibleOrients:
        for c in possibleOrients:
            trajs = np.append(trajs, [[a,b,c,0]], axis=0)

#sail generation
#init conditions -> earth position, velocity must also be vectorized correctly
initPos = sysbds[3].locations.T[0]
initVelVec = (sysbds[3].locations.T[1]-sysbds[3].locations.T[0])/np.linalg.norm(sysbds[3].locations.T[1]-sysbds[3].locations.T[0]) #velocity vector via linearization between point 0 and 1
initVel = initVelVec * 30
sailset = np.array([])

for n in range(len(trajs)):
    newSail = utils.sailGenerator(("sail"+ str(n)), initPos, initVel, 
                                  np.array([timeInt, trajs[n]]), [0, timeSeconds], numSteps)
    sailset = np.append(sailset, newSail)  

utils.animatebodies(np.append(sailset, sysbds), 10)

print(sysbds[0])
print(isinstance(sysbds[0], body.CelestialBody))
print(isinstance(sailset[0], body.CelestialBody))

'''
testing below

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
