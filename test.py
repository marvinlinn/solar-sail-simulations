from time import time
import stateCollection.horizonAPI as horizon
import functions.system as system
import functions.body as body
import functions.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
import stateCollection.spiceInterface as spice
import functions.pretrainingfns as pretrain
from multiprocessing import Pool
import os

#solar_system = system.SolarSystem("solar")
#solar_system.animateBodies()
AU = 1.496e11 /1e3  # astronomical unit in km, distance from sun to earth

def testCone(t):
    if True:
        return -0.6

span = np.arange(0, 3.2e7, 1e5)



"""
for n in range(1000):
    sailorientations = (np.random.random_sample((8,)) * 1.2) - 0.6
    print(sailorientations)
    newSail = utils.sailGenerator(("sail"+ str(n)), np.array([AU,0,0]), np.array([0,30,0]), 
                                  np.array([[0, 2.6e6, 5.2e6, 7.8e6, 1.04e7,1.3e7, 1.56e7, 1.82e7], sailorientations]), [0, 1.82e7], 1e3)
    sailset = np.append(sailset, newSail)  

utils.animatebodies(sailset, 50)
"""

# Deliverable (large scale)

'''
# 6 different points varied between 0.6, 0, -0.6
testTime = spice.Time(1, 1, 2000, 360) # 2 years
timeSeconds = testTime.lengthSeconds

#planet generation
sys = system.SolarSystem("360 day sys", testTime)
sysbds = sys.bodies
numSteps = len(sysbds[0].locations[0])

#trajectories generation

yaws = np.array([[0,0,0,0,0,0,0]])
pitches = np.array([[0.6,0.6,0.6,0.6,0.6,0.6,0.6]])
possibleOrients = np.array([-0.6,0,0.6])
timeInt = np.array([0, timeSeconds/6, (2*timeSeconds)/6, (3*timeSeconds)/6, (4*timeSeconds)/6, (5*timeSeconds)/6, timeSeconds])

for a in possibleOrients:
    for b in possibleOrients:
        for c in possibleOrients:
            for d in possibleOrients:
                for e in possibleOrients:
                    for f in possibleOrients:
                        yaws = np.append(yaws, [[a,b,c,d,e,f,0]], axis=0)
                        pitches = np.append(pitches, [[0.6,0.6,0.6,0.6,0.6,0.6,0.6]], axis=0)

#sail generation
#init conditions -> earth position, velocity must also be vectorized correctly
initPos = sysbds[3].locations.T[0]
initVelVec = (sysbds[3].locations.T[1]-sysbds[3].locations.T[0])/np.linalg.norm(sysbds[3].locations.T[1]-sysbds[3].locations.T[0]) #velocity vector via linearization between point 0 and 1
initVel = initVelVec * 30
sailset = np.array([])

start = time()
for n in range(len(yaws)):
    newSail = utils.sailGenerator(("sail"+ str(n)), initPos, initVel, 
                                  np.array([timeInt, yaws[n], pitches[n]]), [0, timeSeconds], numSteps)
    sailset = np.append(sailset, newSail)  
print(f'{len(yaws)} sail sim takes {time() - start} seconds')

utils.animatebodies(np.append(sailset, sysbds), 5)
#utils.animatebodies(sysbds)                        

#print(np.linalg.norm(initPos))

a = [(1,2), (3,4), (5,6)]
print(a[0], a[0][0])

#print(initPos)
#print(initVelVec)
'''

#Small (light) test
'''

testTime = spice.Time(1, 1, 2000, 360) # 1 years
timeSeconds = testTime.lengthSeconds


#planet generation
sys = system.SolarSystem((testTime.length, "day sys"), testTime)
sysbds = sys.bodies
numSteps = len(sysbds[0].locations[0])


#trajectories generation


pitches = np.array([[0,0,0,0]])
yaws = np.array([[0,0,0,0]])
possibleOrients = np.array([-0.6,0,0.6])
timeInt = np.array([0, timeSeconds/3, (2*timeSeconds)/3, timeSeconds])


for a in possibleOrients:
    for b in possibleOrients:
        for c in possibleOrients:
            pitches = np.append(pitches, [[a,b,c,0]], axis=0)
            yaws = np.append(yaws, [[0.6,0.6,0.6,0.6]], axis=0)


#sail generation
#init conditions -> earth position, velocity must also be vectorized correctly
initPos = sysbds[3].locations.T[0]
initVelVec = (sysbds[3].locations.T[1]-sysbds[3].locations.T[0])/np.linalg.norm(sysbds[3].locations.T[1]-sysbds[3].locations.T[0]) #velocity vector via linearization between point 0 and 1
initVel = initVelVec * 30
sailset = np.array([])


for n in range(len(yaws)):
    newSail = utils.sailGenerator(("sail"+ str(n)), initPos, initVel,
                                  np.array([timeInt, yaws[n], pitches[n]]), [0, timeSeconds], numSteps)
    sailset = np.append(sailset, newSail)  


utils.animatebodies(np.append(sailset, sysbds), 5)
'''

'''
less iterations hopefully less messy
'''
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
yaws = pretrain.permutationGenerator(possibleOrients, 4)
print(yaws[1])
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
                                  np.array([timeInt, yaws[n], trajs[0]]), [0, timeSeconds], numSteps)
    sailset = np.append(sailset, newSail)  

utils.animatebodies(np.append(sailset, sysbds), 10)

print(sysbds[0])
print(isinstance(sysbds[0], body.CelestialBody))
print(isinstance(sailset[0], body.CelestialBody))

'''

#targetbds = system.SolarSystem('targets', timetest).bodies
#sailset, bdys, target = pretrain.packaged2DSim(timetest, [-0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6], 2, targetbds[4])
#print(sailset.shape)
#print(sailset[0].yawAngle.shape)
#print(sailset[0].timeSteps.shape)
#print(sailset[0].locations.shape)
#print(bdys[0].locations.shape)
#utils.animatebodies(np.append(sailset, bdys), tstep=10)
#pretrain.generateBodyCSV(sailset, target, numsails=10, simStartDate='09012000')

#array = pretrain.permutationGenerator([1,2,3], 10)
#print(array)


#fn = pretrain.packagedSimForParallel(1200, [0.6,0,-0.6], 2, 2)
#fn(1,2,2000)
#fn = pretrain.prepackagedWholeSim

#print(utils.mpCone_Angle_Factory([2,4,6,8], [1,2,3,4], 8))
#pretrain.parallelsiming(dates, [0.6,0,-0.6], 3, 3, 1200)


if __name__ == '__main__':
    dates = np.array([[1,1,2000,360], [3,1,2000,360], [6,1,2000,360], [9,1,2000,360], [1,1,2001,360]])
    sailOrientations = [-0.6,-0.3,0,0.3,0.6]
    numSailChanges = 3
    pretrain.multiSailSetGenerator(dates,sailOrientations,numSailChanges)

   


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