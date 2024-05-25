import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import functions.body as body
import functions.system as system
import scipy.integrate as integ
import stateCollection.spiceInterface as spice
import utils

#Pretraining Tools

def generateBodyCSV(sailSet, filename="sails_traj_data"):

    return

#for n different sail orientations and m different points to change sail orientation, n^m sails are generated
def packaged2DSim(simTime, sailorientations, numsailchanges):
    timeSeconds = simTime.lengthSeconds

    #planet generation
    sysname = str(simTime.length) + " day sys"
    sys = system.SolarSystem(sysname, simTime)
    sysbds = sys.bodies
    numSteps = len(sysbds[0].locations[0])


    #trajectories generation
    pitches = np.zeros(numsailchanges+1)
    yaws = permutationGenerator(sailorientations, numsailchanges+1)
    timeInt = np.zeros(numsailchanges+1)
    for n in range(numsailchanges+1):
        timeInt[n] = n * timeSeconds
    print(yaws.shape)

    #sail generation
    #init conditions -> earth position, velocity must also be vectorized correctly
    initPos = sysbds[3].locations.T[0]
    initVelVec = (sysbds[3].locations.T[1]-sysbds[3].locations.T[0])/np.linalg.norm(sysbds[3].locations.T[1]-sysbds[3].locations.T[0]) #velocity vector via linearization between point 0 and 1
    initVel = initVelVec * 30
    sailset = np.array([])


    for n in range(len(yaws)):
        newSail = utils.sailGenerator(("sail"+ str(n)), initPos, initVel,
                                  np.array([timeInt, yaws[n], pitches]), [0, timeSeconds], numSteps)#TODO: verify yaws[n] makes sense
        sailset = np.append(sailset, newSail)

    simset = np.append(sailset, sysbds)
    
    return simset

#generates an array filled with all possible state combinations for a given length
#recursive -> head is passing the already determined pieces down the recursion
def permutationGenerator(states, length, head=np.array([])):
    retArray = np.array([])

    if length == 1:
        for state in states:
            if len(retArray) == 0:
                retArray = np.array([np.append(head, state)])
            else:
                retArray = np.append(retArray, [np.append(head, state)], axis=0)
        return retArray
    
    for state in states:
        newhead = np.append(head, np.array([state]))
        if len(retArray) == 0:
            retArray = permutationGenerator(states, length-1, newhead)
        else:
            retArray = np.append(retArray, permutationGenerator(states, length-1, newhead), axis=0)

    return retArray