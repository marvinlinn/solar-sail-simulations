import numpy as np
import functions.system as system
import functions.utils as utils
import csv

#Pretraining Tools

#generates a CSV of data which takes in a set of sails, and a target bd.
def generateBodyCSV(sailSet, targetbd, filename="sails_traj_data"):
    
    f = filename + ".csv"

    with open(f, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='|')
        
        for sail in sailSet:
            writer.writerow([sail.name + " " + targetbd.name, "sail x", "sail y", "sail z", "time", "yaw", "target x", "target y", "target z", "distance x", "distance y", "distance z", "abs distance"])
            numsteps = len(sail.timeSteps)
            for n in range(numsteps):
                sailPos = sail.locations[:3, n]
                targetPos = targetbd.locations[:3, n]
                distVect = targetPos - sailPos
                distMag = np.linalg.norm(distVect)
                writer.writerow([str(n), sailPos[0], sailPos[1], sailPos[2], sail.timeSteps[n], sail.yawAngle[n], targetPos[0], targetPos[1], targetPos[2], distVect[0], distVect[1], distVect[2], distMag])
    return

#for n different sail orientations and m different points to change sail orientation, n^m sails are generated
#targetbd will be replaced with a neo body, for testing purposes we're gonna just use a planet so target bd will be an index in SolarSystem
def packaged2DSim(simTime, sailorientations, numsailchanges, targetbd):
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
        timeInt[n] = n * (timeSeconds/numsailchanges)

    #sail generation
    #init conditions -> earth position, velocity must also be vectorized correctly
    initPos = sysbds[3].locations.T[0]
    initVelVec = (sysbds[3].locations.T[1]-sysbds[3].locations.T[0])/np.linalg.norm(sysbds[3].locations.T[1]-sysbds[3].locations.T[0]) #velocity vector via linearization between point 0 and 1
    initVel = initVelVec * 30
    sailset = np.array([])

    for n in range(len(yaws)):
        newSail = utils.sailGenerator(("sail "+ str(n)), initPos, initVel,
                                  np.array([timeInt, yaws[n], pitches]), [0, timeSeconds], numSteps, bodies=[sysbds[targetbd]])#TODO: verify yaws[n] makes sense -> it does since we are varrying yaws 
        sailset = np.append(sailset, newSail)

    #simset = np.append(sailset, sysbds)
    
    return sailset, sysbds

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