import numpy as np
import functions.system as system
import functions.utils as utils
import csv
import os

#Pretraining Tools

#generates a CSV of data which takes in a set of sails, and a target bd.
def generateBodyCSV(unProcessedSailSet, targetbd, filename="sails_traj_data", numsails=0, simStartDate=''):
    
    fileHeader = "pretrain_data/" + simStartDate + "/"
    fileDirectory = './' + fileHeader
    if (not os.path.exists(fileDirectory)):
        os.makedirs(fileDirectory)
        print("Made a new directory: " + fileDirectory)

    #generate target body's CSV
    fileAddress = fileHeader + 'target.csv'
    with open(fileAddress, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerow(["target: " + targetbd.name, "target x", "target y", "target z", "time"])
        for n in range(len(targetbd.timeSpan)):
            targetPos = targetbd.locations[:3, n]
            writer.writerow([str(n), targetPos[0], targetPos[1], targetPos[2], targetbd.timeSpan[n]])
    
    ## Write Sail CSVs
    #solve for things such as distance vectors and abs distance, sort based on shortest distance 
    processedSailSet = calculateExtraData(unProcessedSailSet, targetbd)

    #restricts the number of sails being written to the file
    if numsails > 0:
        processedSailSet = processedSailSet[:numsails]

    #write and store each sail in a separatefile
    for sail in processedSailSet:
        fileAddress = fileHeader + sail.name + '.csv'
        with open(fileAddress, 'w', newline='') as file:
            writer = csv.writer(file, delimiter='|')
        
            writer.writerow([sail.name + " " + targetbd.name + "; shortest distance: " + str(sail.closestAbsDistance), "sail x", "sail y", "sail z", "time", "yaw", "sail Vx", "sail Vy", "sail Vz", "distance x", "distance y", "distance z", "abs distance"])
            numsteps = len(sail.timeSpan)
            for n in range(numsteps):
                sailPos = sail.locations[:3, n]
                sailVel = sail.velocity[:3, n]
                distFromTarget = sail.distanceMatrix[:, n]
                writer.writerow([str(n), sailPos[0], sailPos[1], sailPos[2], sail.timeSpan[n], sail.yawAngle[n], sailVel[0], sailVel[1], sailVel[2], distFromTarget[0], distFromTarget[1], distFromTarget[2], distFromTarget[3]])
    '''
    f = "pretrain_data/"+ filename + ".csv"
    with open(f, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='|')
        
        #solve for things such as distance vectors and abs distance, sort based on shortest distance 
        processedSailSet = calculateExtraData(unProcessedSailSet, targetbd)

        #restricts the number of sails being written to the file
        if numsails > 0:
            processedSailSet = processedSailSet[:numsails]

        #write target location information.
        writer.writerow(["target: " + targetbd.name, "target x", "target y", "target z", "time"])
        for n in range(len(targetbd.timeSpan)):
            targetPos = targetbd.locations[:3, n]
            writer.writerow([str(n), targetPos[0], targetPos[1], targetPos[2], targetbd.timeSpan[n]])

        #write sail location information.
        for sail in processedSailSet:
            writer.writerow([sail.name + " " + targetbd.name + "; shortest distance: " + str(sail.closestAbsDistance), "sail x", "sail y", "sail z", "time", "yaw", "sail Vx", "sail Vy", "sail Vz", "distance x", "distance y", "distance z", "abs distance"])
            numsteps = len(sail.timeSpan)
            for n in range(numsteps):
                sailPos = sail.locations[:3, n]
                sailVel = sail.velocity[:3, n]
                distFromTarget = sail.distanceMatrix[:, n]
                writer.writerow([str(n), sailPos[0], sailPos[1], sailPos[2], sail.timeSpan[n], sail.yawAngle[n], sailVel[0], sailVel[1], sailVel[2], distFromTarget[0], distFromTarget[1], distFromTarget[2], distFromTarget[3]])
    '''
    return

def calculateExtraData(sailSet, targetbd):
    
    #create a matrix of distance vectors for display and computation purposes, calculate/store the shortest distance from target 
    for sail in sailSet:
        numsteps = len(sail.timeSpan)        
        sail.distanceMatrix = np.full((4, numsteps), np.inf)
        for n in range(numsteps):
            sailPos = sail.locations[:3, n]
            targetPos = targetbd.locations[:3, n]
            distVect = targetPos - sailPos
            distMag = np.linalg.norm(distVect)
            sail.distanceMatrix[:,n] = np.append(distVect, np.array([distMag]))
        sail.closestAbsDistance = min(sail.distanceMatrix[3,:])
    
    #returns a sorted set of sails based on closest absolute distance
    return sorted(sailSet, key=lambda sail: sail.closestAbsDistance)

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

    print(targetbd.name)

    for n in range(len(yaws)):
        newSail = utils.sailGenerator(("sail "+ str(n)), initPos, initVel,
                                  np.array([timeInt, yaws[n], pitches]), [0, timeSeconds], numSteps, bodies=[targetbd])#TODO: verify yaws[n] makes sense -> it does since we are varrying yaws 
        sailset = np.append(sailset, newSail)

    #simset = np.append(sailset, sysbds)
    
    return sailset, sysbds, targetbd

#generates an array filled with all possible state combinations for a given length
#recursive -> head is passing the already determined pieces down the recursion
def permutationGenerator(states, length, head=np.array([])):
    retArray = np.array([])

    if length == 1:
        retArray = np.array([np.append(head, np.array([0]))])
        return retArray
        '''
        for state in states:
            if len(retArray) == 0:
                retArray = np.array([np.append(head, state)])
            else:
                retArray = np.append(retArray, [np.append(head, state)], axis=0)
        return retArray
        '''
    for state in states:
        newhead = np.append(head, np.array([state]))
        if len(retArray) == 0:
            retArray = permutationGenerator(states, length-1, newhead)
        else:
            retArray = np.append(retArray, permutationGenerator(states, length-1, newhead), axis=0)

    return retArray