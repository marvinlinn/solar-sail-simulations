from multiprocessing import Pool
import numpy as np
import functions.system as system
import functions.utils as utils
import csv
import os
import stateCollection.spiceInterface as spice 
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
        
            writer.writerow([sail.name + " " + targetbd.name + "; shortest distance: " + str(sail.closestAbsDistance) + "; sail yaw orientations: " + str(sail.yaws) + "; sail time intervals" + str(sail.timeInt)])
            writer.writerow(["time step", "sail x", "sail y", "sail z", "time", "yaw", "sail Vx", "sail Vy", "sail Vz", "distance x", "distance y", "distance z", "abs distance"])
            numsteps = len(sail.timeSpan)
            for n in range(numsteps):
                sailPos = sail.locations[:3, n]
                sailVel = sail.velocity[:3, n]
                distFromTarget = sail.distanceMatrix[:, n]
                writer.writerow([str(n), sailPos[0], sailPos[1], sailPos[2], sail.timeSpan[n], sail.yawAngle[n], sailVel[0], sailVel[1], sailVel[2], distFromTarget[0], distFromTarget[1], distFromTarget[2], distFromTarget[3]])
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
def largeScalePermutationSailGenerator(simTime, sailorientations, numsailchanges, targetbd=[], isParallel = False):

    #planet generation
    sysname = str(simTime.length) + " day sys"
    sys = system.SolarSystem(sysname, simTime)
    sysbds = sys.bodies
    numSteps = len(sysbds[0].locations[0])
    timeSeconds = simTime.lengthSeconds


    #trajectories generation
    pitches = np.zeros(numsailchanges+1)
    yaws = permutationGenerator(sailorientations, numsailchanges+1)
    timeInts = np.zeros(numsailchanges+1)
    for n in range(numsailchanges+1):
        timeInts[n] = n * (timeSeconds/numsailchanges)

    #ensure shape makes sense
    timeInts = np.repeat(timeInts, len(yaws))
    pitches = np.repeat(pitches, len(yaws))
    
    Earth = sysbds[3]

    #sail generation
    #init conditions -> earth position, velocity must also be vectorized correctly
    initPos = Earth.locations.T[0]
    initVelVec = (Earth.locations.T[1]-Earth.locations.T[0])/np.linalg.norm(Earth.locations.T[1]-Earth.locations.T[0]) #velocity vector via linearization between point 0 and 1
    initVel = initVelVec * 30
    sailset = np.array([])

    print(targetbd.name)

    if isParallel == False:
        for n in range(len(yaws)):
            newSail = utils.sailGenerator(("sail "+ str(n)), initPos, initVel,
                                      np.array([timeInts[n], yaws[n], pitches[n]]), [0, timeSeconds], numSteps, bodies=[sysbds[4]])#TODO: verify yaws[n] makes sense -> it does since we are varrying yaws 
            sailset = np.append(sailset, newSail)
    else:
        sailset = parallelSailGeneration(initPos, initVel, timeInts, yaws, pitches, timeSeconds, numSteps, targetbd=targetbd)

    #simset = np.append(sailset, sysbds)
    
    return sailset, targetbd

# uses a simple trajectory parameterized by orientationOrder
# generates a bunch of different time intervals (think balls in bins) for the number of sail changes
def largeScaleSimpleTrajSailGen(simTime, orientationOrder, numsailChanges, targetbd=[], isParallel = False):
    #planet generation
    sysname = str(simTime.length) + " day sys"
    sys = system.SolarSystem(sysname, simTime)
    sysbds = sys.bodies
    numSteps = len(sysbds[0].locations[0])
    timeSeconds = simTime.lengthSeconds


    #trajectories generation
    timeInts = generatePartitions(numsailChanges, len(orientationOrder)-1) * (timeSeconds/numsailChanges)
    pitches = np.zeros((len(timeInts), len(orientationOrder)))
    yaw = np.repeat(np.array([orientationOrder]), len(timeInts), axis=0) # typically set to [0.6, 0, -0.6, 0]

    Earth = sysbds[3]

    #sail generation
    #init conditions -> earth position, velocity must also be vectorized correctly
    initPos = Earth.locations.T[0]
    initVelVec = (Earth.locations.T[1]-Earth.locations.T[0])/np.linalg.norm(Earth.locations.T[1]-Earth.locations.T[0]) #velocity vector via linearization between point 0 and 1
    initVel = initVelVec * 30
    sailset = np.array([])

    print(targetbd.name)

    if isParallel == False:
        for n in range(len(timeInts)):
            newSail = utils.sailGenerator(("sail "+ str(n)), initPos, initVel,
                                      np.array([timeInts[n], 
                                                yaw[n], 
                                                pitches[n]]), 
                                                [0, timeSeconds], 
                                                numSteps, 
                                                bodies=[sysbds[4]])#TODO: verify yaws[n] makes sense -> it does since we are varrying yaws 
            sailset = np.append(sailset, newSail)
    else:
        sailset = parallelSailGeneration(initPos, initVel, timeInts, yaw, pitches, timeSeconds, numSteps, targetbd=targetbd) 

    #simset = np.append(sailset, sysbds)
    
    return sailset, targetbd

#generates an array filled with all possible state combinations for a given length
#recursive -> head is passing the already determined pieces down the recursion
def permutationGenerator(states, length, head=np.array([])):
    retArray = np.array([])

    if length == 1:
        retArray = np.array([np.append(head, np.array([0]))])
        return retArray
    
    for state in states:
        newhead = np.append(head, np.array([state]))
        if len(retArray) == 0:
            retArray = permutationGenerator(states, length-1, newhead)
        else:
            retArray = np.append(retArray, permutationGenerator(states, length-1, newhead), axis=0)

    return retArray

#generates an array composed of different "bins" for a given number of "balls" (think balls in bins)
def generatePartitions(numObj, numCategories, head=np.array([0])):
    superArray = np.array([])
    if numCategories == 1:
        num = numObj
        if len(head) != 0:
            num = numObj + head[-1]
        retArray = np.array([np.append(head, np.array([num]))])
        return retArray
    else:
        for n in range(numObj+1):
            num = n
            if len(head) != 0:
                num = num + head[-1]
            newhead = np.append(head, np.array([num]))
            newArray = generatePartitions(numObj-n, numCategories-1, head=newhead) 
            if len(superArray) == 0:
                superArray = newArray
            else:
                superArray = np.append(superArray,newArray, axis=0)
        return superArray

#dates = [month, day, year]
#args = (length, sailOrientations, variations, numsails)
#targetbd is manual changed
def prepackagedWholeSim(dates, args):
    print("prepacksim works")
    timetest = spice.Time(dates[0], dates[1], dates[2], args[0])
    targetSystem = system.SolarSystem('targets', timetest)
    targetBodies = targetSystem.bodies
    sailset, target = largeScalePermutationSailGenerator(timetest.lengthSeconds, args[1], args[2], targetBodies[4], len(targetBodies[0].locations[0]), Earth=targetBodies[3])
    generateBodyCSV(sailset, target, simStartDate=str(dates[0])+str(dates[1])+str(dates[2]), numsails=args[3])
    return 1

def packagedSimForParallel(length, sailOrientations, variations, numsails, targetbd=[]):
    def nested(month, day, year):
        return prepackagedWholeSim(month, day, year, length, sailOrientations, variations, numsails)
    return nested 

# In the future make it possible to set target to a body then search spice
def parallelsiming(dates, sailOrientations, numSailChanges, numsails, length, target=''):
    
    # loop through every date instance and conduct parallel computing in order to solve all the sails
    for date in dates:
        #setup system generation
        timetest = spice.Time(date[0], date[1], date[2], length)
        targetSystem = system.SolarSystem('targets', timetest)
        targetBodies = targetSystem.bodies
        numSymSteps = len(targetBodies[0].locations[0])
        timeSeconds = timetest.lengthSeconds

        targetbd = targetBodies[4]

        #trajectories generation
        pitches = np.zeros(numSailChanges+1)
        yaws = permutationGenerator(sailOrientations, numSailChanges+1)
        timeInt = np.zeros(numSailChanges+1)
        for n in range(numSailChanges+1):
            timeInt[n] = n * (timetest.lengthSeconds/numSailChanges)

        #sail generation
        #init conditions -> earth position, velocity must also be vectorized correctly
        initPos = targetBodies[3].locations.T[0]
        initVelVec = (targetBodies[3].locations.T[1]-targetBodies[3].locations.T[0])/np.linalg.norm(targetBodies[3].locations.T[1]-targetBodies[3].locations.T[0]) #velocity vector via linearization between point 0 and 1
        initVel = initVelVec * 30
        sailset = np.array([])

        if __name__ == '__main__':
            with Pool(os.cpu_count()) as pool:         # start 4 worker processes
                for n in range(len(yaws)):
                    args = (("sail "+ str(n)), initPos, initVel,
                                  np.array([timeInt, yaws[n], pitches]), [0, timeSeconds], numSymSteps, [targetbd])
                    res = pool.apply_async(utils.parallelSailGenerator, args)
                    sailset = np.append(sailset, res)
            generateBodyCSV(sailset, targetbd, filename="sails_traj_data", numsails=numsails, simStartDate='09012000')
        else:
            print('not parallel')
 

# Generates a bunch of sails in parallel 
def parallelSailGeneration(initPos, initVel, timeInt, yaws, pitches, timeSeconds, numSymSteps, targetbd=[]):
    
    argSet = [[("sail "+ str(n)), initPos, initVel,
                    np.array([timeInt[n], yaws[n], pitches[n]]), [0, timeSeconds], numSymSteps, [targetbd]] for n in range(len(yaws))]

    with Pool(os.cpu_count()) as pool:         
        sails = [pool.apply_async(utils.parallelSailGenerator, [args]) for args in argSet]
        sailset = [sail.get() for sail in sails]
    return sailset

# Generates a bunch of sails for a bunch of different dates, stores them in seperate CSV directories
# dates = [[month,day,year,lengthDays],[month,day,year,lengthDays],[month,day,year,lengthDays], ...]
def multiSailSetGenerator(dates, sailOrientations, numChanges):
    for date in dates:
        timeInstance = spice.Time(date[0], date[1], date[2], date[3])
        targetbds = system.SolarSystem('targets', timeInstance).bodies
        target = targetbds[4]
        sailset, target = largeScalePermutationSailGenerator(timeInstance, sailOrientations, numChanges, target, isParallel=True)
        generateBodyCSV(sailset, target, numsails=5, simStartDate=str(date[0]) + str(date[1]) + str(date[2]))
            

''' Parallel Example '''
def func(x):
    return x**2

def paralleltesting(x):
    with Pool(os.cpu_count()) as pool:
        out = [pool.apply_async(func, [val]) for val in x]
        vals = [v.get() for v in out]
    return vals