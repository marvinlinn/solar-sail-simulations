'''
Access to SPICE toolkit using SpiceyPy python wrapper. 

SpicyPy: https://github.com/AndrewAnnex/SpiceyPy

Takes SPK kernel and outputs specific state vectors.

By Marvin Lin
March 2024
'''
import numpy as np
import spiceypy as spice
import stateCollection.horizonAPI as horizon
from enum import Enum

# Standard library of SPK's given in the de430.bsp binary file
STANDARD_LIB = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '199', '299', '301', '399']

'''
[FOR PUBLIC] requestData
Takes in an SPKID, Time object, and step size and return an array of positions and velocities (magnitude)

Inputs:
String spkid - A SPKID designated by NASA JPL Horizons which identifies the body of interest
Time Time - A Time object which defines the time from of interest.
Integer step - Step size between each data point, in hours.

Outputs:
Integer[][] positions - An array of positions with the given step sizes with dimensions (3, # of points)
Integer[] velocities - An array of velocity magnitudies with given step sizes with deminsion (1, #ofpoints)

'''
def requestData(spkid, Time, step):
    saveFile = './data/save.npy'
    existingSPK = np.load(saveFile, allow_pickle=True)
    existingSPK = checkSPK(spkid, existingSPK)

    np.save(saveFile, existingSPK)

    return computeBody(spkid, Time.returnStart(), Time.returnEnd(), points=Time.returnPoints(step))

'''
[FOR PUBLIC] requestDatasat
Takes in an array of SPKIDs, Time object, and step size and return an array of positions and velocities (magnitude). Completes batch processing on any given array of SPKIDs.

Inputs:
String[] spkid - An array of SPKIDs designated by NASA JPL Horizons which identifies the bodies of interest
Time Time - A Time object which defines the time from of interest.
Integer step - Step size between each data point, in hours.

Outputs:
Integer[][][] positions - An array of positions with the given step sizes with dimensions (len(spkid), 3, # of points)
Integer[][] velocities - An array of velocity magnitudies with given step sizes with deminsion (len(spkid), 1, #ofpoints)

'''
def requestDataSet(spkid, Time, step):
    saveFile = './data/save.npy'
    existingSPK = np.load(saveFile)
    for i in spkid:
        existingSPK = checkSPK(i, existingSPK)
    np.save(saveFile, existingSPK)

    returnArray = []
    for i in spkid:
        returnArray.append(computeBody(i, Time.returnStart(), Time.returnEnd(), points=Time.returnPoints(step)))
    return returnArray

# Checks if exisitng SPK exists for given SPKID, If not pulls from API
def checkSPK(spkid, existingSPK):
    if spkid in existingSPK:
        None
    else:
        horizon.getSPK(spkid)
        np.append(existingSPK, spkid)
    return existingSPK

# Create Save File for spkids
def createEmptySave():
    saveFile = './data/save.npy'
    np.save(saveFile, np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '199', '299', '301', '399']))

# Compute the the positon and velocity of a given SPKID, start/end time, and number of calculated points
def computeBody(spkid, start_time = 'Jun 20, 2000', end_time = 'Dec 1, 2030', points = 4000, center = 'SOLAR SYSTEM BARYCENTER'):

    spice.furnsh("./data/metaKernel.txt")
    if spkid not in STANDARD_LIB:
        pathSPK = "./data/" + spkid + ".bsp"
        spice.furnsh(pathSPK.format(spkid))

    utc = [start_time, end_time]
    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])

    times = [x*(etTwo-etOne)/points + etOne for x in range(points)]

    position, velocity = spice.spkpos(spkid, times, 'J2000', 'NONE', center)

    # Positions is shaped (points, 3)--Transposed to (3, points) for easier indexing
    position = position.T

    return position, velocity

# Interval Enum for handling step size calculations
class Intervals(Enum):
    HOUR = 1
    DAY = 24
    WEEK = 168
    MONTH = 730

# Time object for easy handling of start and end times, as well as calculations
class Time():
    MONTHS = [None, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    def __init__(self, month, day, year, lengthInDays):
        if (month > 12 or month < 1):
            raise ValueError("Month must be between 1 and 12")
        if (day > 31 or day < 1):
            raise ValueError("Day must be between 1 and 31")
        if (year < 999 or year > 10000):
            raise ValueError("Year must be four digits")
        self.month = month
        self.day = day
        self.year = year
        self.length = lengthInDays
        self.lengthSeconds = lengthInDays * 24 * 3600

        self.endDay = self.day + lengthInDays
        self.endMonth = self.month + self.endDay // 30
        self.endYear = self.year + self.endMonth // 12
        
        self.endMonth = self.endMonth % 12
        self.endDay = self.endDay % 30

    def returnStart(self):
        outputStr = Time.MONTHS[self.month] + ' {}, {}'
        return outputStr.format(self.day, self.year)

    def returnEnd(self):
        outputStr = Time.MONTHS[self.endMonth] + ' {}, {}'
        return outputStr.format(self.endDay, self.endYear)

    def returnPoints(self, step):
        totalHours = (self.endYear - self.year)*8760 + (self.endMonth - self.month)*730 + (self.endDay - self.day)*24
        return totalHours // step
