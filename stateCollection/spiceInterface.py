'''
Access to SPICE toolkit using SpiceyPy python wrapper. 

SpicyPy: https://github.com/AndrewAnnex/SpiceyPy

Takes SPK kernel and outputs specific state vectors.

By Marvin Lin
March 2024
'''

import spiceypy as spice

def computeBody(spkid, start_time = 'Jun 20, 2000', end_time = 'Dec 1, 2030', step = 4000, center = 'SOLAR SYSTEM BARYCENTER'):

    spice.furnsh("./data/metaKernel.txt")

    utc = [start_time, end_time]
    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])

    times = [x*(etTwo-etOne)/step + etOne for x in range(step)]

    position, velocity = spice.spkpos(spkid, times, 'J2000', 'NONE', center)

    # positions is shaped (4000, 3), let's transpose to (3, 4000) for easier indexing
    position = position.T

    return position

