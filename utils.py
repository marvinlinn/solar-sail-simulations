import numpy as np
import matplotlib.pyplot as plt
import body as bd
#important constants:
G = 6.6743E-11

#calculates the distance between 2 points in 3 dimensions
def distance(p1, p2):
    vect = p2 - p1
    return np.linalg.norm(vect)

#calculates the gravitational acceleration vectors (np arrays) given position and mass
def gravaccel(p1, p2, m1, m2):
    dist = distance(p1, p2)
    unitVect = (p2-p1)/dist
    fgrav = (G * m1 * m2)/(dist*dist)
    accel1 = (fgrav/m1) * unitVect
    accel2 = (fgrav/m2) * unitVect * -1
    return accel1, accel2

def bodyaccel(m1, m2):
    return gravaccel(m1.position, m2.position, m1.mass, m2.mass)

def integrate(bodies, duration, numsteps):
    for n in range(numsteps):
        statechange(bodies)
        for b in bodies:
            b.propogate(duration/(numsteps*1.0))        

def statechange(bodies):
    
    #set the accelerations to zero
    for body in bodies:
        body.acceleration = np.array([0,0,0])

    #calculate the acceleration vectors for the respective bodies
    numbodies = len(bodies)
    for a in range(numbodies):
        b = a + 1 #bodies greater than the initial one
        while b < numbodies:
            accelA, accelB = bodyaccel(bodies[a], bodies[b])
            bodies[a].acceleration = bodies[a].acceleration + accelA
            bodies[b].acceleration = bodies[b].acceleration + accelB
            b += 1

def plotbodies(bodies):
    ax = plt.figure().add_subplot(projection='3d')
    
    for b in bodies:
        pos = b.locations.transpose()
        x = pos[0]
        y = pos[1]
        z = pos[2]
        ax.plot(x,y,z, label=b.name)

    ax.legend()
    plt.show()              
            