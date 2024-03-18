

import numpy as np
import matplotlib.pyplot as plt
import functions.body as bd
import matplotlib.animation as animation
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

'''
Legacy integrator used for determining ALL planet & satellite positions, modified one below
'''

def integrate(bodies, duration, numsteps):
    for n in range(numsteps):
        statechange(bodies)
        for b in bodies:
            b.propagate(duration/(numsteps*1.0))        

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

'''
Modified Integrator used to simulate solar sail trajectories without calculating celestial body posiitons
'''

def rotate(vector, direction, degrees):
    rads = degrees * 2 * np.pi / 360
    if direction == "yaw":
        rotmatrix = np.array([[np.cos(rads), -np.sin(rads), 0],[np.sin(rads), np.cos(rads), 0],[0, 0, 1]])
        newvect = np.dot(rotmatrix, vector)
        return newvect
    elif direction == "pitch":
        rotmatrix = np.array([[1, 0, 0],[0, np.cos(rads), -np.sin(rads)],[0, np.sin(rads), np.cos(rads)]])
        newvect = np.dot(rotmatrix, vector)
        return newvect

'''
Plotting and animation util functions below.
'''
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

def animatebodies(bodies, tstep=1):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    #creating lines for each body
    lines = np.array([])
    duration = bodies[0].locations[0][0].size # for some reason the data is in a redundant array

    print(duration)
    print(bodies[0].locations[0])

    for b in bodies:
        data = b.locations[0]
        ln, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], label=b.name)
        lines = np.append(lines, np.array([ln]))
    
    def update(frame, bodies, lines): # lines and bodies in the same order
        for n in range(len(bodies)):
            ln = lines[n]
            body = bodies[n]
            data = body.locations[0]
            ln.set_data(data[:2, :int(frame*tstep)])
            ln.set_3d_properties(data[2, :int(frame*tstep)])

    # Setting the axes properties
    ax.set_xlim3d([-5E+9, 5E+9])
    ax.set_xlabel('X')

    ax.set_ylim3d([-5E+9, 5E+9])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-5E+9, 5E+9])
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Solar Sytem Animation from Jun 20, 2000 to Dec 1, 2030")

    numframes = int(duration/tstep)

    ani = animation.FuncAnimation(fig, update, numframes, fargs=(bodies, lines), interval=100/numframes, blit=False)
    ani.save('solarsystem.gif')
    plt.show()
            
