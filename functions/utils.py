import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import functions.body as body
import scipy.integrate as integ
#important constants:
G = 6.6743E-11

P0 = 9E-9 #newtons per square meeter -> solar sail effectiveness
mu = 1.327e20 /1e9 # mu in km^3/s^2, sun's gravitational parameter
AU = 1.496e11 /1e3  # astronomical unit in km, distance from sun to earth
beta = 0.15 # ratio of peak solar sail force to sun's gravity

# Style constants:
traj_opacity = 0.1

'''
General Utility functions used for a variety of applications across the project.
'''

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

def solarSailIntegrate(system, sailbodies):
    numsteps = 4000 #TODO: make this not hardcoded!!!!
    for n in range(numsteps):
        solarSailStateChange(system, sailbodies, n)
        for b in sailbodies:
            b.propagate(system.SUN.timeStep)

def solarSailStateChange(system, sailbodies, currStep):
    for sail in sailbodies: # set all the accelerations to zero
        sail.acceleration = np.array([0,0,0])

    for sail in sailbodies: # calculate the gravitational accelerations seen by the different sails
        for planet in system: # we don't mess with planetary acceleration since its on a predetermined path
            planetaccel, sailaccel = planetaryAccel(sail, planet)
            sail.acceleration = sail.acceleration + sailaccel

def planetaryAccel(planet, spacecraft, currStep): #cause position is garbage now
    planetPos = planet.getPosition(currStep)
    return gravaccel(planetPos, spacecraft.pos, planet.mass, spacecraft.mass)

'''
Simulations based on the ODE system
s = [x, y, z, vx, vy, vz]
sret = [vx, vy, vz, ax, ay, az]
'''
mu = 1.327e20 /1e9 # mu in km^3/s^2, sun's gravitational parameter
AU = 1.496e11 /1e3  # astronomical unit in km, distance from sun to earth
beta = 0.15 # ratio of peak solar sail force to sun's gravity

# current system in place for sail calculations
def npSailODE(t, s, sail):
    coneAngle = sail.coneAngle(t, s) #coneAngle is going to be a tuple with a "yaw" and "pitch" component
    #print(f'cone: {coneAngle}')
    r = np.array([s[0], s[1], s[2]])
    v = np.array([s[3], s[4], s[5]])
    asun = (-mu/(np.linalg.norm(r)**3)) * r

    # following calcs will be done in 2d via the transformation matrix
    rplanar = np.matmul(sail.initMatrix, r)
    theta = math.atan2(rplanar[1], rplanar[0])
    phi = math.atan2(rplanar[2], np.sqrt(rplanar[1]**2 + rplanar[0]**2))
    asailMag = (beta * mu / np.dot(rplanar, rplanar) * (np.cos(coneAngle[0]) ** 2) * (np.cos(coneAngle[1]) ** 2))
    asailNorm = np.array([np.cos(coneAngle[1] + phi)*np.cos(coneAngle[0] + theta), np.cos(coneAngle[1] + phi)*np.sin(coneAngle[0]+theta), np.sin(coneAngle[1] + phi)])
    asailplanar = asailMag * asailNorm
    asail = np.matmul(sail.invMatrix, asailplanar)

    atotal = asun + asail
    return np.append(v, atotal)

# sail cone angle function factory
def cone_angle_factory(t_thresholds, yaws, pitches):
    instr_count = len(yaws)
    i_prev = [0]

    assert len(t_thresholds) == instr_count, \
            'There should be one angle per time threshold'

    def cone_angle(t, s):
        if t < t_thresholds[i_prev[0]] and \
                (i_prev[0] == 0 or t > t_thresholds[i_prev[0]]):
            i = i_prev
        else:
            for i in range(instr_count):
                if t < t_thresholds[i]:
                    break
            i_prev[0] = i
        return (yaws[i], pitches[i])

    return cone_angle

# sail creator, creates a sail object and solves the trajectory
# TODO: ngl I think the term trajectory could be a bit confusing since we are 
# controlling the trajectory of the inputs not the actual path of the spacecraft
def sailGenerator(name, initLoc, initVel, trajectory, timeInterval, numsteps):
    
    #generate the transformation matrix needed
    er = initLoc / np.linalg.norm(initLoc)
    ev = initVel / np.linalg.norm(initVel)
    eb = np.cross(er, ev)
    initMatrix = [er, ev, eb]
    
    coneAngle = cone_angle_factory(trajectory[0], trajectory[1], trajectory[2])
    newSail = body.SolarSail(name, initLoc, initVel, 0, 0, coneAngle, path_style='trail', show_traj=False, initMatrix=initMatrix)
    span = np.linspace(timeInterval[0], timeInterval[1], int(numsteps))
    initialconditions = np.append(initLoc, initVel)
    newSailLocs = integ.solve_ivp(npSailODE, timeInterval, initialconditions, rtol=1e-8,t_eval=span, args=[newSail])
    newSail.locations = newSailLocs.y[:3, :]
    return newSail

#another implementation used for odeint based systems
'''
def npODESailGenerator(s, t, sail):
    r = np.array([s[0], s[1], s[2]])
    v = np.array([s[3], s[4], s[5]])
    coneAngle = sail.getCurrentConeAngle(t)
    asun = (-mu/(np.linalg.norm(r)**3)) * r
    theta = math.atan2(r[1], r[0])
    asail = (beta * mu / np.dot(r, r) * np.cos(coneAngle) ** 2) * np.array([np.cos(theta+coneAngle), np.sin(theta+coneAngle), 0])
    atotal = asun + asail
    return np.append(v, atotal)
'''

# reference code for what all the sail odes are based on    
'''
def simpleSailGenerator(t, s, cone):
    rsquared = s[0]**2 + s[1]**2
    rcubed = (rsquared)**(3/2)
    asunx = -mu*s[0]/rcubed
    asuny = -mu*s[1]/rcubed
    theta = math.atan2(s[1],s[0])
    asail = beta*mu/rsquared*math.cos(cone)**2
    asailx = asail*math.cos(theta+cone)
    asaily = asail*math.sin(theta+cone)
    return [s[2], s[3], asunx+asailx, asuny+asaily]
'''

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
    #print(bodies)
    duration = bodies[0].locations.size # for some reason the data is in a redundant array

    #print(duration)
    #print(bodies[0].locations)

    for b in bodies:
        data = b.locations
        if isinstance(b, body.CelestialBody):
            ln, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], 
                      label=b.name, alpha=b.opacity, marker=b.marker, markersize=b.dispSize, color=b.color)
        else:
            ln, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], alpha=b.opacity, marker=b.marker, markersize=b.dispSize)
        if b.show_traj:
            ax.plot(data[0,:], data[1,:], data[2,:], alpha=traj_opacity, color='black')
        lines = np.append(lines, np.array([ln]))
    
    def update(frame, bodies, lines): # lines and bodies in the same order
        for n, b in enumerate(bodies):
            ln = lines[n]
            data = b.locations
            if b.path_style == 'past':
                ln.set_data(data[:2, :int(frame*tstep)])
                ln.set_3d_properties(data[2, :int(frame*tstep)])
            elif b.path_style == 'trail':
                front = int(frame*tstep)
                back = int(max(0, front - b.trail_length))
                ln.set_data(data[:2, back:front])
                ln.set_3d_properties(data[2, back:front])

    # Setting the axes properties
    ax.set_xlim3d([-2.5E+8, 2.5E+8])
    ax.set_xlabel('X')

    ax.set_ylim3d([-2.5E+8, 2.5E+8])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-2.5E+8, 2.5E+8])
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Solar Sytem Animation from Jun 20, 2000 to Dec 1, 2030")

    numframes = int(duration/tstep)

    ani = animation.FuncAnimation(fig, update, numframes, fargs=(bodies, lines), interval=100/numframes, blit=False)
    ani.save('solarsystem.gif')
    plt.show()

'''
The Sim itself, streamlined function which takes in some inputs and synchronizes the planetary and solar sail animation
TODO: figure this out, will hardcode atm
'''
#dumb function used to generate a shit tone of trajectories
#orientations: array of orientations for the sail ex. [-0.6,0.6]
#time: time object dictating the trajectory info
#points: number of points
'''
def orientationVary(orientations, time, points):
    trajectories = np.array([])
    length = time.lengthSeconds
    times = np.linspace(0, length, points)

def orientationVaryHelper(arr, desiredlen, orientations):
    if len(arr) == desiredlen:
        return arr
    else:
        for a in arr:
            for o in orientations:
                orientationVaryHelper(np.append(arr, [o]), desiredlen, orientations)


def generateSim(time):

'''
