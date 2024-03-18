import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from scipy.integrate import solve_ivp
mu = 1.327e20 /1e9 # mu in km^3/s^2, sun's gravitational parameter
AU = 1.496e11 /1e3  # astronomical unit in km, distance from sun to earth
beta = 0.15 # ratio of peak solar sail force to sun's gravity

# style properties:
base_alpha = 0.1
window = 20
loops = 10

# s = [x, y, vx, vy]
# F returns s_dot = [vx, vy, ax, ay]
# r = [x, y]
# a = -mu/|r|**2 r_hat = -mu/|r|**3 [x,y]
def Fsun(t,s):
    rcubed = (s[0]**2+s[1]**2)**(3/2)
    ax = -mu*s[0]/rcubed
    ay = -mu*s[1]/rcubed
    return [s[2], s[3], ax, ay]

def Fsail(t,s,cone):
    cone = cone(t, s)
    pos = np.array(s[0:2])
    rsquared = np.dot(pos,pos)
    rcubed = rsquared**(3/2)
    asun = -mu*pos/rcubed
    theta = math.atan2(s[1],s[0])
    asail = beta*mu/rsquared*math.cos(cone)**2
    asailx = asail*math.cos(theta+cone)
    asaily = asail*math.sin(theta+cone)
    return [s[2], s[3], asun[0]+asailx, asun[1]+asaily]

def cone_angle_factory(a, t_thresholds):
    instr_count = len(a)

    assert len(t_thresholds) == instr_count, 'There should be one angle per time threshold'

    def cone_angle(t, s):
        for i in range(instr_count):
            if t < t_thresholds[i]:
                break
        return a[i]

    return cone_angle

venus = solve_ivp(Fsun, [0, 2e7], [0.72*AU, 0, 0, 35], rtol=1e-8)
earth = solve_ivp(Fsun, [0, 3.2e7], [AU, 0, 0, 30], rtol=1e-8)
mars = solve_ivp(Fsun, [0, 6e7], [1.5*AU, 0, 0, 24], rtol=1e-8)


alpha = lambda t: 0.6 if t > 1e7 else -0.6

sail_in = solve_ivp(Fsail, [0, 3.2e7], [AU, 0, 0, 30], rtol=1e-8, args=[cone_angle_factory([-0.6, 0, 0.6], [1e7, 1.5e7, 2e7])])
sail_out = solve_ivp(Fsail, [0, 3.2e7], [AU, 0, 0, 30], rtol=1e-8, args=[lambda t, s: 0.6])

fig = plt.figure()
ax = plt.axes(projection='3d')

venus_plot, = ax.plot(venus.y[0][0:0], venus.y[1][0:0], [0])
ax.plot(venus.y[0], venus.y[1], alpha=base_alpha)[0]

earth_plot, = ax.plot(earth.y[0][0:0], earth.y[1][0:0], [0])
ax.plot(earth.y[0], earth.y[1], alpha=base_alpha)

mars_plot, = ax.plot(mars.y[0][0:0], mars.y[1][0:0], [0])
ax.plot(mars.y[0], mars.y[1], alpha=base_alpha)

sail_in_plot, = ax.plot(sail_in.y[0][0:0], sail_in.y[1][0:0], [0])
ax.plot(sail_in.y[0], sail_in.y[1], alpha=base_alpha)

sail_out_plot, = ax.plot(sail_out.y[0][0:0], sail_out.y[1][0:0], [0])
ax.plot(sail_out.y[0], sail_out.y[1], alpha=base_alpha)

ax.set_aspect('equal', adjustable='datalim')

venus_plot.set_data(venus.y[0][:100], venus.y[1][:100])

form_back = lambda a_len: lambda x: min(max(0,x-window),a_len)
form_front = lambda a_len: lambda x: min(x, a_len)

def update(frame):
    venus_plot.set_data_3d(venus.y[0][:frame], venus.y[1][:frame], [0 for _ in range(min(len(venus.y[1]), frame))])

    back = form_back(len(earth.y[0]))(frame)
    front = form_front(len(earth.y[0]))(frame)
    earth_plot.set_data_3d(earth.y[0][back:front], earth.y[1][back:front], np.zeros(front-back))

    back = form_back(len(mars.y[0]))(frame)
    front = form_front(len(mars.y[0]))(frame)
    mars_plot.set_data_3d(mars.y[0][back:front], mars.y[1][back:front], np.zeros(front-back))

    back = form_back(len(sail_in.y[0]))(frame)
    front = form_front(len(sail_in.y[0]))(frame)
    sail_in_plot.set_data_3d(sail_in.y[0][back:front], sail_in.y[1][back:front], np.zeros(front-back))

    back = form_back(len(sail_out.y[0]))(frame)
    front = form_front(len(sail_out.y[0]))(frame)
    sail_out_plot.set_data_3d(sail_out.y[0][back:front], sail_out.y[1][back:front], np.zeros(front-back))

    return (venus_plot, earth_plot, mars_plot, sail_in_plot, sail_out_plot)


num_frames = len(sail_in.y[0])
ivl = 50

ani = animation.FuncAnimation(fig=fig, func=update, frames=num_frames, interval=ivl)
#ani.save(filename="./imagemagick_example.gif", writer="imagemagick")
#ani.save(filename="./test.mp4", writer="ffmpeg")
plt.show()
