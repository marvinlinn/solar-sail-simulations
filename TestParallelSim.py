from rl_agent.World import ParallelTrackNEO

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

world = ParallelTrackNEO(dt=1)
positions = []

start=time.time()
for _ in range(world.max_sim_length()):
    _, S = world.advance_simulation(0*np.ones((50,1)))
    S = S[:,:3]
    positions.append(S)
print(time.time()-start)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect([1,1,1])

positions = np.array(positions) # (time, sail, position)
for sail in range(positions.shape[1]):
    x, y, z = positions[:,sail,:].T
    ax.plot(x,y,z)
    ax.scatter([x[0]],[y[0]],[z[0]])
ax.scatter([0], [0], [0])
plt.show()
