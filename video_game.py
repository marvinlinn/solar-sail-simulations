from rl_agent.World import ParallelTrackNEO
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from pynput import keyboard

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

world = ParallelTrackNEO(num_sails=1, control_interval=5, days=2000)

world.reset()

A = np.array([[0]])
R, S = world.advance_simulation(A)

# x, y, z = zip(S[0,:3], S[0,6:9])
scatter = ax.scatter([], [], [])
scatter_sail = ax.scatter([], [], [])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)

last_key = 5

# Function to handle key press events
def on_press(key):
    global last_key
    try:
        new_key = key.char  # Alphanumeric keys
        if new_key in [str(i) for i in range(10)]:
            last_key = int(new_key)
    except AttributeError:
        last_key = last_key

listener = keyboard.Listener(on_press=on_press)
listener.start()

def update(frame):
    global last_key
    if last_key == 0:
        last_key = 5
    A = [-0.6, -0.5, -0.3, -0.1, 0.0, 
         0.1, 0.3, 0.5, 0.6][last_key-1]
    print(A)
    A = np.array([[A]])
    R, S = world.advance_simulation(A)
    x, y, z = zip(S[0,:3], S[0,6:9])
    scatter._offsets3d = (x[1:], y[1:], z[1:])
    scatter_sail._offsets3d = (x[:1] + (0, ), y[:1] + (0, ), z[:1] + (0, ))

ani = FuncAnimation(fig, update, interval=100)

plt.show()
