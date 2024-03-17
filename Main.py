import numpy as np
import matplotlib.pyplot as plt
import body
import utils

# in order to solve the lost in space problem, we must be able to do a few things:
# 1. Identify the locations of the planets (image recognition)
# 2. Know where the planets should be relative to a certain point (eart, sun, etc.)
# 3. Use that information to determine spacecraft location

# 2 -> Developing a model of our solar system using python


# TODO generate ploting info
#print(utils.gravaccel(np.array([0,0,0]), np.array([6378100,0,0]), 100, 5.97219E+24))

b1 = body.Body("earth", np.array([384400000,0,0]), np.array([0,0,0]), np.array([0,0,0]), 5.97219E+24)
b2 = body.Body("moon", np.array([0,0,0]), np.array([0,1023,0]), np.array([0,0,0]), 7.36E+22)
#b1.displayState()
#b2.displayState()
utils.integrate(np.array([b1, b2]), 5000000, 10000)
#b1.displayState()
#b2.displayState()
print(b1.locations)
print(b2.locations)

utils.plotbodies(np.array([b1, b2]))
