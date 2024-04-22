import math

class World:
    # state s := (x, y, z, vx, vy, vz, 
    #             x_target, y_target, z_target, vx_target, vy_target, vz_target)

    def __init__(self):
        pass

    def reset(self):
        # return initial state s
        pass

    def advance_simulation(self, a):
        # return reward and next state (r, s)
        pass

class TrackNEO(World):

    def __init__(self):
        pass

    def reset(self):
        pass

    def advance_simulation(self, a):
        pass

class SimpleTestWorld(World):

    def __init__(self):
        self.reset()

    def reset(self):
        self.s = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        return self.s

    def advance_simulation(self, a):
        dt = 1
        accel = 0.1
        x, y, vx, vy = 0, 1, 3, 4

        x0 = self.s[x]
        y0 = self.s[y]

        accel_x = accel * a # math.cos(a)
        accel_y = 0 # accel * math.sin(a)

        self.s[x] += 1/2 * accel_x * dt**2 + self.s[vx] * dt
        self.s[y] += 1/2 * accel_y * dt**2 + self.s[vy] * dt

        self.s[vx] += accel_x * dt
        self.s[vy] += accel_y * dt

        tx, ty = 6, 7

        d0 = math.sqrt((x0-self.s[tx])**2 + (y0-self.s[ty])**2)
        df = math.sqrt((self.s[x]-self.s[tx])**2 + (self.s[y]-self.s[ty])**2)

        v_error = self.s[vx]**2 + self.s[vy]**2

        pen_neg = lambda x: min(10*df*x, x)

        reward = pen_neg(d0-df) + \
                1/max(1/1000, df**2) + max(7-df,5-df/5, 0)
        
        return reward, self.s