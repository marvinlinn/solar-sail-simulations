import numpy as np
from time import time
import stateCollection.spiceInterface as spice

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

class ParallelWorld:
    # states S in R^{num_sails x 12}:= (x, y, z, vx, vy, vz, 
    #             x_target, y_target, z_target, vx_target, vy_target, vz_target)

    def __init__(self):
        pass

    def reset(self):
        # return initial states S
        pass

    def advance_simulation(self, A):
        # return reward and next state (R, S)
        pass


class TrackNEO(World):

    def __init__(self):
        pass

    def reset(self):
        pass

    def advance_simulation(self, a):
        pass


class ParallelTrackNEO(ParallelWorld):

    G = 6.6743E-11 # gravitational constant

    P0 = 9E-9 #newtons per square meeter -> solar sail effectiveness
    mu_sun = 1.327e20 /1e9 # mu in km^3/s^2, sun's gravitational parameter
    AU = 1.496e11 /1e3  # astronomical unit in km, distance from sun to earth
    beta = 0.15 # ratio of peak solar sail force to sun's gravity
    # Bodies:
    # [ Sun:     mu = 1.327124400189e20 / 1e9,
    #   Mercury: mu = 2.20329e13        / 1e9,
    #   Venus:   mu = 3.248599e14       / 1e9,
    #   Earth:   mu = 3.9860044188e14   / 1e9,
    #   Mars:    mu = 4.2828372e13      / 1e9,
    #   Jupiter: mu = 1.266865349e17    / 1e9,
    #   Saturn:  mu = 3.79311879e16     / 1e9,
    #   Uranus:  mu = 5.7939399e15      / 1e9,
    #   Neptune: mu = 6.8365299e15      / 1e9 ]

    def __init__(self):
        self.time = {'get pos': 0, 'square dists': 0, 'grav accel': 0, 'sail accel': 0, 'update': 0}

        self.num_sails = 100
        dt_hours = 5
        self.dt = dt_hours * 3600

        self.bodies = {}
        self.bodies['name'] = np.array(['Sun', 'Mercury', 'Venus', 'Earth', 
                                        'Mars', 'Jupiter', 'Saturn', 'Uranus',
                                        'Neptune'])
        self.bodies['spkid'] = np.array(['10', '1', '2', '3', '4', '5', '6', 
                                         '7', '8'])
        self.bodies['mu'] = np.array([1.327124400189e20 / 1e9,
                                      2.20329e13        / 1e9,
                                      3.248599e14       / 1e9,
                                      3.9860044188e14   / 1e9,
                                      4.2828372e13      / 1e9,
                                      1.266865349e17    / 1e9,
                                      3.79311879e16     / 1e9,
                                      5.7939399e15      / 1e9,
                                      6.8365299e15      / 1e9])

        self.num_bodies = len(self.bodies['name'])
        timeObj = spice.Time(1, 1, 2000, 360)
        self.bodies['positions'] = \
                np.array([spice.requestData(spkid, timeObj, dt_hours)[0].T 
                          for spkid in self.bodies['spkid']])

        self.bodies['positions'] = np.transpose(self.bodies['positions'], 
                                                axes=(1,0,2))

        self.reset()

    def reset(self):
        self.t = 0 # Index of time. Actual time = self.dt * self.t

        self.body_pos = self.bodies['positions'][self.t:self.t+1]

        self.P = np.random.random((self.num_sails, 3)) # position
        self.V = np.zeros((self.num_sails, 3)) # velocity
        self.Pt = np.random.random((self.num_sails, 3)) # target pos
        self.Vt = np.zeros((self.num_sails, 3)) # target velocity

        return self._get_state()

    def _get_state(self):
        return np.hstack((self.P, self.V, self.Pt, self.Vt))

    def _update_body_pos(self, t):
        assert t < len(self.bodies['positions']), \
               'simulation time exceeds loaded dataset'
        self.body_pos = self.bodies['positions'][t:t+1]

    def advance_simulation(self, A):
        i = time()
        self._update_body_pos(self.t)
        sail_pos = self.P.reshape((self.num_sails, 1, 3))
        self.time['get pos'] += time()-i

        # Compute Gravity Accel
        i = time()
        r = self.body_pos - sail_pos
        r2 = np.sum(r*r, axis=2)
        self.time['square dists'] += time()-i

        i = time()
        n_accel_g = (self.bodies['mu'] / r2) # ||a_g|| TODO: check dims
        accel_g = n_accel_g.reshape(self.num_sails, self.num_bodies, 1) * r # F_g
        accel_g_total = accel_g.sum(axis=1)
        self.time['grav accel'] += time()-i

        # Compute Sail Force
        # --- # sun_norm = sail_pos / \
        # --- #         np.linalg.norm(sail_pos, axis=2)[:,np.newaxis]
        i = time()
        orbit_angle = np.arctan(sail_pos[:,:,1] / sail_pos[:,:,0])
        total_angle = A + orbit_angle
        sail_norm = np.hstack((np.cos(total_angle), 
                               np.sin(total_angle), 
                               np.zeros((self.num_sails, 1))))
        sail_dist2 = np.sum(sail_pos * sail_pos, axis=2)
        sail_accel = ParallelTrackNEO.mu_sun * ParallelTrackNEO.beta \
                / sail_dist2 * np.cos(A) * sail_norm
        self.time['sail accel'] += time()-i

        # Update Pos
        i = time()
        total_accel = accel_g_total + sail_accel
        self.P += 0.5*self.dt**2 * total_accel + self.dt * self.V
        self.V += total_accel * self.dt
        self.time['update'] += time() - i

        # Update time
        self.t += 1

        # Reward
        rewards = 1 # TODO: reward funct

        return rewards, self._get_state()

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
