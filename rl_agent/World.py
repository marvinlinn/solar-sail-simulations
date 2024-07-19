import matplotlib.pyplot as plt
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

    def __init__(self, num_sails=50, dt=5, control_interval=5):
        self.time = {'get pos': 0, 'square dists': 0, 'grav accel': 0, 'sail accel': 0, 'update': 0}

        self.orbit_dist_prev = None

        self.num_sails = num_sails
        self.dt_hours = dt
        self.dt = self.dt_hours * 3600
        self.control_interval=control_interval

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

        # TODO: REMOVE THIS BLOCK TO CONSIDER ALL PLANETS
        selected = ['Sun', 'Mars']
        indices = [i for i in range(len(self.bodies['name'])) 
                   if self.bodies['name'][i] in selected]
        self.bodies = {key: self.bodies[key][indices] for key in self.bodies.keys()}
        # -----------------------------------------------


        self.num_bodies = len(self.bodies['name'])
        timeObj = spice.Time(1, 1, 2000, 360) # jan 01, 2000, 360 days
        self.bodies['positions'] = \
                np.array([spice.requestData(spkid, timeObj, self.dt_hours)[0].T 
                          for spkid in self.bodies['spkid']])

        self.bodies['positions'] = np.transpose(self.bodies['positions'], 
                                                axes=(1,0,2))

        # TODO: SKETCHY DEFAULT INITIAL CONDITIONS FROM TEST.PY
        earth_pos = spice.requestData('3', timeObj, self.dt_hours)[0].T
        self.init_pos = earth_pos[0]
        normalize = lambda a: a / np.linalg.norm(a)
        init_vel_hat = normalize(earth_pos[1]-earth_pos[0])
        self.init_vel = init_vel_hat * 30
        # -----------------------------------------------------

        # TODO: SKETCHY TARGET DEFINITION:
        target_name = 'Mars'
        self.target_body = [i for i in range(self.num_bodies) if self.bodies['name'][i] == target_name][0]
        # -----------------------------------------------------

        # TODO: SKETCHY 2D plane finding
        A = earth_pos
        b = np.ones((earth_pos.shape[0],1))
        x, residuals, rank, s = np.linalg.lstsq(A,b, rcond=None)
        x = np.reshape(x, (-1))

        self.planet_plane = x / np.linalg.norm(x)
        a = self.planet_plane
        self.pp_cross_matrix = np.cross(a, -1*np.identity(3))
        self.pp_outer_prod = np.outer(a, a)
        
        # -----------------------------------------------------

        self.reset()


    def reset(self, new_t0=None):
        self.time = {'get pos': 0, 'square dists': 0, 'grav accel': 0, 'sail accel': 0, 'update': 0}

        if new_t0 is not None:
            timeObj = spice.Time(*new_t0, 360) # jan 01, 2000, 360 days
            self.bodies['positions'] = \
                    np.array([spice.requestData(spkid, timeObj, self.dt_hours)[0].T 
                              for spkid in self.bodies['spkid']])
    
            self.bodies['positions'] = np.transpose(self.bodies['positions'], 
                                                    axes=(1,0,2))
    
            # TODO: SKETCHY DEFAULT INITIAL CONDITIONS FROM TEST.PY
            earth_pos = spice.requestData('3', timeObj, self.dt_hours)[0].T
            self.init_pos = earth_pos[0]
            normalize = lambda a: a / np.linalg.norm(a)
            init_vel_hat = normalize(earth_pos[1]-earth_pos[0])
            self.init_vel = init_vel_hat * 30
        
        self.t = 0 # Index of time. Actual time = self.dt * self.t

        self._update_body_pos(self.t)

        self.mean_target_r = np.mean(np.linalg.norm(
            self.bodies['positions'][:,self.target_body], axis=1))

        self.P = np.tile(self.init_pos, (self.num_sails, 1)) # (num_sails, 3)
        self.V = np.tile(self.init_vel, (self.num_sails, 1)) # (num_sails, 3)

        self.Pt = np.tile(self.body_pos[self.target_body], 
                          (self.num_sails, 1)) # target pos
        self.Vt = np.tile((self.bodies['positions'][self.t+1][self.target_body] 
                           - self.bodies['positions'][self.t][self.target_body]) 
                          / self.dt, (self.num_sails, 1)) # target velocity
        
        return self._get_state()

    def max_sim_length(self):
        return len(self.bodies['positions']) - 1

    def _get_state(self):
        return np.hstack((self.P, self.V, self.Pt, self.Vt)) / ParallelTrackNEO.AU

    def _update_body_pos(self, t):
        assert t < self.max_sim_length(), \
               'simulation time exceeds loaded dataset'
        self.body_pos = self.bodies['positions'][t]

        self.Pt = np.tile(self.body_pos[self.target_body], 
                          (self.num_sails, 1)) # target pos
        self.Vt = np.tile((self.bodies['positions'][self.t+1][self.target_body] 
                           - self.bodies['positions'][self.t][self.target_body]) 
                          / self.dt, (self.num_sails, 1)) # target velocity


    def advance_simulation(self, A):
        for _ in range(self.control_interval):
            control_angle = A[:,0:1]
            
            i = time()
            self._update_body_pos(self.t)
            sail_pos = self.P.reshape((self.num_sails, 1, 3))
            self.time['get pos'] += time()-i

            # Compute Gravity Accel
            i = time()
            r = self.body_pos - sail_pos
            r2 = np.sum(r*r, axis=2)
            r_hat = r / np.linalg.norm(r, axis=2)[:,:,np.newaxis]
            self.time['square dists'] += time()-i

            i = time()
            n_accel_g = (self.bodies['mu'] / r2) # ||a_g|| TODO: check dims
            accel_g = n_accel_g.reshape(self.num_sails, self.num_bodies, 1) * r_hat # F_g
            accel_g_total = accel_g.sum(axis=1)
            self.time['grav accel'] += time()-i

            # Compute Sail Force
            cos_angle = np.cos(control_angle)
            cos_angle_extd = cos_angle[:,:,np.newaxis]
            sin_angle = np.sin(control_angle)[:,:,np.newaxis]
            rotation_matrix = cos_angle_extd * np.identity(3) + \
                              sin_angle * self.pp_cross_matrix + \
                              (1 - cos_angle_extd) * self.pp_outer_prod
            sail_norm = sail_pos / np.linalg.norm(sail_pos, axis=2)[:,np.newaxis]

            # equivalent to finding A_i @ b_i for all i. elementwise matrix product
            # equivalent to [rotation_matrix[i] @ sail_norm[i][0] for i in range(num_sails)]
            sail_norm = np.einsum('ijk,i...k->ij', rotation_matrix, sail_norm, optimize=True)        
            sail_dist2 = np.sum(sail_pos * sail_pos, axis=2)
            sail_accel = ParallelTrackNEO.mu_sun * ParallelTrackNEO.beta \
                    / sail_dist2 * cos_angle * sail_norm
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
        r_scalar = np.sqrt(r2)
        reward_tdist = np.exp(-10/ParallelTrackNEO.AU * \
            r_scalar[:,self.target_body:self.target_body+1])

        orbit_dist = (r_scalar[:,0:1] - self.mean_target_r) / ParallelTrackNEO.AU

        if self.orbit_dist_prev is not None:
            reward_orbit_error = orbit_dist - self.orbit_dist_prev
        else:
            reward_orbit_error = np.zeros(orbit_dist.shape)
        reward_orbit_error[reward_orbit_error > 0] *= 10
        reward_orbit_error[reward_orbit_error < 0] *= 5
        reward_orbit_error = np.tanh(reward_orbit_error)

        self.orbit_dist_prev = orbit_dist

        rewards = reward_tdist - reward_orbit_error

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
