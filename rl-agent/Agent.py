import World
import numpy as np
import tensorflow as tf
from rich.progress import track

class Agent:
    def __init__(self, world, policy, Q, learning_rate_policy=0.00001, learning_rate_Q=0.0005, decay_rate=0.9):
        self.world = world
        self.policy = policy
        self.Q = Q

        self.learning_rate_policy = learning_rate_policy
        self.learning_rate_Q = learning_rate_Q
        self.decay_rate = decay_rate
        
        self.p_optimizer = tf.keras.optimizers.Adam()
        self.q_optimizer = tf.keras.optimizers.Adam()

    def train(self, max_cycle, cycles_per_epoch, epochs):
        r_mean = 0
        for epoch in range(cycles_per_epoch):
            c_prev = -1
            for cycle in track(range(cycles_per_epoch), description=f'Epoch {epoch}'):
                r_mean = self.training_step(max_cycle)
                c_prev = cycle
            print(f'epoch {epoch}: Mean Reward = {r_mean}')

    def training_step(self, max_cycle):
        expand_policy = lambda t: (t[0], t[1], t[2].numpy()[0][0])
        apply_policy = lambda x: expand_policy(self.policy(np.expand_dims(x, axis=0)))
        expand_Q = lambda t: (t, t.numpy()[0][0])
        apply_Q = lambda x: expand_Q(self.Q(np.expand_dims(x, axis=0)))
        
        r_total = 0
        s = self.world.reset()
        with tf.GradientTape() as policy_tape:
            dist, a, a_raw = apply_policy(s)
            log_prob = tf.reduce_mean(dist.log_prob(a))

        
        for t in range(max_cycle):
            # sample reward and get next state
            r, s_next = self.world.advance_simulation(a)
            r_total += r

            # sample next action from policy
            with tf.GradientTape() as policy_tape_next:
                dist_next, a_next, a_next_raw = apply_policy(s)
                log_prob_next = tf.reduce_mean(dist.log_prob(a_next))
            # get Q(s,a)
            sa = np.append(s,a)
            with tf.GradientTape() as Q_tape:
                Q_sa, Q_sa_raw = apply_Q(sa)

            # update policy parameters theta += alpha * Q(s,a) * grad log pi(a|s)
            policy_grads = policy_tape.gradient(log_prob, self.policy.trainable_variables)
            policy_update = [self.learning_rate_policy * Q_sa_raw * grad for grad in policy_grads]
            self.p_optimizer.apply_gradients(zip(policy_update, self.policy.trainable_variables))

            # compute TD error
            sa_next = np.append(s_next, a_next)
            Q_sa_next, Q_sa_next_raw = apply_Q(sa_next)
            td_error = r * self.decay_rate * Q_sa_next_raw - Q_sa_raw

            # update weights of Q: w += alpha * delta * grad Q(s,a)
            Q_grads = Q_tape.gradient(Q_sa, self.Q.trainable_variables)
            Q_update = [self.learning_rate_Q * td_error * grad for grad in Q_grads]
            self.q_optimizer.apply_gradients(zip(Q_update, self.Q.trainable_variables))

            # advance s and a
            a, a_raw, log_prob = a_next, a_next_raw, log_prob_next
            dist = dist_next
            s = s_next
            policy_tape = policy_tape_next

        return r_total/max_cycle
