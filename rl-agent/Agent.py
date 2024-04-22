import World
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rich.progress import track
import multiprocessing

class Agent():

    def __init__(self, world, policy, Q, learning_rate_policy=0.00001, 
                 learning_rate_Q=0.0005, decay_rate=0.9):
        self.world = world
        self.policy = policy
        self.Q = Q

        self.learning_rate_policy = learning_rate_policy
        self.learning_rate_Q = learning_rate_Q
        self.decay_rate = decay_rate
        
        self.p_optimizer = tf.keras.optimizers.Adam()
        self.q_optimizer = tf.keras.optimizers.Adam()

    def train(self, max_duration, episodes_per_epoch, epochs):
        pass

    def training_step(self, max_duration):
        pass


class SyncAgent(Agent):

    def __init__(self, world, policy, Q, learning_rate_policy=0.00001, 
                 learning_rate_Q=0.0005, decay_rate=0.9):
        super().__init__(world, policy, Q, learning_rate_policy, 
                         learning_rate_Q, decay_rate)

    def train(self, max_duration, episodes_per_epoch, epochs):
        r_mean = 0
        states = None
        for epoch in range(episodes_per_epoch):
            c_prev = -1
            for cycle in track(range(episodes_per_epoch), description=f'Epoch {epoch}'):
                r_mean, states = self.training_step(max_duration)
                c_prev = cycle
            print(f'epoch {epoch}: Mean Reward = {r_mean}')
        return states

    def _apply_policy(self, x):
        x = np.expand_dims(x, axis=0)
        mu, sigma = self.policy(x)
        normal = tfp.distributions.Normal(mu, sigma)
        sample = tf.squeeze(normal.sample(1), axis=0)
        sample_clipped = tf.clip_by_value(sample, 
                                          self.policy.min_action, 
                                          self.policy.max_action)
        raw_out = sample_clipped.numpy()[0][0]
        return normal, sample, raw_out

    def training_step(self, max_duration):
        expand_Q = lambda t: (t, t.numpy()[0][0])
        apply_Q = lambda x: expand_Q(self.Q(np.expand_dims(x, axis=0)))
        
        r_total = 0
        states = [None for _ in range(max_duration)]
        s = self.world.reset()
        with tf.GradientTape() as policy_tape:
            dist, a, a_raw = self._apply_policy(s)
            log_prob = tf.reduce_mean(dist.log_prob(a))

        
        for t in range(max_duration):
            # sample reward and get next state
            states[t] = np.array(s)
            r, s_next = self.world.advance_simulation(a_raw)
            r_total += r

            # sample next action from policy
            with tf.GradientTape() as policy_tape_next:
                dist_next, a_next, a_next_raw = self._apply_policy(s)
                log_prob_next = tf.reduce_mean(dist.log_prob(a_next))
            # get Q(s,a)
            sa = np.append(s,a)
            with tf.GradientTape() as Q_tape:
                Q_sa, Q_sa_raw = apply_Q(sa)

            # update policy parameters theta += alpha * Q(s,a) * grad log pi(a|s)
            policy_grads = policy_tape.gradient(log_prob, self.policy.trainable_variables)
            # print(policy_grads[0].numpy()[0][0])
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

        print(f'mean: {r_total/max_duration}, r: {r_total}, md: {max_duration}')
        return r_total/max_duration, states

class AsyncAgent(Agent):

    def __init__(self, world, policy, Q, learning_rate_policy=0.00001, 
                 learning_rate_Q=0.0005, decay_rate=0.9):
        super().__init__(world, policy, Q, learning_rate_policy, 
                         learning_rate_Q, decay_rate)

    
    def _policy_server(self, senders, receivers):
            active_connections = [True for i in range(len(senders))]
            num_active = len(senders)
            while num_active > 0:
                for i, rec in enumerate(receivers):
                    if not active_connections[i]:
                        # print('NOT ACTIVE')
                        continue
                    if rec.poll():
                        data = rec.recv()
                        if data == 'COMPLETE':
                            # print(f'Task {i} Complete!')
                            active_connections[i] = False
                            num_active -= 1
                            # print(num_active)
                        else:
                            res = self._sample_policy(data)
                            senders[i].send(res)


    def train(self, max_duration, episodes_per_epoch, epochs):
        available_threads = multiprocessing.cpu_count()
        proc_count = min(available_threads, episodes_per_epoch)

        to_pipes = [multiprocessing.Pipe() for _ in range(episodes_per_epoch)]
        from_pipes = [multiprocessing.Pipe() for _ in range(episodes_per_epoch)]
        pipes = zip(to_pipes, from_pipes)

        res_q = multiprocessing.Queue()
        episodes_async = [(ps, pr, 
                           multiprocessing.Process(target=self._sar_list, 
                                                   args=(max_duration, res_q, cs, cr))) 
                          for (ps, cr), (cs, pr) in pipes]

        senders, receivers, results = zip(*episodes_async)

        for res in results:
            res.start()

        self._policy_server(senders, receivers)

        for res in results:
            res.join()
            print("JOINED")

        results = [res_q.get() for _ in results]

        for s, a, r in results:
            # compute Q(s,a) for batch
            with tf.GradientTape() as Q_tape:
                Qsa = self.Q(np.hstack((s, a)))
            
            # compute log prob policy(a|s)
            with tf.GradientTape() as policy_tape:
                mu, sigma = self.policy(s[:-1])
                normal = tfp.distributions.Normal(mu, sigma)
                log_prob = normal.log_prob(a[:-1])
                policy_grad_target = \
                        self.learning_rate_policy * Qsa[:-1] * log_prob

            # update policy params
            policy_grads = policy_tape.gradient(policy_grad_target, 
                                                self.policy.trainable_variables)

            # compute td error for batch
            td_error = (r + self.decay_rate * (Qsa[1:] - Qsa[:-1])).numpy()

            # update Q function
            with Q_tape:
                Q_grad_target = self.learning_rate_Q * td_error * Qsa[:-1]
            Q_grads = Q_tape.gradient(Q_grad_target, self.Q.trainable_variables)
            self.q_optimizer.apply_gradients(zip(Q_grads, self.Q.trainable_variables))
            print(f'mean reward: {np.mean(r)}')

    def _sample_policy(self, x):
        x = np.expand_dims(x, axis=0)
        mu, sigma = self.policy(x)
        normal = tfp.distributions.Normal(mu, sigma)
        sample = tf.squeeze(normal.sample(1), axis=0)
        return sample.numpy()[0][0]

    def _sar_list(self, max_duration, res_q, sender, rec):
        states = []
        actions = []
        rewards = []

        s = self.world.reset()

        sender.send(s)
        a = rec.recv()
        
        for t in range(max_duration):
            states.append(s)
            actions.append(a)

            # sample reward and get next state
            r, s_next = self.world.advance_simulation(a)
            rewards.append(r)

            # sample next action from policy
            sender.send(s)
            a_next = rec.recv()

            # advance s and a
            a = a_next
            s = s_next

        states.append(s)
        actions.append(a)

        states = np.array(states)
        actions = np.array(actions).reshape(-1,1)
        rewards = np.array(rewards).reshape(-1,1)

        sender.send('COMPLETE')

        res_q.put((states, actions, rewards))
