import os
import numpy as np
import pandas as pd
import tensorflow_probability as tfp
import tensorflow as tf

from rich.progress import track
import multiprocessing
from multiprocessing import Process

class Agent():

    def __init__(self, world, policy, Q, learning_rate_policy=0.00001, 
                 learning_rate_Q=0.0005, decay_rate=0.9):
        self.world = world
        self.policy = policy
        self.Q = Q

        self.learning_rate_policy = learning_rate_policy
        self.learning_rate_Q = learning_rate_Q
        self.decay_rate = decay_rate
        
        self.p_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_policy)
        self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_Q)

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
            td_error = r + self.decay_rate * Q_sa_next_raw - Q_sa_raw

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

class ParallelAgent(Agent):

    def __init__(self, world, policy, Q, learning_rate_policy=0.00001,
                 learning_rate_Q=0.0005, decay_rate=0.9):
        super().__init__(world, policy, Q, learning_rate_policy, 
                         learning_rate_Q, decay_rate)

    def train(self, max_duration, episodes_per_epoch, epochs, added_uncertainty=0):
        r_mean = 0
        states = None
        for epoch in range(epochs):
            self.policy.save_weights('./checkpoints/policy.weights.h5')
            self.Q.save_weights('./checkpoints/Q.weights.h5')
            for cycle in track(range(episodes_per_epoch), 
                               description=f'Epoch {epoch}'):
                u = added_uncertainty
                if callable(added_uncertainty):
                    u = added_uncertainty(epoch, cycle)
                r_mean, states = self.training_step(max_duration, 
                                                    added_uncertainty=u)
            print(f'epoch {epoch}: Mean Reward = {np.mean(r_mean)}')
        return states

    def training_step(self, max_duration, added_uncertainty=0):
        R_total = np.zeros((self.world.num_sails, 1))
        all_S = []
        
        year = np.random.randint(2000, 2020)
        month = np.random.randint(1,12)
        day = np.random.randint(1,28)
        S = self.world.reset(new_t0=(month, day, year))

        with tf.GradientTape() as policy_tape:
            mu, sigma = self.policy(S)
            normal = tfp.distributions.Normal(mu, sigma)
            A = normal.sample()
            log_prob = normal.log_prob(A)

        # get Q(s,a)
        SA = np.hstack((S, A))
        with tf.GradientTape() as Q_tape:
            Q_SA = self.Q(SA)

        
        for t in range(max_duration):
            # sample reward and get next state
            R, S_next = self.world.advance_simulation(A)

            # Totals
            R_total += R
            all_S.append(S)

            # sample next action from policy
            with tf.GradientTape() as policy_tape_next:
                mu, sigma = self.policy(S)
                sigma += added_uncertainty
                normal = tfp.distributions.Normal(mu, sigma)
                A_next = normal.sample()
                log_prob_next = normal.log_prob(A_next)

            # update policy parameters theta += alpha * Q(s,a) * grad log pi(a|s)
            with policy_tape:
                policy_grad_target = -Q_SA * log_prob
            policy_grads = policy_tape.gradient(policy_grad_target, 
                                                self.policy.trainable_variables)

            if not any([tf.math.reduce_any(tf.math.is_nan(a)) for a in policy_grads]):
                self.p_optimizer.apply_gradients(zip(policy_grads, 
                                                     self.policy.trainable_variables))

            # compute TD error
            SA_next = np.hstack((S_next, A_next))
            with tf.GradientTape() as Q_tape_next:
                Q_SA_next = self.Q(SA_next)
            td_error = R + self.decay_rate * Q_SA_next - Q_SA
            with Q_tape:
                Q_grad_target = -td_error.numpy() * Q_SA # TODO: confirm td error not differentiated
            Q_grads = Q_tape.gradient(Q_SA, self.Q.trainable_variables)

            # update weights of Q: w += alpha * delta * grad Q(s,a)
            if not any([tf.math.reduce_any(tf.math.is_nan(a)) for a in Q_grads]):
                self.q_optimizer.apply_gradients(zip(Q_grads, 
                                                     self.Q.trainable_variables))

            # advance s and a TODO: confirm all vars prop correctly
            # a, a_raw, log_prob = a_next, a_next_raw, log_prob_next
            A, log_prob = A_next, log_prob_next
            S = S_next

            del policy_tape
            policy_tape = policy_tape_next

            Q_SA = Q_SA_next
            
            del Q_tape
            Q_tape = Q_tape_next

        r_total = np.sum(R_total, axis=1)
        print(f'mean: {np.mean(r_total/max_duration)}, r: {np.mean(r_total)}, md: {max_duration}')
        return r_total/max_duration, all_S

    def _pretrain_SAR_iterator(self, dirs, batch_size, load_threshold=2):
        # TODO: untested! also maybe shuffle? NEED EVAL METRIC FOR OVERFIT
        def get_SAR(file, target_df, queue=None):
            df = pd.read_csv(file, delimiter='|')
            print(df)
            print(queue)
            S = np.copy(np.array(df[['sail x', 'sail y', 'sail z', 
                             'sail Vx', 'sail Vy', 'sail Vz']]))
            A = np.copy(np.array(df[['yaw']]))
            R = 1 / np.array(df[['abs distance']])
            res = (S,A,R)
            if queue is not None:
                queue.put(res)
            else:
                return res
        
        def SAR_batches(S,A,R,p=None):
            no_batches = -(len(S) // -batch_size)
            for i in range(no_batches):
                if p is not None and \
                  p[0] is None and \
                  i >= no_batches - load_threshold:
                    p[0] = Process(target=get_SAR, 
                                   args=(dir + file, target_df), 
                                   kwargs={'queue': queue})
                    p[0].start()
                    print(p[0])

                end = min(i+batch_size, len(S))
                yield S[i:end], A[i:end], R[i:end]

        queue = multiprocessing.Queue(maxsize=2)
        p = [None]

        for dir in dirs:
            target_df = pd.read_csv(dir + 'target.csv', delimiter='|')
            files = os.listdir(dir)
            for file_i, file in enumerate(files):
                if file == 'target.csv':
                    continue
                if p[0] is None:
                    S, A, R = get_SAR(dir + file, target_df)
                else:
                    # if p[0].is_alive():
                    #     print('SAR not ready! increasing load threshold')
                    #     load_threshold += 1
                    print(p[0])
                    p[0].join()
                    print('ASJLJSDALKJ')
                    S, A, R = queue.get()
                    p[0] = None

                yield from SAR_batches(S,A,R, p=p)

            if p[0] is not None:
                p[0].join()
                S, A, R = queue.get()
                yield from SAR_batches(S,A,R)

    def pretrain(self, batch_size=1024, epochs=5):
        for epoch in range(epochs):
            for s, a, r in self._pretrain_SAR_iterator('', batch_size):
                # compute Q(s,a) for batch
                with tf.GradientTape() as Q_tape:
                    Qsa = self.Q(np.hstack((s, a)))
                
                # compute log prob policy(a|s)
                with tf.GradientTape() as policy_tape:
                    mu, sigma = self.policy(s[:-1])
                    normal = tfp.distributions.Normal(mu, sigma)
                    log_prob = normal.log_prob(a[:-1])
                    policy_grad_target = -Qsa[:-1] * log_prob

                # update policy params
                policy_grads = policy_tape.gradient(policy_grad_target, 
                                                    self.policy.trainable_variables)

                # compute td error for batch
                td_error = (r + self.decay_rate * (Qsa[1:] - Qsa[:-1])).numpy()

                # update Q function
                with Q_tape:
                    Q_grad_target = -td_error * Qsa[:-1]
                Q_grads = Q_tape.gradient(Q_grad_target, self.Q.trainable_variables)
                self.q_optimizer.apply_gradients(zip(Q_grads, self.Q.trainable_variables))
        

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
