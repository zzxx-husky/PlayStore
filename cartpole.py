import tensorflow as tf
import gym
import numpy as np
import random

class DQN:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.num_input = self.env.observation_space.shape
        self.num_output = self.env.action_space.n
        self.num_epoc = 50000
        self.num_step_per_epoc = 10000
        self.action_eps = 1.0
        self.action_eps_bound = 0.1
        self.action_decay = 0.001
        self.tran_capacity = 1e6
        self.tran_idx = 0
        self.train_batch_size = 100
        self.gamma = 0.9
        self.max_step = 0
        self.learning_rate = 0.001
        self.momentum = 0.5
        self.num_skip_frame = 1

        self.network()
        self.sess = tf.Session()
        self.tran_list = list()
        self.sess.run(tf.global_variables_initializer())

    def network(self):
        # relu, dropout ...
        self.input = tf.placeholder(tf.float32, (None,) + self.num_input)
        self.ground_truth = tf.placeholder(tf.float32, (None, self.num_output))

        self.fc1_w = tf.Variable(tf.zeros(self.num_input + (128,)))
        self.fc1_b = tf.Variable(tf.zeros([128]))
        self.fc1 = tf.matmul(self.input, self.fc1_w) + self.fc1_b

        self.fc2_w = tf.Variable(tf.zeros([128, 128]))
        self.fc2_b = tf.Variable(tf.zeros([128]))
        self.fc2 = tf.matmul(self.fc1, self.fc2_w) + self.fc2_b

        self.out_w = tf.Variable(tf.zeros([128, self.num_output]))
        self.out_b = tf.Variable(tf.zeros([self.num_output]))
        self.out = tf.matmul(self.fc2, self.out_w) + self.out_b

        self.Q = tf.reduce_sum(self.out, axis=0)
        self.action = tf.argmax(tf.reduce_sum(self.out, axis=0))
        self.loss = tf.square(self.ground_truth - self.out)
        self.max_output = tf.reduce_max(self.out)
        self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.loss)

    def select_action(self, obs):
        if (self.action_eps > self.action_eps_bound):
            self.action_eps -= self.action_decay
        if (random.random() >= self.action_eps):
            return self.sess.run(self.action, feed_dict={self.input: np.array([obs])})
        else:
            return random.randint(0, self.num_output - 1)

    def maxQ(self, obs):
        return self.sess.run(self.max_output, feed_dict={self.input: np.array([obs])})

    def calcQ(self, obs):
        return self.sess.run(self.Q, feed_dict={self.input: np.array([obs])})

    def ins_new_transition(self, tran):
        if (len(self.tran_list) == self.tran_capacity):
            self.tran_list[self.tran_idx] = tran
            self.tran_idx = (self.tran_idx + 1) % self.tran_capacity
        else:
            self.tran_list.append(tran)

    def mini_batch(self):
        if (len(self.tran_list) < self.train_batch_size):
            return self.tran_list
        else:
            return [self.tran_list[i]
                for i in random.sample(xrange(len(self.tran_list)), self.train_batch_size)]

    def update(self, lst, idx, val):
        lst[idx] = val
        return lst

    def train(self):
        for epoc in xrange(self.num_epoc):
            pre_obs = self.env.reset()
            self.env.render()
            for step in xrange(self.num_step_per_epoc):
                act = self.select_action(pre_obs)

                for i in xrange(self.num_skip_frame):
                    obs, r, done, info = self.env.step(act)
                    self.env.render()

                    self.ins_new_transition(((pre_obs, act, r, obs), done))
                    pre_obs = obs

                    if done:
                        break

                train = [
                    (p_obs, act, r + (0 if term else self.gamma * self.maxQ(obs)))
                    for ((p_obs, act, r, obs), term) in self.mini_batch()]

                train_gd = [self.update(self.calcQ(p_obs), act, r) for (p_obs, act, r) in train]
                train_in = [i[0] for i in train]

                self.sess.run(self.train_step, feed_dict={self.input: train_in, self.ground_truth: train_gd})

                if done:
                    self.max_step = max(self.max_step, step + 1)  
                    print "Epoc: %d, survive for %d steps, max step: %d" % (epoc + 1, step + 1, self.max_step)
                    break

if __name__ == '__main__':
    DQN('CartPole-v0').train()
    # DQN('SpaceInvaders-v0').train()
