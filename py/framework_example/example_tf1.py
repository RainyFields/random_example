import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import seaborn as sns
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
# get_ipython().magic(u'matplotlib inline')
# from helper import *

from random import choice
from time import sleep
from time import time

# ### Helper Functions

# %% In[2]:

isreversal = 'no_reversal'
LSTM_num = 48
stim_num = 8
target_num = 6
episode_num = 1000
MAX_hist = int(100*episode_num*2/stim_num)

import scipy.misc
import pandas as pd

data = {'episode_count':[],
        'trial_index':[],
        'target_index':[],
        'stim_index':[],
        'reward': [],
        'action': [],
        'prev_reward':[],
        'prev_action':[],
        'pre_post': [],
        'neuron_1':[]}
df_data = pd.DataFrame(data, columns = ['trial_index','target_index', 'stim_index','reward','action','prev_reward','prev_action','pre_post','neuron_1'])

for i in range(2,49):
    col_name = "neuron_%d"%i
    df_data[col_name] = [0]


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


# This code allows gifs to be saved of the training episode for use in the Control Center.
def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration, verbose=False)


def set_image_bandit(values, probs, selection, trial):
    bandit_image = Image.open('./resources/bandit.png')
    draw = ImageDraw.Draw(bandit_image)
    font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
    draw.text((40, 10), str(float("{0:.2f}".format(probs[0]))), (0, 0, 0), font=font)
    draw.text((130, 10), str(float("{0:.2f}".format(probs[1]))), (0, 0, 0), font=font)
    draw.text((60, 370), 'Trial: ' + str(trial), (0, 0, 0), font=font)
    bandit_image = np.array(bandit_image)
    bandit_image[115:115 + floor(values[0] * 2.5), 20:75, :] = [0, 255.0, 0]
    bandit_image[115:115 + floor(values[1] * 2.5), 120:175, :] = [0, 255.0, 0]
    bandit_image[101:107, 10 + (selection * 95):10 + (selection * 95) + 80, :] = [80.0, 80.0, 225.0]
    return bandit_image


def set_image_context(correct, observation, values, selection, trial):
    obs = observation * 225.0
    obs_a = obs[:, 0:1, :]
    obs_b = obs[:, 1:2, :]
    cor = correct * 225.0
    obs_a = scipy.misc.imresize(obs_a, [100, 100], interp='nearest')
    obs_b = scipy.misc.imresize(obs_b, [100, 100], interp='nearest')
    cor = scipy.misc.imresize(cor, [100, 100], interp='nearest')
    bandit_image = Image.open('./resources/c_bandit.png')
    draw = ImageDraw.Draw(bandit_image)
    font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
    draw.text((50, 360), 'Trial: ' + str(trial), (0, 0, 0), font=font)
    draw.text((50, 330), 'Reward: ' + str(values), (0, 0, 0), font=font)
    bandit_image = np.array(bandit_image)
    bandit_image[120:220, 0:100, :] = obs_a
    bandit_image[120:220, 100:200, :] = obs_b
    bandit_image[0:100, 50:150, :] = cor
    bandit_image[291:297, 10 + (selection * 95):10 + (selection * 95) + 80, :] = [80.0, 80.0, 225.0]
    return bandit_image


def set_image_gridworld(frame, color, reward, step):
    a = scipy.misc.imresize(frame, [200, 200], interp='nearest')
    b = np.ones([400, 200, 3]) * 255.0
    b[0:200, 0:200, :] = a
    b[200:210, 0:200, :] = np.array(color) * 255.0
    b = Image.fromarray(b.astype('uint8'))
    draw = ImageDraw.Draw(b)
    font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
    draw.text((40, 280), 'Step: ' + str(step), (0, 0, 0), font=font)
    draw.text((40, 330), 'Reward: ' + str(reward), (0, 0, 0), font=font)
    c = np.array(b)
    return c


# see if I can adapt it to a different paradigm
class dependent_bandit_PT():
    def __init__(self, difficulty=None):
        self.num_actions = 2  # number of objects

        self.stim_nfeat = 2
        self.stim_ndim = 3
        self.maxstim = np.power(self.stim_nfeat, self.stim_ndim)
        self.multiplier = np.power(self.stim_nfeat * np.ones((1, self.stim_ndim)),
                                   np.arange(self.stim_ndim)).transpose()
        self.trial_num = 100
        self.reset()

    def set_restless_prob(self):
        pass

    #  without reversal version
    def reset(self):
        self.timestep = 0
        bandit_prob = 1.0  # make it really easy by being deterministic
        self.bandit = np.array([bandit_prob, 1 - bandit_prob])
        self.target_dim = np.random.randint(0, self.stim_ndim)
        self.target_val = np.float32(np.random.randint(0, self.stim_nfeat))  # for comparisons
        # self.second_target_val = np.float32(np.random.randint(0, self.stim_nfeat))
        # self.trial_num = np.random.randint(30, 70)

    def pullArm(self, action, stimconfig):
        # Get a random number.
        self.timestep += 1
        chosen_stim = stimconfig[:, action]
        # print(chosen_stim)

        if chosen_stim[self.target_dim] == self.target_val:
            #   print(chosen_stim[self.target_dim])
            bandit = self.bandit[0]
        else:
            bandit = self.bandit[1]

        result = np.random.uniform()
        if result < bandit:
            # return a positive reward.
            reward = 1
        else:
            # return a negative reward.
            reward = 0
        if self.timestep > 100:
            done = True
        else:
            done = False
        return reward, done, self.timestep


    # def reset(self):
    #     self.timestep = 0
    #     bandit_prob = 1.0  # make it really easy by being deterministic
    #     self.bandit = np.array([bandit_prob, 1 - bandit_prob])
    #     self.target_dim = np.random.randint(0, self.stim_ndim)
    #     self.target_val = np.float32(np.random.randint(0, self.stim_nfeat))  # for comparisons
    #     self.second_target_val = np.float32(np.random.randint(0, self.stim_nfeat))
    #     self.trial_num = np.random.randint(30, 70)
    #
    # def pullArm(self, action, stimconfig):
    #     # Get a random number.
    #     self.timestep += 1
    #     chosen_stim = stimconfig[:, action]
    #     # print(chosen_stim)
    #     if self.timestep < self.trial_num:
    #         if chosen_stim[self.target_dim] == self.target_val:
    #             #   print(chosen_stim[self.target_dim])
    #             bandit = self.bandit[0]
    #         else:
    #             bandit = self.bandit[1]
    #     else:  # assumpe both target val has the same reward for same feature
    #         if chosen_stim[self.target_dim] == self.second_target_val:
    #             #   print(chosen_stim[self.target_dim])
    #             bandit = self.bandit[0]
    #         else:
    #             bandit = self.bandit[1]
    #     result = np.random.uniform()
    #     if result < bandit:
    #         # return a positive reward.
    #         reward = 1
    #     else:
    #         # return a negative reward.
    #         reward = 0
    #     if self.timestep > 59:
    #         done = True
    #     else:
    #         done = False
    #     return reward, done, self.timestep

    def dec2bin(self, value):
        if self.stim_ndim == 3:  # make this more general
            formatstr = '03b'
            x = format(value, formatstr)

        return [int(char) for char in x]

    def CreateStim(self):
        stimnum = np.random.randint(0, self.maxstim)
        # print(stimnum)
        stimconfig = np.zeros((self.stim_ndim, 2))
        stimconfig[:, 0] = self.dec2bin(stimnum)
        stimconfig[:, 1] = 1 - stimconfig[:, 0]
        # we can add contrast by adding 1 (so 1 and 2)
        # or we can make representation of hot one coding
        # of both stims i.e. twice times 8 in this case
        stimid = np.int32(np.sum(stimconfig * self.multiplier, axis=0))
        # stimhot=np.zeros((self.maxstim,2))
        # stimhot[stimid[0],0]=1
        # stimhot[stimid[1],1]=1
        # we can also just give one stim, because the second one is redundant
        stimhot = np.zeros((self.maxstim, 1))
        stimhot[stimid[0], 0] = 1
        return stimconfig, stimhot


# %% test new stimulus generator


# ### Actor-Critic Network

# In[3]:


class AC_Network_PT():
    def __init__(self, a_size, scope, trainer, stimdim=6):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32)
            # stimdim=6  #number of dimensions x number of objects -- other encodings are possible
            # first see how we need to do this
            self.stimconfig = tf.placeholder(shape=[None, stimdim], dtype=tf.float32)
            self.timestep = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.prev_actions_onehot = tf.one_hot(self.prev_actions, a_size, dtype=tf.float32)

            hidden = tf.concat([self.stimconfig, self.prev_rewards, self.prev_actions_onehot, self.timestep], 1)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(48, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.prev_rewards)[:1]
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 48])

# check the value of rnn_out
            self.rnn_out = rnn_out


            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.05

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 50.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


# ### Worker Agent


class Worker_PT():
    def __init__(self, game, name, a_size, trainer, model_path, global_episodes, stimdim):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
# not reversal
        # self.episode_prevrl_rewards = []
        # self.episode_postrl_rewards = []

        #self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))
# try self.summary_writer for the testing phase
        self.summary_writer = tf.summary.FileWriter('%s_test_' % isreversal+ str(self.number))
        # self.summary_writer = tf.train.SummaryWriter("train_"+str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network_PT(a_size, self.name, trainer, stimdim)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        actions = rollout[:, 0]
        rewards = rollout[:, 1]

        timesteps = rollout[:, 2]
# not reversal
        # prev_rewards = [0] + rewards[:-1].tolist()
        # prev_actions = [0] + actions[:-1].tolist()
        values = rollout[:, 4]
        star = rollout[:, 5:]
        # print(star)
# not reversal
#         self.pr = prev_rewards
#         self.pa = prev_actions

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.prev_rewards: np.vstack(prev_rewards),
                     self.local_AC.prev_actions: prev_actions,
                     self.local_AC.actions: actions,
                     self.local_AC.stimconfig: np.vstack(star),
                     self.local_AC.timestep: np.vstack(timesteps),
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n


    def work(self, gamma, sess, coord, saver, train):
        episode_count = sess.run(self.global_episodes)
        old_episode_count = episode_count
        total_steps = 0
        counts = np.zeros((LSTM_num, stim_num,1))


        global df_data

        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []

                episode_reward = [0, 0]
                episode_step_count = 0
                d = False
                r = 0
                a = 0
                t = 0
                prev_action = a
                prev_reward = r
                self.env.reset()
                rnn_state = self.local_AC.state_init


                while d == False:
                    # Take an action using probabilities from policy network output.
                    stimc, stimhot = self.env.CreateStim()
                    # st=np.ndarray.flatten(stimc)
                    st = np.ndarray.flatten(stimhot)



                    a_dist, v, rnn_state_new, rnn_out, stimconfig = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out, self.local_AC.rnn_out, self.local_AC.stimconfig],
                        feed_dict={
                            self.local_AC.prev_rewards: [[r]],
                            self.local_AC.timestep: [[t]],
                            self.local_AC.prev_actions: [a],
                            self.local_AC.stimconfig: [st],
                            self.local_AC.state_in[0]: rnn_state[0],
                            self.local_AC.state_in[1]: rnn_state[1]})



                    _, stim_index = np.where(stimconfig == 1)
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    rnn_state = rnn_state_new
                    r, d, t = self.env.pullArm(a, stimc)
                    episode_buffer.append(np.hstack((a, r, t, d, v[0, 0], st)))
                    episode_values.append(v[0, 0])
                    episode_reward[a] += r
                    total_steps += 1




                    episode_step_count += 1

                    summary = tf.Summary()
                    summary.value.add(tag='Regular/episode_count', simple_value = float(episode_count))
                    summary.value.add(tag='Regular/step_count', simple_value = float(episode_step_count))
                    summary.value.add(tag='Regular/stimulu_type', simple_value = float(stim_index))

                    # store lstm cell response for each stimulus pattern
                    # if r!= 0:
                    #     for lstm_num in range(48):
                    #         tag_name = 'LSTM/lstm_number %.1f'% lstm_num
                    #         summary.value.add(tag = tag_name, simple_value = float(rnn_out[0,lstm_num]))
                    #         count = np.where(lstm_activity[lstm_num,int(stim_index),:]==0)[0][0]
                    #         lstm_activity[lstm_num, int(stim_index),int(count)] = float(rnn_out[0,lstm_num])
                    if self.env.target_dim == 0:
                        if self.env.target_val == 0:
                            target_index = 0
                        else:

                            target_index = 1
                    elif self.env.target_dim == 1:
                        if self.env.target_val == 0:

                            target_index = 2
                        else:

                            target_index = 3
                    elif self.env.target_dim == 2:
                        if self.env.target_val == 0:

                            target_index = 4
                        else:

                            target_index = 5
                    # ['trial_index', 'target_index', 'stim_index', 'reward', 'action', 'pre_post', 'neuron_1']
                    if episode_step_count < self.env.trial_num:
                        temp = 0
                    else:
                        temp = 1
                    new_row = {
                        'episode_count': episode_count - old_episode_count,
                        'trial_index': episode_step_count,
                        'target_index': target_index,
                        'stim_index': int(stim_index),
                        'reward': r,
                        'action': a,
                        'prev_reward':prev_reward,
                        'prev_action':prev_action,
                        'pre_post': temp

                    }



                    for i in range(1,49):
                        col_name = 'neuron_%d'%i
                        new_row[col_name] = float(rnn_out[0,i-1])

                    df_data = df_data.append(new_row, ignore_index = True)



                    prev_action = a
                    prev_reward = r
                    self.summary_writer.add_summary(summary, total_steps)

                    self.summary_writer.flush()

# no reversal
                # self.episode_prevrl_rewards.append(sum(episode_prevrl_reward) / float(self.env.trial_num))
                # self.episode_postrl_rewards.append(
                #     sum(episode_postrl_reward) / (episode_step_count - self.env.trial_num + 1e-7))

                self.episode_rewards.append(np.sum(episode_reward))
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0 and train == True:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 50 == 0 and episode_count != 0:
                    print("current episode is ", episode_count)

                    print("current cul_rewards is ", self.episode_rewards[-1])
                    # if episode_count % 100 == 0 and self.name == 'worker_0' and train == True:
                    #     saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                    #     print("Saved Model")

                    # if episode_count % 100 == 0 and self.name == 'worker_0':
                    #     self.images = np.array(episode_frames)
                    #     make_gif(self.images,'./frames/image'+str(episode_count)+'.gif',
                    #         duration=len(self.images)*0.1,true_image=True)

                    mean_reward = np.mean(self.episode_rewards[-50:])
                    mean_length = np.mean(self.episode_lengths[-50:])
                    mean_value = np.mean(self.episode_mean_values[-50:])
# no reversal
                    # mean_prev_rl_rewards = np.mean(self.episode_prevrl_rewards[-50:])
                    # mean_post_rl_rewards = np.mean(self.episode_postrl_rewards[-50:])
                    # summary = tf.Summary()
                    # summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    # summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    # summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
# no reversal
#                     summary.value.add(tag='Perf/prev_rl_rewards', simple_value=float(mean_prev_rl_rewards))
#                     summary.value.add(tag='Perf/post_rl_rewards', simple_value=float(mean_post_rl_rewards))

                    # if train == True:
                    #     summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    #     summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    #     summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    #     summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    #     summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))



                    # self.summary_writer.add_summary(summary, episode_count)
                    #
                    # self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                if episode_count > old_episode_count + episode_num:
                    df_data.to_pickle('df_data_norl.pkl')
                    #np.save('df_data_norl', df_data)
                    coord.request_stop()



