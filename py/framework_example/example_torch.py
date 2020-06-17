import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import time
import scipy.signal


STATE_DIM = 6
STATE_NUM = 8
ACTION_DIM = 2
REWARD_DIM = 1
INPUT_DIM = STATE_DIM

EPISODES = 10000

BATCH_SIZE = 1
SEQ_LEN = 30
HIDDEN_SIZE = 256
GAMMA = 0.9
LR = 0.0001
beta_v = 0.5
beta_e = 0.05


class ACNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ACNetwork, self).__init__()
        self.lstm_layer = torch.nn.LSTM(input_size, hidden_size,batch_first=True)
        self.la = torch.nn.Linear(hidden_size, 2)
        self.lv = torch.nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_size = 1
        self.hidden = self.init_hidden()

    def init_hidden(self):
        h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        c0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        h0 = nn.init.xavier_normal(h0)
        c0 = nn.init.xavier_normal(c0)
        return (Variable(h0), Variable(c0))

    def forward(self, x):
        #lstm_out, self.hidden = self.lstm_layer(x, self.hidden)
        lstm_out, (hn, cn) = self.lstm_layer(x, self.hidden)

        actions = self.la(lstm_out.squeeze())
        actions = F.softmax(actions, dim=1)
        v = self.lv(lstm_out.squeeze())
        return actions, v

    def loss(self, Loss, states, softmax_actions,onehot_actions, rewards, vs, agent, gamma):
        totloss = Loss.forward(states, softmax_actions,onehot_actions, rewards, vs, agent, gamma)
        return totloss


class Combined_Loss(torch.nn.Module):

    def __init__(self):
        super(Combined_Loss, self).__init__()

    def forward(self, states, softmax_actions,onehot_actions, rewards, vs, agent, gamma):
        vs = vs.squeeze().tolist()
        vs.insert(0,0.)

        discounted_rewards = agent.discount_reward(rewards, gamma)[:-1]

        advantages = rewards + gamma * np.asarray(vs[1:]) - vs[:-1]
        advantages = advantages.tolist()
        advantages = agent.discount_reward(advantages, gamma)

        advantages = advantages.tolist()
        advantages = Variable(torch.Tensor(advantages))
        onehot_actions = torch.stack(onehot_actions)

        responsible_output = torch.sum(softmax_actions * onehot_actions, 1)

        policy_loss =  torch.sum(responsible_output * advantages)
        value_loss = 0.5 * torch.sum(torch.mul(advantages, advantages))
        entropy = - torch.sum(softmax_actions * torch.log(softmax_actions))


        loss = policy_loss + beta_v * value_loss + beta_e * entropy

        return loss



class Agent():
    def __init__(self):
        self.states = {
            "states1": [0, 0, 0, 1, 1, 1],
            "states2": [1, 0, 0, 0, 1, 1],
            "states3": [0, 1, 0, 1, 0, 1],
            "states4": [0, 0, 1, 1, 1, 0],
            "states5": [1, 1, 0, 0, 0, 1],
            "states6": [1, 0, 1, 0, 1, 0],
            "states7": [0, 1, 1, 1, 0, 0],
            "states8": [1, 1, 1, 0, 0, 0]
        }

    def reset(self):
        key = random.choice(list(self.states))
        self.state = self.states[key]
        return self.state

    def states_sampler(self,seq_len):
        # return states list of size seq_len * f_dim
        keys = random.choices(list(self.states),k=seq_len)
        states = [self.states[key] for key in keys]
        return states

    def states_cont(self, states):
        states = states[:-1]
        key = random.choices(list(self.states), k=1)
        states.append(self.states[key[0]])

        return states

    def rewards(self, states, actions):
        # input: states, actions (one-hot)
        # return: rewards
        rewards = []
        for l in range(len(states)):
            if states[l][0] == actions[l][0]:
                rewards.append(1)
            else:
                rewards.append(0)
        return rewards

    # def discount_reward(self, r, gamma, vs):
    #     discounted_r = vs.clone()
    #     running_add = vs[-1]
    #     for t in reversed(range(0, len(r))):
    #         running_add = running_add * gamma + r[t]
    #         discounted_r[t] = running_add
    #     return discounted_r

    def discount_reward(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def roll_out(self, acnetwork, agent, seq_len,states):
        # return the s,a,r,final_r tuple list


        #states = agent.states_cont(states)
        states = agent.states_sampler(seq_len)

        states = agent.states_cont(states)
        # states = agent.states_sampler(seq_len)

        states_ts = Variable(torch.Tensor(states).view(1, -1, INPUT_DIM))
        softmax_actions, vs = acnetwork(states_ts)
        onehot_actions = [(action >= max(action)).to(torch.int) for action in softmax_actions]
        rewards = agent.rewards(states, onehot_actions)

        return states, softmax_actions, onehot_actions, rewards, vs




def main():
    # init a task generator for data fetching
    agent = Agent()
    states = agent.states_sampler(seq_len=SEQ_LEN)
    Loss = Combined_Loss()
    # init ac network
    acnetwork = ACNetwork(input_size=STATE_DIM, hidden_size=HIDDEN_SIZE, output_size=ACTION_DIM)

    ac_optim = torch.optim.RMSprop(acnetwork.parameters(), lr=LR)

    cul_rewards_tracker = []
    loss_tracker = []
    # vs_tracker = []
    for episode in tqdm(range(EPISODES)):

        time.sleep(0.001)
        states, softmax_actions,onehot_actions, rewards, vs = agent.roll_out(acnetwork, agent, SEQ_LEN, states)

        loss = acnetwork.loss(Loss,states, softmax_actions, onehot_actions, rewards, vs, agent, GAMMA)
        real_actions = torch.ones_like(torch.stack(onehot_actions)) - torch.stack(onehot_actions)
        real_rewards = agent.rewards(states, real_actions)

        if episode % 100 == 0:
            cul_rewards_tracker.append(sum(real_rewards))
            loss_tracker.append(loss)

            print(softmax_actions)
            print("current_loss is:", loss)
            print("current_reward is:", cul_rewards_tracker[-1])

        ac_optim.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(acnetwork.parameters(),0.5)
        ac_optim.step()

    # save the model
    torch.save(acnetwork.state_dict(), "./torchmodel")


    loss_tracker = (torch.stack(loss_tracker)).detach().numpy()

    # print(vs_tracker)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(cul_rewards_tracker)
    plt.title("reward tracking")
    plt.xlabel("episodes")
    plt.ylabel("rewards acquired in 30 trials")
    plt.subplot(1,2,2)
    plt.title("loss tracking")
    plt.plot(loss_tracker)
    plt.xlabel("episodes")
    plt.ylabel("combined loss")

    plt.show()



if __name__ == '__main__':
    main()
