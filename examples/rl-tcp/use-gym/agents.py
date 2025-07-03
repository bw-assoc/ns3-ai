# Copyright (c) 2023 Huazhong University of Science and Technology
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Author: Muyuan Shen <muyuan_shen@hust.edu.cn>


import torch
import numpy as np
import torch.nn as nn
import random


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 4),
        )

    def forward(self, x):
        return self.layers(x)


class DQN(object):
    def __init__(self):
        self.eval_net = net()
        self.target_net = net()
        self.learn_step = 0
        self.batchsize = 32
        self.observer_shape = 5
        self.target_replace = 100
        self.memory_counter = 0
        self.memory_capacity = 2000
        self.memory = np.zeros((2000, 2 * 5 + 2))  # s, a, r, s'
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=0.0001)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.Tensor(x)
        if np.random.uniform() > 0.99 ** self.memory_counter:  # choose best
            action = self.eval_net.forward(x)
            action = torch.argmax(action, 0).numpy()
        else:  # explore
            action = np.random.randint(0, 4)
        return action

    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = np.hstack((s, [a, r], s_))
        self.memory_counter += 1

    def learn(self, ):
        self.learn_step += 1
        if self.learn_step % self.target_replace == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        sample_list = np.random.choice(self.memory_capacity, self.batchsize)
        # choose a mini batch
        sample = self.memory[sample_list, :]
        s = torch.Tensor(sample[:, :self.observer_shape])
        a = torch.LongTensor(
            sample[:, self.observer_shape:self.observer_shape + 1])
        r = torch.Tensor(
            sample[:, self.observer_shape + 1:self.observer_shape + 2])
        s_ = torch.Tensor(sample[:, self.observer_shape + 2:])
        q_eval = self.eval_net(s).gather(1, a)
        q_next = self.target_net(s_).detach()
        q_target = r + 0.8 * q_next.max(1, True)[0].data

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class TcpNewRenoAgent:

    def __init__(self):
        self.new_cWnd = 0
        self.new_ssThresh = 0
        pass

    def get_action(self, obs, reward, done, info):
        # current ssThreshold
        ssThresh = obs[4]
        # current contention window size
        cWnd = obs[5]
        # segment size
        segmentSize = obs[6]
        # number of acked segments
        segmentsAcked = obs[9]
        # estimated bytes in flight
        bytesInFlight = obs[7]

        self.new_cWnd = 1
        if cWnd < ssThresh:
            # slow start
            if segmentsAcked >= 1:
                self.new_cWnd = cWnd + segmentSize
        if cWnd >= ssThresh:
            # congestion avoidance
            if segmentsAcked > 0:
                adder = 1.0 * (segmentSize * segmentSize) / cWnd
                adder = int(max(1.0, adder))
                self.new_cWnd = cWnd + adder

        self.new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))
        return [self.new_ssThresh, self.new_cWnd]


class TcpDeepQAgent:

    def __init__(self):
        self.dqn = DQN()
        self.new_cWnd = None
        self.new_ssThresh = None
        self.s = None
        self.a = None
        self.r = None
        self.s_ = None  # next state

    def get_action(self, obs, reward, done, info):
        # current ssThreshold
        ssThresh = obs[4]
        # current contention window size
        cWnd = obs[5]
        # segment size
        segmentSize = obs[6]
        # number of acked segments
        segmentsAcked = obs[9]
        # estimated bytes in flight
        bytesInFlight = obs[7]

        # update DQN
        self.s = self.s_
        #self.s_ = [ssThresh, cWnd, segmentsAcked, segmentSize, bytesInFlight]
        self.s_ = [cWnd, 0, 0, 0, 0]
        if self.s is not None:  # not first time
            self.r = 1 if self.a == 1 else 0
            self.dqn.store_transition(self.s, self.a, self.r, self.s_)
            if self.dqn.memory_counter > self.dqn.memory_capacity:
                self.dqn.learn()

        # choose action
        self.a = self.dqn.choose_action(self.s_)
        print("Action:", self.a)
        if self.a & 1:
            self.new_cWnd = cWnd + segmentSize
        else:
            if cWnd > 0:
                self.new_cWnd = cWnd + int(max(1, (segmentSize * segmentSize) / cWnd))
        if self.a < 3:
            self.new_ssThresh = 2 * segmentSize
        else:
            self.new_ssThresh = int(bytesInFlight / 2)

        return [self.new_ssThresh, self.new_cWnd]


class TcpQAgent:

    def discretize(self, metric, minval, maxval):
        metric = max(metric, minval)
        metric = min(metric, maxval)
        return int((metric - minval) * (self.discrete_level - 1) / (maxval - minval))

    def __init__(self):
        self.update_times = 0
        self.learning_rate = None
        self.discount_rate = 0.5
        self.discrete_level = 15
        self.epsilon = 0.1  # exploration rate
        self.state_size = 3
        self.action_size = 1
        self.action_num = 4
        self.actions = np.arange(self.action_num, dtype=int)
        self.q_table = np.zeros((*((self.discrete_level, ) * self.state_size), self.action_num))
        # print(self.q_table.shape)
        self.new_cWnd = None
        self.new_ssThresh = None
        self.s = None
        self.a = np.zeros(self.action_size, dtype=int)
        self.r = None
        self.s_ = None  # next state

    def get_action(self, obs, reward, done, info):
        # current ssThreshold
        # ssThresh = obs[4]
        # current contention window size
        cWnd = obs[5]
        # segment size
        segmentSize = obs[6]
        # number of acked segments
        segmentsAcked = obs[9]
        # estimated bytes in flight
        bytesInFlight = obs[7]

        cWnd_d = self.discretize(cWnd, 0., 50000.)
        segmentsAcked_d = self.discretize(segmentsAcked, 0., 64.)
        bytesInFlight_d = self.discretize(bytesInFlight, 0., 1000000.)

        self.s = self.s_
        self.s_ = [cWnd_d, segmentsAcked_d, bytesInFlight_d]
        if self.s:  # not first time
            # update Q-table
            self.learning_rate = 0.3 * (0.995 ** (self.update_times // 10))
            self.r = segmentsAcked - bytesInFlight - cWnd
            self.q_table[tuple(self.s)][tuple(self.a)] = (
                    (1 - self.learning_rate) * self.q_table[tuple(self.s)][tuple(self.a)] +
                    self.learning_rate * (self.r + self.discount_rate * np.max(self.q_table[tuple(self.s_)]))
            )
            self.update_times += 1

        # epsilon-greedy
        if random.uniform(0, 1) < 0.1:
            self.a[0] = np.random.choice(self.actions)
        else:
            self.a[0] = np.argmax(self.q_table[tuple(self.s_)])

        # map action to cwnd and ssthresh
        if self.a[0] & 1:
            self.new_cWnd = cWnd + segmentSize
        else:
            if cWnd > 0:
                self.new_cWnd = cWnd + int(max(1, (segmentSize * segmentSize) / cWnd))
        if self.a[0] < 3:
            self.new_ssThresh = 2 * segmentSize
        else:
            self.new_ssThresh = int(bytesInFlight / 2)

        return [self.new_ssThresh, self.new_cWnd]



class DeepQNetwork(nn.Module):
    def __init__(self, learning_rate, input_dimensions, output_dimensions, hidden_dimensions=20):
        super(DeepQNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions

        self.layer1 = nn.Linear(self.input_dimensions, hidden_dimensions)
        self.activ1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.activ2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_dimensions, self.output_dimensions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        if torch.cuda.is_available(): self.to('cuda')
    
    def forward(self, input):
        x = self.layer1(input)
        x = self.activ1(x)
        x = self.layer2(x)
        x = self.activ2(x)
        return self.layer3(x)

class Ben_Agent():
    # gamma is weighting of future rewards
    # epsilon is tradeoff between exploration and best action
    def __init__(self):
        self.gamma = 0.15
        self.learning_rate = 0.0003
        self.input_dimensions = 3
        self.num_actions = 3
        self.actions = [i for i in range(self.num_actions)]
        self.batch_size = 64
        self.memory_size = 2000
        self.memory_counter = 0
        self.epsilon = 1
        self.minimum_epsilon = 0.01
        self.epsilon_decrement = 5e-4
        self.current_state = None # Compatibility layer
        self.previous = {
                "throughput": None,
                "avgRtt": None
            }

        self.eval = DeepQNetwork(self.learning_rate, input_dimensions=self.input_dimensions, output_dimensions=self.num_actions, hidden_dimensions=128)
        self.initial_state_memory = np.zeros((self.memory_size, self.input_dimensions), dtype=np.float32)
        self.resulting_state_memory = np.zeros((self.memory_size, self.input_dimensions), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)

    def store_transition(self, initial_state, action, reward, resulting_state):
        index = self.memory_counter % self.memory_size
        self.initial_state_memory[index] = initial_state
        self.resulting_state_memory[index] = resulting_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.memory_counter += 1
    
    def choose_action(self, observation):
        self.learn() # Compatibility layer
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float32)
            if torch.cuda.is_available(): state = state.to('cuda')
            actions = self.eval.forward(state)
            return torch.argmax(actions).item()
        else:
            return np.random.choice(self.actions)

    def learn(self):
        if self.memory_counter < self.batch_size: return

        self.eval.optimizer.zero_grad()
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        initial_states_batch = torch.tensor(self.initial_state_memory[batch])
        resulting_states_batch = torch.tensor(self.resulting_state_memory[batch])
        actions_batch = torch.tensor(self.action_memory[batch], dtype=torch.int64)
        rewards_batch = torch.tensor(self.reward_memory[batch])
        if torch.cuda.is_available():
            initial_states_batch = initial_states_batch.to('cuda')
            resulting_states_batch = resulting_states_batch.to('cuda')
            actions_batch = actions_batch.to('cuda')
            rewards_batch = rewards_batch.to('cuda')
        initial_predictions = self.eval.forward(initial_states_batch)[batch_index, actions_batch] # Get the predicted outcome of the action taken for each initial state
        next_predictions = self.eval.forward(resulting_states_batch)
        improved_predictions = rewards_batch + self.gamma * torch.max(next_predictions, dim=1)[0]
        loss = self.eval.loss(improved_predictions, initial_predictions)
        if torch.cuda.is_available():
            loss = loss.to('cuda')
        loss.backward()
        self.eval.optimizer.step()

        if self.epsilon > self.minimum_epsilon: self.epsilon = max(self.epsilon - self.epsilon_decrement, self.minimum_epsilon)
    
    # Compatibility layer
    def get_action(self, observations, garbage1, garbage2, garbage3):
        ssThresh = observations[4]      # current ssThreshold
        cWnd = observations[5]          # current congestion window size
        segmentSize = observations[6]   # Segment size
        segmentsAcked = observations[9] # Segments acknowledged
        bytesInFlight = observations[7] # Bytes in flight
        avgRtt = observations[11]       # avgRtt in microseconds
        throughput = observations[15]   # throughput in bytes/sec

        self.previous_state = self.current_state
        self.current_state = [cWnd, bytesInFlight, segmentsAcked]
        if self.previous_state is not None:
            reward = 0
            if throughput > self.previous["throughput"]: reward = 1
            if avgRtt > self.previous["avgRtt"]: reward = -1
            #print("Reward:", reward)
            self.store_transition(self.previous_state, self.action, reward, self.current_state)
        self.previous["throughput"] = throughput
        self.previous["avgRtt"] = avgRtt
        self.action = self.choose_action(self.current_state)
        print("Action:", self.action)
        new_cWnd = cWnd
        if self.action == 1:
            new_cWnd = min(cWnd + segmentSize, 50000)
        elif self.action == 2:
            new_cWnd = max(cWnd - segmentSize, 10 * segmentSize)
        return [int(max(2 * segmentSize, bytesInFlight / 2)), new_cWnd]

