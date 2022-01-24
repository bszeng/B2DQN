import os
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
import time
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import loss as gloss
from collections import deque
from environment import radio_environment

class QNetwork(gluon.nn.Block):
    def __init__(self, n_action):
        super(QNetwork, self).__init__()
        self.n_action = n_action

        self.dense0 = gluon.nn.Dense(24, activation='relu')
        self.dense1 = gluon.nn.Dense(24, activation='relu')
        self.dense2 = gluon.nn.Dense(self.n_action)

    def forward(self, state):
        q_value = self.dense2(self.dense1(self.dense0(state)))
        return q_value

class DQN:
    def __init__(self,
                 n_action,
                 init_epsilon,
                 final_epsilon,
                 gamma,
                 replay_buffer_size,
                 batch_size,
                 target_update,
                 annealing,
                 learning_rate,
                 replay_start_size,
                 max_episode,
                 ctx,
                 seed
                 ):
        self.n_action = n_action
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.gamma = gamma # discount factor
        self.replay_buffer_size = replay_buffer_size
        self.replay_start_size = replay_start_size
        self.max_episode = max_episode
        self.batch_size = batch_size
        self.target_update = target_update # replace the parameters of the target network every T time steps
        self.annealing = annealing # The number of step it will take to linearly anneal the epsilon to its min value
        self.learning_rate = learning_rate
        self.ctx = ctx
        self.total_steps = 0        
        self.replay_buffer = MemoryBuffer(self.replay_buffer_size, ctx)
        self.seed = seed
        # build the network
        self.target_network = QNetwork(n_action)
        self.main_network = QNetwork(n_action)
        self.target_network.collect_params().initialize(init.Normal(0.02), ctx=ctx)
        self.main_network.collect_params().initialize(init.Normal(0.02), ctx=ctx)
        # optimize the main network
        self.optimizer = gluon.Trainer(self.main_network.collect_params(), 'adam', {'learning_rate': self.learning_rate})

    def choose_action(self, state):
        state = nd.array([state], ctx=self.ctx)
        if nd.random.uniform(0, 1) > self.epsilon:
            # choose the best action
            q_value = self.main_network(state)
            action = int(nd.argmax(q_value, axis=1).asnumpy())
        else:
            # random choice
            action = random.choice(range(self.n_action))
        # anneal
        self.epsilon = max(self.final_epsilon,
                           self.epsilon - (self.init_epsilon - self.final_epsilon) / self.annealing)
        self.total_steps += 1
        return action

    def update(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        with autograd.record():

            # get the Q(s,a)
            all_current_q_value = self.main_network(state_batch)
            main_q_value = nd.pick(all_current_q_value, action_batch)

            # different from DQN
            # get next action from main network, 
            # then get its Q value from target network
            all_next_q_value = self.target_network(next_state_batch).detach()  # only get gradient of main network
            max_action = nd.argmax(all_current_q_value, axis=1)
            
            target_q_value = nd.pick(all_next_q_value, max_action).detach()
            target_q_value = reward_batch + (1 - done_batch) * self.gamma * target_q_value
            
            # record loss
            loss = gloss.L2Loss()
            value_loss = loss(target_q_value, main_q_value)
        self.main_network.collect_params().zero_grad()
        value_loss.backward()
        self.optimizer.step(batch_size=self.batch_size)

    def replace_parameters(self):
        self.main_network.save_parameters('./ddqn_'+str(self.seed)+'/ddqn_temp_params')
        self.target_network.load_parameters('./ddqn_'+str(self.seed)+'/ddqn_temp_params')
        print('ddqn parameters replaced')
    
    def save_parameters(self):
        self.target_network.save_parameters('./ddqn_'+str(self.seed)+'/ddqn_target_network_parameters')
        self.main_network.save_parameters('./ddqn_'+str(self.seed)+'/ddqn_main_network_parameters')

    def load_parameters(self):
        self.target_network.load_parameters('./ddqn_'+str(self.seed)+'/ddqn_target_network_parameters')
        self.main_network.load_parameters('./ddqn_'+str(self.seed)+'/ddqn_main_network_parameters')

class MemoryBuffer:
    def __init__(self, buffer_size, ctx):
        self.buffer = deque(maxlen=buffer_size)
        self.maxsize = buffer_size
        self.ctx = ctx

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        assert len(self.buffer) > batch_size
        minibatch = random.sample(self.buffer, batch_size)
        state_batch = nd.array([data[0] for data in minibatch], ctx=self.ctx)
        action_batch = nd.array([data[1] for data in minibatch], ctx=self.ctx)
        reward_batch = nd.array([data[2] for data in minibatch], ctx=self.ctx)
        next_state_batch = nd.array([data[3] for data in minibatch], ctx=self.ctx)
        done = nd.array([data[4] for data in minibatch], ctx=self.ctx)
        return state_batch, action_batch, reward_batch, next_state_batch, done

    def store_transition(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

### Train the model
np.random.seed(0)
seeds = np.random.randint(1,100,3).tolist()  # [45, 48, 65]

max_frame = 10
for seed in seeds:

    command = 'mkdir ddqn_'+str(seed) # Creat a direcotry to store models and scores.
    os.system(command)

    mx.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = radio_environment(seed=seed) #env = gym.make('CartPole-v0').unwrapped
    M_ULA = env.M_ULA

    agent = DQN(n_action=env.action_space.n,
                init_epsilon=1,
                final_epsilon=0.1,
                gamma=0.99,
                replay_buffer_size=20000,
                batch_size=32,
                replay_start_size=5000,
                annealing=30000,
                learning_rate=0.01,
                target_update=1000,
                max_episode=100000,
                ctx=mx.cpu(),
                seed=seed
                )

    print('--'*27)
    print('replay_buffer_size(rbz):' + str(agent.replay_buffer_size))
    print('replay_start_size(rsz):' + str(agent.replay_start_size))
    print('batch_size(bs):' + str(agent.batch_size))
    print('Target_update(tu):' + str(agent.target_update))
    print('--'*27)

    f_n = 'M'+str(M_ULA)+'_ddqn_' + str(agent.max_episode) + '_seed' + str(seed) +'_rbz'+str(agent.replay_buffer_size)+'rsz'+str(agent.replay_start_size)+'bs'+str(agent.batch_size)+'tu'+str(agent.target_update)+ '.txt' 
    #f_n = 'results_dqn_' + str(episodes) + '_seed' + str(seed) + '.txt' 
    f = open(f_n,'w')
    titles = 'episode,reward,timesteps,successful'
    print(titles)
    f.write('\n' + titles)
    epis_count = 0 # Counts the number episodes so far
    episode_reward_list = []
    
    for epis_count in range(1, agent.max_episode+1):
        successful = False        
        episode_reward = 0
        state = env.reset()
        time_steps = 0
        for timestep_index in range(1,max_frame+1):       
            time_steps = time_steps + 1
            action = agent.choose_action(state)
            next_state, reward, done, abort = env.step(action)
            episode_reward += reward
            agent.replay_buffer.store_transition(state, action, reward, next_state, done)
            state = next_state

            if agent.total_steps >= agent.replay_start_size:
                agent.update()
                if agent.total_steps % agent.target_update == 0:
                    agent.replace_parameters()
                    print('ddqn parameters replaced')

            successful = done and (episode_reward > 0) and (abort == False)

            if abort == True:
                break

        results = '%d,%d,%d,%d' % (epis_count, episode_reward, timestep_index, successful)
        if epis_count % 1000 == 0:
            local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('ddqn:'+'M'+str(M_ULA)+',seed'+str(seed) + ',' + results+','+str(local_time))
        f.write('\n' + results)
    f.close()
    #agent.save_parameters()
