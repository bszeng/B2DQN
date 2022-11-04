import os
import random
import gym
import numpy as np
import mxnet as mx
import time
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import loss as gloss
from collections import namedtuple
from environment_1 import radio_environment

### Set the hyper-parameters
class Options:
    def __init__(self):
        self.scene=1
        
        #Articheture
        self.batch_size = 32 # batch_size=32
        self.state_size = 8 if self.scene==1 else 10
        #Trickes
        self.learning_frequency = 1 # With Freq of 1 step update the Q-network
        self.frame_len = 1
        self.replay_buffer_size = 20000 
        self.Target_update = 1000 
        self.epsilon_min = 0.1 # final_epsilon=0.1
        self.init_epsilon=1
        self.annealing_end = 30000. # annealing=3000
        self.gamma = 0.99 # gamma=0.99
        self.replay_start_size = 5000 # Start to backpropagated through the network, learning starts        
        self.K = 10 #number of DQN
        #Optimization
        self.lr = 0.01 # learning_rate=0.01
        self.ctx = mx.gpu() # Enables gpu if available, if not, set it to mx.cpu()
        self.max_episode = 100000
opt = Options()
env_name = '5G'
env = radio_environment(seed=0)
num_action = env.action_space.n # Extract the number of available action from the environment setting
M_ULA = env.M_ULA
attrs = vars(opt)

### Define the DQN model
## DQN   
DQN = gluon.nn.Sequential()
with DQN.name_scope():
    #first layer
    DQN.add(gluon.nn.Dense(24, activation='relu'))
    #second layer
    DQN.add(gluon.nn.Dense(24, activation='relu'))
def HEAD():
    head = gluon.nn.Sequential()
    with head.name_scope():
        #third layer
        head.add(gluon.nn.Dense(num_action))
    head.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
    return head
    
dqn = DQN
dqn.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
DQN_trainer = gluon.Trainer(dqn.collect_params(), 'adam', {'learning_rate': opt.lr})
heads = gluon.nn.Sequential()
for _ in range(opt.K):
    heads.add(HEAD())
heads_trainer = []
for i in range(opt.K):
    heads_trainer.append(gluon.Trainer(heads[i].collect_params(),'adam',{'learning_rate': opt.lr}))
dqn.collect_params().zero_grad()
for i in range(opt.K):
    heads[i].collect_params().zero_grad()

## Target_DQN    
Target_DQN = gluon.nn.Sequential()
with Target_DQN.name_scope():
    #first layer
    Target_DQN.add(gluon.nn.Dense(24, activation='relu'))
    #second layer
    Target_DQN.add(gluon.nn.Dense(24, activation='relu'))
target_dqn = Target_DQN
target_dqn.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
heads_target = gluon.nn.Sequential()
for _ in range(opt.K):
    #third layer
    heads_target.add(HEAD())

### Replay buffer
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward','done','head'))
class Replay_Buffer():
    def __init__(self, replay_buffer_size):
        self.replay_buffer_size = replay_buffer_size
        self.memory = []
        self.position = 0
    def push(self, *args):
        if len(self.memory) < self.replay_buffer_size:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.replay_buffer_size
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

### Train the model
np.random.seed(0)
seeds = np.random.randint(1,100,3).tolist()  # [45, 48, 65]
seeds = {45, 48, 65}

for seed in seeds:

    mx.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)    
    env = radio_environment(seed=seed)        

    print('--'*18)
    print('replay_buffer_size(rbz):' + str(opt.replay_buffer_size))
    print('replay_start_size(rsz):' + str(opt.replay_start_size))
    print('batch_size(bs):' + str(opt.batch_size))
    print('Target_update(tu):' + str(opt.Target_update))
    print('--'*18)

    f_n = 'M'+str(M_ULA)+'_BoDQN_'+str(opt.scene)+'_' + str(opt.max_episode) + '_seed' + str(seed) + '_k' + str(opt.K)  +'_rbz'+str(opt.replay_buffer_size)+'rsz'+str(opt.replay_start_size)+'bs'+str(opt.batch_size)+'tu'+str(opt.Target_update)+'.txt' 
    f = open(f_n,'w')
    titles = 'episode,reward,timesteps,duration'
    f.write(titles)

    command = 'mkdir BoDQN_'+str(opt.scene)+'_M'+str(M_ULA)+'_'+'k'+str(opt.K)+'_s'+str(seed) # Creat a direcotry to store models and scores.
    os.system(command) 

    total_steps = 0
    batch_state = nd.empty((opt.batch_size,opt.frame_len,opt.state_size), opt.ctx)
    batch_state_next = nd.empty((opt.batch_size,opt.frame_len,opt.state_size), opt.ctx)
    l2loss = gluon.loss.L2Loss(batch_axis=0)
    replay_memory = Replay_Buffer(opt.replay_buffer_size) # Initialize the replay buffer
    annealing_count = 0

    batch_state = nd.empty((opt.batch_size,opt.frame_len,opt.state_size), opt.ctx)
    batch_state_next = nd.empty((opt.batch_size,opt.frame_len,opt.state_size), opt.ctx)
    batch_reward = nd.empty((opt.batch_size),opt.ctx)
    batch_action = nd.empty((opt.batch_size),opt.ctx)
    batch_done = nd.empty((opt.batch_size),opt.ctx)
    batch_head = nd.empty((opt.batch_size),opt.ctx)

    for  epis_count in range(1, opt.max_episode+1):
        episode_reward = 0
        state = env.reset()
        start_time = time.time()
        timestep_index = 0

        k = random.randint(0,opt.K-1)

        while True:
            timestep_index += 1

            sample = random.random()
            if total_steps > opt.replay_start_size:
                annealing_count += 1
            if total_steps == opt.replay_start_size:
                print('annealing and learning are started')
            eps = np.maximum(1.-annealing_count/opt.annealing_end,opt.epsilon_min)
            effective_eps = eps
            # epsilon greedy policy
            if sample < effective_eps:
                action = random.randint(0, num_action - 1)
            else:
                data = nd.array(state.reshape([1,opt.frame_len,opt.state_size]),opt.ctx)
                action = int(nd.argmax(heads[k](dqn(data)),axis=1).as_in_context(opt.ctx).asscalar())

            next_state, reward, done, abort = env.step(action)
            replay_memory.push(state,action,next_state,reward,done,k)
            episode_reward += reward
            total_steps += 1
            state = next_state

            # Train
            if total_steps > opt.replay_start_size:        
                if total_steps % opt.learning_frequency == 0:                    
                    batch = replay_memory.sample(opt.batch_size)
                    loss = 0
                    head_batch = np.zeros(opt.K)
                    for j in range(opt.batch_size):
                        batch_state[j] = batch[j].state.astype('float32')
                        batch_state_next[j] = batch[j].next_state.astype('float32')
                        batch_reward[j] = batch[j].reward
                        batch_action[j] = batch[j].action
                        batch_done[j] = batch[j].done
                        head_update = batch[j].head                  
                        head_batch[head_update] += 1

                        with autograd.record():
                            # get the Q(s,a)
                            all_current_q_value = heads[head_update](dqn(batch_state[j]))
                            main_q_value = nd.pick(all_current_q_value,batch_action[j],1)                            
                            # get next action from main network
                            all_next_q_value = heads_target[head_update](target_dqn(batch_state_next[j])).detach() # only get gradient of main network
                            max_action = nd.argmax(all_current_q_value, axis=1)                            
                            # then get its Q value from target network
                            target_q_value = nd.pick(all_next_q_value, max_action).detach()
                            target_q_value = batch_reward[j] + (1-batch_done[j])*opt.gamma *target_q_value
                            # record loss
                            loss = loss + nd.mean(l2loss(main_q_value, target_q_value))
                            
                    loss.backward()
                    DQN_trainer.step(opt.batch_size)
                    for h in range(opt.K):
                        if (head_batch[h]>0):
                            heads_trainer[h].step(head_batch[h])

            # Save the model and update Target model
            if total_steps > opt.replay_start_size:
                if total_steps % opt.Target_update == 0 :
                    check_point = total_steps / (opt.Target_update *100)
                    fdqn = './BoDQN_%d_M%d_k%d_s%d/Boots_target_%s_%d' % (opt.scene,M_ULA,opt.K, seed,env_name,int(check_point))
                    dqn.save_parameters(fdqn)
                    target_dqn.load_parameters(fdqn, opt.ctx)
                    print('BoDQN parameters replaced')
                    for h in range(opt.K):
                        fdqn = './BoDQN_%d_M%d_k%d_s%d/Boots_target_head_%s_%d_%d' % (opt.scene,M_ULA,opt.K, seed, env_name,int(check_point),h)
                        heads[h].save_parameters(fdqn)
                        heads_target[h].load_parameters(fdqn, opt.ctx)
            
            if abort == True:
                break
                
        end_time = time.time()
        duration = 1000. * (end_time - start_time)
        results = '%d,%d,%d,%.2f' % (epis_count, episode_reward, timestep_index,duration/timestep_index)
        if epis_count % 1000 == 0:
            local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('Scene_'+str(opt.scene)+': BoDQN,'+'M'+str(M_ULA)+',seed'+str(seed) + ',' + results+','+str(local_time))
        f.write('\n' + results)
    f.close()
