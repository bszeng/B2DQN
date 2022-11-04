import os
import random
import gym
import numpy as np
import mxnet as mx
import time
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import loss as gloss
from collections import namedtuple
from environment_2 import radio_environment

### Set the hyper-parameters
class Options:
    def __init__(self):
        self.scene=2

        #Articheture
        self.batch_size = 32 # batch_size=32
        self.state_size = 8 if self.scene==1 else 10
        #Trickes
        self.learning_frequency = 1 # With Freq of 1 step update the Q-network
        self.frame_len = 1
        self.replay_buffer_size = 20000 
        self.Target_update = 1000 
        self.gamma = 0.99 # gamma=0.99
        self.replay_start_size = 5000 # Start to backpropagated through the network, learning starts        
        #Optimization
        self.lr = 0.01 # learning_rate=0.01
        self.ctx = mx.gpu() # Enables gpu if available, if not, set it to mx.cpu()
        self.max_episode = 100000
        self.K = 10 #number of DQN        
        self.lastlayer = 24 # Dimensionality of feature space
        self.alpha = .01 # forgetting factor 1->forget
        self.alpha_target = 1 # forgetting factor 1->forget
        self.target_batch_size = 250 #target update sample batch
        self.target_W_update = 5
        self.lambda_W = 0.1 #update on W = lambda W_new + (1-lambda) W
        self.sigma = 0.001 # W prior variance
        self.sigma_n = 1 # noise variacne
opt = Options()
env_name = '5G'
env = radio_environment(seed=0)
num_action = env.action_space.n # Extract the number of available action from the environment setting
M_ULA = env.M_ULA
attrs = vars(opt)

### Define the feature representation model with heads
DQN = gluon.nn.Sequential()
with DQN.name_scope():
    #first layer
    DQN.add(gluon.nn.Dense(24, activation='relu'))    
def HEAD():
    head = gluon.nn.Sequential()
    with head.name_scope():
        #final layer
        head.add(gluon.nn.Dense(opt.lastlayer, activation='relu'))
    head.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
    return head
dqn_ = DQN
dqn_.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
DQN_trainer = gluon.Trainer(dqn_.collect_params(), 'adam', {'learning_rate': opt.lr})

heads = gluon.nn.Sequential()
for _ in range(opt.K):
    heads.add(HEAD())

heads_trainer = []

for i in range(opt.K):
    heads_trainer.append(gluon.Trainer(heads[i].collect_params(),'adam',{'learning_rate': opt.lr}))

dqn_.collect_params().zero_grad()

for i in range(opt.K):
    heads[i].collect_params().zero_grad()

###  Target_DQN   ### 
Target_DQN = gluon.nn.Sequential()
with Target_DQN.name_scope():
    #first layer
    Target_DQN.add(gluon.nn.Dense(24, activation='relu'))
target_dqn_ = Target_DQN
target_dqn_.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)

heads_target = gluon.nn.Sequential()
for _ in range(opt.K):
    #final layer
    heads_target.add(HEAD())

### Replay buffer
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward','done', 'head'))
class Replay_Buffer():
    def __init__(self, replay_buffer_size):
        self.replay_buffer_size = replay_buffer_size
        self.memory = []
        self.memory_blr = [[]for i in range(num_action)]
        self.position = 0
    def push(self, *args):
        if len(self.memory) < self.replay_buffer_size:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.replay_buffer_size
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
sigma = opt.sigma
sigma_n = opt.sigma_n

def BayesReg(phiphiT,phiY,alpha,batch_size):
    phiphiT *= (1-alpha) #Forgetting parameter alpha suggest how much of the moment from the past can be used, we set alpha to 1 which means do not use the past moment
    phiY *= (1-alpha)
    for j in range(batch_size):
        transitions = replay_memory.sample(1) # sample a minibatch of size one from replay buffer
        bat_state[0] = nd.array(transitions[0].state.astype('float32'),opt.ctx)
        bat_state_next[0] = nd.array(transitions[0].next_state.astype('float32'),opt.ctx)
        bat_reward = transitions[0].reward 
        bat_action = transitions[0].action 
        bat_done = transitions[0].done
        bat_h = transitions[0].head
        phiphiT[bat_h][int(bat_action)] += nd.dot(heads[bat_h](dqn_(bat_state)).T, heads[bat_h](dqn_(bat_state)))
        temp_output = nd.max(nd.dot(E_W_target[bat_h],heads_target[bat_h](target_dqn_(bat_state_next)).T))
        phiY[bat_h][int(bat_action)] += (heads[bat_h](dqn_(bat_state)).T*(bat_reward +(1.-bat_done) * opt.gamma * temp_output))[0]
    for h in range(opt.K):
        for i in range(num_action):
            inv = np.linalg.inv((phiphiT[h][i]/sigma_n + 1/sigma*eye).asnumpy())
            E_W[h][i] = nd.array(np.dot(inv,phiY[h][i].asnumpy())/sigma_n, ctx = opt.ctx)
            Cov_W[h][i] = sigma * inv
    return phiphiT,phiY,E_W,Cov_W 

# Thompson sampling, sample model W form the posterior.
def sample_W(E_W,U,k):
    for i in range(num_action):
        sam = nd.normal(loc=0, scale=1, shape=(opt.lastlayer,1),ctx = opt.ctx)
        E_W_[k][i] = E_W[k][i] + nd.dot(U[k][i],sam)[:,0]
    return E_W_

### Train the model
np.random.seed(0)
seeds = np.random.randint(1,100,3).tolist()  # [45, 48, 65]
seeds = {45, 48, 65}

for seed in seeds:
    
    mx.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)    
    env = radio_environment(seed=seed)

    l2loss = gluon.loss.L2Loss(batch_axis=0)
    replay_memory = Replay_Buffer(opt.replay_buffer_size) # Initialize the replay buffer
    epis_count = 0 # Counts the number episodes so far

    print('--'*18)
    print('replay_buffer_size:' + str(opt.replay_buffer_size))
    print('replay_start_size:' + str(opt.replay_start_size))
    print('batch_size:' + str(opt.batch_size))
    print('Target_update:' + str(opt.Target_update))
    print('target_W_update:' + str(opt.target_W_update))
    print('target_batch_size:' + str(opt.target_batch_size))
    print('--'*18)
    f_n = 'M'+str(M_ULA)+'_B2DQN_'+str(opt.scene)+'_' + str(opt.max_episode) + '_seed' + str(seed) + '_k' + str(opt.K)  +'_rbz'+str(opt.replay_buffer_size)+'rsz'+str(opt.replay_start_size)+'bs'+str(opt.batch_size)+'tu'+str(opt.Target_update)+'twu'+str(opt.target_W_update)+'tbs'+str(opt.target_batch_size)+'.txt' 
    f = open(f_n,'w')
    titles = 'episode,reward,timesteps,duration'
    f.write(titles)

    command = 'mkdir B2DQN_'+str(opt.scene)+'_M'+str(M_ULA)+'_'+'k'+str(opt.K)+'_s'+str(seed) # Creat a direcotry to store models and scores.
    os.system(command) 

    ## Initialized BLR matrices
    bat_state = nd.empty((1,opt.frame_len,opt.state_size), opt.ctx)
    bat_state_next = nd.empty((1,opt.frame_len,opt.state_size), opt.ctx)
    bat_reward = nd.empty((1), opt.ctx)
    bat_action = nd.empty((1), opt.ctx)
    bat_done = nd.empty((1), opt.ctx)
    eye = nd.zeros((opt.lastlayer,opt.lastlayer), opt.ctx)
    for i in range(opt.lastlayer):
        eye[i,i] = 1
    E_W = nd.empty((opt.K,num_action,opt.lastlayer), opt.ctx)
    E_W_target = nd.empty((opt.K,num_action,opt.lastlayer), opt.ctx)
    E_W_ = nd.empty((opt.K,num_action,opt.lastlayer), opt.ctx)
    Cov_W = nd.empty((opt.K,num_action,opt.lastlayer,opt.lastlayer), opt.ctx)
    Cov_W_decom = nd.empty((opt.K,num_action,opt.lastlayer,opt.lastlayer), opt.ctx)
    phiphiT = nd.empty((opt.K,num_action,opt.lastlayer,opt.lastlayer), opt.ctx)
    phiY = nd.empty((opt.K,num_action,opt.lastlayer), opt.ctx)
    for h in range(opt.K):
        E_W[h] = nd.normal(loc=0, scale=.01, shape=(num_action,opt.lastlayer),ctx = opt.ctx)
        E_W_target[h] = nd.normal(loc=0, scale=.01, shape=(num_action,opt.lastlayer),ctx = opt.ctx)
        E_W_[h] = nd.normal(loc=0, scale=.01, shape=(num_action,opt.lastlayer),ctx = opt.ctx)
        Cov_W[h] = nd.normal(loc=0, scale= 1, shape=(num_action,opt.lastlayer,opt.lastlayer),ctx = opt.ctx)+eye
        Cov_W_decom[h] = Cov_W[h]
        for i in range(num_action):
            Cov_W[h][i] = eye
            Cov_W_decom[h][i] = nd.array(np.linalg.cholesky(((Cov_W[h][i]+nd.transpose(Cov_W[h][i]))/2.).asnumpy()),ctx = opt.ctx)
        Cov_W_target = Cov_W
        phiphiT[h] = nd.zeros((num_action,opt.lastlayer,opt.lastlayer), opt.ctx)
        phiY[h] = nd.zeros((num_action,opt.lastlayer), opt.ctx)    

    batch_state = nd.empty((opt.batch_size,opt.frame_len,opt.state_size), opt.ctx)
    batch_state_next = nd.empty((opt.batch_size,opt.frame_len,opt.state_size), opt.ctx)
    batch_reward = nd.empty((opt.batch_size),opt.ctx)
    batch_action = nd.empty((opt.batch_size),opt.ctx)
    batch_done = nd.empty((opt.batch_size),opt.ctx)
    batch_head = nd.empty((opt.batch_size),opt.ctx)

    c_t = 0
    total_steps = 0

    for  epis_count in range(1, opt.max_episode+1):
        episode_reward = 0
        state = env.reset()
        timestep_index = 0
        start_time = time.time()

        k = random.randint(0,opt.K-1)
        while True:
            timestep_index += 1

            data = nd.array(state.reshape([1,opt.frame_len,opt.state_size]),opt.ctx) 
            a = nd.dot(E_W_[k], heads[k](dqn_(data)).T)
            action = np.argmax(a.asnumpy()).astype(np.uint8)

            next_state, reward, done, abort = env.step(action)
            replay_memory.push(state,action,next_state,reward,done,k) 
            episode_reward += reward
            total_steps += 1
            state = next_state
            E_W_ = sample_W(E_W,Cov_W_decom,k)
            
            # Train
            if total_steps > opt.replay_start_size:        
                if total_steps % opt.learning_frequency == 0:
                    batch = replay_memory.sample(opt.batch_size)
                    loss = 0
                    head_batch = np.zeros(opt.K)
                    #update network                
                    for j in range(opt.batch_size):
                        batch_state[j] = batch[j].state.astype('float32')
                        batch_state_next[j] = batch[j].next_state.astype('float32')
                        batch_reward[j] =batch[j].reward
                        batch_action[j] = batch[j].action
                        batch_done[j] = batch[j].done
                        head_update = batch[j].head 
                        head_batch[head_update] += 1                        
                        with autograd.record():
                            # get the Q(s,a)
                            all_current_q_value = nd.dot(heads[head_update](dqn_(batch_state[j])), E_W[head_update].T)
                            main_q_value = nd.pick(all_current_q_value, batch_action[j],1)
                            # get next action from main network
                            all_next_q_value = nd.dot(heads_target[head_update](target_dqn_(batch_state_next[j])), (E_W_target[head_update]).T).detach() # only get gradient of main network
                            max_action = nd.argmax(all_current_q_value, axis=1)
                            #  get its Q value from target network
                            target_q_value = nd.pick(all_next_q_value, max_action).detach()
                            target_q_value = batch_reward[j] + (1-batch_done[j])*opt.gamma *target_q_value
                            # record loss
                            loss = loss + nd.mean(l2loss(main_q_value, target_q_value))
                    loss.backward()
                    DQN_trainer.step(opt.batch_size)
                    for h in range(opt.K):
                        if(head_batch[h]>0): 
                            heads_trainer[h].step(head_batch[h]) 

            # Save the model and update Target model
            if total_steps > opt.replay_start_size:
                if total_steps % opt.Target_update == 0 :
                    # Synchronize parameters 
                    check_point = total_steps / (opt.Target_update *100)
                    fdqn = './B2DQN_%d_M%d_k%d_s%d/Boots_target_%s_%d' % (opt.scene,M_ULA,opt.K, seed,env_name,int(check_point))
                    dqn_.save_parameters(fdqn)
                    target_dqn_.load_parameters(fdqn, opt.ctx)
                    print('network parameters replaced')
                    for h in range(opt.K): 
                        fdqn = './B2DQN_%d_M%d_k%d_s%d/Boots_target_head_%s_%d_%d' % (opt.scene,M_ULA,opt.K, seed,env_name,int(check_point),h)
                        heads[h].save_parameters(fdqn)
                        heads_target[h].load_parameters(fdqn, opt.ctx)
                    # Update
                    c_t += 1
                    if c_t == opt.target_W_update:                    
                        phiphiT,phiY,E_W,Cov_W = BayesReg(phiphiT,phiY,opt.alpha_target,opt.target_batch_size)
                        E_W_target = E_W
                        Cov_W_target = Cov_W
                        c_t = 0
                        for h in range(opt.K):
                            for ii in range(num_action):
                                Cov_W_decom[h][ii] = nd.array(np.linalg.cholesky(((Cov_W[h][ii]+nd.transpose(Cov_W[h][ii]))/2.).asnumpy()),ctx = opt.ctx)
                        print('posterior w updated')
            if abort == True:
                break

        end_time = time.time()
        duration = 1000. * (end_time - start_time)

        results = '%d,%d,%d,%.2f' % (epis_count, episode_reward, timestep_index, duration/timestep_index)
        if epis_count % 1000 == 0:
            local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('Scene_'+str(opt.scene)+': B2DQN,'+'M'+str(M_ULA)+',seed'+str(seed) + ',' + results+','+str(local_time))
        f.write('\n' + results)
    f.close()
