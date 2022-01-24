import os
import random
import gym
import numpy as np
import mxnet as mx
import time
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import loss as gloss
from collections import namedtuple
from environment import radio_environment

### Set the hyper-parameters
class Options:
    def __init__(self):
        #Articheture
        self.batch_size = 32 # batch_size=32
        self.state_size = 8
        #Trickes
        self.learning_frequency = 1 # With Freq of 1 step update the Q-network
        self.frame_len = 1
        self.replay_buffer_size = 20000 # buffer_size=2000
        self.Target_update = 1000 # replace_iter=1000
        self.gamma = 0.99 # gamma=0.99
        self.replay_start_size = 5000 # Start to backpropagated through the network, learning starts        
        #Optimization
        self.max_frame = 10 # radio_frame = 10
        self.lr = 0.01 # learning_rate=0.01
        self.ctx = mx.gpu() # Enables gpu if available, if not, set it to mx.cpu()
        self.max_episode = 100000
        self.lastlayer = 24 # Dimensionality of feature space
        # self.f_sampling =1000 # frequency sampling E_W_ (Thompson Sampling)
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

### Define the feature representation model
def DQN_gen():
    DQN = gluon.nn.Sequential()
    with DQN.name_scope():
        #first layer
        DQN.add(gluon.nn.Dense(24, activation='relu'))      #second layer
        #final layer
        DQN.add(gluon.nn.Dense(opt.lastlayer, activation='relu'))
    DQN.collect_params().initialize(mx.init.Normal(0.02), ctx=opt.ctx)
    return DQN

dqn_ = DQN_gen()
target_dqn_ = DQN_gen()

DQN_trainer = gluon.Trainer(dqn_.collect_params(), 'adam', {'learning_rate': opt.lr})
dqn_.collect_params().zero_grad()

### Replay buffer
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward','done'))
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

### BLR posteriro update
sigma = opt.sigma
sigma_n = opt.sigma_n

def BayesReg(phiphiT,phiY,alpha,batch_size):
    phiphiT *= (1-alpha) #Forgetting parameter alpha suggest how much of the moment from the past can be used, we set alpha to 1 which means do not use the past moment
    phiY *= (1-alpha)
    for j in range(batch_size):
        transitions = replay_memory.sample(1) # sample a minibatch of size one from replay buffer
        bat_state[0] = transitions[0].state
        bat_state_next[0] = transitions[0].next_state
        bat_reward = transitions[0].reward 
        bat_action = transitions[0].action 
        bat_done = transitions[0].done 
        phiphiT[int(bat_action)] += nd.dot(dqn_(bat_state).T,dqn_(bat_state))
        phiY[int(bat_action)] += (dqn_(bat_state)[0].T*(bat_reward +(1.-bat_done) * opt.gamma * nd.max(nd.dot(E_W_target,target_dqn_(bat_state_next)[0].T))))
    for i in range(num_action):
        inv = np.linalg.inv((phiphiT[i]/sigma_n + 1/sigma*eye).asnumpy())
        E_W[i] = nd.array(np.dot(inv,phiY[i].asnumpy())/sigma_n, ctx = opt.ctx)
        Cov_W[i] = sigma * inv
    return phiphiT,phiY,E_W,Cov_W 

# Thompson sampling, sample model W form the posterior.
def sample_W(E_W,U):
    for i in range(num_action):
        sam = nd.normal(loc=0, scale=1, shape=(opt.lastlayer,1),ctx = opt.ctx)
        E_W_[i] = E_W[i] + nd.dot(U[i],sam)[:,0]
    return E_W_

### Train the model
np.random.seed(0)
seeds = np.random.randint(1,100,3).tolist()  # [45, 48, 65]

for seed in seeds:

    mx.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)    
    env = radio_environment(seed=seed)
    
    command = 'mkdir ba_'+str(seed) # Creat a direcotry to store models and scores.
    os.system(command)
    
    # Initialized BLR matrices
    bat_state = nd.empty((1,opt.frame_len,opt.state_size), opt.ctx)
    bat_state_next = nd.empty((1,opt.frame_len,opt.state_size), opt.ctx)
    bat_reward = nd.empty((1), opt.ctx)
    bat_action = nd.empty((1), opt.ctx)
    bat_done = nd.empty((1), opt.ctx)

    eye = nd.zeros((opt.lastlayer,opt.lastlayer), opt.ctx)
    for i in range(opt.lastlayer):
        eye[i,i] = 1

    E_W = nd.normal(loc=0, scale=.01, shape=(num_action,opt.lastlayer),ctx = opt.ctx)
    E_W_target = nd.normal(loc=0, scale=.01, shape=(num_action,opt.lastlayer),ctx = opt.ctx)
    E_W_ = nd.normal(loc=0, scale=.01, shape=(num_action,opt.lastlayer),ctx = opt.ctx)
    Cov_W = nd.normal(loc=0, scale= 1, shape=(num_action,opt.lastlayer,opt.lastlayer),ctx = opt.ctx)+eye
    Cov_W_decom = Cov_W
    
    for i in range(num_action):
        Cov_W[i] = eye
        Cov_W_decom[i] = nd.array(np.linalg.cholesky(((Cov_W[i]+nd.transpose(Cov_W[i]))/2.).asnumpy()),ctx = opt.ctx)
    Cov_W_target = Cov_W
    phiphiT = nd.zeros((num_action,opt.lastlayer,opt.lastlayer), opt.ctx)
    phiY = nd.zeros((num_action,opt.lastlayer), opt.ctx)


    print('--'*18)
    print('replay_buffer_size(rbz):' + str(opt.replay_buffer_size))
    print('replay_start_size(rsz):' + str(opt.replay_start_size))
    print('batch_size(bs):' + str(opt.batch_size))
    # print('f_sampling(fs):'+str(opt.f_sampling))
    print('Target_update(tu):' + str(opt.Target_update))
    print('target_W_update(twu):' + str(opt.target_W_update))
    print('target_batch_size(tbs):' + str(opt.target_batch_size))
    print('--'*18)
    f_n = 'M'+str(M_ULA)+'_badqn_' + str(opt.max_episode) + '_seed' + str(seed) +'_rbz'+str(opt.replay_buffer_size)+'rsz'+str(opt.replay_start_size)+'bs'+str(opt.batch_size)+'tu'+str(opt.Target_update)+'twu'+str(opt.target_W_update)+'tbs'+str(opt.target_batch_size)+ '.txt' 
    f = open(f_n,'w')
    titles = 'episode,reward,timesteps,successful'
    print(titles)
    f.write('\n' + titles)

    c_t = 0
    total_steps = 0
    total_not_positive_definite = 0
    batch_state = nd.empty((opt.batch_size,opt.frame_len,opt.state_size), opt.ctx)
    batch_state_next = nd.empty((opt.batch_size,opt.frame_len,opt.state_size), opt.ctx)
    batch_reward = nd.empty((opt.batch_size),opt.ctx)
    batch_action = nd.empty((opt.batch_size),opt.ctx)
    batch_done = nd.empty((opt.batch_size),opt.ctx)

    l2loss = gluon.loss.L2Loss(batch_axis=0)
    replay_memory = Replay_Buffer(opt.replay_buffer_size) # Initialize the replay buffer
    epis_count = 0 # Counts the number episodes so far

    for  epis_count in range(1, opt.max_episode+1):
        episode_reward = 0
        successful = False        
        state = env.reset() #next_frame = env.reset()
        time_steps = 0

        for timestep_index in range(1, opt.max_frame+1):             
            time_steps = time_steps + 1
            # sample = random.random()

            # if total_steps == opt.replay_start_size:
            #     print('annealing and learning are started')

            data = nd.array(state.reshape([1,opt.frame_len,opt.state_size]),opt.ctx)
            a = nd.dot(E_W_,dqn_(data)[0].T)
            action = np.argmax(a.asnumpy()).astype(np.uint8)

            next_state, reward, done, abort = env.step(action)
            replay_memory.push(state,action,next_state,reward,done)
            episode_reward += reward
            total_steps += 1
            state = next_state
            # Thompson Sampling
            # if total_steps % opt.f_sampling:
            E_W_ = sample_W(E_W,Cov_W_decom)
                
            # Train
            if total_steps > opt.replay_start_size:        
                if total_steps % opt.learning_frequency == 0:
                    batch = replay_memory.sample(opt.batch_size)

                    #update network  
                    for j in range(opt.batch_size):
                        batch_state[j] = batch[j].state.astype('float32')
                        batch_state_next[j] = batch[j].next_state.astype('float32')
                        batch_reward[j] = batch[j].reward
                        batch_action[j] = batch[j].action
                        batch_done[j] = batch[j].done

                    with autograd.record():
                        # get the Q(s,a)
                        all_current_q_value = nd.dot(dqn_(batch_state),E_W.T)
                        main_q_value = nd.pick(all_current_q_value, batch_action,1)
                        # get next action from main network
                        all_next_q_value = nd.dot(target_dqn_(batch_state_next), E_W_target.T).detach()
                        max_action = nd.argmax(all_current_q_value, axis=1)
                        #  get its Q value from target network
                        target_q_value = nd.pick(all_next_q_value, max_action).detach()
                        target_q_value = batch_reward + (1-batch_done)*opt.gamma *target_q_value
                        # record loss
                        loss = nd.mean(l2loss(main_q_value,  target_q_value))
                        
                    loss.backward()
                    DQN_trainer.step(opt.batch_size)                    

            # Save the model and update Target model
            if total_steps > opt.replay_start_size:
                if total_steps % opt.Target_update == 0 :
                    check_point = total_steps / (opt.Target_update *100)
                    fdqn = './ba_%d/Boots_target_%s_%d' % (seed, env_name,int(check_point))
                    dqn_.save_parameters(fdqn)
                    target_dqn_.load_parameters(fdqn, opt.ctx)
                    print('badqn parameters replaced, c_t='+str(c_t))
                    c_t += 1

                    if c_t == opt.target_W_update:                        
                        while (True):                            
                            c_npd = 0
                            phiphiT,phiY,E_W,Cov_W = BayesReg(phiphiT,phiY,opt.alpha_target,opt.target_batch_size)
                            E_W_target = E_W
                            Cov_W_target = Cov_W                            
                            for ii in range(num_action):
                                try:
                                    cholesky_output = np.linalg.cholesky(((Cov_W[ii]+nd.transpose(Cov_W[ii]))/2.).asnumpy())
                                except:
                                    total_not_positive_definite += 1
                                    print('Matrix is not positive definite, Cov_W_decom will not be udpated')
                                    break
                                else:
                                    Cov_W_decom[ii] = nd.array(cholesky_output, ctx = opt.ctx)
                                    c_npd += 1
                            if (c_npd==num_action):
                                c_t = 0
                                print('target_W_update')
                                break
                                
            successful = done and (episode_reward > 0) and (abort == False)

            if abort == True:
                break

        results = '%d,%d,%d,%d' % (epis_count, episode_reward, timestep_index, successful)
        if epis_count % 1000 == 0:
            local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('badqn:'+'M'+str(M_ULA)+',seed'+str(seed) + ',' + results+','+str(local_time))
        f.write('\n' + results)
    f.close()

