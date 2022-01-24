import math, random
import gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from collections import deque
from environment import radio_environment

# Use Cuda
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))        
        self.reset_parameters()
        self.reset_noise()    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu        
        return F.linear(x, weight, bias)    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
    
class NoisyDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(NoisyDQN, self).__init__()        
        self.linear =  nn.Linear(env.observation_space.shape[0], 24)
        self.noisy1 = NoisyLinear(24, 24)
        self.noisy2 = NoisyLinear(24, env.action_space.n)        
    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x    
    def act(self, state):
        state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        q_value = self.forward(state)
        action  = q_value.max(1)[1].data[0]
        return action    
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()
        
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)            
        self.buffer.append((state, action, reward, next_state, done))    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done    
    def __len__(self):
        return len(self.buffer)
    
# Computing Temporal Difference Loss
def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))
    q_values      = current_model(state)
    next_q_values = target_model(next_state)
    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)    
    loss  = (q_value - expected_q_value.detach()).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    current_model.reset_noise()
    target_model.reset_noise()    
    return loss

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict()) 


### Train the model
np.random.seed(0)
seeds = np.random.randint(1,100,3).tolist()  # [45, 48, 65]

for seed in seeds:    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)    
    env = radio_environment(seed=seed)        
    current_model = NoisyDQN(env.observation_space.shape[0], env.action_space.n)
    target_model  = NoisyDQN(env.observation_space.shape[0], env.action_space.n)
    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()    
    optimizer = optim.Adam(current_model.parameters(), lr=0.01)  
    update_target(current_model, target_model)
    
    replay_start_size = 5000
    replay_buffer_size=20000    
    max_episode = 100000
    max_frame = 10
    target_update = 1000
    batch_size = 32
    gamma= 0.99
    losses = []
    all_rewards = []
    episode_reward = 0

    M_ULA = env.M_ULA
    replay_buffer = ReplayBuffer(replay_buffer_size)
    f_n = 'M'+str(M_ULA)+'_noisynet_' + str(max_episode) + '_seed' + str(seed) +'_rbz'+str(replay_buffer_size)+'rsz'+str(replay_start_size)+'bs'+str(batch_size)+'tu'+str(target_update)+'.txt' 
    f = open(f_n,'w')
    titles = 'episode,reward,timesteps,successful'
    print(titles)
    f.write('\n' + titles)

    for  epis_count in range(1, max_episode+1):
        losses = []
        all_rewards = []
        episode_reward = 0
        successful = False
        
        state = env.reset()
        for frame_idx in range(1, max_frame + 1):
            action = current_model.act(state)

            next_state, reward, done, abort = env.step(action.cpu().numpy())
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(replay_buffer) > replay_start_size:
                loss = compute_td_loss(batch_size)
                losses.append(loss.data.item())

            if frame_idx % target_update == 0:
                update_target(current_model, target_model)
        
            successful = done and (episode_reward > 0) and (abort == False)

            if abort == True:
                break

        results = '%d,%d,%d,%d' % (epis_count, episode_reward, frame_idx, successful)
        if epis_count % 1000 == 0:
            local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('noisynet:'+'M'+str(M_ULA)+',seed'+str(seed) + ',' + results+','+str(local_time))
        f.write('\n' + results)
    f.close()