from actor import GaussianPolicy
from buffer import ReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

class SACAgent: 
    def __init__(self, state_dim, action_dim, action_bound, gamma=0.99, tau=0.005, alpha=0.2):
        print("Initializing SAC Agent...")
        self.actor = GaussianPolicy(state_dim, action_dim)
        self.q1 = MLP(state_dim + action_dim, 1)
        self.q2 = MLP(state_dim + action_dim, 1)
        self.q1_target = MLP(state_dim + action_dim, 1)
        self.q2_target = MLP(state_dim + action_dim, 1)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=3e-4)

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-4)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_bound = action_bound

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        if evaluate:
            mean, _ = self.actor.forward(state)
            action = torch.tanh(mean)
        else:
            action, _ = self.actor.sample(state)
        return (action * self.action_bound).detach().numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            next_q1 = self.q1_target(torch.cat([next_state, next_action], 1))
            next_q2 = self.q2_target(torch.cat([next_state, next_action], 1))
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * next_q

        current_q1 = self.q1(torch.cat([state, action], 1))
        current_q2 = self.q2(torch.cat([state, action], 1))

        q1_loss = nn.MSELoss()(current_q1, target_q)
        q2_loss = nn.MSELoss()(current_q2, target_q)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        new_action, log_prob = self.actor.sample(state)
        q1_new = self.q1(torch.cat([state, new_action], 1))
        q2_new = self.q2(torch.cat([state, new_action], 1))
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_new).mean()

        

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().detach()

        

        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        print(f"Q1 Loss: {q1_loss.item():.4f}, Q2 Loss: {q2_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}, Alpha Loss: {alpha_loss.item():.4f}")

