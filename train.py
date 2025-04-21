# gail_bc_wandb.py
import os
import yaml
import wandb
import torch
import pickle
import random
import numpy as np
import enlighten
import argparse

from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from loguru import logger
from copy import deepcopy

# === Load Config from YAML ===
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

try:
    WANDB_ENTITY=os.environ('WANDB_ENTITY')
except:
    logger.error(
        'Please define the WANDB_ENTITY environment variable for logging.'
    )
    exit()

wandb.init(
    entity=WANDB_ENTITY,
    project="cs8803-drl-project",
    name=f'{cfg["env_name"]}_{cfg["algorithm"]}_seed_{cfg["seed"]}',
    config=cfg)
config = wandb.config

# === Set seeds ===
SEED = config.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# === Constants ===
ENV_NAME = config.env_name
EPOCHS = config.epochs
GAMMA = config.gamma
N_ENVS = config.n_envs
N_EVAL_ENVS = config.n_eval_envs
BATCH_SIZE = config.batch_size
HIDDEN_DIM = config.hidden_dim
EVAL_EPISODES = config.eval_episodes
GAE_LAMBDA = config.gae_lambda
PPO_EPSILON = config.ppo_epsilon
LEARNING_RATE = config.learning_rate
GENERATOR_ITERATIONS = config.generator_iterations
DISCRIMINATOR_ITERATIONS = config.discriminator_iterations
ALGORITHM = config.algorithm
LOG_DIR = config.log_dir
GAMMA_D = config.gamma_d if "gamma_d" in config else None
GAMMA_G = config.gamma_g if "gamma_g" in config else None
EVAL_EVERY = config.eval_every if "eval_every" in config else 100

# === Helper ===
def make_network(in_dim, out_dim, hidden_dim=HIDDEN_DIM, device=device):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, out_dim)
    ).to(device)

# === Dataset ===
class TrajData:
    def __init__(self, n_steps, n_envs, n_obs, n_actions):
        s, e, o, a = n_steps, n_envs, n_obs, n_actions
        from torch import zeros

        self.states = zeros((s, e, o), device=device)
        self.actions = zeros((s, e, a), device=device)
        self.rewards = zeros((s, e), device=device)
        self.not_dones = zeros((s, e), device=device)

        self.log_probs = zeros((s, e), device=device)
        self.returns = zeros((s, e), device=device)
        self.advantages = zeros((s, e), device=device)

        self.n_steps = s

    def detach(self):
        self.actions = self.actions.detach()
        self.log_probs = self.log_probs.detach()

    def store(self, t, s, a, r, lp, d):
        self.states[t] = torch.tensor(s, dtype=torch.float, device=device)
        self.actions[t] = a.to(device)
        self.rewards[t] = torch.tensor(r, dtype=torch.float, device=device)

        self.log_probs[t] = lp.to(device)
        self.not_dones[t] = 1 - torch.tensor(d, dtype=torch.float, device=device)

    def calc_returns(self, values, last_value, gamma = GAMMA, gae_lambda = GAE_LAMBDA ):
        self.returns = deepcopy(self.rewards.detach())
        self.values = deepcopy(values)
        last_value = last_value.squeeze()
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps-1:
                delta = self.rewards[t] + gamma * last_value * self.not_dones[t] - self.values[t]
                self.advantages[t] = delta
            else:
                delta = self.rewards[t] + gamma * self.values[t+1] * self.not_dones[t] - self.values[t]
                self.advantages[t] = delta + gamma * gae_lambda * self.not_dones[t] * self.advantages[t+1]
            self.returns[t] = self.advantages[t] + self.values[t]

class R3GAILTrajData(TrajData):
    def __init__(self, n_steps, n_envs, n_obs, n_actions,
                 expert_dataset, gail_discriminator):
        super().__init__(n_steps, n_envs, n_obs, n_actions)

        self.expert_dataset = expert_dataset
        self.gail_discriminator = gail_discriminator

    def sample_expert_data(self, n):
        states, actions = self.expert_dataset.sample_batch(n)
        return torch.cat([states, actions], dim=-1)

    def update_rewards(self):
        if len(self.actions.shape) == 2:
            actions = self.actions.unsqueeze(-1)
        else:
            actions = self.actions
        sa = torch.cat([self.states, actions], dim=-1)
        expert_sa = self.sample_expert_data(sa.shape[0])
        self.rewards = self.gail_discriminator.get_rewards(sa, expert_sa).detach()

class GAILTrajData(TrajData):
    def __init__(self, n_steps, n_envs, n_obs, n_actions,
                 expert_dataset, gail_discriminator):
        super().__init__(n_steps, n_envs, n_obs, n_actions)

        self.expert_dataset = expert_dataset
        self.gail_discriminator = gail_discriminator

    def sample_expert_data(self, n):
        states, actions = self.expert_dataset.sample_batch(n)
        return torch.cat([states, actions], dim=-1)

    def update_rewards(self):
        if len(self.actions.shape) == 2:
            actions = self.actions.unsqueeze(-1)
        else:
            actions = self.actions
        sa = torch.cat([self.states, actions], dim=-1)
        self.rewards = self.gail_discriminator.get_rewards(sa).detach()

# R3GAIL Class
class R3GANGAILDiscriminator(torch.nn.Module):
    def __init__(self, state_dim, action_dim, gamma_D=0.001, gamma_G=0.001):
        super(R3GANGAILDiscriminator, self).__init__()
        self.model = make_network(
            state_dim + action_dim, 1, device=device
        )
        self.gamma_D = gamma_D
        self.gamma_G = gamma_G

    def preprocess(self, state, action):
        return torch.cat([state, action], dim=-1)

    def forward(self, state_action):
        shape = state_action.shape
        result = self.model(state_action.reshape(-1, shape[-1]))
        return result.reshape(*shape[:-1])

    def get_rewards(self, learner_sa, expert_sa):
        learner_sa.requires_grad_(True)
        learner_logits = self.forward(learner_sa)
        # expert_logits = self.forward(expert_sa)
        GP_penalty = self.compute_ZeroCenteredGP(learner_sa, learner_logits)

        # RelativisticLogits = expert_logits - learner_logits
        # AdversarialLoss = F.softplus(-RelativisticLogits)

        learner_sa.requires_grad_(False)
        return -F.logsigmoid(-learner_logits) - self.gamma_G * GP_penalty
        # return -AdversarialLoss - self.gamma_G * GP_penalty
        
    
    def get_loss(self, traj_data, writer, i):
        states, actions = traj_data.states, traj_data.actions
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(-1)
        learner_sa = torch.cat([states, actions], dim=-1)
        learner_shape = learner_sa.shape
        learner_sa = learner_sa.reshape(-1, learner_shape[-1])
        expert_sa = traj_data.sample_expert_data(learner_sa.shape[0])

        learner_sa.requires_grad_(True)
        expert_sa.requires_grad_(True)
        expert_logits = self.forward(expert_sa)
        expert_loss = F.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits))

        agent_logits = self.forward(learner_sa)
        agent_loss = F.binary_cross_entropy_with_logits(agent_logits, torch.zeros_like(agent_logits))


        RelativisticLogits = expert_logits - agent_logits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)

        accuracy = ((expert_logits > 0.5).float().mean() + (agent_logits < 0.5).float().mean()) / 2

        learner_sa.requires_grad_(True)
        expert_sa.requires_grad_(True)

        learner_sa.grad = None
        expert_sa.grad = None
        expert_GP = self.compute_ZeroCenteredGP(learner_sa, agent_logits)
        learner_GP = self.compute_ZeroCenteredGP(expert_sa, expert_logits)
        
        learner_sa.requires_grad_(False)
        expert_sa.requires_grad_(False)
        # we can reduce the 
        # loss = AdversarialLoss.mean() + self.gamma_D * (expert_GP + learner_GP).mean()
        loss = agent_loss + expert_loss + self.gamma_D * (expert_GP + learner_GP).mean()
        # loss = loss.mean()

        writer.add_scalar("accuracy", accuracy, i)
        wandb.log({"accuracy": accuracy}, step=i)
        return loss
    
    @staticmethod
    def compute_ZeroCenteredGP(sa, logits):
        Gradient, = torch.autograd.grad(outputs=logits.sum(), inputs=sa, create_graph=True)
        return Gradient.square().sum(
            list(range(len(Gradient.shape)))[1:]
        )

# GAIL Class
class GAILDiscriminator(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GAILDiscriminator, self).__init__()
        self.model = make_network(
            state_dim + action_dim, 1, device=device
        )

    def preprocess(self, state, action):
        return torch.cat([state, action], dim=-1)

    def forward(self, state_action):
        shape = state_action.shape
        result = self.model(state_action.reshape(-1, shape[-1]))
        return result.reshape(*shape[:-1])

    def get_rewards(self, state_action):
        logits = self.forward(state_action)
        return -F.logsigmoid(-logits)

    def get_loss(self, traj_data, writer, i):
        states, actions = traj_data.states, traj_data.actions
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(-1)
        learner_sa = torch.cat([states, actions], dim=-1)
        learner_shape = learner_sa.shape
        learner_sa = learner_sa.reshape(-1, learner_shape[-1])
        expert_sa = traj_data.sample_expert_data(learner_sa.shape[0])

        expert_logits = self.forward(expert_sa)
        expert_loss = F.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits))

        agent_logits = self.forward(learner_sa)
        agent_loss = F.binary_cross_entropy_with_logits(agent_logits, torch.zeros_like(agent_logits))

        accuracy = ((expert_logits > 0.5).float().mean() + (agent_logits < 0.5).float().mean()) / 2

        writer.add_scalar("accuracy", accuracy, i)
        return expert_loss + agent_loss

# PPO Implementation
class PPO(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.name = 'PPO'

        torch.manual_seed(SEED)  # needed before network init for fair comparison
        self.value = make_network(
            n_obs, 1, device=device
        )
        self.policy = make_network(
            n_obs, 2*n_actions, device=device
        )

    def get_loss(self, traj_data, epsilon=PPO_EPSILON):

        predicted_values = self.value(traj_data.states).squeeze(-1)
        returns = traj_data.returns
        loss_fn = nn.MSELoss()
        value_loss = loss_fn(predicted_values, traj_data.returns.detach()).mean()
        _, probs = self.get_action(traj_data.states)
        log_probs = probs.log_prob(traj_data.actions)
        old_log_probs = traj_data.log_probs.detach()
        ratio = torch.exp(log_probs - old_log_probs)
        advantage = traj_data.advantages
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        policy_loss = -torch.min(ratio * advantage.detach(), clipped_ratio * advantage.detach()).mean()
        loss = value_loss + policy_loss
        return loss

    def get_action(self, obs):
        logits = self.policy(obs)
        mean, std = torch.chunk(logits, 2, dim=-1)
        mean = torch.tanh(mean)
        # probs = categorical.Categorical(logits=logits)

        cov_mat = torch.diag_embed(F.softplus(std))#torch.diag(std)#unsqueeze(dim=0)
        probs = torch.distributions.MultivariateNormal(mean, cov_mat)
        actions = probs.rsample()
        return actions, probs

# R3GAIL Trainer
class R3GAILRunner:
    def __init__(self, expert_dataset):
        self.n_envs = BATCH_SIZE
        self.n_steps = BATCH_SIZE

        self.envs = gym.make_vec(ENV_NAME, num_envs=self.n_envs, vectorization_mode="sync")
        self.eval_envs = gym.make_vec(ENV_NAME, num_envs=N_EVAL_ENVS, vectorization_mode="sync")
        N_OBS: int=self.eval_envs.observation_space.shape[-1]
        N_ACTIONS: int=self.eval_envs.action_space.shape[-1]
        self.n_obs = N_OBS
        self.n_actions = N_ACTIONS

        self.learner = PPO(self.n_obs, n_actions=self.n_actions)  # 2 action choices are available

        self.discriminator = R3GANGAILDiscriminator(self.n_obs, self.n_actions, gamma_D=GAMMA_D, gamma_G=GAMMA_G)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=LEARNING_RATE)
        self.optimizer = Adam(self.learner.parameters(), lr=LEARNING_RATE)

        self.traj_data = R3GAILTrajData(self.n_steps, self.n_envs, self.n_obs, n_actions=self.n_actions,
                                      expert_dataset=expert_dataset, gail_discriminator=self.discriminator) # 1 action choice is made

        self.writer = SummaryWriter(log_dir=f'{LOG_DIR}/R3GAIL/seed_{SEED}')

    def rollout(self, i):
        obs, _ = self.envs.reset(seed=SEED)
        obs = torch.tensor(obs, dtype=torch.float, device=device)

        for t in range(self.n_steps):
            # PPO doesnt use gradients here, but REINFORCE and VPG do.
            with torch.no_grad() if self.learner.name == 'PPO' else torch.enable_grad():
                actions, probs = self.learner.get_action(obs)
            log_probs = probs.log_prob(actions)
            next_obs, rewards, done, truncated, infos = self.envs.step(actions.detach().cpu().numpy())
            done = done | truncated  # episode doesnt truncate till t = 500, so never
            self.traj_data.store(t, obs, actions, rewards, log_probs, done)
            obs = torch.tensor(next_obs,dtype=torch.float,  device=device)
        last_value = self.learner.value(obs).detach()
        values = self.learner.value(self.traj_data.states).detach().squeeze()
        self.writer.add_scalar("Reward/original", self.traj_data.rewards.mean(), i)
        wandb.log({"original_reward": self.traj_data.rewards.mean()}, step=i)
        self.traj_data.update_rewards()
        self.traj_data.calc_returns(values, last_value=last_value)

        self.writer.add_scalar("Reward/GAIL", self.traj_data.rewards.clone().detach().mean(), i)
        wandb.log({"gail_reward": self.traj_data.rewards.clone().detach().mean()}, step=i)
        self.writer.flush()

    def update(self, i):
        learner_epochs = GENERATOR_ITERATIONS
        disc_epochs = DISCRIMINATOR_ITERATIONS

        disc_losses = []
        learner_losses = []
        for _ in range(disc_epochs):
            disc_loss = self.discriminator.get_loss(self.traj_data, self.writer, i)
            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            self.discriminator_optimizer.step()
            disc_losses.append(disc_loss.detach().item())

        for _ in range(learner_epochs):
            learner_loss = self.learner.get_loss(self.traj_data)
            self.optimizer.zero_grad()
            learner_loss.backward()
            self.optimizer.step()
            learner_losses.append(learner_loss.detach().item())

        self.writer.add_scalar("loss/learner_loss", sum(learner_losses) / len(learner_losses), i)
        wandb.log({"learner_loss": sum(learner_losses) / len(learner_losses)}, step=i)
        self.writer.add_scalar("loss/disc_loss", sum(disc_losses) / len(disc_losses), i)
        wandb.log({"disc_loss": sum(disc_losses) / len(disc_losses)}, step=i)
        self.writer.flush()
        self.traj_data.detach()

    def evaluate_policy(self, i, n_eval_episodes = 5):
        obs, _ = self.eval_envs.reset(seed=SEED)
        obs = torch.tensor(obs, dtype=torch.float, device=device)
        episode_counts = np.zeros(self.n_envs, dtype="int")
        episode_count_targets = np.array([(n_eval_episodes + i) // self.n_envs for i in range(self.n_envs)], dtype="int")
        rewardsum_current = np.zeros(self.n_envs)
        rewardsum_untildone =[]
        dones = np.zeros(self.n_envs, dtype="bool")
        while (episode_counts < episode_count_targets).any():
            with torch.no_grad() if self.learner.name == 'PPO' else torch.enable_grad():
                actions, probs = self.learner.get_action(obs)
            next_obs, rewards, done, truncated, infos = self.eval_envs.step(actions.detach().cpu().numpy())
            done = done | truncated  # episode doesnt truncate till t = 500, so never
            rewardsum_current += rewards
            obs = torch.tensor(next_obs, dtype=torch.float, device=device)
            for env in range(self.n_envs):
                if episode_counts[env] < episode_count_targets[env]:
                    if done[env]:
                        rewardsum_untildone.append(rewardsum_current[env])
                        rewardsum_current[env] = 0
                        episode_counts[env] += 1
        if rewardsum_untildone:
            mean_rewardsum = np.mean(rewardsum_untildone)
            std_rewardsum = np.std(rewardsum_untildone)
            self.writer.add_scalar("Reward/evaluation", mean_rewardsum, i)
            wandb.log({"evaluation_reward": mean_rewardsum, 'evaluation_reward_std': std_rewardsum}, step=i)
        else:
            mean_rewardsum = 0
            std_rewardsum = 0
            self.writer.add_scalar("Reward/evaluation", 0, i)
            wandb.log({"evaluation_reward": mean_rewardsum, 'evaluation_reward_std': std_rewardsum}, step=i)

        return mean_rewardsum, std_rewardsum

# GAIL Runner
class GAILRunner:
    def __init__(self, expert_dataset):
        self.n_envs = BATCH_SIZE
        self.n_steps = BATCH_SIZE

        # update to match with R3GAIL
        self.envs = gym.make_vec(ENV_NAME, num_envs=self.n_envs, vectorization_mode="sync")
        self.eval_envs = gym.make_vec(ENV_NAME, num_envs=N_EVAL_ENVS, vectorization_mode="sync")
        N_OBS: int=self.eval_envs.observation_space.shape[-1]
        N_ACTIONS: int=self.eval_envs.action_space.shape[-1]
        self.n_obs = N_OBS
        self.n_actions = N_ACTIONS
        self.learner = PPO(self.n_obs, n_actions=self.n_actions)  # 2 action choices are available

        self.discriminator = GAILDiscriminator(self.n_obs, self.n_actions)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=LEARNING_RATE)
        self.optimizer = Adam(self.learner.parameters(), lr=LEARNING_RATE)

        self.traj_data = GAILTrajData(self.n_steps, self.n_envs, self.n_obs, n_actions=self.n_actions,
                                      expert_dataset=expert_dataset, gail_discriminator=self.discriminator) # 1 action choice is made

        self.writer = SummaryWriter(log_dir=f'{LOG_DIR}/GAIL/seed_{SEED}')

    def rollout(self, i):
        obs, _ = self.envs.reset()
        obs = torch.tensor(obs, dtype=torch.float, device=device)

        for t in range(self.n_steps):
            # PPO doesnt use gradients here, but REINFORCE and VPG do.
            with torch.no_grad() if self.learner.name == 'PPO' else torch.enable_grad():
                actions, probs = self.learner.get_action(obs)
            log_probs = probs.log_prob(actions)
            next_obs, rewards, done, truncated, infos = self.envs.step(actions.detach().cpu().numpy())
            done = done | truncated  # episode doesnt truncate till t = 500, so never
            self.traj_data.store(t, obs, actions, rewards, log_probs, done)
            obs = torch.tensor(next_obs,dtype=torch.float,  device=device)
        last_value = self.learner.value(obs).detach()
        values = self.learner.value(self.traj_data.states).detach().squeeze()
        self.writer.add_scalar("Reward/original", self.traj_data.rewards.mean(), i)
        wandb.log({"original_reward": self.traj_data.rewards.mean()}, step=i)
        self.traj_data.update_rewards()
        self.traj_data.calc_returns(values, last_value=last_value)

        self.writer.add_scalar("Reward/GAIL", self.traj_data.rewards.clone().detach().mean(), i)
        wandb.log({"gail_reward": self.traj_data.rewards.clone().detach().mean()}, step=i)
        self.writer.flush()

    def update(self, i):
        learner_epochs = GENERATOR_ITERATIONS
        disc_epochs = DISCRIMINATOR_ITERATIONS

        disc_losses = []
        learner_losses = []
        for _ in range(disc_epochs):
            disc_loss = self.discriminator.get_loss(self.traj_data, self.writer, i)
            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            self.discriminator_optimizer.step()
            disc_losses.append(disc_loss.detach().item())

        for _ in range(learner_epochs):
            learner_loss = self.learner.get_loss(self.traj_data)
            self.optimizer.zero_grad()
            learner_loss.backward()
            self.optimizer.step()
            learner_losses.append(learner_loss.detach().item())

        self.writer.add_scalar("loss/learner_loss", sum(learner_losses) / len(learner_losses), i)
        wandb.log({"learner_loss": sum(learner_losses) / len(learner_losses)}, step=i)
        self.writer.add_scalar("loss/disc_loss", sum(disc_losses) / len(disc_losses), i)
        wandb.log({"disc_loss": sum(disc_losses) / len(disc_losses)}, step=i)
        self.writer.flush()
        self.traj_data.detach()

    def evaluate_policy(self, i, n_eval_episodes = 5):
        obs, _ = self.eval_envs.reset(seed=SEED)
        obs = torch.tensor(obs, dtype=torch.float, device=device)
        episode_counts = np.zeros(self.n_envs, dtype="int")
        episode_count_targets = np.array([(n_eval_episodes + i) // self.n_envs for i in range(self.n_envs)], dtype="int")
        rewardsum_current = np.zeros(self.n_envs)
        rewardsum_untildone =[]
        dones = np.zeros(self.n_envs, dtype="bool")
        while (episode_counts < episode_count_targets).any():
            with torch.no_grad() if self.learner.name == 'PPO' else torch.enable_grad():
                actions, probs = self.learner.get_action(obs)
            next_obs, rewards, done, truncated, infos = self.eval_envs.step(actions.detach().cpu().numpy())
            done = done | truncated  # episode doesnt truncate till t = 500, so never
            rewardsum_current += rewards
            obs = torch.tensor(next_obs, dtype=torch.float, device=device)
            for env in range(self.n_envs):
                if episode_counts[env] < episode_count_targets[env]:
                    if done[env]:
                        rewardsum_untildone.append(rewardsum_current[env])
                        rewardsum_current[env] = 0
                        episode_counts[env] += 1
        if rewardsum_untildone:
            mean_rewardsum = np.mean(rewardsum_untildone)
            std_rewardsum = np.std(rewardsum_untildone)
            self.writer.add_scalar("Reward/evaluation", mean_rewardsum, i)
            wandb.log({"evaluation_reward": mean_rewardsum, 'evaluation_reward_std': std_rewardsum}, step=i)
        else:
            mean_rewardsum = 0
            std_rewardsum = 0
            self.writer.add_scalar("Reward/evaluation", 0, i)
            wandb.log({"evaluation_reward": mean_rewardsum, 'evaluation_reward_std': std_rewardsum}, step=i)

        return mean_rewardsum, std_rewardsum

# BC Deterministic Runner
class BCDeterministicRunner:
    def __init__(self, expert_dataset):
        self.n_envs = BATCH_SIZE
        self.n_steps = BATCH_SIZE
    
        self.envs = gym.make_vec(ENV_NAME, num_envs=self.n_envs, vectorization_mode="sync")
        self.eval_envs = gym.make_vec(ENV_NAME, num_envs=N_EVAL_ENVS, vectorization_mode="sync")
        N_OBS: int=self.eval_envs.observation_space.shape[-1]
        N_ACTIONS: int=self.eval_envs.action_space.shape[-1]
        self.n_obs = N_OBS
        self.n_actions = N_ACTIONS

        self.policy = BCPolicy(n_obs=self.n_obs, n_actions=self.n_actions, stochastic=False).to(device)
        self.optimizer = Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.loader = DataLoader(expert_dataset, batch_size=BATCH_SIZE, shuffle=True)

        self.writer = SummaryWriter(log_dir=f"{LOG_DIR}/BC_Det/seed_{SEED}")
        self.manager = enlighten.get_manager()

    def evaluate_policy(self, i, n_eval_episodes = 5):
        obs, _ = self.envs.reset()
        obs = torch.tensor(obs, dtype=torch.float, device=device)
        episode_counts = np.zeros(self.n_envs, dtype="int")
        episode_count_targets = np.array([(n_eval_episodes + i) // self.n_envs for i in range(self.n_envs)], dtype="int")
        rewardsum_current = np.zeros(self.n_envs)
        rewardsum_untildone =[]
        dones = np.zeros(self.n_envs, dtype="bool")
        self.policy.eval()
        while (episode_counts < episode_count_targets).any():
            with torch.no_grad():
                actions, probs = self.policy.get_action(obs)
            next_obs, rewards, done, truncated, infos = self.envs.step(actions.detach().cpu().numpy())
            done = done | truncated  # episode doesnt truncate till t = 500, so never
            rewardsum_current += rewards
            obs = torch.tensor(next_obs, dtype=torch.float, device=device)
            for env in range(self.n_envs):
                if episode_counts[env] < episode_count_targets[env]:
                    if done[env]:
                        rewardsum_untildone.append(rewardsum_current[env])
                        rewardsum_current[env] = 0
                        episode_counts[env] += 1
        if rewardsum_untildone:
            mean_rewardsum = np.mean(rewardsum_untildone)
            std_rewardsum = np.std(rewardsum_untildone)
            self.writer.add_scalar("Reward/evaluation", mean_rewardsum, i)
            wandb.log({"evaluation_reward": mean_rewardsum, 'evaluation_reward_std': std_rewardsum}, step=i)
        else:
            mean_rewardsum = 0
            std_rewardsum = 0
            self.writer.add_scalar("Reward/evaluation", 0, i)
            wandb.log({"evaluation_reward": mean_rewardsum, 'evaluation_reward_std': std_rewardsum}, step=i)

        return mean_rewardsum, std_rewardsum
    
    def update(self, i):
        # perform one epoch of training
        self.policy.train()
        for batch_i, (s, a) in enumerate(self.loader):
            s = s.to(device)
            a = a.to(device)

            # Forward pass
            pred = self.policy(s)                  # shape [B, action_dim]
            loss = F.mse_loss(pred, a)        # MSE for continuous deterministic BC

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log to TensorBoard
            self.writer.add_scalar("loss/mse", loss.item(), i)

            wandb.log({"mse_loss": loss.item()}, step=i)

# BC Stochastic Runner
class BCStochasticRunner:
    def __init__(self, expert_dataset):
        self.n_envs = BATCH_SIZE
        self.n_steps = BATCH_SIZE
    
        self.envs = gym.make_vec(ENV_NAME, num_envs=self.n_envs, vectorization_mode="sync")
        self.eval_envs = gym.make_vec(ENV_NAME, num_envs=N_EVAL_ENVS, vectorization_mode="sync")
        N_OBS: int=self.eval_envs.observation_space.shape[-1]
        N_ACTIONS: int=self.eval_envs.action_space.shape[-1]
        self.n_obs = N_OBS
        self.n_actions = N_ACTIONS

        self.policy = BCPolicy(n_obs=self.n_obs, n_actions=self.n_actions, stochastic=True).to(device)
        self.optimizer = Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.loader = DataLoader(expert_dataset, batch_size=BATCH_SIZE, shuffle=True)

        self.writer = SummaryWriter(log_dir=f"{LOG_DIR}/BC_Stoch/seed_{SEED}")
        self.manager = enlighten.get_manager()

    def evaluate_policy(self, i, n_eval_episodes = 5):
        obs, _ = self.envs.reset()
        obs = torch.tensor(obs, dtype=torch.float, device=device)
        episode_counts = np.zeros(self.n_envs, dtype="int")
        episode_count_targets = np.array([(n_eval_episodes + i) // self.n_envs for i in range(self.n_envs)], dtype="int")
        rewardsum_current = np.zeros(self.n_envs)
        rewardsum_untildone =[]
        dones = np.zeros(self.n_envs, dtype="bool")
        self.policy.eval()
        while (episode_counts < episode_count_targets).any():
            with torch.no_grad():
                actions, probs = self.policy.get_action(obs)
            next_obs, rewards, done, truncated, infos = self.envs.step(actions.detach().cpu().numpy())
            done = done | truncated  # episode doesnt truncate till t = 500, so never
            rewardsum_current += rewards
            obs = torch.tensor(next_obs, dtype=torch.float, device=device)
            for env in range(self.n_envs):
                if episode_counts[env] < episode_count_targets[env]:
                    if done[env]:
                        rewardsum_untildone.append(rewardsum_current[env])
                        rewardsum_current[env] = 0
                        episode_counts[env] += 1
        if rewardsum_untildone:
            mean_rewardsum = np.mean(rewardsum_untildone)
            std_rewardsum = np.std(rewardsum_untildone)
            self.writer.add_scalar("Reward/evaluation", mean_rewardsum, i)
            wandb.log({"evaluation_reward": mean_rewardsum, 'evaluation_reward_std': std_rewardsum}, step=i)
        else:
            mean_rewardsum = 0
            std_rewardsum = 0
            self.writer.add_scalar("Reward/evaluation", 0, i)
            wandb.log({"evaluation_reward": mean_rewardsum, 'evaluation_reward_std': std_rewardsum}, step=i)

        return mean_rewardsum, std_rewardsum
    
    def update(self, i):
        # perform one epoch of training
        self.policy.train()
        for batch_i, (s, a) in enumerate(self.loader):
            s = s.to(device)
            a = a.to(device)

            # Forward pass => (mean, std)
            mean, std = self.policy(s)
            dist = torch.distributions.Normal(mean, std)

            # Negative log-likelihood of the expert actions
            loss = -dist.log_prob(a).mean()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log to TensorBoard
            self.writer.add_scalar("loss/nll", loss.item(), i)
            
            wandb.log({"nll_loss": loss.item()}, step=i)

        


# Another dataset class for expert data
class ExpertDataset(Dataset):
    def __init__(self, states, actions):
        super().__init__()
        assert len(states) == len(actions)
        self.states = torch.from_numpy(states).float().to(device)
        self.actions = torch.from_numpy(actions).float().to(device)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

    def sample_batch(self, batch_size):
        """Sample a batch of states and actions"""
        indices = torch.randint(0, len(self), (batch_size,))
        return self.states[indices], self.actions[indices]


# BC Policy
class BCPolicy(nn.Module):
    """
    Same as before, but we’ll keep it here for clarity.
    If stochastic=True, outputs (mean, std).
    If stochastic=False, outputs just the deterministic action.
    """
    def __init__(self, n_obs, n_actions, stochastic=False):
        super().__init__()
        self.stochastic = stochastic

        torch.manual_seed(SEED)  # needed before network init for fair comparison
        self.net = make_network(
            n_obs, HIDDEN_DIM, device=device
        )

        if stochastic:
            self.head = nn.Linear(HIDDEN_DIM, 2*n_actions)  # mean + log_std
        else:
            self.head = nn.Linear(HIDDEN_DIM, n_actions)

    def forward(self, states):
        x = self.net(states)
        out = self.head(x)
        if self.stochastic:
            mean, log_std = torch.chunk(out, 2, dim=-1)
            # We'll do a softplus so std is positive
            std = F.softplus(log_std)
            return mean, std
        else:
            return out

    @torch.no_grad()
    def get_action(self, obs):
        """
        For env rollout. If stochastic => sample from Normal(mean, std).
        If deterministic => just output the direct action from the net.
        """
        if self.stochastic:
            mean, std = self.forward(obs)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
        else:
            action = self.forward(obs)
        return action, None  # mimic the signature from PPO

# begin the training logic
print(f"expert_data/{ENV_NAME}_25.pkl")
with open(f"expert_data/{ENV_NAME}_25.pkl", 'rb') as f:
    expert_dataset = pickle.load(f)

num_expert_trajs = 2
exp_states = np.stack(expert_dataset['states'][:num_expert_trajs])
exp_actions = np.stack(expert_dataset['actions'][:num_expert_trajs])

if len(exp_actions.shape) == 2:
    exp_actions = np.expand_dims(exp_actions, axis=-1)

# Prepare data for ExpertDataset
exp_states_flat = exp_states.reshape(-1, exp_states.shape[-1])
exp_actions_flat = exp_actions.reshape(-1, exp_actions.shape[-1])

# Create ExpertDataset instance
expert_dataset = ExpertDataset(exp_states_flat, exp_actions_flat)

# training logic

if ALGORITHM == "R3GAIL":
    gail = R3GAILRunner(expert_dataset=expert_dataset)

    # Create enlighten manager for progress tracking
    manager = enlighten.get_manager()
    epochs_pbar = manager.counter(total=EPOCHS, desc="Training epochs", unit="epochs")

    returns_by_epoch = np.empty((0,3))

    for i in range(EPOCHS):
        gail.rollout(i)
        gail.update(i)
        epochs_pbar.update()
        wandb.log({"epoch": i}, step=i)
        if i % EVAL_EVERY == 0 or i == EPOCHS - 1:
            [mean, std] = gail.evaluate_policy(i)
            returns_by_epoch = np.append(returns_by_epoch, np.array([[i, mean, std]]), axis=0)
    
    # TODO: do we need to save the model?

elif ALGORITHM == "GAIL":
    gail = GAILRunner(expert_dataset=expert_dataset)

    # Create enlighten manager for progress tracking
    manager = enlighten.get_manager()
    epochs_pbar = manager.counter(total=EPOCHS, desc="Training epochs", unit="epochs")

    returns_by_epoch = np.empty((0,3))

    for i in range(EPOCHS):
        gail.rollout(i)
        gail.update(i)
        epochs_pbar.update()
        wandb.log({"epoch": i}, step=i)
        if i % EVAL_EVERY == 0 or i == EPOCHS - 1:
            [mean, std] = gail.evaluate_policy(i)
            returns_by_epoch = np.append(returns_by_epoch, np.array([[i, mean, std]]), axis=0)
    # TODO: do we need to save the model?

elif ALGORITHM == "BC_Det":
    bc_det = BCDeterministicRunner(expert_dataset=expert_dataset)

    # Create enlighten manager for progress tracking
    manager = enlighten.get_manager()
    epochs_pbar = manager.counter(total=EPOCHS, desc="Training epochs", unit="epochs")

    returns_by_epoch = np.empty((0,3))

    for i in range(EPOCHS):
        bc_det.update(i)
        epochs_pbar.update()
        wandb.log({"epoch": i}, step=i)
        if i % EVAL_EVERY == 0 or i == EPOCHS - 1:
            [mean, std] = bc_det.evaluate_policy(i)
            returns_by_epoch = np.append(returns_by_epoch, np.array([[i, mean, std]]), axis=0)


elif ALGORITHM == "BC_Stoch":
    bc_stoch = BCStochasticRunner(expert_dataset=expert_dataset)

    # Create enlighten manager for progress tracking
    manager = enlighten.get_manager()
    epochs_pbar = manager.counter(total=EPOCHS, desc="Training epochs", unit="epochs")

    returns_by_epoch = np.empty((0,3))

    for i in range(EPOCHS):
        bc_stoch.update(i)
        epochs_pbar.update()
        wandb.log({"epoch": i}, step=i)
        if i % EVAL_EVERY == 0 or i == EPOCHS - 1:
            [mean, std] = bc_stoch.evaluate_policy(i)
            returns_by_epoch = np.append(returns_by_epoch, np.array([[i, mean, std]]), axis=0)
