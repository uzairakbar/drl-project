# %% [markdown]
# # Group project for CS8803-DRL

# %%
import os
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"
# os.environ["MUJOCO_GL"] = "omesa"  # Alternatively, try "osmesa" if you still face issues
# os.environ["MUJOCO_PY_MUJOCO_PATH"] = "/Users/uzair/.mujoco/mujoco210"

# %%
import torch
import pickle
import random
import enlighten
import numpy as np
from torch import nn
import gymnasium as gym
from loguru import logger
from copy import deepcopy
from torch.optim import Adam
from torch.nn import functional as F
from torch.distributions import categorical
from torch.utils.tensorboard import SummaryWriter
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# get SEED from cli args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
SEED = args.seed
# random seeds for reproducability
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)
logger.info(f'Random seed set as {SEED}.')

# %%

#@title Device check
def test_device_is_gpu():
    if("cuda" in device.type):
        logger.info(f"Test passed: Device is GPU!.")
    else:
        logger.info(f"Test failed: Device is not GPU! Continuing with CPU.")

# Run the test
test_device_is_gpu()

# %% [markdown]
# ## Tensorboard

# %%
# # Launch TensorBoard
# %load_ext tensorboard
# %tensorboard --logdir runs

# %%
#@title Hyperparameters
ENV_NAME: str="Ant-v5" # or "HalfCheetah-v2"
LOG_DIR: str=f"{ENV_NAME.split('-')[0]}_runs"
EPOCHS: int=1000
GAMMA: float=0.99
N_ENVS: int=64
N_EVAL_ENVS: int=64
BATCH_SIZE: int=64
HIDDEN_DIM: int=64
EVAL_EPISODES: int=16
GAE_LAMBDA: float=0.95
PPO_EPSILON: float=0.2
LEARNING_RATE: float=4e-4
GENERATOR_ITERATIONS: int=32
DISCRIMINATOR_ITERATIONS: int=1

eval_envs = gym.make_vec(ENV_NAME, num_envs=N_EVAL_ENVS, vectorization_mode="sync")

N_OBS: int=eval_envs.observation_space.shape[-1]
N_ACTIONS: int=eval_envs.action_space.shape[-1]
# I've defined this function here to reduce code duplicaiton
def make_network(in_dim, out_dim, hidden_dim=HIDDEN_DIM, device=device):
    """
    Returns a NN with the specified dimensions.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim)
    ).to(device)

# %%
# Let's build upon the course PPO implementation
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
        expert_sa = self.sample_expert_data(sa.shape[0])
        self.rewards = self.gail_discriminator.get_rewards(sa, expert_sa).detach()

# %% [markdown]
# # **GAIL Discriminator**
# The **GAIL Discriminator** is a neural network that distinguishes expert trajectories from trajectories generated by the policy. This discriminator acts as a reward function for the agent in adversarial imitation learning.
# 1. **Network Architecture**</br>
# The discriminator is a neural network that takes state-action pairs $(s,a)$ as input. Outputs a logit score indicating whether the input comes from an expert (1) or from the agent (0).
# Mathematically, the discriminator is modeled as:
# $$D_Θ(s,a)=σ(f_Θ(s,a)).$$
# where:
# - $f_\theta(s, a)$ is the neural network output (logits).
# - $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid activation function.
# 
# 2. **Loss Function** </br>
# The **discriminator loss** is computed using **binary cross-entropy**: $$L_D = \mathbb{E}_{(s, a) \sim \pi} \left[ \log(1 - D(s, a)) \right] + \mathbb{E}_{(s, a) \sim \pi_E} \left[ \log D(s, a) \right]$$ which is implemented as: $$
# L_D = \text{BCEWithLogits}(D(s,a), 0) + \text{BCEWithLogits}(D(s_E,a_E), 1)
# $$ where:
# - $\pi$ is the agent policy.
# - $\pi_E$ is the expert policy.
# - $\text{BCEWithLogits}$ applies **binary cross-entropy with logits**.
# 
# 3. **Accuracy Metric** </br>
# The discriminator's **accuracy** is defined as: $$\text{Accuracy} = \frac{1}{2} \left( \mathbb{E}_{(s_E, a_E)} [D(s_E, a_E) > 0.5] + \mathbb{E}_{(s, a)} [D(s, a) < 0.5] \right)$$ which measures how well the discriminator separates expert and agent samples.
# 
# 

# %%
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
        return loss
    
    @staticmethod
    def compute_ZeroCenteredGP(sa, logits):
        Gradient, = torch.autograd.grad(outputs=logits.sum(), inputs=sa, create_graph=True)
        return Gradient.square().sum(
            list(range(len(Gradient.shape)))[1:]
        )

# %% [markdown]
# ## **Proximal Policy Optimization (PPO)**
# In GAIL, the Proximal Policy Optimization (PPO) algorithm is used as the policy optimizer (the generator in the adversarial framework). The generator is responsible for producing trajectories that closely resemble expert demonstrations.

# %%
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

# %% [markdown]
# ## **Training GAIL**
# 
# The `GAILRunner` class orchestrates the **training loop for GAIL** by managing trajectory collection, reward computation, and policy updates. In `rollout()`, the agent interacts with the environment using a **PPO policy**, storing states, actions, and log probabilities. Instead of using environment rewards, the **discriminator assigns rewards** based on expert similarity: $$
# r_{\text{GAIL}}(s, a) = -\log \sigma(-D_\phi(s, a))
# $$ where \( D_\phi(s, a) \) is the discriminator’s logit output, and \( \sigma(x) = \frac{1}{1 + e^{-x}} \) is the sigmoid function.
# 
# In `update()`, the **discriminator** is trained using **binary cross-entropy loss** to differentiate between expert and agent-generated trajectories. The **policy (generator)** is optimized using **PPO’s clipped objective**: $$
# L^{\text{CLIP}}(\theta) = \mathbb{E} \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right) \right]
# $$ where $ r_t(\theta) $ is the probability ratio of new and old policy actions. By iteratively improving both the **discriminator** and **policy**, `GAILRunner` ensures the agent progressively mimics expert behavior while avoiding direct reliance on environment rewards.

# %%
class R3GAILRunner:
    def __init__(self, expert_dataset):
        self.n_envs = BATCH_SIZE
        self.n_steps = BATCH_SIZE
        self.n_obs = N_OBS
        self.n_actions = N_ACTIONS

        self.envs = gym.make_vec(ENV_NAME, num_envs=self.n_envs, vectorization_mode="sync")
        self.eval_envs = gym.make_vec(ENV_NAME, num_envs=N_EVAL_ENVS, vectorization_mode="sync")

        self.learner = PPO(self.n_obs, n_actions=self.n_actions)  # 2 action choices are available

        self.discriminator = R3GANGAILDiscriminator(self.n_obs, self.n_actions)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=LEARNING_RATE)
        self.optimizer = Adam(self.learner.parameters(), lr=LEARNING_RATE)

        self.traj_data = GAILTrajData(self.n_steps, self.n_envs, self.n_obs, n_actions=self.n_actions,
                                      expert_dataset=expert_dataset, gail_discriminator=self.discriminator) # 1 action choice is made

        self.writer = SummaryWriter(log_dir=f'{LOG_DIR}/R3GAIL/seed_{SEED}')

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
        self.traj_data.update_rewards()
        self.traj_data.calc_returns(values, last_value=last_value)

        self.writer.add_scalar("Reward/GAIL", self.traj_data.rewards.clone().detach().mean(), i)
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
        self.writer.add_scalar("loss/disc_loss", sum(disc_losses) / len(disc_losses), i)
        self.writer.flush()
        self.traj_data.detach()

    def evaluate_policy(self, i, n_eval_episodes = 5):
        obs, _ = self.eval_envs.reset()
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
        else:
            mean_rewardsum = 0
            std_rewardsum = 0
            self.writer.add_scalar("Reward/evaluation", 0, i)

        return mean_rewardsum, std_rewardsum

# %% [markdown]
# ## **Expert Dataset**
# The provided code defines and loads an **expert dataset** for training a **GAIL discriminator** or a **Behavior Cloning (BC)** policy. The `ExpertDataset` class stores expert state-action pairs, converting them into PyTorch tensors for efficient sampling. It provides methods for retrieving individual samples and randomly sampling batches for training. The dataset is loaded from a pre-saved file (`HalfCheetah-v2_25.pkl`), extracting the first two expert trajectories of states and actions. The actions are reshaped if necessary to ensure correct dimensionality. The extracted data is then flattened to remove sequence dependencies, making it compatible with neural network training. Finally, an `ExpertDataset` instance is created, which serves as input to the GAIL discriminator (for adversarial training) or the BC policy (for supervised learning).

# %%
from torch.utils.data import Dataset, DataLoader

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

# %%
# Load the expert data
print(f"{ENV_NAME}_25.pkl")
with open(f"./{ENV_NAME}_25.pkl", 'rb') as f:
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

# %%
gail = R3GAILRunner(expert_dataset=expert_dataset)

# Create enlighten manager for progress tracking
manager = enlighten.get_manager()
epochs_pbar = manager.counter(total=EPOCHS, desc="Training epochs", unit="epochs")

returns_by_epoch = np.empty((0,3))

for i in range(EPOCHS):
    gail.rollout(i)
    gail.update(i)
    epochs_pbar.update()
    if i%10 == 0:
      [mean, std] = gail.evaluate_policy(i)
      returns_by_epoch = np.append(returns_by_epoch, np.array([[i, mean, std]]), axis=0)

# %% [markdown]
# ## **Behavior Cloning**
# 
# 
# 
# 

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


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

# %%
def train_bc_deterministic(
    expert_dataset,
    n_obs,
    n_actions,
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    run_name="BC_Det"
):
    # 1) Create policy
    policy = BCPolicy(n_obs=n_obs, n_actions=n_actions, stochastic=False).to(device)
    optimizer = Adam(policy.parameters(), lr=LEARNING_RATE)

    # 2) Create DataLoader
    loader = DataLoader(expert_dataset, batch_size=batch_size, shuffle=True)

    # 3) Create TensorBoard writer
    writer = SummaryWriter(log_dir=f"{LOG_DIR}/{run_name}/seed_{SEED}")

    # 4) Training Loop
    global_step = 0
    policy.train()

    # Create enlighten manager for progress tracking
    manager = enlighten.get_manager()
    epochs_counter = manager.counter(total=n_epochs, desc="Epochs", unit="epochs")

    for epoch in range(n_epochs):
        # Create counter for batches within this epoch
        batches_counter = manager.counter(
            total=len(loader),
            desc=f"Epoch {epoch+1} batches",
            unit="batches",
            leave=False
        )

        for batch_i, (s, a) in enumerate(loader):
            s = s.to(device)
            a = a.to(device)

            # Forward pass
            pred = policy(s)                  # shape [B, action_dim]
            loss = F.mse_loss(pred, a)        # MSE for continuous deterministic BC

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log to TensorBoard
            writer.add_scalar("loss/mse", loss.item(), global_step)
            global_step += 1

            # Update batch counter
            batches_counter.update()

        # Update epoch counter
        batches_counter.close()
        epochs_counter.update()

    writer.close()
    return policy

# %%
def train_bc_stochastic(
    expert_dataset,
    n_obs,
    n_actions,
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    run_name="BC_Stoch"
):
    # 1) Create policy
    policy = BCPolicy(n_obs=n_obs, n_actions=n_actions, stochastic=True).to(device)
    optimizer = Adam(policy.parameters(), lr=LEARNING_RATE)

    # 2) Create DataLoader
    loader = DataLoader(expert_dataset, batch_size=batch_size, shuffle=True)

    # 3) Create TensorBoard writer
    writer = SummaryWriter(log_dir=f"{LOG_DIR}/{run_name}/seed_{SEED}")

    # 4) Training Loop
    global_step = 0
    policy.train()

    # Create enlighten manager for progress tracking
    manager = enlighten.get_manager()
    epochs_counter = manager.counter(total=n_epochs, desc="Epochs", unit="epochs")

    for epoch in range(n_epochs):
        # Create counter for batches within this epoch
        batches_counter = manager.counter(
            total=len(loader),
            desc=f"Epoch {epoch+1} batches",
            unit="batches",
            leave=False
        )

        for batch_i, (s, a) in enumerate(loader):
            s = s.to(device)
            a = a.to(device)

            # Forward pass => (mean, std)
            mean, std = policy(s)
            dist = torch.distributions.Normal(mean, std)

            # Negative log-likelihood of the expert actions
            loss = -dist.log_prob(a).mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log to TensorBoard
            writer.add_scalar("loss/nll", loss.item(), global_step)
            global_step += 1

            # Update batch counter
            batches_counter.update()

        # Update epoch counter
        batches_counter.close()
        epochs_counter.update()

    writer.close()
    return policy
exit()
# %%
# Train Deterministic BC
bc_deterministic = train_bc_deterministic(
    expert_dataset=expert_dataset,
    n_obs=N_OBS,
    n_actions=N_ACTIONS,
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    run_name="BC_Det"
)

# %%
# Train Stochastic BC
bc_stochastic = train_bc_stochastic(
    expert_dataset=expert_dataset,
    n_obs=N_OBS,
    n_actions=N_ACTIONS,
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    run_name="BC_Stoch"
)

# %% [markdown]
# ## Visualization and Results

# %%
# @title Policy visualization / demo code.

from gymnasium.wrappers import RecordVideo
from IPython.display import Video, display, clear_output

def visualize(agent, policy_name="agent", env_name=ENV_NAME):
    video_dir = "./videos"  # Directory to save videos
    os.makedirs(video_dir, exist_ok=True)

    # Create environment with proper render_mode
    env = gym.make(env_name, render_mode="rgb_array")

    # name_prefix ensures each video file is distinct
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda e_id: True,
        name_prefix=policy_name  # This gets appended to "rl-video-step-..."
    )

    obs, _ = env.reset(seed=SEED)

    for t in range(1000):
        # Get action from policy
        actions, _ = agent.get_action(
            torch.tensor(obs, dtype=torch.float, device=device)[None, :]
        )
        obs, _, done, truncated, _ = env.step(actions.cpu().detach().numpy()[0])
        if done or truncated:
            break

    env.close()

    # The RecordVideo wrapper names the file automatically with the prefix + step info
    # We'll grab the latest video with our given prefix
    # e.g. "agent_rl-video-episode-0.mp4" or similar
    filtered_videos = sorted(
        f for f in os.listdir(video_dir)
        if f.endswith(".mp4") and policy_name in f
    )
    if len(filtered_videos) == 0:
        logger.warning("No videos found!")
        return

    video_path = os.path.join(video_dir, filtered_videos[-1])  # the newest file

    clear_output(wait=True)
    display(Video(video_path, embed=True))

# %%
logger.info("Visualizing GAIL Policy...")
visualize(gail.learner, policy_name="GAIL")

# %%
logger.info("Visualizing Deterministic BC Policy...")
visualize(bc_deterministic, policy_name="BC_Deterministic")

# %%
logger.info("Visualizing Stochastic BC Policy...")
visualize(bc_stochastic, policy_name="BC_Stochastic")

# %%
#@title Evaluate policies and compare

def collect_eval_data(
        agent,
        env_name=ENV_NAME,
        n_episodes=EVAL_EPISODES
    ):
    """
    Runs 'n_eval_episodes' rollouts in the given env_name, returning list of episode returns.
    """
    agent.eval()
    returns = []

    # Create a progress bar for evaluation episodes
    manager = enlighten.get_manager()
    eval_pbar = manager.counter(total=n_episodes, desc="Evaluating", unit="episodes")

    for episode_i in range(n_episodes):
        env = gym.make(env_name)

        obs, _ = env.reset()

        total_reward = 0.0
        terminated, truncated = False, False
        while not (terminated or truncated):
            # Wrap obs in a torch tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0)
            action, _ = agent.get_action(obs_tensor)
            # Convert action back to numpy for the env
            obs, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy()[0])
            total_reward += reward
        returns.append(total_reward)
        env.close()

        # Update progress bar
        eval_pbar.update()

    return returns

# %%
# Collect evaluation data for each policy
logger.info("Collecting evaluation data.")
gail_returns = collect_eval_data(gail.learner)
bc_det_returns = collect_eval_data(bc_deterministic)
bc_stoch_returns = collect_eval_data(bc_stochastic)

# %%
# Get summary evaluation statistics
gail_mean, gail_std = float(np.mean(gail_returns)), float(np.std(gail_returns))
bc_det_mean, bc_det_std = float(np.mean(bc_det_returns)), float(np.std(bc_det_returns))
bc_stoch_mean, bc_stoch_std = float(np.mean(bc_stoch_returns)), float(np.std(bc_stoch_returns))

logger.info(f"GAIL policy:     return={gail_mean:.1f} ± {gail_std:.1f}")
logger.info(f"BC Deter policy: return={bc_det_mean:.1f} ± {bc_det_std:.1f}")
logger.info(f"BC Stoch policy: return={bc_stoch_mean:.1f} ± {bc_stoch_std:.1f}")

# %%
#@title Plot returns for all policies

import matplotlib.pyplot as plt
import seaborn as sns

FS_TICK: int = 12
FS_LABEL: int = 18
PLOT_DPI: int=1200
PLOT_FORMAT: str='pdf'
RC_PARAMS: dict = {
    # Set background and border settings
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 2,
    'xtick.color': 'black',
    'ytick.color': 'black',
}

def plot_returns(data, xlabel, ylabel, title):
    plt.rcParams.update(RC_PARAMS);
    sns.set_palette('deep')
    fig = plt.figure()
    ax = sns.boxplot(
        data=data,
        palette='deep',
        orient='v',
        showmeans=True,
        meanprops={
            'markerfacecolor': 'white',
            'markeredgecolor': 'black'
            },
        flierprops={'marker': 'x'}
    )

    # sns.stripplot(
    #     data=data,
    #     alpha=0.5,
    #     color="black",
    #     jitter=True
    # )

    plt.xlabel(xlabel, fontsize=FS_LABEL)
    plt.ylabel(ylabel, fontsize=FS_LABEL)
    plt.yticks(fontsize=FS_TICK)
    plt.xticks(fontsize=FS_TICK)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.show()
    fig.savefig(f"{title}.{PLOT_FORMAT}", dpi=PLOT_DPI, format=PLOT_FORMAT)
    plt.close(fig)




def plot_returns_by_epoch(data, xlabel, ylabel, title):
    plt.rcParams.update(RC_PARAMS);
    sns.set_palette('deep')
    fig = plt.figure()
    xdata = data['epoch']
    ydata = data['return_mean']
    ybar = data['return_std']

    plt.plot(xdata, ydata, label="Mean Return", color="blue")
    plt.fill_between(xdata, ydata - ybar, ydata + ybar, alpha=0.2, color="blue", label="±1 Std Dev")


    plt.xlabel(xlabel, fontsize=FS_LABEL)
    plt.ylabel(ylabel, fontsize=FS_LABEL)
    plt.yticks(fontsize=FS_TICK)
    plt.xticks(fontsize=FS_TICK)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.show()
    fig.savefig(f"{title}.{PLOT_FORMAT}", dpi=PLOT_DPI, format=PLOT_FORMAT)
    plt.close(fig)


# %%
return_data = {
    "GAIL": gail_returns,
    "BC Deterministic": bc_det_returns,
    "BC Stochastic": bc_stoch_returns
}
plot_returns(
    return_data,
    xlabel='Policy',
    ylabel='Return',
    title='Evaluation'
);

# %%
return_by_epoch_data = {
    "epoch": returns_by_epoch[:,0],
    "return_mean": returns_by_epoch[:,1],
    "return_std": returns_by_epoch[:,2]
}

plot_returns_by_epoch(
    return_by_epoch_data,
    xlabel='Epoch',
    ylabel='Return',
    title='Evaluation by Epochs'
)

# %%


# %% [markdown]
# ***


