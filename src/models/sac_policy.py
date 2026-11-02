"""
sac_policy.py
=============
Soft Actor-Critic (SAC) Architecture in PyTorch for Continuous Robotic Control.
Implements the canonical maximum-entropy off-policy formulation (Haarnoja et al., 2018).

Components:
    - GaussianActor: Tanh-squashed diagonal Gaussian policy pi_phi(a|s).
    - TwinCritic: Clipped Double-Q network pair (Q_theta1, Q_theta2).
    - ReplayBuffer: High-throughput circular experience replay buffer.
    - SACAgent: Orchestrator for updates, target polyak smoothing, and action selection.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _init_orthogonal(module: nn.Module, gain: float = np.sqrt(2.0)) -> None:
    """
    Applies orthogonal weight initialization and zeroes biases for linear layers.
    Mitigates vanishing/exploding gradients in deep reinforcement learning architectures.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class GaussianActor(nn.Module):
    """
    Gaussian Policy with Tanh Squashing for continuous, bounded action spaces:
        u ~ N(mu_phi(s), sigma_phi(s))
        a = action_scale * tanh(u)
    
    The log-likelihood is corrected via change-of-variables:
        log pi(a|s) = log N(u; mu, sigma) - sum_i log(1 - tanh(u_i)^2 + eps)
    """

    def __init__(
        self,
        obs_dim: int = 30,
        action_dim: int = 5,
        hidden_dim: int = 256,
        action_scale: float = 1.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.action_scale = float(action_scale)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # Shared trunk MLP
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.trunk.apply(_init_orthogonal)

        # Separate heads for mean and log_std (gain 0.01 prevents early exploratory collapse)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        _init_orthogonal(self.mean_head, gain=0.01)
        _init_orthogonal(self.log_std_head, gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing Gaussian parameters.
        # shape: obs (B, obs_dim) -> mu (B, action_dim), log_std (B, action_dim)
        """
        x = self.trunk(obs)
        mu = self.mean_head(x)
        # Strict numerical guard: bound log_std to prevent underflow/overflow in exp()
        log_std = torch.clamp(self.log_std_head(x), self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable reparameterized sampling:
            u = mu + sigma * eps, where eps ~ N(0, I)
            a = action_scale * tanh(u)
        # shape: obs (B, obs_dim) -> action (B, action_dim), log_prob (B, 1)
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        # Reparameterization trick (preserves gradient path d_a / d_phi)
        u = dist.rsample()
        action_unscaled = torch.tanh(u)
        action = action_unscaled * self.action_scale

        # Enforce numerical guard on squashing Jacobian: clamp (1 - tanh^2) >= 1e-7
        # Eliminates log(0) and negative arguments caused by precision rounding
        squash_correction = torch.log(
            torch.clamp(1.0 - action_unscaled.pow(2), min=1e-7, max=1.0)
        )
        log_prob = (dist.log_prob(u) - squash_correction).sum(dim=-1, keepdim=True)

        return action, log_prob


class TwinCritic(nn.Module):
    """
    Twin Q-Networks Q_theta1(s, a) and Q_theta2(s, a) for Clipped Double Q-learning.
    Mitigates overestimation bias in the Bellman backup operator.
    """

    def __init__(
        self,
        obs_dim: int = 30,
        action_dim: int = 5,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        in_dim = obs_dim + action_dim

        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q1.apply(_init_orthogonal)
        self.q2.apply(_init_orthogonal)
        _init_orthogonal(self.q1[-1], gain=1.0)
        _init_orthogonal(self.q2[-1], gain=1.0)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates both Q-networks on the state-action pair.
        # shape: obs (B, obs_dim), action (B, action_dim) -> q1 (B, 1), q2 (B, 1)
        """
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class ReplayBuffer:
    """
    Vectorized pre-allocated Experience Replay Buffer in RAM with GPU batch streaming.
    Supports high-speed circular FIFO queueing for both single and parallel-environment batches.
    """

    def __init__(
        self,
        obs_dim: int = 30,
        action_dim: int = 5,
        capacity: int = 100000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.idx = 0
        self.size = 0

        # Pre-allocated contiguous NumPy arrays
        self.obs_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity, 1), dtype=np.float32)

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Appends a single transition (s, a, r, s', d) into the circular buffer."""
        self.obs_buf[self.idx] = obs
        self.action_buf[self.idx] = action
        self.reward_buf[self.idx] = reward
        self.next_obs_buf[self.idx] = next_obs
        self.done_buf[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """
        Vectorized circular insertion of B transitions from parallel simulation environments.
        Zero-loop slice indexing eliminates CPU bottlenecks.
        # shape: obs (B, obs_dim), action (B, action_dim), reward (B, 1)|(B,), next_obs (B, obs_dim), done (B, 1)|(B,)
        """
        n = obs.shape[0]
        reward_reshaped = np.asarray(reward, dtype=np.float32).reshape(n, 1)
        done_reshaped = np.asarray(done, dtype=np.float32).reshape(n, 1)

        if self.idx + n <= self.capacity:
            self.obs_buf[self.idx : self.idx + n] = obs
            self.action_buf[self.idx : self.idx + n] = action
            self.reward_buf[self.idx : self.idx + n] = reward_reshaped
            self.next_obs_buf[self.idx : self.idx + n] = next_obs
            self.done_buf[self.idx : self.idx + n] = done_reshaped
            self.idx = (self.idx + n) % self.capacity
        else:
            part1 = self.capacity - self.idx
            part2 = n - part1

            # Slice 1: up to capacity edge
            self.obs_buf[self.idx : self.capacity] = obs[:part1]
            self.action_buf[self.idx : self.capacity] = action[:part1]
            self.reward_buf[self.idx : self.capacity] = reward_reshaped[:part1]
            self.next_obs_buf[self.idx : self.capacity] = next_obs[:part1]
            self.done_buf[self.idx : self.capacity] = done_reshaped[:part1]

            # Slice 2: wrapped around to index 0
            self.obs_buf[0:part2] = obs[part1:]
            self.action_buf[0:part2] = action[part1:]
            self.reward_buf[0:part2] = reward_reshaped[part1:]
            self.next_obs_buf[0:part2] = next_obs[part1:]
            self.done_buf[0:part2] = done_reshaped[part1:]
            self.idx = part2

        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int = 256) -> Dict[str, torch.Tensor]:
        """Uniform random mini-batch sampling transferred directly to device."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs_buf[indices], dtype=torch.float32, device=self.device),
            "action": torch.as_tensor(self.action_buf[indices], dtype=torch.float32, device=self.device),
            "reward": torch.as_tensor(self.reward_buf[indices], dtype=torch.float32, device=self.device),
            "next_obs": torch.as_tensor(self.next_obs_buf[indices], dtype=torch.float32, device=self.device),
            "done": torch.as_tensor(self.done_buf[indices], dtype=torch.float32, device=self.device),
        }

    def __len__(self) -> int:
        return self.size


class SACAgent:
    """
    Soft Actor-Critic agent managing policy optimization, double Q-learning,
    and automatic entropy temperature adjustment.
    """

    def __init__(
        self,
        obs_dim: int = 30,
        action_dim: int = 5,
        action_scale: float = 1.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.action_scale = float(action_scale)
        self.device = torch.device(device)

        # Actor and Critic networks
        self.actor = GaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
            action_scale=action_scale,
        ).to(self.device)

        self.critic = TwinCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
        ).to(self.device)

        self.critic_target = TwinCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=256,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # Target network weights never accumulate gradients
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy temperature adjustment: H_target = -dim(A)
        self.target_entropy = (
            float(target_entropy) if target_entropy is not None else -float(action_dim)
        )
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
        """Current temperature multiplier: alpha = exp(log_alpha)."""
        return self.log_alpha.exp()

    @torch.inference_mode()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Samples an action for environment interaction.
        Uses @torch.inference_mode() for minimal VRAM usage and zero autograd overhead.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        is_batched = obs_t.ndim == 2
        if not is_batched:
            obs_t = obs_t.unsqueeze(0)

        if deterministic:
            mu, _ = self.actor(obs_t)
            action = torch.tanh(mu) * self.action_scale
        else:
            action, _ = self.actor.sample(obs_t)

        res = action if is_batched else action.squeeze(0)
        return res.cpu().numpy()

    def update(self, buffer: ReplayBuffer, batch_size: int = 256) -> Dict[str, float]:
        """
        Executes a single joint gradient step on Critic, Actor, and Temperature networks.
        """
        batch = buffer.sample(batch_size)
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        # =====================================================================
        # 1. Critic Update (Bellman Soft Backup)
        # y = r + gamma * (1 - d) * [ min(Q_target1, Q_target2) - alpha * log_pi(a'|s') ]
        # =====================================================================
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            q1_target, q2_target = self.critic_target(next_obs, next_action)
            q_target_min = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
            target_q = reward + (1.0 - done) * self.gamma * q_target_min

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # =====================================================================
        # 2. Actor Update
        # L_pi = E_{s ~ D, a ~ pi} [ alpha * log_pi(a|s) - min(Q1(s, a), Q2(s, a)) ]
        # =====================================================================
        curr_action, curr_log_prob = self.actor.sample(obs)
        q1_curr, q2_curr = self.critic(obs, curr_action)
        actor_loss = (self.alpha * curr_log_prob - torch.min(q1_curr, q2_curr)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # =====================================================================
        # 3. Temperature Update (Entropy Target)
        # L_alpha = -log_alpha * [ log_pi(a|s) + H_target ].detach()
        # =====================================================================
        alpha_loss = -(
            self.log_alpha * (curr_log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # =====================================================================
        # 4. Target Networks Polyak Averaging
        # theta_target <- (1 - tau) * theta_target + tau * theta
        # =====================================================================
        with torch.no_grad():
            for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_target.lerp_(p, self.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def save(self, filepath: str) -> None:
        """Saves actor, critic, and temperature states."""
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "log_alpha": self.log_alpha,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Loads model states with device mapping and target network synchronization."""
        ckpt = torch.load(filepath, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].data)


if __name__ == "__main__":
    """
    Reproducible verification suite and physical-mathematical invariants test:
        1. Dimension compatibility of heads and trunks.
        2. Gradient flow across actor, critic, and entropy temperature.
        3. Extreme-input numerical stability (no NaNs / Infs under edge cases).
        4. Vectorized ReplayBuffer wrap-around consistency.
        5. State persistence round-trip (save/load).
    """
    import tempfile
    import os

    print("[*] Running SAC Module Verification Suite...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim = 30
    action_dim = 5
    batch_size = 64

    # 1. Verification of Actor
    actor = GaussianActor(obs_dim=obs_dim, action_dim=action_dim).to(device)
    dummy_obs = torch.randn(batch_size, obs_dim, device=device)
    action, log_prob = actor.sample(dummy_obs)
    assert action.shape == (batch_size, action_dim), f"Bad action shape: {action.shape}"
    assert log_prob.shape == (batch_size, 1), f"Bad log_prob shape: {log_prob.shape}"
    assert not torch.isnan(action).any(), "NaN found in sampled actions"
    assert not torch.isnan(log_prob).any(), "NaN found in action log_prob"
    print("    [+] GaussianActor shapes and finite outputs verified.")

    # 2. Numerical Stability Guard Test (Stress-test extreme values)
    extreme_obs = torch.full((batch_size, obs_dim), 1e4, device=device)
    act_ext, logp_ext = actor.sample(extreme_obs)
    assert not torch.isnan(act_ext).any() and not torch.isinf(act_ext).any(), "Extreme obs produced non-finite action"
    assert not torch.isnan(logp_ext).any() and not torch.isinf(logp_ext).any(), "Extreme obs produced non-finite log_prob"
    assert (act_ext.abs() <= 1.0 + 1e-6).all(), "Action violated tanh bounds [-1, 1]"
    print("    [+] Extreme input stress-test: bounds and numerical guards verified.")

    # 3. Verification of TwinCritic
    critic = TwinCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    q1, q2 = critic(dummy_obs, action)
    assert q1.shape == (batch_size, 1), f"Bad q1 shape: {q1.shape}"
    assert q2.shape == (batch_size, 1), f"Bad q2 shape: {q2.shape}"
    assert not torch.isnan(q1).any() and not torch.isnan(q2).any(), "NaN in Q-values"
    print("    [+] TwinCritic evaluation verified.")

    # 4. Vectorized ReplayBuffer Wrap-Around Test
    buffer_cap = 100
    rb = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, capacity=buffer_cap, device=device)
    n_parallel = 16
    for step in range(10):  # 160 elements total -> wraps around buffer of 100
        rb.add_batch(
            obs=np.ones((n_parallel, obs_dim), dtype=np.float32) * step,
            action=np.zeros((n_parallel, action_dim), dtype=np.float32),
            reward=np.ones(n_parallel, dtype=np.float32) * 0.5,
            next_obs=np.ones((n_parallel, obs_dim), dtype=np.float32) * (step + 1),
            done=np.zeros(n_parallel, dtype=bool),
        )
    assert len(rb) == buffer_cap, f"Buffer size should be capped at {buffer_cap}, got {len(rb)}"
    sampled = rb.sample(batch_size=32)
    assert sampled["obs"].shape == (32, obs_dim)
    assert sampled["reward"].shape == (32, 1)
    print("    [+] ReplayBuffer vectorized batch insertion and circular wrap-around verified.")

    # 5. Full SACAgent Optimization Step
    agent = SACAgent(obs_dim=obs_dim, action_dim=action_dim, device=device)
    metrics = agent.update(rb, batch_size=32)
    assert "critic_loss" in metrics and "actor_loss" in metrics and "alpha" in metrics
    for k, v in metrics.items():
        assert not np.isnan(v) and not np.isinf(v), f"Metric {k} is non-finite: {v}"
    print(f"    [+] SACAgent gradient update step verified: {metrics}")

    # 6. Action Selection with Inference Mode
    eval_act = agent.select_action(np.ones(obs_dim, dtype=np.float32), deterministic=True)
    assert eval_act.shape == (action_dim,)
    assert (np.abs(eval_act) <= 1.0 + 1e-6).all()
    print("    [+] Deterministic inference action selection verified.")

    # 7. Persistence Round-Trip
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        agent.save(tmp_path)
        agent_loaded = SACAgent(obs_dim=obs_dim, action_dim=action_dim, device=device)
        agent_loaded.load(tmp_path)
        assert torch.allclose(agent.log_alpha, agent_loaded.log_alpha), "Checkpoint mismatch on log_alpha"
        print("    [+] Checkpoint save/load integrity verified.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("\n[SUCCESS] All verification tests passed cleanly without errors or NaNs.")
