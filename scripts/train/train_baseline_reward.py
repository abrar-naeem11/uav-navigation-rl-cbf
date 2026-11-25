from typing import Any, ClassVar, Optional, TypeVar, Union, Dict, List, Tuple

import os
import numpy as np
import torch as th
from gymnasium import spaces
from collections import deque

from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.td3.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback

from enums import ObservationType, ActionType

from AOATTS import AOATTS

SelfTD3 = TypeVar("SelfTD3", bound="TD3")


class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param n_steps: When n_step > 1, uses n-step return (with the NStepReplayBuffer) when updating the Q-value network.
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`td3_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: TD3Policy
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        grad_clip: float = 0.5,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.grad_clip = grad_clip

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        # Multi Replay Buffer
        self.replay_buffer = MultiReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            optimize_memory_usage=self.optimize_memory_usage,
            gamma = self.gamma
        )
        self._create_aliases()

        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip)
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self: SelfTD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTD3:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []

class MultiReplayBuffer:
    def __init__(self, buffer_size: int, observation_space:spaces.Space, action_space:spaces.Space,
                 device: str = "auto", optimize_memory_usage: bool = False,
                 sampling_prob: Dict[str, float] = None, gamma: float = 0.99, beta: float = 0.85):
        self.mt = 200
        self.ms = 1_000_000
        self.mf = 1_000_000

        self.buffers = {
            "success": ReplayBuffer(self.ms, observation_space, action_space, device=device,
                                    optimize_memory_usage=optimize_memory_usage),
            "failure": ReplayBuffer(self.mf, observation_space, action_space, device=device,
                                    optimize_memory_usage=optimize_memory_usage)
        }
        self.temporary_pool = deque(maxlen=self.mt)  # Temporary experience pool with FIFO overflow
        self.beta = beta  # Sampling ratio parameter
        self.device = device
        self.gamma = gamma

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
            done: np.ndarray, infos: List[Dict[str, Any]]) -> None:
        for i in range(len(obs)):
            if len(self.temporary_pool) == self.temporary_pool.maxlen:
                # remove oldest and push to success buffer
                p_obs, p_next_obs, p_action, p_reward, p_done, p_infos = self.temporary_pool.popleft()
                self.buffers["success"].add(p_obs, p_next_obs, p_action, p_reward, p_done, p_infos)

            experience = (obs[i:i + 1], next_obs[i:i + 1], action[i:i + 1], reward[i:i + 1], done[i:i + 1], [infos[i]])
            self.temporary_pool.append(experience)

            if done[i]:
                # Distribute remaining temporary pool data based on episode outcome
                final_info = infos[i]
                target_key = "success" if final_info.get("success", False) else "failure"
                for exp in list(self.temporary_pool):  # Convert to list to avoid modification during iteration
                    ep_obs, ep_next_obs, ep_action, ep_reward, ep_done, ep_infos = exp
                    self.buffers[target_key].add(ep_obs, ep_next_obs, ep_action, ep_reward, ep_done, ep_infos)
                self.temporary_pool.clear()


    def sample(self, batch_size: int, env=None):
        d_s = self.buffers["success"].size()
        d_f = self.buffers["failure"].size()

        if d_s + d_f == 0:
            # Buffers empty, return zero-filled batch
            obs_shape = list(self.buffers["success"].observation_space.shape)
            act_shape = list(self.buffers["success"].action_space.shape)
            ReplaySample = type(self.buffers["success"].sample(1, env=env))
            return ReplaySample(
                observations=th.zeros((batch_size, *obs_shape), device=self.device),
                actions=th.zeros((batch_size, *act_shape), device=self.device),
                next_observations=th.zeros((batch_size, *obs_shape), device=self.device),
                dones=th.zeros((batch_size, 1), device=self.device),
                rewards=th.zeros((batch_size, 1), device=self.device),
                discounts=th.full((batch_size, 1), self.gamma, device=self.device)
            )

        # Calculate batch sizes with beta
        b1 = min(d_s, int(self.beta * batch_size))  # Success samples
        b2 = batch_size - b1  # Failure samples

        # Adjust if failure pool has less than b2
        if d_f < b2:
            b2 = d_f
            b1 = batch_size - b2
            b1 = min(b1, d_s)  # Ensure not exceeding success pool size

        success_data = self.buffers["success"].sample(b1, env) if b1 > 0 and d_s > 0 else None
        failure_data = self.buffers["failure"].sample(b2, env) if b2 > 0 and d_f > 0 else None

        # Concatenate samples
        observations = th.cat([success_data.observations, failure_data.observations]) if success_data and failure_data else \
                       (success_data.observations if success_data else failure_data.observations)
        actions = th.cat([success_data.actions, failure_data.actions]) if success_data and failure_data else \
                  (success_data.actions if success_data else failure_data.actions)
        next_observations = th.cat([success_data.next_observations, failure_data.next_observations]) if success_data and failure_data else \
                            (success_data.next_observations if success_data else failure_data.next_observations)
        dones = th.cat([success_data.dones, failure_data.dones]) if success_data and failure_data else \
                (success_data.dones if success_data else failure_data.dones)
        rewards = th.cat([success_data.rewards, failure_data.rewards]) if success_data and failure_data else \
                  (success_data.rewards if success_data else failure_data.rewards)
        discounts_list = []
        if success_data:
            discounts_list.append(success_data.discounts if hasattr(success_data, 'discounts') and success_data.discounts is not None else th.full((b1, 1), self.gamma, device=self.device))
        if failure_data:
            discounts_list.append(failure_data.discounts if hasattr(failure_data, 'discounts') and failure_data.discounts is not None else th.full((b2, 1), self.gamma, device=self.device))
        discounts = th.cat(discounts_list, dim=0)

        ReplaySample = type(self.buffers["success"].sample(1, env=env))
        return ReplaySample(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
            discounts=discounts
        )

    def size(self) -> int:
        return sum(buffer.size() for buffer in self.buffers.values())

    def save(self, path: str) -> None:
        for key, buffer in self.buffers.items():
            buffer.save(f"{path}_{key}")



log_dir = "logs/td3_runs_new"
os.makedirs(log_dir, exist_ok=True)

#Make Env
env = AOATTS(gui=False,
                  obs=ObservationType.KIN,
                  act=ActionType.VEL,
                  )
env = Monitor(env, filename=f"{log_dir}/monitor.csv", info_keywords=("success",))


save_dir = os.path.join(log_dir, "best")           # e.g. logs/td3_run2/best
eval_log_dir = os.path.join(log_dir, "eval")       # e.g. logs/td3_run2/eval
os.makedirs(save_dir, exist_ok=True)
os.makedirs(eval_log_dir, exist_ok=True)

# Save the model whenever mean reward over eval episodes improves
eval_callback = EvalCallback(
    env,
    best_model_save_path=save_dir,
    log_path=eval_log_dir,
    eval_freq=10_000,          # evaluate every N training steps (tune this)
    n_eval_episodes=10,        # average over 10 eval episodes
    deterministic=True,
    render=False,
    verbose=1,
)

# (optional) also save periodic checkpoints, regardless of performance
ckpt_callback = CheckpointCallback(
    save_freq=50_000,          # every N training steps
    save_path=os.path.join(log_dir, "checkpoints"),
    name_prefix="td3_ckpt"
)

class BufferMonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        buffer_sizes = {key: buffer.size() for key, buffer in self.model.replay_buffer.buffers.items()}
        for key, size in buffer_sizes.items():
            self.logger.record(f"buffer_size/{key}", size)
        return True

# Callback to decay sigma
class NoiseDecayCallback(BaseCallback):
    def __init__(self, initial_sigma, final_sigma, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma  # e.g., 0.1 * np.ones(act_dim)
        self.total_timesteps = total_timesteps

    def _on_step(self):
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        new_sigma = self.initial_sigma * (1 - progress) + self.final_sigma * progress
        self.model.action_noise.sigma = new_sigma  # Update sigma in place
        return True


#Action Noise
act_dim = env.action_space.shape[0]
initial_sigma = 0.5 * np.ones(act_dim)
action_noise = NormalActionNoise(mean=np.zeros(act_dim), sigma=initial_sigma)
total_steps = 1_000_000

# Instantiate callback
noise_decay_callback = NoiseDecayCallback(
    initial_sigma=initial_sigma,
    final_sigma=0.05 * np.ones(act_dim),
    total_timesteps=total_steps,
    verbose=1
)

callback = CallbackList([eval_callback, ckpt_callback, noise_decay_callback, BufferMonitorCallback()])

model = TD3(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-5,
    buffer_size=1_000_000,
    learning_starts=10000,
    batch_size=1024,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=4,
    action_noise=action_noise,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    policy_delay=3,
    verbose=1,
    tensorboard_log=log_dir,
    grad_clip=0.5,
    policy_kwargs = dict(net_arch=dict(pi=[400, 300], qf=[400, 300])),
    device="cuda:1"
)

model.learn(total_timesteps=total_steps, log_interval=10, callback=callback)
model.save("td3_aoatt_new")
rb_path = r"logs/td3_runs_new/replay_buffer.pkl"
model.save_replay_buffer(rb_path)