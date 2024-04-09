# cc. https://minigrid.farama.org/content/training/


import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn
import torch

list_envs = ['MiniGrid-Empty-Random-5x5-v0', 'MiniGrid-Empty-Random-6x6-v0', 'MiniGrid-Empty-8x8-v0', 'MiniGrid-Empty-16x16-v0',
             'MiniGrid-LavaGapS5-v0', 'MiniGrid-LavaGapS6-v0', 'MiniGrid-LavaGapS7-v0',
             'MiniGrid-DistShift1-v0', 'MiniGrid-DistShift2-v0',
             'MiniGrid-DoorKey-5x5-v0', 'MiniGrid-DoorKey-6x6-v0', 'MiniGrid-DoorKey-8x8-v0', 'MiniGrid-DoorKey-16x16-v0']


class MinigridFeaturesExtractor128(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(128, 64, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 16, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor128,
    features_extractor_kwargs=dict(features_dim=128),
)

env = gym.make('MiniGrid-DoorKey-5x5-v0')
env = ImgObsWrapper(env)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, n_steps=1024 * 10)
model.learn(5e6, progress_bar=True)

model.save(path='./trained_model/minigrid_doorkey_5x5')

print("=================================================================================")
print('result - MiniGrid-DoorKey-5x5-v0')
for e in list_envs:
    env = gym.make(e)
    env = ImgObsWrapper(env)

    rwd_mean, rwd_std = evaluate_policy(model, env, n_eval_episodes=10)
    print(e, rwd_mean, rwd_std)

print("=================================================================================")

env = gym.make('MiniGrid-DoorKey-6x6-v0')
env = ImgObsWrapper(env)

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor128,
    features_extractor_kwargs=dict(features_dim=128),
)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, n_steps=1024 * 10)
model.learn(5e6, progress_bar=True)

model.save(path='./trained_model/minigrid_doorkey_6x6')

print("=================================================================================")
print('result - MiniGrid-DoorKey-6x6-v0')
for e in list_envs:
    env = gym.make(e)
    env = ImgObsWrapper(env)

    rwd_mean, rwd_std = evaluate_policy(model, env, n_eval_episodes=10)
    print(e, rwd_mean, rwd_std)

print("=================================================================================")

env = gym.make('MiniGrid-DoorKey-8x8-v0')
env = ImgObsWrapper(env)

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor128,
    features_extractor_kwargs=dict(features_dim=128),
)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, n_steps=1024 * 10)
model.learn(5e6, progress_bar=True)

model.save(path='./trained_model/minigrid_doorkey_8x8')

print("=================================================================================")
print('result - MiniGrid-DoorKey-8x8-v0')
for e in list_envs:
    env = gym.make(e)
    env = ImgObsWrapper(env)

    rwd_mean, rwd_std = evaluate_policy(model, env, n_eval_episodes=10)
    print(e, rwd_mean, rwd_std)

print("=================================================================================")

env = gym.make('MiniGrid-DoorKey-16x16-v0')
env = ImgObsWrapper(env)

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor128,
    features_extractor_kwargs=dict(features_dim=128),
)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, n_steps=1024 * 10)
model.learn(5e6, progress_bar=True)

model.save(path='./trained_model/minigrid_doorkey_16x16')

print("=================================================================================")
print('result - MiniGrid-DoorKey-16x16-v0')
for e in list_envs:
    env = gym.make(e)
    env = ImgObsWrapper(env)

    rwd_mean, rwd_std = evaluate_policy(model, env, n_eval_episodes=10)
    print(e, rwd_mean, rwd_std)

print("=================================================================================")