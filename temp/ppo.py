import math
import random
from collections import deque
import airsim
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from env import DroneEnv 
from torch.utils.tensorboard import SummaryWriter
import time
import glob

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.schedule import LinearSchedule
from stable_baselines3.common.utils import get_schedule_fn

writer = SummaryWriter()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PPO_Agent:
    def __init__(self, useDepth=False):
        self.useDepth = useDepth
        self.env = DummyVecEnv([lambda: DroneEnv(useDepth)])  # Vectorized environment wrapper for parallel training
        self.max_episodes = 10000
        self.save_interval = 2
        self.test_interval = 10
        self.max_steps = 1000
        self.episode = -1
        self.best_score = -np.inf
        self.early_stop_patience = 100
        self.best_mean_reward = -np.inf
        self.no_improvement_steps = 0

        # Learning rate schedule
        self.lr_schedule = LinearSchedule(1.0, 0.0, self.max_episodes)

        # KL coefficient schedule
        self.kl_coef_schedule = get_schedule_fn(0.2)

        # PPO model
        self.model = PPO('MlpPolicy', self.env, verbose=1, device=device,
                         learning_rate=self.lr_schedule,
                         kl_coef_schedule=self.kl_coef_schedule,
                         normalize_advantage=True,
                         max_grad_norm=0.5,
                         ent_coef=0.01,  # Entropy coefficient
                         normalize_rewards=True)

        # LOGGING
        cwd = os.getcwd()
        self.save_dir = os.path.join(cwd, "saved_models")
        if not os.path.exists(self.save_dir):
            os.mkdir("saved_models")
        if not os.path.exists(os.path.join(cwd, "videos")):
            os.mkdir("videos")

        files = glob.glob(self.save_dir + '/*.zip')
        if len(files) > 0:
            files.sort(key=os.path.getmtime)
            file = files[-1]
            self.model = PPO.load(file, env=self.env)
            self.episode = int(file.split('EPISODE')[-1].split('.zip')[0])
            print("Saved parameters loaded"
                  "\nModel: ", file,
                  "\nEpisode: ", self.episode)
        else:
            if os.path.exists("log.txt"):
                open('log.txt', 'w').close()
            if os.path.exists("last_episode.txt"):
                open('last_episode.txt', 'w').close()
            if os.path.exists("saved_model_params.txt"):
                open('saved_model_params.txt', 'w').close()

        obs = self.env.reset()
        writer.add_graph(self.model.policy, torch.tensor(obs).float())

    def train(self):
        print("Starting training...")
        score_history = []
        reward_history = []

        if self.episode == -1:
            self.episode = 1

        for e in range(self.episode, self.max_episodes + 1):
            start = time.time()
            try:
                self.model.learn(total_timesteps=self.max_steps)
                self.model.save(f"{self.save_dir}/EPISODE{e}.zip")

                # Evaluate the model
                obs = self.env.reset()
                score = 0
                steps = 0
                while True:
                    action, _states = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.env.step(action)
                    score += reward
                    steps += 1
                    if done:
                        break

                reward = score / steps
                score_history.append(score)
                reward_history.append(reward)

                if score > self.best_score:
                    self.best_score = score
                    self.model.save(f"{self.save_dir}/BEST_MODEL.zip")

                print(f"episode: {e}, reward: {reward}, score: {score}, total steps: {steps}")
                with open('log.txt', 'a') as file:
                    file.write(f"episode: {e}, reward: {reward}, score: {score}, total steps: {steps}\n")

                writer.add_scalar('reward_history', reward, e)
                writer.add_scalar('score_history', score, e)
                writer.add_scalar('Total steps', steps, e)
                writer.add_scalars('General Look', {'score_history': score, 'reward_history': reward}, e)

                # Early stopping
                mean_reward = np.mean(reward_history[-100:])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_steps = 0
                else:
                    self.no_improvement_steps += 1

                if self.no_improvement_steps >= self.early_stop_patience:
                    print(f"Early stopping at episode {e} due to no improvement.")
                    break

            except Exception as ex:
                print(f"Error during episode {e}: {ex}")

            end = time.time()
            print(f"Episode {e} done, episode time: {end - start}")

            if e % self.test_interval == 0:
                self.test()

        writer.close()

    def test(self):
        start = time.time()
        steps = 0
        score = 0
        image_array = []
        obs = self.env.reset()
        image_array.append(obs)

        while True:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            image_array.append(obs)

            steps += 1
            score += reward

            if done or steps >= self.max_steps:
                break

        reward = score / steps
        print(f"TEST, reward: {reward}, score: {score}, total steps: {steps}")
        with open('tests.txt', 'a') as file:
            file.write(f"TEST, reward: {reward}, score: {score}, total steps: {steps}\n")

        writer.add_scalars('Test', {'score': score, 'reward': reward}, self.episode)

        # Save the trajectory video
        video_path = os.path.join('videos', f'Test_Episode_{self.episode}.mp4')
        self.save_video(image_array, video_path)

        end = time.time()
        print(f"Test done, test time: {end - start}")

    def save_video(self, image_array, video_path):
        # Save the sequence of images as a video
        frames = [Image.fromarray(obs.squeeze()) for obs in image_array]
        frames[0].save(video_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

if __name__ == "__main__":
    agent = PPO_Agent()
    agent.train()

