import torch.nn as nn
import torch
import random as rd
import numpy as np

from custom.utils.embedding import ObservationEmbedding
from custom.envs.custom_empty import CustomEmptyEnv
from custom.envs.custom_doorkey import CustomDoorKeyEnv
from custom.envs.custom_lavagap import CustomLavaGapEnv
from custom.envs.custom_distshift import CustomDistShiftEnv
from custom.utils.padding import unknown_masking


class RLTransformer(nn.Module):
    def __init__(self, d_embed=50, n_head=10,
                 num_encoder_layer=6, num_decoder_layer=6, num_action=7,
                 env_size=5, max_step=200, small_value=0.00001):

        super().__init__()

        self.embedding_layer = nn.Sequential(
            nn.Linear(d_embed * 3, d_embed),
            nn.ReLU(),
        )

        self.decoder_fn = nn.Sequential(
            nn.Linear(d_embed, num_action),
        )

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_embed, nhead=n_head)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_embed, nhead=n_head)

        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                             num_layers=num_encoder_layer)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer,
                                             num_layers=num_decoder_layer,
                                             norm=self.decoder_fn)

        self.max_step = max_step
        self.d_embed = d_embed
        self.num_action = num_action
        self.sel_actions = {}
        for i in range(self.num_action):
            self.sel_actions[i] = 0
        self.sel_actions[11] = 1
        self.last_action = 0
        self.small_value = small_value

        self.get_key = False
        self.open_door = False

        self.env_size = env_size

        self.env = CustomDoorKeyEnv(size=env_size, max_steps=self.max_step)
        self.embedding = ObservationEmbedding()

        mission = [8, 1, 0]

        self.goal_info = self.embedding.random_goaL_embedding(goal_object=mission)
        self.color_to_idx = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
        self.steps = 0

    def episode(self, mode='exploitation', pre_epi_steps=1):
        epi_step = 0
        done = False
        obs, _ = self.env.reset()

        if self.open_door:
            self.goal_info = self.embedding.random_goaL_embedding(goal_object=[8, 1, 0])

        elif self.get_key:
            self.goal_info = self.embedding.random_goaL_embedding(goal_object=[4, 4, 2])

        else:
            self.goal_info = self.embedding.random_goaL_embedding(goal_object=[5, 4, 0])

        episode_buffer = torch.zeros([self.max_step, self.d_embed])
        epi_obs_buffer = torch.zeros([self.max_step, self.d_embed])
        actions = torch.zeros([self.max_step, 1])
        actions.fill_(11.)

        self.encoder.eval()
        self.decoder.eval()
        self.embedding_layer.eval()

        while not done:
            masked_obs = unknown_masking(obs['image'], prev_ep_length=pre_epi_steps, type='dropout')
            obs = self.embedding.tensor_to_longtensor(masked_obs)
            obs = self.embedding_layer.forward(obs)

            if mode == 'exploration':
                action = rd.choice(range(self.num_action))

            else:
                emb_obs = obs.reshape([1, 1, self.d_embed])
                action = self.predict(emb_obs)
                epi_obs_buffer[epi_step] = emb_obs
                self.sel_actions[action] += 1
                self.sel_actions[11] += 1
                self.last_action = action

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.get_key = self.env.get_key
            self.open_door = self.env.open_door

            episode_buffer[epi_step] = obs
            actions[epi_step] = action

            epi_step += 1

            done = terminated or truncated
            obs = next_obs

            epi_reward = reward
            pre_epi_steps = epi_step

        epi_step -= 1

        return episode_buffer, epi_obs_buffer, actions, epi_reward, epi_step

    def predict(self, observation):
        emb_goal = self.embedding_layer.forward(self.goal_info)
        context = self.encoder.forward(emb_goal)

        action_softmax = self.decoder.forward(observation, context)
        action = action_softmax.squeeze()
        action[self.last_action] = action[self.last_action]

        action = torch.exp(action)
        ex_sum = action.sum()
        action = torch.div(action, ex_sum)

        r = rd.choices(population=range(len(action)), weights=action)[0]
        output = r

        return output

    def decoder_predict(self, observation, goal_info):
        action_softmax = self.decoder.forward(observation, goal_info)

        action = action_softmax.squeeze()

        arg = 0

        for a in range(len(action)):
            if action[a] > action[arg]:
                arg = a

        output = arg

        return output

    def decoder_backward(self, observation, goal_info):
        decoder_norm = self.decoder.forward(observation, goal_info)
        decoder_norm = decoder_norm.squeeze()

        return decoder_norm

    def train(self, optimizer, scheduler, tot_steps):
        steps = self.steps
        rewards = []
        epi = 0
        pre_epi_step = 1
        success = 0

        while steps < tot_steps:
            mode = 'exploitation'
            episode_buffer, epi_obs_buffer, actions, epi_reward, epi_step = self.episode(mode=mode, pre_epi_steps=1)

            self.encoder.train()
            self.decoder.train()
            self.embedding_layer.train()

            if epi_step > self.max_step * 0.2:
                train_start = rd.choice((range(int(self.max_step * 0.2) - 1)))
                train_end = min(train_start + int(self.max_step * 0.2), epi_step)

            else:
                train_start = 0
                train_end = epi_step + 1

            for e in range(train_start, train_end):
                optimizer.zero_grad()
                output = self.decoder.forward(epi_obs_buffer[e].reshape([1, 1, self.d_embed]),
                                              self.encoder.forward(epi_obs_buffer[e].reshape([1, 1, self.d_embed])))
                mask = torch.Tensor(output.shape)
                for i in range(len(output[0, 0, :])):
                    if i == actions[e]:
                        time_reward = round(-epi_reward * (0.99 ** (epi_step - e)), 4)
                        mask[0, 0, i] = time_reward * time_reward * time_reward
                    else:
                        mask[0, 0, i] = self.small_value
                output.backward(gradient=mask, retain_graph=True)
                optimizer.step()

            rewards.append(epi_reward)
            epi += 1
            steps += epi_step + 1
            pre_epi_step = epi_step
            if epi_reward > 0:
                success += 1

            if epi % 100 == 0:
                print('----------------{', epi, '}-----------------')
                print('processing :', steps)
                print('success rate: ', round((success / 100) * 100, 2), '%')
                print('last 10 rewards :', rewards[-10:])
                print('last 10th mean rewards :', np.sum(rewards[-10:]) / 10)
                print('last 100th mean rewards :', np.sum(rewards[-100:]) / 100)

                if round((success / 100) * 100, 2) > 50:
                    print('early evaluation')
                    torch.save(self.embedding_layer, './custom/trained_model/doorkey_embedding.pt')
                    torch.save(self.decoder, './custom/trained_model/doorkey_rl_trans_decoder.pt')
                    torch.save(self.encoder, './custom/trained_model/doorkey_rl_trans_encoder.pt')

                    self.eval()

                rewards = []
                success = 0

    def eval(self):
        self.embedding_layer.eval()
        self.encoder.eval()
        self.decoder.eval()

        self.goal_info = self.embedding.random_goaL_embedding(goal_object=[8, 1, 0])

        size_step = [(5, 20), (6, 40), (8, 80), (16, 120)]

        for size, step in size_step:
            rewards = []

            for i in range(10):
                env = CustomEmptyEnv(size=size, max_steps=step)
                obs, _ = env.reset()

                env.agent_pos = (rd.choice(range(1, size - 2)),
                                 rd.choice(range(1, size - 2)))
                env.agent_dir = rd.choice(range(4))

                done = False
                while not done:
                    obs = self.embedding.tensor_to_longtensor(obs['image'])
                    obs = self.embedding_layer.forward(obs)
                    obs = obs.reshape([1, 1, self.d_embed])
                    action = self.predict(obs)
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    done = terminated or truncated

                    if done:
                        rewards.append(reward)

                    obs = next_obs

            print(f'empty size {size}x{size} step {step}')
            print(rewards)
            print(np.mean(rewards))
            print(np.std(rewards))

        size_step = [(5, 60), (6, 100), (8, 160), (16, 200)]

        for size, step in size_step:
            rewards = []

            for i in range(10):
                env = CustomDoorKeyEnv(size=size, max_steps=step)
                obs, _ = env.reset()

                env.agent_pos = (rd.choice(range(1, size - 2)),
                                 rd.choice(range(1, size - 2)))
                env.agent_dir = rd.choice(range(4))

                done = False
                while not done:
                    obs = self.embedding.tensor_to_longtensor(obs['image'])
                    obs = self.embedding_layer.forward(obs)
                    obs = obs.reshape([1, 1, self.d_embed])
                    action = self.predict(obs)
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    done = terminated or truncated

                    if done:
                        rewards.append(reward)

                    obs = next_obs

            print(f'empty size {size}x{size} step {step}')
            print(rewards)
            print(np.mean(rewards))
            print(np.std(rewards))

        size_step = [(5, 30), (6, 40), (7, 60)]

        for size, step in size_step:
            env = CustomLavaGapEnv(size=size, max_steps=step)
            rewards = []

            for i in range(10):
                obs, _ = env.reset()

                env.agent_pos = (rd.choice(range(1, size - 2)),
                                 rd.choice(range(1, size - 2)))
                env.agent_dir = rd.choice(range(4))

                done = False
                while not done:
                    obs = self.embedding.tensor_to_longtensor(obs['image'])
                    obs = self.embedding_layer.forward(obs)
                    obs = obs.reshape([1, 1, self.d_embed])
                    action = self.predict(obs)
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    done = terminated or truncated

                    if done:
                        rewards.append(reward)

                    obs = next_obs

            print(f'lava gaps size {size}x{size} step {step}')
            print(rewards)
            print(np.mean(rewards))
            print(np.std(rewards))

        step_row = [(80, 2), (80, 4)]

        for step, row in step_row:
            env = CustomDistShiftEnv(width=10, height=8, strip2_row=row, max_steps=step)
            rewards = []

            for i in range(10):
                obs, _ = env.reset()

                env.agent_pos = (rd.choice(range(1, 3)), rd.choice(range(1, 3)))
                env.agent_dir = rd.choice(range(4))

                done = False
                while not done:
                    obs = self.embedding.tensor_to_longtensor(obs['image'])
                    obs = self.embedding_layer.forward(obs)
                    obs = obs.reshape([1, 1, self.d_embed])
                    action = self.predict(obs)
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    done = terminated or truncated

                    if done:
                        rewards.append(reward)

                    obs = next_obs

            print(f'dist shift size {size}x{size} step {step}')
            print(rewards)
            print(np.mean(rewards))
            print(np.std(rewards))

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)

        return output

    def load(self, embedding_layer_path, encoder_path, decoder_path):
        self.embedding_layer = torch.load(embedding_layer_path)
        self.encoder = torch.load(encoder_path)
        self.decoder = torch.load(decoder_path)