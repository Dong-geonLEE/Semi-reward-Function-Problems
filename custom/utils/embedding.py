import torch
import torch.nn as nn
import numpy as np
import math

# Input Embedding
class ObservationEmbedding:
    # image: Tensor([7, 7, 3])
    # image: ('object_idx', 'color_idx', 'state')
    # agent position at observation: [3][6]
    # position info: (rad, dist) from agent
    # n_embed: 49 (agent_view_size * agent_view_size)
    def __init__(self):
        super().__init__()

        self.pos_dict = self._position_dict()
        self.embedding_vec = torch.zeros(49)

    def _position_dict(self):
        obs_tensor = torch.zeros(7, 7, 1)
        agent_position = [(3,0), (0,3), (3,6), (6,3)]
        pos_dict = {}

        # rotate via agent_direction

        for agent_dir in range(4):
            agent_pos = agent_position[agent_dir]
            pos_list = []
            for n in range(len(obs_tensor)):
                _pos = []
                for m in range(len(obs_tensor[0])):
                    vert = n - agent_pos[0]
                    length = m - agent_pos[1]
                    rad = round(math.atan2(length, vert), 4)
                    dist = round(math.sqrt(abs(vert) + abs(length)), 4)

                    _pos.append([rad, dist])

                pos_list.append(_pos)

            pos_dict[agent_dir] = pos_list

        return pos_dict

    def tensor_to_vector(self, image, agent_dir):
        num_object_idx = 12
        num_color_idx = 6
        num_state_idx = 3

        pos_list = self.pos_dict[2]

        for n in range(len(image)):
            for m in range(len(image[0])):
                max_dist = math.sqrt(6 + 6)
                whe = n * len(image[0]) + m
                ob = image[n][m][0] * 100
                ob += image[n][m][1] * 10
                ob += image[n][m][2]
                ob = ob * pos_list[n][m][0] * (max_dist - pos_list[n][m][1])
                obb = ob / ((num_object_idx + 1) * 100)

                self.embedding_vec[whe] = obb

        return self.embedding_vec

    def tensor_to_longtensor(self, image, reward=0):
        image = image.reshape([49 * 3])
        image = np.append(image, [reward, reward, reward])
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        return image

    def goal_tensor(self):
        goal = torch.zeros([7, 7, 3], dtype=torch.int)
        m = round(7 / 2)
        n = 7 - 1
        goal[m][n] = torch.tensor([8, 1, 0])
        goal = goal.reshape([49 * 3])
        goal = goal.type(torch.FloatTensor)

        return goal

    def random_goaL_embedding(self, num_goal=20, goal_object=[8, 1, 0]):
        goal = torch.zeros([7, 7, 3])
        goal[:, :, 0] = 11

        goal[3, 6] = torch.tensor(goal_object)

        goal = goal.reshape([49 * 3])
        goal = torch.cat([goal, torch.ones([3])])
        goal = goal.reshape([1, 1, 150])

        return goal


    def goal_embedding(self):
        goal = [
            [[11, 0, 0], [11, 0, 0], [11, 0, 0], [11, 0, 0], [11, 0, 0], [11, 0, 0], [11, 0, 0]],
            [[11, 0, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0]],
            [[11, 0, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0]],
            [[11, 0, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0]],
            [[11, 0, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0]],
            [[11, 0, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0], [8, 1, 0]],
            [[11, 0, 0], [11, 0, 0], [11, 0, 0], [11, 0, 0], [11, 0, 0], [11, 0, 0], [11, 0, 0]],
        ]
        goal = self.tensor_to_vector(goal, agent_dir=0)
        return torch.cat([goal, torch.Tensor([1])])

    def make_tgt(self, n_batch, max_step, d_embed):
        goal = torch.zeros([n_batch, max_step, d_embed])

        goal_info = self.goal_embedding()
        for n in range(n_batch):
            for m in range(max_step):
                goal[n][m] = goal_info

        return goal

    def goal_embedding_by_size(self, env_size):
        goal = torch.zeros([env_size, env_size, 3], dtype=torch.int)
        m = round(env_size / 2)
        n = env_size - 1
        goal[m][n] = torch.tensor([1, 0, 0])
        goal[m][n-1] = torch.tensor([8, 1, 0])

        goal = self.tensor_to_vector(goal, agent_dir=0)

        return torch.cat([goal, torch.Tensor([1])])


    def delayed_reward(self, replay_buffer: torch.Tensor, eoe_step: int):
        eoe_reward = replay_buffer[eoe_step][-1].item()

        for step in range(eoe_step):
            reward = round(eoe_reward * (0.99 ** (eoe_step - step)), 4)
            replay_buffer[step][-1] = reward

        return replay_buffer

    def constant_reward(self, replay_buffer: torch.Tensor, eoe_step: int):
        eoe_reward = replay_buffer[eoe_step][-1].item()

        for step in range(eoe_step):
            replay_buffer[step][-1] = eoe_reward

        return replay_buffer

    def loss_delayed_reward(self, replay_buffer: torch.Tensor, idx_buffer, eoe_step: int):
        eoe_reward = replay_buffer[eoe_step][-1].item()

        for step in range(eoe_step):
            idx_loss = idx_buffer[step].item()
            reward = round(eoe_reward * (0.99 ** (eoe_step - step) / max(idx_loss, 0.1)), 4)
            replay_buffer[step][-1] = reward

        return replay_buffer

    def encoder_input(self, image, agent_direction):
        return self.tensor_to_vector(image, agent_direction)

    def decoder_input(self, image, agent_direction, reward):
        observation = self.tensor_to_vector(image, agent_direction)
        reward = torch.Tensor([reward])
        return torch.cat([observation, reward])


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        # embedding vec size = agent view size * agent view size
        self.position_encoding = torch.zeros(50)

    def forward(self, observation_embedding, steps):
        position_encoding = self.position_encoding + 1 - (1 / 500) * steps
        return observation_embedding + position_encoding