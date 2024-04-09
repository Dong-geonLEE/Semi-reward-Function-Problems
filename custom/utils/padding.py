import torch
import math
import random as rd

def scene_padding(num_steps, replay_buffer, embedding_length=50, max_steps=100):
    pad = torch.zeros([embedding_length])
    num_padding = max_steps - num_steps

    while num_padding > 0:
        replay_buffer.append(pad)

    return replay_buffer

def unknown_masking(obs, prev_ep_length, type: str):
    if type == 'reward':
        idx = 11

    elif type == 'dropout':
        idx = 0

        pad_num = round(math.log2(prev_ep_length))
        pad_list = rd.choices(population=range(49), k=pad_num)

        for pad in pad_list:
            n = pad // 7
            m = pad % 7

            obs[n][m][0] = idx

    return obs