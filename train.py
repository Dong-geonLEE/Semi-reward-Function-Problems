import torch

from custom.utils.rl_transformer import RLTransformer

max_step = 20
num_action = 3
env_size = 5
d_embed = 7 ** 2 + 1
num_decoder_layer = 2
num_encoder_layer = 2
small_value = 0.00001
tot_step = 4e5

rl_trans = RLTransformer(d_embed=d_embed, n_head=10, num_action=num_action,
                         num_encoder_layer=num_encoder_layer, num_decoder_layer=num_decoder_layer,
                         env_size=env_size, max_step=max_step, small_value=small_value)

lr = 0.00005
optimizer = torch.optim.SGD(rl_trans.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

print("####################################################")

print("env: custom empty env")
print(f"num encoder: {num_encoder_layer}, num decoder: {num_decoder_layer}")
print(f"num action: {num_action}, env size: {env_size}, lr: {lr}")
print(f"max steps: {max_step}")
print(f"total steps: {tot_step}")

print("####################################################")

# torch.autograd.set_detect_anomaly(True)
rl_trans.train(optimizer, scheduler, tot_steps=tot_step)

rl_trans.eval()