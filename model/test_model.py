#%%
from U_net import Conv_block,Conv_block_2x,Encoder,Decoder,UNet
import torch
from torchsummary import summary
#%%
model = UNet()
model.to('cuda:0')
summary(model, (17,5000))
input =  torch.rand((10,17,5000)).to('cuda:0')
output = model(input)
print('Done')
# %%
