import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from vit import *
x = torch.randn(8, 3, 256, 256)

patch_size_h = 256 # 256 pixels
patch_size_w = 1


model = ViT_Generator(
    image_size= 256,
    patch_size_w=1,
    patch_size_h=256,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
# summary(model, (3,256,256),device='cpu')
preds = model(x)
print(preds.shape)
# print(preds)