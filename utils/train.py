import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from parso.python.tree import Class
from torch.utils.data import Dataset, DataLoader

from decoder import Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoded = torch.rand(1, 128, 256).to(device)

decoder =Decoder(text_embed_dim=256, depth=3)

decoder.to(device)

output = decoder(encoded)