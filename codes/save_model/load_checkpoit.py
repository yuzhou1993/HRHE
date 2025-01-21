import numpy as np
import torch

data = torch.load("checkpoint", map_location=torch.device('cpu'))
print(data)