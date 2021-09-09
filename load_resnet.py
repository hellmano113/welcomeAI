import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#下载模型 C:\Users\HaoLu\.cache\torch\hub\checkpoints

inception =models.inception_v3(pretrained=True)

print(inception)