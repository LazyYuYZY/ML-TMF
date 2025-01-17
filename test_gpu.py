import torch

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())
print(torch.cuda_version)
print(torch.backends.cudnn.version())
