import torch
import numpy as np


output_data = np.array([0.1,0.2,0.3])
Y = torch.tensor(output_data)
print(f"Y = {Y}, Shape = {Y.shape}")

X = torch.tensor([0.1,0.2,0.3])
print(f"X={X}, Shape = {X.shape}")


